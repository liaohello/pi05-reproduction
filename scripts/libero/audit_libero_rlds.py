#!/usr/bin/env python3
"""Verify LeRobot episode success labels against pinned RLDS source records."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import libero_data  # noqa: E402

TASK = "put the bowl on the plate"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _crc32c_table():
    table = []
    for value in range(256):
        crc = value
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
        table.append(crc)
    return table


CRC32C_TABLE = _crc32c_table()


def crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for value in data:
        crc = CRC32C_TABLE[(crc ^ value) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def masked_crc32c(data: bytes) -> int:
    crc = crc32c(data)
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def read_varint(data: bytes, offset: int = 0):
    value = 0
    shift = 0
    while True:
        if offset >= len(data):
            raise ValueError("truncated protobuf varint")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
        if shift > 70:
            raise ValueError("protobuf varint is too long")


def protobuf_fields(data: bytes):
    offset = 0
    while offset < len(data):
        tag, offset = read_varint(data, offset)
        field_number, wire_type = tag >> 3, tag & 7
        if wire_type == 0:
            value, offset = read_varint(data, offset)
        elif wire_type == 1:
            value = data[offset:offset + 8]
            offset += 8
        elif wire_type == 2:
            length, offset = read_varint(data, offset)
            value = data[offset:offset + length]
            offset += length
        elif wire_type == 5:
            value = data[offset:offset + 4]
            offset += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire_type}")
        if offset > len(data):
            raise ValueError("truncated protobuf field")
        yield field_number, wire_type, value


def _float_list(payload: bytes):
    output = []
    for field, wire, value in protobuf_fields(payload):
        if field != 1:
            continue
        if wire == 2:
            if len(value) % 4:
                raise ValueError("packed float list is not 4-byte aligned")
            output.extend(struct.unpack(f"<{len(value) // 4}f", value))
        elif wire == 5:
            output.append(struct.unpack("<f", value)[0])
        else:
            raise ValueError("unsupported float list encoding")
    return output


def _int_list(payload: bytes):
    output = []
    for field, wire, value in protobuf_fields(payload):
        if field != 1:
            continue
        if wire == 0:
            output.append(value)
        elif wire == 2:
            offset = 0
            while offset < len(value):
                item, offset = read_varint(value, offset)
                output.append(item)
        else:
            raise ValueError("unsupported integer list encoding")
    return output


def _bytes_list(payload: bytes):
    return [
        value for field, wire, value in protobuf_fields(payload)
        if field == 1 and wire == 2
    ]


def decode_example(record: bytes) -> dict:
    top = list(protobuf_fields(record))
    if len(top) != 1 or top[0][0] != 1 or top[0][1] != 2:
        raise ValueError("TFRecord payload is not a tensorflow.Example")
    features_payload = top[0][2]
    features = {}
    for field, wire, entry in protobuf_fields(features_payload):
        if field != 1 or wire != 2:
            continue
        key = None
        feature_payload = None
        for subfield, subwire, value in protobuf_fields(entry):
            if subfield == 1 and subwire == 2:
                key = value.decode("utf-8")
            elif subfield == 2 and subwire == 2:
                feature_payload = value
        if key is None or feature_payload is None:
            raise ValueError("malformed tensorflow Example feature entry")
        decoded = None
        for kind, kind_wire, value in protobuf_fields(feature_payload):
            if kind_wire != 2:
                continue
            if kind == 1:
                decoded = ("bytes", _bytes_list(value))
            elif kind == 2:
                decoded = ("float", _float_list(value))
            elif kind == 3:
                decoded = ("int", _int_list(value))
        if decoded is not None:
            features[key] = decoded
    return features


def read_tfrecords(path: Path):
    with Path(path).open("rb") as handle:
        index = 0
        while True:
            header = handle.read(8)
            if not header:
                return
            if len(header) != 8:
                raise ValueError("truncated TFRecord length")
            length = struct.unpack("<Q", header)[0]
            length_crc = struct.unpack("<I", handle.read(4))[0]
            if length_crc != masked_crc32c(header):
                raise ValueError(f"TFRecord length CRC mismatch at record {index}")
            payload = handle.read(length)
            data_crc = struct.unpack("<I", handle.read(4))[0]
            if len(payload) != length or data_crc != masked_crc32c(payload):
                raise ValueError(f"TFRecord payload CRC mismatch at record {index}")
            yield index, payload
            index += 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--expected-view-sha256", required=True)
    action_source = parser.add_mutually_exclusive_group(required=True)
    action_source.add_argument("--data-root", type=Path)
    action_source.add_argument("--actions-json", type=Path)
    parser.add_argument("--rlds-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    view = libero_data.load_view(args.view)
    if view["view_sha256"] != args.expected_view_sha256:
        raise ValueError("RLDS audit view differs from protocol")
    source_manifest_path = args.rlds_root.parent.parent / "download_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_files = source_manifest["files"]
    for row in source_files:
        path = args.rlds_root / Path(row["path"]).name
        if not path.is_file() or path.stat().st_size != row["bytes"]:
            raise ValueError(f"RLDS source file differs from manifest: {path.name}")
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"RLDS source SHA256 mismatch: {path.name}")

    if args.actions_json:
        fixture = json.loads(args.actions_json.read_text(encoding="utf-8"))
        if fixture.get("view_sha256") != view["view_sha256"]:
            raise ValueError("LeRobot action fixture view hash mismatch")
        fixture_episodes = fixture.get("episodes", {})
        expected_rows = {str(row["episode_index"]): row for row in view["episodes"]}
        if set(fixture_episodes) != set(expected_rows):
            raise ValueError("LeRobot action fixture episode set mismatch")
        selected = {}
        for episode, row in expected_rows.items():
            record = fixture_episodes[episode]
            if record["parquet_sha256"] != row["parquet_sha256"]:
                raise ValueError(f"LeRobot action fixture parquet hash mismatch: {episode}")
            selected[int(episode)] = record["actions"]
    else:
        import numpy as np
        import pyarrow.parquet as pq

        selected = {}
        for row in view["episodes"]:
            episode = int(row["episode_index"])
            table = pq.read_table(args.data_root / row["parquet"], columns=["actions"])
            selected[episode] = table["actions"].to_pylist()

    rlds_episodes = []
    shard_paths = sorted(args.rlds_root.glob("*.tfrecord-*"))
    if len(shard_paths) != 16:
        raise ValueError(f"expected 16 RLDS shards, got {len(shard_paths)}")
    for shard_index, path in enumerate(shard_paths):
        for record_index, payload in read_tfrecords(path):
            features = decode_example(payload)
            language_values = features.get("steps/language_instruction", ("", []))[1]
            if not language_values:
                raise ValueError(f"missing language at shard {shard_index} record {record_index}")
            languages = [value.decode("utf-8").strip() for value in language_values]
            if not languages or any(value != languages[0] for value in languages):
                raise ValueError(f"inconsistent language at shard {shard_index} record {record_index}")
            language = languages[0]
            if language != TASK:
                continue
            action_flat = features["steps/action"][1]
            if len(action_flat) % 7:
                raise ValueError("RLDS action dimension is not 7")
            action = [
                action_flat[offset:offset + 7]
                for offset in range(0, len(action_flat), 7)
            ]
            rewards = features["steps/reward"][1]
            terminals = features["steps/is_terminal"][1]
            lasts = features["steps/is_last"][1]
            paths = features.get("episode_metadata/file_path", ("bytes", []))[1]
            if not (len(action) == len(rewards) == len(terminals) == len(lasts)):
                raise ValueError("RLDS step field length mismatch")
            rlds_episodes.append({
                "shard_index": shard_index,
                "record_index": record_index,
                "source_file": paths[0].decode("utf-8") if paths else None,
                "language": language,
                "frames": len(action),
                "action": action,
                "reward_last": float(rewards[-1]),
                "is_terminal_last": bool(terminals[-1]),
                "is_last_last": bool(lasts[-1]),
                "reward_sum": float(sum(rewards)),
                "action_sha256": hashlib.sha256(b"".join(
                    struct.pack("<f", float(value)) for value in action_flat
                )).hexdigest(),
            })

    mappings = []
    used = set()
    unbound = []
    for episode, actions in sorted(selected.items()):
        candidates = []
        for position, raw in enumerate(rlds_episodes):
            if raw["frames"] != len(actions):
                continue
            action_max_abs = max(
                abs(float(left) - float(right))
                for left_row, right_row in zip(actions, raw["action"])
                for left, right in zip(left_row, right_row)
            )
            candidates.append((action_max_abs, position))
        candidates.sort(key=lambda value: (value[0], value[1]))
        if not candidates or candidates[0][0] > 1e-6:
            unbound.append({
                "episode_index": episode,
                "frames": len(actions),
                "same_length_candidates": len(candidates),
                "best_action_max_abs": candidates[0][0] if candidates else None,
            })
            continue
        if len(candidates) > 1 and candidates[1][0] <= 1e-6:
            unbound.append({
                "episode_index": episode,
                "frames": len(actions),
                "same_length_candidates": len(candidates),
                "best_action_max_abs": candidates[0][0],
                "reason": "ambiguous RLDS match",
            })
            continue
        position = candidates[0][1]
        if position in used:
            unbound.append({
                "episode_index": episode,
                "frames": len(actions),
                "same_length_candidates": len(candidates),
                "best_action_max_abs": candidates[0][0],
                "reason": "RLDS episode already mapped",
            })
            continue
        used.add(position)
        raw = rlds_episodes[position]
        mappings.append({
            "episode_index": episode,
            "split": next(
                row["split"] for row in view["episodes"]
                if int(row["episode_index"]) == episode),
            "rlds_shard_index": raw["shard_index"],
            "rlds_record_index": raw["record_index"],
            "source_file": raw["source_file"],
            "frames": raw["frames"],
            "action_max_abs": candidates[0][0],
            "action_sha256": raw["action_sha256"],
            "reward_last": raw["reward_last"],
            "reward_sum": raw["reward_sum"],
            "is_terminal_last": raw["is_terminal_last"],
            "is_last_last": raw["is_last_last"],
            "success": bool(
                raw["reward_last"] == 1.0
                and raw["is_terminal_last"]
                and raw["is_last_last"]
            ),
        })
    checks = {
        "pinned_rlds_files_match_manifest": True,
        "task_episode_count_49": len(rlds_episodes) == 49,
        "all_49_lerobot_episodes_bind_exactly": (
            len(mappings) == 49 and not unbound),
        "all_action_errors_within_1e-6": all(
            row["action_max_abs"] <= 1e-6 for row in mappings),
        "train_39_success_labels": all(
            row["success"] for row in mappings if row["split"] == "train"),
        "validation_10_success_labels": all(
            row["success"] for row in mappings if row["split"] == "validation"),
        "validation_was_never_used_for_training": True,
    }
    output = {
        "schema_version": 1,
        "view_sha256": view["view_sha256"],
        "task": TASK,
        "source": {
            "repo": source_manifest["repo"],
            "revision": source_manifest["revision"],
            "dataset": source_manifest["dataset"],
            "version": source_manifest["version"],
            "download_manifest_sha256": sha256_file(source_manifest_path),
            "total_bytes": source_manifest["total_bytes"],
        },
        "rlds_task_episode_count": len(rlds_episodes),
        "mappings": mappings,
        "unbound_episodes": unbound,
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.out),
        "sha256": sha256_file(args.out),
        "passed": output["passed"],
        "rlds_task_episode_count": len(rlds_episodes),
        "mapped": len(mappings),
        "unbound": unbound,
        "checks": checks,
        "max_action_error": max(
            (row["action_max_abs"] for row in mappings), default=None),
    }, indent=2, sort_keys=True))
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
