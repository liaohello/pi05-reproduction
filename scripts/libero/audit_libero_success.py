#!/usr/bin/env python3
"""Bind selected LeRobot episodes to raw LIBERO demos and success flags."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import libero_data  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--expected-view-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--hdf5-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import h5py
    import numpy as np
    import pyarrow.parquet as pq

    view = libero_data.load_view(args.view)
    if view["view_sha256"] != args.expected_view_sha256:
        raise ValueError("success audit view differs from protocol")
    candidates = sorted(args.hdf5_root.rglob("put_the_bowl_on_the_plate_demo.hdf5"))
    if len(candidates) != 1:
        raise ValueError(f"expected one raw task HDF5, found {len(candidates)}")
    hdf5_path = candidates[0]
    selected_episodes = sorted(int(row["episode_index"]) for row in view["episodes"])
    selected_actions = {}
    for row in view["episodes"]:
        episode = int(row["episode_index"])
        table = pq.read_table(args.data_root / row["parquet"], columns=["actions"])
        selected_actions[episode] = np.asarray(
            table["actions"].to_pylist(), dtype=np.float32)

    raw = []
    with h5py.File(hdf5_path, "r") as handle:
        for name in sorted(handle["data"].keys(), key=lambda value: int(value.split("_")[1])):
            group = handle["data"][name]
            actions = np.asarray(group["actions"], dtype=np.float32)
            dones = np.asarray(group["dones"], dtype=np.uint8)
            rewards = np.asarray(group["rewards"], dtype=np.uint8)
            if actions.shape[1:] != (7,) or dones.shape != (len(actions),) or rewards.shape != (len(actions),):
                raise ValueError(f"{name} raw action/status shape mismatch")
            raw.append({
                "demo": name,
                "demo_index": int(name.split("_")[1]),
                "frames": len(actions),
                "actions": actions,
                "done_last": int(dones[-1]),
                "reward_last": int(rewards[-1]),
                "done_any": bool(dones.any()),
                "reward_sum": int(rewards.sum()),
                "success": bool(dones[-1] and rewards[-1]),
                "action_sha256": hashlib.sha256(
                    actions.astype("<f4").tobytes()).hexdigest(),
            })

    mappings = []
    unbound = []
    used_demos = set()
    for episode in selected_episodes:
        actions = selected_actions[episode]
        matches = []
        for row in raw:
            if row["frames"] != len(actions):
                continue
            max_abs = float(np.max(np.abs(actions - row["actions"]))) if len(actions) else 0.0
            matches.append((max_abs, row))
        if not matches:
            unbound.append({
                "episode_index": episode,
                "frames": len(actions),
                "same_length_candidates": 0,
                "reason": "no_same_length_raw_demo",
            })
            continue
        matches.sort(key=lambda value: (value[0], value[1]["demo_index"]))
        max_abs, raw_row = matches[0]
        second = matches[1][0] if len(matches) > 1 else None
        if max_abs > 1e-6:
            unbound.append({
                "episode_index": episode,
                "frames": len(actions),
                "same_length_candidates": len(matches),
                "best_raw_demo": raw_row["demo"],
                "best_action_max_abs": max_abs,
                "second_best_action_max_abs": second,
                "reason": "no_exact_same_length_action_match",
            })
            continue
        if second is not None and second <= 1e-6:
            unbound.append({
                "episode_index": episode,
                "frames": len(actions),
                "same_length_candidates": len(matches),
                "best_raw_demo": raw_row["demo"],
                "best_action_max_abs": max_abs,
                "second_best_action_max_abs": second,
                "reason": "ambiguous_exact_action_match",
            })
            continue
        if raw_row["demo_index"] in used_demos:
            unbound.append({
                "episode_index": episode,
                "frames": len(actions),
                "same_length_candidates": len(matches),
                "best_raw_demo": raw_row["demo"],
                "best_action_max_abs": max_abs,
                "second_best_action_max_abs": second,
                "reason": "raw_demo_already_mapped",
            })
            continue
        used_demos.add(raw_row["demo_index"])
        mappings.append({
            "episode_index": episode,
            "split": next(
                row["split"] for row in view["episodes"]
                if int(row["episode_index"]) == episode),
            "raw_demo": raw_row["demo"],
            "raw_demo_index": raw_row["demo_index"],
            "frames": raw_row["frames"],
            "action_max_abs": max_abs,
            "second_best_action_max_abs": second,
            "done_last": raw_row["done_last"],
            "reward_last": raw_row["reward_last"],
            "done_any": raw_row["done_any"],
            "reward_sum": raw_row["reward_sum"],
            "success": raw_row["success"],
        })

    unmapped = [
        {
            key: value for key, value in row.items() if key != "actions"
        }
        for row in raw if row["demo_index"] not in used_demos
    ]
    checks = {
        "raw_demo_count_50": len(raw) == 50,
        "selected_episode_count_49": len(mappings) == 49,
        "unique_raw_demo_per_episode": len(used_demos) == 49,
        "all_49_episodes_bind_exactly": len(mappings) == 49 and not unbound,
        "all_actions_bind_within_1e-6": all(
            row["action_max_abs"] <= 1e-6 for row in mappings),
        "all_selected_demos_terminal_success": (
            len(mappings) == 49 and all(row["success"] for row in mappings)),
        "train_39_all_success": all(
            row["success"] for row in mappings if row["split"] == "train"),
        "validation_10_all_success": all(
            row["success"] for row in mappings if row["split"] == "validation"),
    }
    output = {
        "schema_version": 1,
        "view_sha256": view["view_sha256"],
        "task": view["task"],
        "raw_hdf5": hdf5_path.name,
        "raw_hdf5_sha256": sha256_file(hdf5_path),
        "raw_demo_count": len(raw),
        "selected_episode_count": len(mappings),
        "unmapped_raw_demos": unmapped,
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
        "checks": checks,
        "unmapped": unmapped,
        "mapping_preview": mappings[:3],
        "max_binding_error": max(row["action_max_abs"] for row in mappings),
    }, indent=2, sort_keys=True))
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
