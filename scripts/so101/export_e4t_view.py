#!/usr/bin/env python3
"""E4-T export: merge 20 audited clean episodes with 5 new clean and 25 recovery episodes.

The unified training view is 50 independent scenes: 25 clean, 25 recovery, 40 train and
10 validation.  Normalisation uses the 40 train episodes only.  Event-crossing windows are
removed from training and are recorded in metadata so the evaluation sampler can inspect
before/after correction segments without giving labels as inputs.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_common as common  # noqa: E402
import export_training_view as old_export  # noqa: E402
import collect_e4t as collector  # noqa: E402
import verify_e4t as audit_module  # noqa: E402

ACTION_HORIZON = 10

# Fields scripts/so101/train_so101.py reads before it accepts an E4-T admission document.
# The third element marks the fields that must literally be true, not merely present.
ADMISSION_CONTRACT = (("passed", None, False), ("data_gate", "passed", True),
                      ("view_sha256", None, False), ("totals", "episodes", False),
                      ("totals", "train", False), ("totals", "validation", False))


def admission_contract_problems(admission: dict) -> list[str]:
    """Names the admission fields the training entry point requires but the document lacks.

    The exporter and the training container never share a Python object, so a key the entry point
    reads has to be produced here; this check runs before the file is written.
    """
    problems = []
    for top, nested, require_true in ADMISSION_CONTRACT:
        value = admission.get(top)
        if nested is None:
            if value is None:
                problems.append(f"missing field '{top}'")
        elif not isinstance(value, dict) or nested not in value:
            problems.append(f"missing field '{top}.{nested}'")
        elif require_true and value[nested] is not True:
            problems.append(f"field '{top}.{nested}' is not true")
    return problems


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def camera_directories(meta: dict) -> list[str]:
    files = meta.get("camera_files")
    return list(files.values()) if isinstance(files, dict) else list(meta["cameras"])


def _eligible_starts(length: int, perturbation_frame: int | None, horizon: int) -> list[int]:
    starts = list(range(max(0, length - horizon + 1)))
    if perturbation_frame is None:
        return starts
    return [frame for frame in starts if not frame <= int(perturbation_frame) <= frame + horizon - 1]


def _evenly_spaced(candidates: list[int], count: int) -> list[int]:
    if not candidates:
        return []
    count = min(int(count), len(candidates))
    return [candidates[round(index * (len(candidates) - 1) / max(1, count - 1))]
            for index in range(count)] if count > 1 else [candidates[0]]


def _stage_frames(events: list[dict], trace_times: np.ndarray) -> list[int]:
    frames = []
    for row in events:
        if row.get("type") not in ("stage", "plan_start", "perturbation", "expert_replan"):
            continue
        stamp = float(row.get("sim_time", row.get("time", 0.0)))
        values = np.asarray(trace_times, dtype=float)
        if values.size:
            index = int(np.searchsorted(values, stamp - 1e-9))
            frames.append(max(0, min(index, len(values) - 1)))
    return frames


def _event_windows(length: int, perturbation_frame: int | None, events: list[dict],
                   trace_times: np.ndarray, starts: list[int]) -> list[int]:
    eligible = set(starts)
    candidates = set()
    if perturbation_frame is not None:
        candidates.update(int(perturbation_frame) + offset for offset in
                          (-10, -5, -1, 1, 2, 5, 10, 20, 40, 80))
    candidates.update(_stage_frames(events, trace_times))
    for offset in (-5, -2, 0, 2, 5, 10, 20, 40, 80):
        candidates.update(max(0, value + offset) for value in _stage_frames(events, trace_times))
    result = sorted(value for value in candidates if value in eligible and 0 <= value < length)
    return result


def _unified_mapping(new_episodes: list[dict], protocol: dict) -> tuple[list[dict], dict]:
    expected = protocol["collection"]["target_successes_per_profile"]
    groups = ["clean_new", "recovery_A", "recovery_B", "recovery_C"]
    rows = []
    next_id = 20
    for group in groups:
        selected = [row for row in new_episodes if row["profile"] == group]
        if len(selected) != expected[group]:
            raise ValueError(f"new group {group} has {len(selected)}, expected {expected[group]}")
        selected.sort(key=lambda row: (row["split"] != "train", row["seed"]))
        for row in selected:
            event = row.get("recovery_audit", {}).get("perturbation_event") or {}
            unified = {
                "unified_episode_index": next_id,
                "source_episode_index": int(row["source_episode_index"]),
                "attempt_id": row["attempt_id"], "profile": row["profile"],
                "category": "clean" if group == "clean_new" else group[-1],
                "subgroup": "clean" if group == "clean_new" else group[-1],
                "mechanism": event.get("mechanism"),
                "seed": int(row["seed"]),
                "scene_sha256": row["scene_sha256"],
                "split": row["split"], "frames": int(row["frames"]),
                "diagnostics": row["diagnostics"],
                "perturbation_frame": event.get("control_frame"),
                "perturbation_substep": event.get("substep"),
                "perturbation_stage": event.get("stage"),
                "source_dataset": "new",
                "source_collection": protocol["reuse"]["old_collection"] if False else "E4-T new collection",
                "source_sha256": protocol["source_patch"]["patched_source_sha256"],
            }
            rows.append(unified)
            next_id += 1
    train = [row["unified_episode_index"] for row in rows if row["split"] == "train"]
    validation = [row["unified_episode_index"] for row in rows if row["split"] == "validation"]
    return rows, {"new_train": train, "new_validation": validation}


def make_admission(*, meta: dict, meta_path: Path, collections: list, new_audits: list,
                   audit_rows: dict, old_replay_record: dict) -> dict:
    """Seal the frozen view into the document the training entry point reads.

    Everything here is derived from already written artifacts, so the same document can also be
    rebuilt later by scripts/so101/e4t_seal_admission.py and compared hash for hash.
    """
    split = meta["split"]
    new_ids = [row["attempt_id"] for row in meta["episodes"] if row.get("source_dataset") == "new"]
    missing = [aid for aid in new_ids if aid not in audit_rows]
    if missing:
        raise RuntimeError(f"new episodes without an audit record: {missing}")
    included = [audit_rows[aid] for aid in new_ids]
    failed = [row["attempt_id"] for row in included if not row["passed"]]
    if failed:
        raise RuntimeError(f"E4-T admission refused, audit-failed episodes are in the view: {failed}")
    if old_replay_record["replayed"] != old_replay_record["episodes"]:
        raise RuntimeError("E4-T admission refused: the reused episodes did not all replay")
    excluded = [row["attempt_id"] for row in audit_rows.values() if row.get("excluded_by_protocol")]
    return {
        "schema_version": 1, "phase": "E4-T-data-gate", "passed": True,
        "data_gate": {
            "passed": True,
            "reason": "every shard audit passed, all 50 episodes have physical replay evidence, "
                      "the split is frozen and the statistics come from the 40 train episodes only",
            "shard_audits": [{"collection": str(path), "audit": str(audit_path),
                              "audit_sha256": common.sha256_file(audit_path)}
                             for path, audit_path in zip(collections, new_audits)],
            "excluded_by_protocol": excluded,
            "new_episode_replay": {
                row["attempt_id"]: {"passed": row["passed"],
                                    "max_state_error": row.get("replay_max_state_error"),
                                    "max_object_error_m": row.get("replay_max_object_error_m")}
                for row in included},
            "old_episode_replay": old_replay_record,
            "train_only_statistics": True,
            "norm_stats_canonical_sha256": meta["norm_stats_canonical_sha256"],
        },
        "created": _now(), "dataset_sha256": meta["new_dataset_sha256"],
        "split_sha256": common.sha256_file(meta_path),
        "view_sha256": meta["view_sha256"], "train_stats_sha256": meta["norm_stats_sha256"],
        "norm_stats_canonical_sha256": meta["norm_stats_canonical_sha256"],
        "totals": {"episodes": len(meta["episodes"]), "train": len(split["train"]),
                   "validation": len(split["validation"]),
                   "clean": split["categories"]["clean"]["total"],
                   "recovery": split["categories"]["recovery"]["total"],
                   "eligible_windows": meta["eligible_windows"]},
        "counts": split, "old_statistics_remain_historical": True,
        "audit_excluded_by_protocol": excluded,
        "recovery_category_note": meta["recovery_category_note"],
        "audit_sha256": meta["audit_sha256"],
    }


def _copy_old_episode(source_root: Path, destination: Path, index: int, row: dict,
                      source_sha: str) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    for item in (source_root / f"episode_{index:03d}").iterdir():
        target = destination / item.name
        if item.is_dir():
            os.system(f"cp -a {str(item)!r} {str(target)!r}")
        else:
            os.system(f"cp -a {str(item)!r} {str(target)!r}")
    with np.load(destination / "episode.npz", allow_pickle=False) as data:
        state = np.asarray(data["state"])
        action = np.asarray(data["action"])
        timestamp = np.asarray(data["timestamp"])
    starts = _eligible_starts(len(state), None, ACTION_HORIZON)
    meta = {
        "episode_index": row["unified_episode_index"], "source_episode_index": row["source_episode_index"],
        "attempt_id": row["attempt_id"], "profile": "clean_reused", "category": "clean",
        "subgroup": "clean", "mechanism": None, "split": row["split"], "seed": int(row["seed"]),
        "scene_sha256": row["scene_sha256"], "frames": int(len(state)), "state_shape": list(state.shape),
        "action_shape": list(action.shape), "image_shape": [480, 640, 3],
        "state_min": state.min(axis=0).tolist(), "state_max": state.max(axis=0).tolist(),
        "action_min": action.min(axis=0).tolist(), "action_max": action.max(axis=0).tolist(),
        "max_state_error_vs_trace": None, "max_action_error_vs_trace": None,
        "replay_evidence": "reused E1 episodes are replayed as a set; the aggregate result is "
                           "recorded once in the view meta under old_episode_replay",
        "first_timestamp": float(timestamp[0]), "last_timestamp": float(timestamp[-1]),
        "perturbation_frame": None, "eligible_windows": len(starts),
        "excluded_cross_event_windows": 0, "fixed_windows": _evenly_spaced(starts, 16 if row["split"] == "train" else 64),
        "event_windows": [], "source_dataset": "old_E1",
        "source_sha256": source_sha,
    }
    common.write_json(destination / "meta.json", meta)
    return meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-view", type=Path, required=True)
    parser.add_argument("--old-manifest", type=Path, required=True)
    parser.add_argument("--old-replay", type=Path, default=None,
                        help="verify_demos.py output for the 20 reused E1 episodes; its aggregate "
                             "replay result is the only replay evidence for those episodes")
    parser.add_argument("--new-collection", type=Path, action="append", required=True,
                        help="repeatable: one directory per parallel collection shard")
    parser.add_argument("--new-audit", type=Path, action="append", required=True)
    parser.add_argument("--protocol", type=Path, default=common.REPO_ROOT / "configs/so101/e4t_protocol.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--public-manifest", type=Path, default=None)
    args = parser.parse_args()

    old_view = args.old_view.expanduser().resolve()
    if len(args.new_collection) != len(args.new_audit):
        raise SystemExit("every --new-collection needs exactly one --new-audit")
    collections = [path.expanduser().resolve() for path in args.new_collection]
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError("choose a new training-view directory")
    protocol = common.load_json(args.protocol)
    old_manifest = common.load_json(args.old_manifest)
    if args.old_replay is None:
        raise RuntimeError("the 20 reused episodes need their replay evidence: pass --old-replay")
    old_replay = common.load_json(args.old_replay.expanduser().resolve())
    if not old_replay.get("ok") or int(old_replay.get("replay_passed", -1)) != 20 \
            or int(old_replay.get("replay_total", -1)) != 20:
        raise RuntimeError(f"reused E1 episodes did not all replay: "
                           f"{old_replay.get('replay_passed')}/{old_replay.get('replay_total')}")
    old_replay_record = {
        "episodes": int(old_replay["replay_total"]), "replayed": int(old_replay["replay_passed"]),
        "frames": int(old_replay.get("frames", 0)),
        "verification_sha256": common.sha256_file(args.old_replay.expanduser().resolve()),
        "expert_source_sha256": old_replay.get("expert_source_sha256"),
        "certificate_hashes": old_replay.get("certificate_hashes"),
    }
    quotas = protocol["collection"]["target_successes_per_profile"]
    shards = []
    for path, audit_path in zip(collections, args.new_audit):
        manifest = common.load_json(path / "e4t_manifest.json")
        audit = common.load_json(audit_path)
        if not audit.get("passed"):
            raise RuntimeError(f"E4-T data audit did not pass for {path}")
        allowed = protocol["source_patch"].get("collection_source_sha256") or [
            protocol["source_patch"]["patched_source_sha256"]]
        if isinstance(allowed, str):
            allowed = [allowed]
        used = manifest["source_fingerprint"]["source_sha256"]
        if used not in allowed:
            raise RuntimeError(f"collection source {used} is not one of the registered snapshot hashes")
        shards.append((path, manifest))
    # Parallel shards can together overshoot a pre-registered category quota. Keep exactly the
    # registered number per profile, chosen deterministically, and record what was set aside.
    ordered_rows = []
    excluded_rows = []
    audit_replay_rows = {}
    for (path, manifest), audit_path in zip(shards, args.new_audit):
        audit_document = common.load_json(audit_path)
        for row_audit in audit_document.get("episodes", []):
            audit_replay_rows[row_audit["attempt_id"]] = row_audit
            if row_audit.get("excluded_by_protocol"):
                excluded_rows.append({"attempt_id": row_audit["attempt_id"],
                                      "collection": str(path),
                                      "reason": row_audit.get("exclusion_reason"),
                                      "problems": row_audit.get("problems", []),
                                      "replay_max_state_error": row_audit.get("replay_max_state_error")})
        passed = {row["attempt_id"] for row in audit_document.get("episodes", [])
                  if row.get("passed") and not row.get("excluded_by_protocol")}
        for row in manifest["episodes"]:
            if row["attempt_id"] not in passed:
                continue  # only episodes the shard audit passed, replay included, may enter the view
            ordered_rows.append((path, manifest, row))
    ordered_rows.sort(key=lambda item: (item[2].get("profile", ""), str(item[2].get("seed", 0)),
                                        item[2]["attempt_id"]))
    kept, excess = [], []
    seen_counts = {name: 0 for name in quotas}
    for path, manifest, row in ordered_rows:
        profile = row.get("profile")
        if profile in seen_counts and seen_counts[profile] >= quotas[profile]:
            excess.append({"attempt_id": row["attempt_id"], "profile": profile,
                           "collection": str(path)})
            continue
        seen_counts[profile] = seen_counts.get(profile, 0) + 1
        kept.append((path, manifest, row))
    registered = audit_module.admitted_mechanisms(protocol)
    for path, manifest, row in kept:
        event = (row.get("recovery_audit") or {}).get("perturbation_event") or {}
        category = (row.get("profile") or "")[-1]
        mechanism = event.get("mechanism")
        if category in registered and mechanism and mechanism not in registered[category]:
            raise RuntimeError(f"episode {row['attempt_id']} uses {mechanism}, which the protocol no longer "
                               f"registers for category {category}")
    if any(seen_counts[name] != quotas[name] for name in quotas):
        raise RuntimeError(f"collected episodes do not meet the category quotas: {seen_counts}")
    new_collection = kept[0][0]
    new_manifest = kept[0][1]
    per_collection = {}
    merged_episodes = []
    for path, manifest, row in kept:
        per_collection.setdefault(str(path), manifest)
        merged_episodes.append({**row, "_collection": str(path)})

    old_meta = common.load_json(old_view / "meta.json")
    # Each shard assigned its own split when it finished; the frozen split must be global, so
    # discard the per-shard labels and re-derive them once over the merged set.
    for row in merged_episodes:
        row.pop("split", None)
    global_split = collector.assign_split(merged_episodes, protocol)
    if global_split["problems"]:
        raise RuntimeError("merged E4-T split invalid: " + "; ".join(global_split["problems"]))
    unified_new, split_new = _unified_mapping(merged_episodes, protocol)
    unified = []
    next_id = 0
    for row in old_meta["episodes"]:
        unified.append({
            "unified_episode_index": next_id, "source_episode_index": int(row["episode_index"]),
            "attempt_id": f"old_E1_{int(row['seed']):04d}", "profile": "clean_reused",
            "category": "clean", "subgroup": "clean", "mechanism": None,
            "split": row["split"], "seed": int(row["seed"]),
            "scene_sha256": row["scene_sha256"], "frames": int(row["frames"]),
            "source_dataset": "old_E1", "source_sha256": old_meta["source_sha256"],
        })
        next_id += 1
    unified.extend(unified_new)
    if len(unified) != 50:
        raise RuntimeError(f"unified view has {len(unified)} episodes, expected 50")
    if [row["unified_episode_index"] for row in unified] != list(range(50)):
        raise RuntimeError("unified episode ids are not contiguous")
    train_ids = [row["unified_episode_index"] for row in unified if row["split"] == "train"]
    val_ids = [row["unified_episode_index"] for row in unified if row["split"] == "validation"]
    if len(train_ids) != 40 or len(val_ids) != 10:
        raise RuntimeError(f"global split is {len(train_ids)}/{len(val_ids)}, expected 40/10")
    if len({row["scene_sha256"] for row in unified}) != 50:
        raise RuntimeError("scene identities are not independent")

    output.mkdir(parents=True)
    (output / "episodes").mkdir()

    copy_jobs = []
    for row in unified:
        if row["source_dataset"] == "old_E1":
            copy_jobs.append(row)
    export_rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        # shutil-like copying with os.system is process-safe per unique destination; workers are
        # deliberately one for filesystem metadata consistency.
        futures = [pool.submit(_copy_old_episode, old_view / "episodes",
                               output / "episodes" / f"episode_{row['unified_episode_index']:03d}",
                               row["source_episode_index"], row, row["source_sha256"]) for row in copy_jobs]
        for future in concurrent.futures.as_completed(futures):
            export_rows.append(future.result())

    repo_ids = {manifest["repo_id"] for _, manifest in shards}
    if len(repo_ids) != 1:
        raise RuntimeError(f"parallel collections must share one repo id, found {sorted(repo_ids)}")
    new_repo_id = repo_ids.pop()
    shard_records = []
    for path, manifest in shards:
        shard_records.append({"collection": str(path),
                              "dataset_sha256": manifest["dataset_sha256"],
                              "episodes": len(manifest["episodes"])})
    composite_dataset_sha = common.sha256_json(sorted(shard_records, key=lambda r: r["collection"]))
    jobs = []
    by_attempt = {row["attempt_id"]: row for row in unified}
    for new_row in merged_episodes:
        row = by_attempt[new_row["attempt_id"]]
        job = {
            "dataset_root": str(Path(new_row["_collection"]) / "dataset"),
            "repo_id": new_repo_id,
            "episode": {
                "episode_index": row["unified_episode_index"],
                "source_episode_index": int(new_row["source_episode_index"]),
                "seed": int(new_row["seed"]), "scene_sha256": new_row["scene_sha256"],
                "split": row["split"], "frames": int(new_row["frames"]),
                "diagnostics": new_row["diagnostics"], "attempt": new_row,
            },
            "output": str(output), "jpeg_quality": args.jpeg_quality,
            "source_sha256": row["source_sha256"],
        }
        jobs.append(job)
    new_rows = []
    # Workers cache the dataset they first open, so submitting a shard's jobs together keeps the
    # number of LeRobot dataset constructions at one per worker per shard.
    jobs.sort(key=lambda job: (job["dataset_root"], int(job["episode"]["source_episode_index"])))
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(old_export._export_one, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(f"exported view episode {result['episode_index']} "
                  f"({result['frames']} frames)", flush=True)
            match = next(row for row in unified if row["unified_episode_index"] == result["episode_index"])
            result["attempt_id"] = match["attempt_id"]
            result["profile"] = match["profile"]
            result["category"] = match["category"]
            result["subgroup"] = match["subgroup"]
            result["mechanism"] = match["mechanism"]
            result["perturbation_frame"] = match["perturbation_frame"]
            new_rows.append(result)
    export_rows.extend(new_rows)
    export_rows.sort(key=lambda row: row["episode_index"])
    if len({int(row["episode_index"]) for row in export_rows}) != len(export_rows):
        raise RuntimeError("two episodes were exported to the same view index")

    # Every episode directory must hold exactly the frames and the scene its metadata claims; this
    # is what catches an index collision between the reused episodes and the new collection.
    for row in export_rows:
        directory = output / "episodes" / f"episode_{int(row['episode_index']):03d}"
        with np.load(directory / "episode.npz", allow_pickle=False) as data:
            if str(data["scene_sha256"]) != row["scene_sha256"]:
                raise RuntimeError(f"view index {row['episode_index']} holds scene "
                                   f"{data['scene_sha256']}, metadata says {row['scene_sha256']}")
            frames = len(np.asarray(data["state"]))
        if frames != int(row["frames"]):
            raise RuntimeError(f"view index {row['episode_index']}: npz has {frames} frames, "
                               f"metadata says {row['frames']}")
        for camera in camera_directories(old_meta):
            written = len(list((directory / camera).glob("*.jpg")))
            if written != frames:
                raise RuntimeError(f"view index {row['episode_index']} camera {camera}: "
                                   f"{written} frames written, {frames} expected")

    # Enrich every episode with window audit and replace exported metadata.
    eval_windows = {}
    for row in export_rows:
        index = int(row["episode_index"])
        unified_row = next(item for item in unified if item["unified_episode_index"] == index)
        directory = output / "episodes" / f"episode_{index:03d}"
        with np.load(directory / "episode.npz", allow_pickle=False) as data:
            length = len(np.asarray(data["state"]))
            times = np.asarray(data["timestamp"], dtype=float)
        events_path = Path(unified_row.get("diagnostics") or "") / "events.jsonl"
        events = []
        if events_path.is_file():
            events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        perturbation_frame = unified_row.get("perturbation_frame")
        starts = _eligible_starts(length, perturbation_frame, ACTION_HORIZON)
        fixed_count = 16 if unified_row["split"] == "train" else 64
        event_windows = _event_windows(length, perturbation_frame, events, times, starts)
        row["attempt_id"] = unified_row["attempt_id"]
        row["profile"] = unified_row["profile"]
        row["category"] = unified_row["category"]
        row["subgroup"] = unified_row["subgroup"]
        row["mechanism"] = unified_row["mechanism"]
        row["perturbation_frame"] = perturbation_frame
        row["perturbation_substep"] = unified_row.get("perturbation_substep")
        row["perturbation_stage"] = unified_row.get("perturbation_stage")
        row["source_dataset"] = unified_row["source_dataset"]
        row["eligible_windows"] = len(starts)
        row["excluded_cross_event_windows"] = length - ACTION_HORIZON + 1 - len(starts) if length >= ACTION_HORIZON else 0
        row["fixed_windows"] = _evenly_spaced(starts, fixed_count)
        row["event_windows"] = sorted(set(event_windows + row["fixed_windows"]))
        common.write_json(directory / "meta.json", row)
        eval_windows[str(index)] = row["event_windows"]

    train_rows = [row for row in export_rows if row["split"] == "train"]
    if len(train_rows) != 40:
        raise RuntimeError("training export lost train episodes")
    state_arrays = []
    action_arrays = []
    for row in train_rows:
        with np.load(output / "episodes" / f"episode_{row['episode_index']:03d}" / "episode.npz",
                     allow_pickle=False) as data:
            state_arrays.append(np.asarray(data["state"]))
            action_arrays.append(np.asarray(data["action"]))
    train_state = np.concatenate(state_arrays, axis=0)
    train_action = np.concatenate(action_arrays, axis=0)
    norm_stats = {"norm_stats": {"state": old_export._stats(train_state),
                                 "actions": old_export._stats(train_action)}}
    assets = output / "assets" / "so101"
    common.write_json(assets / "norm_stats.json", norm_stats)
    canonical_payload = {
        key: {field: values[field] for field in ("mean", "std", "q01", "q99")}
        for key, values in norm_stats["norm_stats"].items()}
    canonical_sha = common.sha256_json(canonical_payload)
    train_stats = {
        "created": _now(), "scope": "40 train episodes only",
        "train_episodes": train_ids, "validation_episodes": val_ids,
        "train_frames": int(train_state.shape[0]),
        "state_shape": list(train_state.shape), "action_shape": list(train_action.shape),
        "norm_stats_sha256": common.sha256_file(assets / "norm_stats.json"),
        "norm_stats_canonical_sha256": canonical_sha, "stats": norm_stats["norm_stats"],
        "old_statistics_remain_historical": True,
    }
    common.write_json(output / "train_stats.json", train_stats)

    eligible_total = sum(int(row["eligible_windows"]) for row in export_rows)
    view_hash = common.sha256_tree(output / "episodes")
    split = {"rule": protocol["split"]["rule"], "train": train_ids, "validation": val_ids,
             "categories": {
                 "clean": {"total": sum(row["category"] == "clean" for row in export_rows),
                           "train": sum(row["split"] == "train" and row["category"] == "clean" for row in export_rows),
                           "validation": sum(row["split"] == "validation" and row["category"] == "clean" for row in export_rows)},
                 "recovery": {"total": sum(row["category"] != "clean" for row in export_rows),
                              "train": sum(row["split"] == "train" and row["category"] != "clean" for row in export_rows),
                              "validation": sum(row["split"] == "validation" and row["category"] != "clean" for row in export_rows)},
             },
             "perturbation_categories": {
                 key: {"train": sum(row["split"] == "train" and row["subgroup"] == key for row in export_rows),
                       "validation": sum(row["split"] == "validation" and row["subgroup"] == key for row in export_rows)}
                 for key in ("A", "B", "C")},
             "mechanisms": {
                 key: sum(row.get("mechanism") == key for row in export_rows)
                 for key in sorted({row["mechanism"] for row in export_rows if row.get("mechanism")})}}
    meta = {
        "schema_version": 2, "phase": "E4-T", "created": _now(),
        "dataset_label": "so101_e4t_50", "repo_id": new_repo_id,
        "claim": "50 independent scripted-expert episodes (25 clean, 25 recovery) for offline training",
        "protocol_sha256": common.sha256_file(args.protocol),
        "source_sha256": kept[0][1]["source_fingerprint"]["source_sha256"],
        "expert_sha256": kept[0][1]["source_fingerprint"]["expert_sha256"],
        "patch_sha256": kept[0][1]["patch_sha256"],
        "collections": shard_records, "collected_but_not_used": excess,
        "old_episode_replay": old_replay_record,
        "audit_excluded_by_protocol": excluded_rows,
        "recovery_category_note": protocol["perturbations"].get(
            "mechanism_amendment_v21", {}).get("decision", "category B contributes zero episodes this round"),
        "old_source_sha256": old_manifest["source_fingerprint"]["source_sha256"],
        "new_dataset_sha256": composite_dataset_sha,
        "audit_sha256": common.sha256_json([common.sha256_file(path) for path in args.new_audit]),
        "old_view_sha256": old_meta["view_sha256"],
        "view_sha256": view_hash["sha256"], "view_files": view_hash["files"], "view_bytes": view_hash["bytes"],
        "total_episodes": len(export_rows), "eligible_windows": eligible_total,
        "excluded_cross_event_windows": sum(int(row["excluded_cross_event_windows"]) for row in export_rows),
        "split": split, "episodes": export_rows, "feature_names": old_meta["feature_names"],
        "cameras": old_meta["cameras"], "camera_files": old_meta["camera_files"],
        "state_action_dim": 12, "fps": 50, "action_horizon": ACTION_HORIZON,
        "action_semantics": old_meta["action_semantics"], "feature_units": old_meta["feature_units"],
        "norm_stats_sha256": train_stats["norm_stats_sha256"],
        "norm_stats_canonical_sha256": canonical_sha,
        "old_statistics_remain_historical": True,
    }
    common.write_json(output / "meta.json", meta)
    common.write_json(output / "e4t_windows.json", eval_windows)
    audit_pass_rows = [audit_replay_rows[row["attempt_id"]] for _, _, row in kept
                       if row["attempt_id"] in audit_replay_rows]
    admission = make_admission(meta=meta, meta_path=output / "meta.json",
                               collections=collections, new_audits=args.new_audit,
                               audit_rows=audit_replay_rows,
                               old_replay_record=old_replay_record)
    problems = admission_contract_problems(admission)
    if problems:
        raise RuntimeError("E4-T admission is not accepted by the training entry point: "
                           + "; ".join(problems))
    common.write_json(output / "e4t_admission.json", admission)
    if args.public_manifest:
        common.write_json(args.public_manifest, {
            "schema_version": 1, "phase": "E4-T-train-view", "created": meta["created"],
            "view_sha256": meta["view_sha256"], "source_sha256": meta["source_sha256"],
            "expert_sha256": meta["expert_sha256"], "patch_sha256": meta["patch_sha256"],
            "new_dataset_sha256": meta["new_dataset_sha256"], "audit_sha256": meta["audit_sha256"],
            "norm_stats_canonical_sha256": canonical_sha,
            "totals": {"episodes": 50, "train": len(train_ids), "validation": len(val_ids),
                       "clean": 25, "recovery": 25, "eligible_windows": eligible_total,
                       "excluded_cross_event_windows": meta["excluded_cross_event_windows"]},
            "split": split, "recovery_category_note": meta["recovery_category_note"],
            "audit_excluded_by_protocol": [row["attempt_id"] for row in excluded_rows],
            "episodes": [{
                key: row.get(key) for key in ("episode_index", "source_episode_index", "attempt_id",
                                              "profile", "category", "mechanism", "split", "seed",
                                              "scene_sha256", "frames", "eligible_windows",
                                              "excluded_cross_event_windows", "perturbation_frame")}
                for row in export_rows],
            "claim": meta["claim"],
        })
    print(json.dumps({"episodes": len(export_rows), "train": len(train_ids),
                      "validation": len(val_ids), "eligible_windows": eligible_total,
                      "excluded_cross_event_windows": meta["excluded_cross_event_windows"],
                      "view_sha256": meta["view_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
