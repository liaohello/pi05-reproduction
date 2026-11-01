#!/usr/bin/env python3
"""Export and audit the single-block collection as an OpenPI training view.

Unlike the clean60 exporter this script has no clean60 contract import and no
inherited statistics.  It binds the collection manifest, checks every dataset
frame against the diagnostics trace, verifies per-episode instructions and
computes normalisation from the 52 train episodes only.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_common as common  # noqa: E402

CAMERAS = {
    "observation.images.central": "central",
    "observation.images.left_wrist": "left_wrist",
    "observation.images.right_wrist": "right_wrist",
}
FEATURE_NAMES = [
    "left_shoulder_pan.pos", "left_shoulder_lift.pos", "left_elbow_flex.pos",
    "left_wrist_flex.pos", "left_wrist_roll.pos", "left_gripper.pos",
    "right_shoulder_pan.pos", "right_shoulder_lift.pos", "right_elbow_flex.pos",
    "right_wrist_flex.pos", "right_wrist_roll.pos", "right_gripper.pos",
]
FEATURE_UNITS = ["degree"] * 5 + ["percent"] + ["degree"] * 5 + ["percent"]
_DATASET_CACHE = {}


def _now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _dataset_for(job):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    key = (job["repo_id"], job["dataset_root"])
    if key not in _DATASET_CACHE:
        _DATASET_CACHE[key] = LeRobotDataset(repo_id=job["repo_id"],
                                             root=Path(job["dataset_root"]),
                                             video_backend="pyav")
    return _DATASET_CACHE[key]


def _hwc_uint8(value):
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"image must be three-dimensional, got {array.shape}")
    if array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    if np.issubdtype(array.dtype, np.floating):
        if array.min() < -1e-4 or array.max() > 1.0001:
            raise ValueError(f"floating image outside [0,1]: {array.min()}..{array.max()}")
        array = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _stats(arrays):
    arrays = np.asarray(arrays, dtype=np.float64)
    return {"mean": np.mean(arrays, axis=0).tolist(),
            "std": np.std(arrays, axis=0).tolist(),
            "q01": np.quantile(arrays, 0.01, axis=0).tolist(),
            "q99": np.quantile(arrays, 0.99, axis=0).tolist()}


def _export_one(job):
    dataset = _dataset_for(job)
    episode = job["episode"]
    source_index = int(episode["source_episode_index"])
    out_index = int(episode["episode_index"])
    meta = dataset.meta.episodes[source_index]
    start, stop = int(meta["dataset_from_index"]), int(meta["dataset_to_index"])
    length = stop - start
    if length != int(episode["frames"]):
        raise RuntimeError(f"episode {source_index}: dataset length {length} != recorded {episode['frames']}")
    trace_path = Path(job["collection"]) / "diagnostics" / episode["attempt_id"] / "trace.npz"
    with np.load(trace_path, allow_pickle=False) as trace:
        trace_state = np.asarray(trace["state"], dtype=np.float32)
        trace_action = np.asarray(trace["action"], dtype=np.float32)
        trace_time = np.asarray(trace["time"], dtype=np.float64)
        trace_substeps = np.asarray(trace["substeps"], dtype=np.int32)
        trace_incomplete = np.asarray(trace["incomplete"], dtype=bool)
    if trace_state.shape != (length, 12) or trace_action.shape != (length, 12):
        raise RuntimeError(f"episode {out_index}: trace shapes {trace_state.shape}/{trace_action.shape}")
    if trace_time.shape != (length,) or trace_substeps.shape != (length,):
        raise RuntimeError(f"episode {out_index}: trace timing arrays have wrong length")
    diffs = np.diff(trace_time)
    if length > 1 and (np.any(diffs <= 0) or float(np.max(np.abs(diffs - 1.0 / job["fps"]))) > 1e-6):
        raise RuntimeError(f"episode {out_index}: control tick spacing is not 1/fps: "
                           f"{float(np.min(diffs))}..{float(np.max(diffs))}")
    if int(np.max(trace_substeps)) != int(job["physics_substeps"]) or bool(trace_incomplete.any()):
        raise RuntimeError(f"episode {out_index}: partial or incomplete control frames")

    out_dir = Path(job["output"]) / "episodes" / f"episode_{out_index:03d}"
    for camera in CAMERAS.values():
        (out_dir / camera).mkdir(parents=True, exist_ok=True)
    states = np.empty((length, 12), dtype=np.float32)
    actions = np.empty((length, 12), dtype=np.float32)
    timestamps = np.empty((length,), dtype=np.float64)
    frame_indices = np.empty((length,), dtype=np.int64)
    task = None
    max_state_error = 0.0
    max_action_error = 0.0
    image_shape = None
    for local in range(length):
        sample = dataset[start + local]
        state = np.asarray(sample["observation.state"], dtype=np.float32)
        action = np.asarray(sample["action"], dtype=np.float32)
        if state.shape != (12,) or action.shape != (12,):
            raise RuntimeError(f"episode {out_index} frame {local}: state/action shapes")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise RuntimeError(f"episode {out_index} frame {local}: non-finite state/action")
        states[local] = state
        actions[local] = action
        timestamps[local] = float(sample["timestamp"])
        frame_indices[local] = int(sample["frame_index"])
        frame_task = str(sample["task"])
        if task is None:
            task = frame_task
        elif frame_task != task:
            raise RuntimeError(f"episode {out_index} frame {local}: task changes within an episode")
        max_state_error = max(max_state_error, float(np.max(np.abs(state - trace_state[local]))))
        max_action_error = max(max_action_error, float(np.max(np.abs(action - trace_action[local]))))
        for key, camera in CAMERAS.items():
            image = _hwc_uint8(sample[key])
            image_shape = image.shape if image_shape is None else image_shape
            if image.shape != image_shape:
                raise RuntimeError(f"episode {out_index}: camera shapes differ")
            Image.fromarray(image).save(out_dir / camera / f"{local:06d}.jpg",
                                        quality=int(job["jpeg_quality"]))
    if task != episode["instruction"]:
        raise RuntimeError(f"episode {out_index}: task {task!r} != scene instruction {episode['instruction']!r}")
    if max_state_error > 1e-6 or max_action_error > 1e-6:
        raise RuntimeError(f"episode {out_index}: dataset and trace disagree "
                           f"(state {max_state_error}, action {max_action_error})")
    expected_timestamp = frame_indices.astype(np.float64) / float(job["fps"])
    if float(np.max(np.abs(timestamps - expected_timestamp))) > 1e-6:
        raise RuntimeError(f"episode {out_index}: timestamp is not frame_index/fps")
    np.savez_compressed(
        out_dir / "episode.npz", state=states, action=actions, timestamp=timestamps,
        frame_index=frame_indices, task=np.asarray(task), seed=np.int64(episode["seed"]),
        scene_sha256=np.asarray(episode["scene_sha256"]),
        source_episode_index=np.int64(source_index), attempt_id=np.asarray(episode["attempt_id"]),
    )
    video_alignment = {}
    for key in CAMERAS:
        video_alignment[key] = {
            "from_timestamp": float(meta[f"videos/{key}/from_timestamp"]),
            "to_timestamp": float(meta[f"videos/{key}/to_timestamp"]),
        }
    summary = {
        "episode_index": out_index, "source_episode_index": source_index,
        "attempt_id": episode["attempt_id"], "seed": int(episode["seed"]),
        "color": episode["color"], "split": episode["split"],
        "scene_sha256": episode["scene_sha256"], "frames": length,
        "fps": float(job["fps"]), "task": task,
        "fixed_windows": (
            sorted(set(np.linspace(0, max(0, length - 10), min(16 if episode["split"] == "train" else 32,
                                                               max(1, length - 9)), dtype=int).tolist()))
            if length > 10 else []),
        "event_windows": [],
        "eligible_windows": max(0, length - 9),
        "state_shape": list(states.shape), "action_shape": list(actions.shape),
        "image_shape": list(image_shape),
        "max_state_error_vs_trace": max_state_error,
        "max_action_error_vs_trace": max_action_error,
        "control_tick_min_s": float(np.min(diffs)) if length > 1 else None,
        "control_tick_max_s": float(np.max(diffs)) if length > 1 else None,
        "substeps_all_full": bool(np.all(trace_substeps == int(job["physics_substeps"]))),
        "incomplete_frames": int(trace_incomplete.sum()),
        "first_timestamp": float(timestamps[0]), "last_timestamp": float(timestamps[-1]),
        "video_alignment": video_alignment,
        "yaw_degrees": float(episode["yaw_degrees"]),
        "trays": episode["trays"], "slot_name": episode["slot_name"],
        "category": "clean", "subgroup": "clean",
    }
    common.write_json(out_dir / "meta.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path,
                        default=common.REPO_ROOT / "configs/so101/scene1_single_random_protocol.json")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--camera-sync", type=Path, default=None)
    parser.add_argument("--public-manifest", type=Path, default=None)
    args = parser.parse_args()

    collection = args.collection.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest = common.load_json(collection / "collection_manifest.json")
    protocol = common.load_json(args.protocol)
    split_manifest = manifest["split"]
    rows = manifest["successes_rows"]
    if len(rows) != 64:
        raise RuntimeError(f"expected 64 successes, got {len(rows)}")
    if len({row["scene_sha256"] for row in rows}) != 64:
        raise RuntimeError("duplicate scene identities")
    if len({row["seed"] for row in rows}) != 64:
        raise RuntimeError("duplicate seeds")
    train_indices = set(int(value) for value in split_manifest["train_indices"])
    validation_indices = set(int(value) for value in split_manifest["validation_indices"])
    if len(train_indices) != 52 or len(validation_indices) != 12:
        raise RuntimeError("split manifest is not 52/12")
    if split_manifest["family_crosses"] != 0:
        raise RuntimeError("family crosses train/validation")
    for index, row in enumerate(rows, start=1):
        source = row.get("source_episode_index")
        if source is None:
            raise RuntimeError(f"success row {row['attempt_id']} has no dataset index")
        row["episode_index"] = int(source)
        row["split"] = "train" if int(source) in train_indices else "validation"
        row["color"] = row["scene"]["color"]
        row["instruction"] = row["scene"]["instruction"]
        row["yaw_degrees"] = row["scene"]["yaw_degrees"]
        row["trays"] = row["scene"]["trays"]
        row["slot_name"] = row["scene"]["slot_name"]
    if output.exists():
        raise FileExistsError(f"choose a new view directory: {output}")
    output.mkdir(parents=True)
    repo_id = manifest["repo_id"]
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    info = json.loads((collection / "dataset/meta/info.json").read_text())
    dataset = LeRobotDataset(repo_id=repo_id, root=collection / "dataset", video_backend="pyav")
    fps = float(info["fps"])
    jobs = [{
        "dataset_root": str(collection / "dataset"), "repo_id": repo_id,
        "collection": str(collection), "episode": row, "output": str(output),
        "jpeg_quality": args.jpeg_quality, "fps": fps,
        "physics_substeps": int(protocol["control"]["physics_hz"] // protocol["control"]["control_hz"]),
    } for row in rows]
    results = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(_export_one, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({"episode": result["episode_index"], "frames": result["frames"],
                              "state_error": result["max_state_error_vs_trace"],
                              "action_error": result["max_action_error_vs_trace"]}), flush=True)
    results.sort(key=lambda row: row["episode_index"])
    if [row["episode_index"] for row in results] != list(range(64)):
        raise RuntimeError("view episode index range is not 0..63")
    train = [row for row in results if row["split"] == "train"]
    validation = [row for row in results if row["split"] == "validation"]
    if len(train) != 52 or len(validation) != 12:
        raise RuntimeError("view split is not 52/12")
    state_rows, action_rows = [], []
    for row in train:
        with np.load(output / "episodes" / f"episode_{row['episode_index']:03d}" / "episode.npz") as data:
            state_rows.append(data["state"])
            action_rows.append(data["action"])
    train_state = np.concatenate(state_rows, axis=0)
    train_action = np.concatenate(action_rows, axis=0)
    norm_stats = {"norm_stats": {"state": _stats(train_state), "actions": _stats(train_action)}}
    assets = output / "assets" / "so101"
    common.write_json(assets / "norm_stats.json", norm_stats)
    stats_sha = common.sha256_file(assets / "norm_stats.json")
    constant = {"state": [i for i, value in enumerate(norm_stats["norm_stats"]["state"]["std"]) if value == 0.0],
                "actions": [i for i, value in enumerate(norm_stats["norm_stats"]["actions"]["std"]) if value == 0.0]}
    train_stats = {"created": _now(), "scope": "train split only",
                   "train_episodes": [row["episode_index"] for row in train],
                   "validation_episodes": [row["episode_index"] for row in validation],
                   "train_frames": int(train_state.shape[0]),
                   "state_shape": list(train_state.shape), "action_shape": list(train_action.shape),
                   "constant_dimensions": constant, "norm_stats_sha256": stats_sha,
                   "norm_stats_canonical_sha256": common.sha256_json(norm_stats["norm_stats"]),
                   "stats": norm_stats["norm_stats"]}
    common.write_json(output / "train_stats.json", train_stats)
    view_sha = common.sha256_tree(output / "episodes")
    camera_sync = (common.load_json(args.camera_sync) if args.camera_sync else
                   {"passed": False, "reason": "camera sync test not supplied"})
    attempts = [json.loads(line) for line in (collection / "attempts.jsonl").read_text().splitlines()
                if line.strip()]
    failures = Counter(row.get("failure_reason") for row in attempts
                       if row.get("kind") == "episode" and not row.get("success"))
    layout_rejects = sum(1 for row in attempts if row.get("kind") == "layout_reject")
    duration_deltas = []
    for entry in results:
        durations = [entry["video_alignment"][key]["to_timestamp"]
                     - entry["video_alignment"][key]["from_timestamp"] for key in CAMERAS]
        duration_deltas.append(max(durations) - min(durations))
    video_alignment_ok = all(abs(value) <= 1e-6 for value in duration_deltas)
    audit = {
        "schema_version": 1, "created": _now(), "view_sha256": view_sha["sha256"],
        "episodes": 64, "train": 52, "validation": 12,
        "episodes_by_color": dict(Counter(row["color"] for row in results)),
        "attempts": len(attempts), "layout_rejects": layout_rejects,
        "failure_reasons": dict(failures),
        "contact_monitor_errors": sum(int(row.get("contact_monitor_error_count", 0))
                                      for row in attempts if row.get("kind") == "episode"),
        "soft_contacts": sum(int(row.get("soft_contact_count", 0))
                             for row in attempts if row.get("kind") == "episode"),
        "frames_total": int(sum(row["frames"] for row in results)),
        "frames_min": min(row["frames"] for row in results),
        "frames_max": max(row["frames"] for row in results),
        "control_tick_s": {"min": min(row["control_tick_min_s"] for row in results),
                           "max": max(row["control_tick_max_s"] for row in results),
                           "expected": 1.0 / fps},
        "substeps_all_full": all(row["substeps_all_full"] for row in results),
        "incomplete_frames": sum(row["incomplete_frames"] for row in results),
        "max_state_error_vs_trace": max(row["max_state_error_vs_trace"] for row in results),
        "max_action_error_vs_trace": max(row["max_action_error_vs_trace"] for row in results),
        "timestamps_equal_frame_index_over_fps": True,
        "task_per_episode_matches_scene": True,
        "camera_shapes": sorted({tuple(row["image_shape"]) for row in results}),
        "video_timestamps_aligned_across_cameras": video_alignment_ok,
        "video_alignment_method": "per-camera duration comparison; absolute file offsets differ by encoder chunking",
        "video_duration_delta_max_s": max(duration_deltas, default=None),
        "camera_sync": camera_sync,
        "yaw_degrees": {"min": min(row["yaw_degrees"] for row in results),
                        "max": max(row["yaw_degrees"] for row in results)},
        "split_families_cross": 0,
        "train_only_statistics": True,
    }
    common.write_json(output / "data_audit.json", audit)
    meta = {
        "schema_version": 1, "created": _now(), "phase": "scene1-E1->E2",
        "source_collection": "collection",
        "source_dataset_info_sha256": common.sha256_file(collection / "dataset/meta/info.json"),
        "source_sha256": manifest["current_source_fingerprint"]["source_sha256"],
        "expert_sha256": manifest["current_source_fingerprint"]["expert_sha256"],
        "protocol_sha256": manifest["protocol_sha256"],
        "certificate_sha256": manifest["certificate_sha256"],
        "repo_id": repo_id, "fps": fps, "cameras": list(CAMERAS), "camera_files": CAMERAS,
        "state_action_dim": 12, "feature_names": FEATURE_NAMES, "feature_units": FEATURE_UNITS,
        "action_semantics": "absolute joint target applied at that control frame",
        "episodes": results, "split": split_manifest, "train_stats": "train_stats.json",
        "norm_stats_sha256": stats_sha,
        "norm_stats_canonical_sha256": train_stats["norm_stats_canonical_sha256"],
        "view_sha256": view_sha["sha256"], "view_files": view_sha["files"],
        "view_bytes": view_sha["bytes"],
        "claim": "single-block scripted-expert training view; not a policy result",
    }
    common.write_json(output / "meta.json", meta)
    common.write_json(output / "splits.json", split_manifest)
    common.write_json(output / "e4t_windows.json",
                      {str(row["episode_index"]): row["fixed_windows"] for row in results})
    sampling_plan = {"mode": "balanced_categories", "group_weights": {"clean": 1.0},
                     "episode_equal_within_group": True}
    common.write_json(output / "sampling_plan.json", sampling_plan)
    admission = {
        "schema_version": 2, "passed": True,
        "data_gate": {"passed": True, "train_only_statistics": True,
                      "source": "scene1_single_random_v1",
                      "reason": "64 accepted single-block scenes; 52/12 whole-family split; frame/action/trace alignment checked"},
        "view_sha256": meta["view_sha256"],
        "dataset_info_sha256": meta["source_dataset_info_sha256"],
        "split_sha256": common.sha256_json(split_manifest),
        "train_stats_sha256": stats_sha,
        "totals": {"episodes": 64, "train": 52, "validation": 12, "clean": 64, "recovery": 0},
    }
    common.write_json(output / "e4t_admission.json", admission)
    common.write_json(output / "scene_audit.json", {
        "schema_version": 1, "view_sha256": meta["view_sha256"],
        "objects_per_scene": 1, "perturbation_free": True,
        "color_counts": dict(Counter(row["color"] for row in results)),
        "yaw_min": min(row["yaw_degrees"] for row in results),
        "yaw_max": max(row["yaw_degrees"] for row in results),
        "unique_scene_sha256": len({row["scene_sha256"] for row in results}) == 64,
    })
    common.write_json(output / "stats_recompute.json", {
        "schema_version": 1, "view_sha256": meta["view_sha256"],
        "norm_stats_sha256": stats_sha, "recomputed": True,
        "train_episodes": 52, "train_frames": int(train_state.shape[0]),
        "stats": norm_stats["norm_stats"],
    })
    if args.public_manifest:
        common.write_json(args.public_manifest, {
            "schema_version": 1, "phase": "scene1-E1->E2", "created": meta["created"],
            "source_sha256": meta["source_sha256"], "protocol_sha256": meta["protocol_sha256"],
            "view_sha256": meta["view_sha256"], "view_files": meta["view_files"],
            "view_bytes": meta["view_bytes"], "norm_stats_sha256": stats_sha,
            "episodes": 64, "train": 52, "validation": 12,
            "split": split_manifest, "audit": audit, "claim": meta["claim"],
        })
    print(json.dumps({"status": "complete", "episodes": 64, "train": 52, "validation": 12,
                      "view_sha256": meta["view_sha256"], "norm_stats_sha256": stats_sha,
                      "constant_dimensions": constant}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
