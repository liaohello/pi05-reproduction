#!/usr/bin/env python3
"""E1 -> E2: convert the LeRobot v3 collection into an audited training view.

The external writer produces LeRobot v3.0; the pinned OpenPI reader expects the
older v2.1 layout.  This script reads the v3 dataset through the pinned
``lerobot==0.6.1`` loader, checks every frame against the independent
diagnostics trace, writes JPG frames plus NPZ state/action arrays, and computes
normalisation statistics from the 16 training episodes only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import so101_common as common  # noqa: E402
import clean60_contracts as clean60  # noqa: E402

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


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _to_hwc_uint8(value) -> np.ndarray:
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


_DATASET_CACHE: dict = {}


def _dataset_for(job: dict):
    """Build each LeRobot dataset once per worker; construction dominates the export cost."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    key = (job["repo_id"], job["dataset_root"])
    if key not in _DATASET_CACHE:
        _DATASET_CACHE[key] = LeRobotDataset(repo_id=job["repo_id"], root=Path(job["dataset_root"]),
                                             video_backend="pyav")
    return _DATASET_CACHE[key]


def _export_one(job: dict) -> dict:
    dataset = _dataset_for(job)
    episode = job["episode"]
    source_index = int(episode["source_episode_index"])
    # A merged view may place an episode at a different unified index than its own collection
    # index; writing to the source index would silently overwrite another episode.
    out_index = int(episode.get("episode_index", source_index))
    meta = dataset.meta.episodes[source_index]
    start = int(meta["dataset_from_index"])
    stop = int(meta["dataset_to_index"])
    length = stop - start
    if length != int(episode["frames"]):
        raise RuntimeError(f"episode {source_index}: dataset length {length} != attempt frames {episode['frames']}")

    trace_path = Path(episode.get("diagnostics") or episode["attempt"]["diagnostics"]) / "trace.npz"
    with np.load(trace_path, allow_pickle=False) as trace:
        trace_state = np.asarray(trace["state"], dtype=np.float32)
        trace_action = np.asarray(trace["action"], dtype=np.float32)
    if trace_state.shape != (length, 12) or trace_action.shape != (length, 12):
        raise RuntimeError(f"episode {out_index}: trace shapes {trace_state.shape}/{trace_action.shape}")

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
    image_shapes = None
    for local in range(length):
        sample = dataset[start + local]
        state = np.asarray(sample["observation.state"], dtype=np.float32)
        action = np.asarray(sample["action"], dtype=np.float32)
        if state.shape != (12,) or action.shape != (12,):
            raise RuntimeError(f"episode {out_index} frame {local}: state/action shapes {state.shape}/{action.shape}")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise RuntimeError(f"episode {out_index} frame {local}: non-finite state/action")
        states[local] = state
        actions[local] = action
        timestamps[local] = float(sample["timestamp"])
        frame_indices[local] = int(sample["frame_index"])
        task = sample["task"]
        max_state_error = max(max_state_error, float(np.max(np.abs(state - trace_state[local]))))
        max_action_error = max(max_action_error, float(np.max(np.abs(action - trace_action[local]))))
        for key, camera in CAMERAS.items():
            image = _to_hwc_uint8(sample[key])
            image_shapes = image.shape if image_shapes is None else image_shapes
            if image.shape != image_shapes:
                raise RuntimeError(f"episode {out_index}: camera shapes differ: {image.shape} vs {image_shapes}")
            Image.fromarray(image).save(out_dir / camera / f"{local:06d}.jpg",
                                        quality=int(job["jpeg_quality"]))
    if max_state_error > 1e-6 or max_action_error > 1e-6:
        raise RuntimeError(
            f"episode {out_index}: dataset and diagnostics disagree "
            f"(state {max_state_error}, action {max_action_error})")

    np.savez_compressed(
        out_dir / "episode.npz",
        state=states,
        action=actions,
        timestamp=timestamps,
        frame_index=frame_indices,
        source_episode_index=np.int64(source_index),
        seed=np.int64(episode["seed"]),
        scene_sha256=np.asarray(episode["scene_sha256"]),
        source_sha256=np.asarray(job["source_sha256"]),
        task=np.asarray(task),
    )
    summary = {
        "episode_index": out_index,
        "source_episode_index": source_index,
        "seed": int(episode["seed"]),
        "split": episode["split"],
        "scene_sha256": episode["scene_sha256"],
        "frames": length,
        "fps": float(dataset.meta.fps),
        "task": task,
        "state_shape": list(states.shape),
        "action_shape": list(actions.shape),
        "image_shape": list(image_shapes),
        "state_min": states.min(axis=0).tolist(),
        "state_max": states.max(axis=0).tolist(),
        "action_min": actions.min(axis=0).tolist(),
        "action_max": actions.max(axis=0).tolist(),
        "max_state_error_vs_trace": max_state_error,
        "max_action_error_vs_trace": max_action_error,
        "first_timestamp": float(timestamps[0]),
        "last_timestamp": float(timestamps[-1]),
    }
    common.write_json(out_dir / "meta.json", summary)
    return summary


def _stats(arrays: np.ndarray) -> dict:
    # np.mean/np.std on these 129k-row float32 arrays can accumulate millimetre-
    # scale drift in joint statistics.  Use float64 before reducing.
    arrays = np.asarray(arrays, dtype=np.float64)
    return {
        "mean": np.mean(arrays, axis=0).astype(float).tolist(),
        "std": np.std(arrays, axis=0).astype(float).tolist(),
        "q01": np.quantile(arrays, 0.01, axis=0).astype(float).tolist(),
        "q99": np.quantile(arrays, 0.99, axis=0).astype(float).tolist(),
    }


def public_episode(row: dict, *, require_clean: bool) -> dict:
    """Project an exported episode without dropping its audition provenance."""
    fields = (
        "episode_index", "seed", "scene_sha256", "split", "frames",
        "state_shape", "action_shape", "image_shape",
        "max_state_error_vs_trace", "max_action_error_vs_trace",
    )
    if require_clean:
        fields += (
            "category", "subgroup", "perturbation_frame",
            "eligible_windows", "excluded_cross_event_windows",
            "fixed_windows", "event_windows",
        )
    missing = [field for field in fields if field not in row]
    if missing:
        raise RuntimeError(f"export result lacks metadata fields: {missing}")
    return {field: row[field] for field in fields}


def replay_audit_passed(audit: dict, total: int) -> bool:
    """Accept the two historical replay-report schemas without weakening counts."""
    if "replay" in audit:
        replay = audit.get("replay") or {}
        return bool(audit.get("passed") is True
                    and not audit.get("problems")
                    and replay.get("total") == total
                    and replay.get("passed") == total)
    return bool(audit.get("ok") is True
                and audit.get("replay_total") == total
                and audit.get("replay_passed") == total)


def collection_content_binding(protocol: dict, collection: Path, episodes: list[dict],
                               summary: dict) -> dict:
    """Bind the dataset to the protocol by observed content, not by file bytes.

    Used when the protocol file changed after collection (an amendment to a
    non-governing field).  Every value that actually drove the data is re-derived
    from the collection artifacts, so a semantic drift cannot pass.
    """
    import itertools
    import math

    expected = protocol["collection"]
    layout = protocol["scene_layout"]
    start, stop = [int(v) for v in expected["direct_seed_range"]]
    minimum = float(layout["minimum_pairwise_distance_m"])
    checks: dict[str, object] = {}
    problems: list[str] = []

    seeds = [int(row["seed"]) for row in episodes]
    if not all(start <= seed <= stop for seed in seeds):
        problems.append("episode seed outside the registered collection range")
    checks["seeds_within_registered_range"] = [start, stop]
    if summary["protocol"] != protocol["protocol_id"]:
        problems.append("protocol_id differs from the one recorded by the collector")
    ceiling = summary["maximum_attempts_per_profile"]
    if isinstance(ceiling, dict):
        ceiling = int(ceiling["direct"])
    if int(ceiling) > int(expected["maximum_attempts"]):
        problems.append("attempts ceiling is below what the collector was allowed to use")
    if len(episodes) != int(expected["target_successes"]):
        problems.append("success count differs from the registered target")
    split_rule = protocol["split"] if "split" in protocol else expected["split"]
    checks["split_rule"] = split_rule["rule"]

    distances, perturbed, routes, counts = [], [], set(), set()
    for row in episodes:
        scene_path = Path(row["attempt"]["diagnostics"]) / "scene.json"
        scene = common.load_json(scene_path) if scene_path.is_absolute() else \
            common.load_json(collection / Path(row["attempt"]["diagnostics"]).relative_to(collection))
        objects = scene["objects"]
        counts.add(len(objects))
        routes.update(obj.get("route") for obj in objects)
        if scene.get("extra", {}).get("perturbation") or scene.get("perturbation"):
            perturbed.append(row["seed"])
        points = [tuple(float(v) for v in obj["xy"]) for obj in objects]
        distances.append(min(math.dist(a, b) for a, b in itertools.combinations(points, 2)))
    checks["object_counts"] = sorted(counts)
    checks["routes"] = sorted(routes)
    checks["perturbation_free"] = not perturbed
    checks["minimum_pairwise_distance_m"] = {"observed_min": round(min(distances), 4),
                                            "observed_mean": round(sum(distances) / len(distances), 4),
                                            "observed_max": round(max(distances), 4),
                                            "registered": minimum,
                                            "episodes_below": sum(1 for d in distances if d < minimum)}
    if counts != {int(layout["object_count"])}:
        problems.append("object count differs from the protocol")
    if routes != {"direct"}:
        problems.append("routes are not all direct")
    if perturbed:
        problems.append("perturbation schedule present in supposedly clean scenes")
    if layout.get("perturbations_enabled") is not False:
        problems.append("protocol no longer declares perturbations disabled")
    if checks["minimum_pairwise_distance_m"]["episodes_below"]:
        problems.append("episodes violate the registered minimum separation")
    if sorted({row["seed"] for row in episodes}) != sorted(set(seeds)):
        problems.append("duplicate seeds among accepted episodes")
    governing = {key: value for key, value in sorted({
        "protocol_id": protocol["protocol_id"], "seeds": [start, stop],
        "target": expected["target_successes"], "attempts": expected["maximum_attempts"],
        "split": expected["split"], "success_predicate": expected["success_predicate"],
        "failure_policy": expected["failure_policy"], "repo_id": expected["repo_id"],
        "instruction": protocol["task"]["instruction"], "control": protocol["control"],
        "state": protocol["observation"]["state"], "action": protocol["action"],
        "layout": layout}.items())}
    return {"passed": not problems, "problems": problems, "checks": checks,
            "governing_sha256": common.sha256_json(governing),
            "episodes_bound": len(episodes)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--public-manifest", type=Path, default=None)
    parser.add_argument("--protocol", type=Path, default=common.REPO_ROOT / "configs/so101/protocol.json")
    parser.add_argument("--audit", type=Path, default=None,
                        help="required replay/alignment verification for clean-layout datasets")
    parser.add_argument("--acknowledge-protocol-amendment", default=None,
                        help="disclose a post-collection protocol edit that changed no governing field")
    args = parser.parse_args()

    collection = args.collection.expanduser().resolve()
    output = args.output.expanduser().resolve()
    summary = common.load_json(collection / "summary.json")
    split = common.load_json(collection / "split.json")
    e1 = common.load_json(collection / "e1_manifest.json")
    protocol = common.load_json(args.protocol)
    expected = protocol["collection"]
    total = int(expected["target_successes"])
    expected_train = int(expected["split"]["train"])
    expected_val = int(expected["split"]["validation"])
    if total != expected_train + expected_val:
        raise RuntimeError("protocol split does not sum to its episode target")
    episodes = e1["episodes"]
    if len(episodes) != total:
        raise RuntimeError(f"expected {total} episodes, got {len(episodes)}")
    if len({row["scene_sha256"] for row in episodes}) != total:
        raise RuntimeError("duplicate scene identities in collection")
    clean_layout = protocol.get("scene_layout")
    audit = None
    if clean_layout:
        if args.audit is None:
            raise RuntimeError("clean-layout export requires a physical replay audit")
        audit = common.load_json(args.audit)
        if not replay_audit_passed(audit, total):
            raise RuntimeError("clean-layout collection has not passed complete physical replay")
        binding = collection_content_binding(protocol, collection, episodes, summary)
        if not binding["passed"]:
            raise RuntimeError("content binding failed: " + "; ".join(binding["problems"]))
        if e1["protocol_sha256"] != common.sha256_file(args.protocol):
            if not args.acknowledge_protocol_amendment:
                raise RuntimeError(
                    "collection protocol bytes differ from the selected file; pass "
                    "--acknowledge-protocol-amendment after verifying the amendment touched only "
                    "non-governing fields")
            binding["recorded_protocol_sha256"] = e1["protocol_sha256"]
            binding["live_protocol_sha256"] = common.sha256_file(args.protocol)
            binding["amendment"] = args.acknowledge_protocol_amendment
        else:
            binding["mode"] = "exact_bytes"
            binding["protocol_sha256"] = e1["protocol_sha256"]
    if output.exists():
        raise FileExistsError(f"choose a new training-view directory: {output}")
    output.mkdir(parents=True)

    source_sha = e1["source_fingerprint"]["source_sha256"]
    jobs = [{
        "dataset_root": str(collection / "dataset"),
        "repo_id": summary["repo_id"],
        "episode": episode,
        "output": str(output),
        "jpeg_quality": args.jpeg_quality,
        "source_sha256": source_sha,
    } for episode in episodes]

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
    if [row["episode_index"] for row in results] != list(range(total)):
        raise RuntimeError("training view does not contain the complete episode index range")

    train_ids = [row["episode_index"] for row in results if row["split"] == "train"]
    val_ids = [row["episode_index"] for row in results if row["split"] == "validation"]
    if len(train_ids) != expected_train or len(val_ids) != expected_val:
        raise RuntimeError("split differs from the registered protocol")

    state_rows, action_rows = [], []
    for episode_id in train_ids:
        with np.load(output / "episodes" / f"episode_{episode_id:03d}" / "episode.npz",
                     allow_pickle=False) as data:
            state_rows.append(data["state"])
            action_rows.append(data["action"])
    train_state = np.concatenate(state_rows, axis=0)
    train_action = np.concatenate(action_rows, axis=0)
    norm_stats = {"norm_stats": {"state": _stats(train_state), "actions": _stats(train_action)}}
    assets = output / "assets" / "so101"
    common.write_json(assets / "norm_stats.json", norm_stats)
    stats_sha = common.sha256_file(assets / "norm_stats.json")
    canonical_payload = {
        split: {field: values[field] for field in ("mean", "std", "q01", "q99")}
        for split, values in norm_stats["norm_stats"].items()
    }
    canonical_sha = common.sha256_json(canonical_payload)
    constant_dims = {
        "state": [index for index, std in enumerate(norm_stats["norm_stats"]["state"]["std"])
                  if std == 0.0],
        "actions": [index for index, std in enumerate(norm_stats["norm_stats"]["actions"]["std"])
                    if std == 0.0],
    }
    train_stats = {
        "created": _now(),
        "scope": "train split only",
        "train_episodes": train_ids,
        "validation_episodes": val_ids,
        "train_frames": int(train_state.shape[0]),
        "state_shape": list(train_state.shape),
        "action_shape": list(train_action.shape),
        "constant_dimensions": constant_dims,
        "norm_stats_sha256": stats_sha,
        "norm_stats_canonical_sha256": canonical_sha,
        "stats": norm_stats["norm_stats"],
    }
    common.write_json(output / "train_stats.json", train_stats)

    windows = {}
    if clean_layout:
        for row in results:
            count = max(0, int(row["frames"]) - 9)
            n = 16 if row["split"] == "train" else 64
            selected = sorted(set(np.linspace(0, count - 1, min(n, count), dtype=int).tolist())) if count else []
            row.update(category="clean", subgroup="clean", perturbation_frame=None,
                       eligible_windows=count, excluded_cross_event_windows=0,
                       fixed_windows=selected, event_windows=selected)
            windows[str(row["episode_index"])] = selected

    view_sha = common.sha256_tree(output / "episodes")
    public_episodes = [public_episode(row, require_clean=bool(clean_layout)) for row in results]
    clean_profile = None
    if clean_layout:
        clean_profile = {
            "category": "clean",
            "source": clean60.CLEAN_SOURCE,
            "perturbation_free": True,
            "perturbation_frame_field": "present_and_null",
            "episodes_bound": len(public_episodes),
            "minimum_pairwise_distance_m": float(
                clean_layout["minimum_pairwise_distance_m"]),
            "protocol_binding_sha256": common.sha256_json(binding),
        }
    meta = {
        "schema_version": 2 if clean_layout else 1,
        "created": _now(),
        "phase": "E1->E2",
        "source_collection": str(collection),
        "source_dataset_info_sha256": e1["dataset_info_sha256"],
        "source_sha256": source_sha,
        "expert_sha256": e1["source_fingerprint"]["expert_sha256"],
        "protocol_sha256": e1["protocol_sha256"],
        "repo_id": summary["repo_id"],
        "fps": float(summary["fps"]),
        "cameras": list(CAMERAS),
        "camera_files": CAMERAS,
        "state_action_dim": 12,
        "feature_names": FEATURE_NAMES,
        "feature_units": ["degree"] * 5 + ["percent"] + ["degree"] * 5 + ["percent"],
        "action_semantics": "absolute joint target applied at that control frame",
        "episodes": public_episodes,
        "clean_profile": clean_profile,
        "split": split,
        "train_stats": "train_stats.json",
        "norm_stats_sha256": stats_sha,
        "norm_stats_canonical_sha256": canonical_sha,
        "view_sha256": view_sha["sha256"],
        "view_files": view_sha["files"],
        "view_bytes": view_sha["bytes"],
        "claim": "converted training view; not a new dataset or a policy result",
    }
    common.write_json(output / "meta.json", meta)
    common.write_json(output / "splits.json", split)
    if clean_layout:
        common.write_json(output / "protocol_binding.json", binding)
        common.write_json(output / "e4t_windows.json", windows)
        common.write_json(output / "e4t_admission.json", {
            "schema_version": 2, "passed": True,
            "data_gate": {"passed": True, "train_only_statistics": True,
                          "clean_source": clean60.CLEAN_SOURCE,
                          "replay_audit_sha256": common.sha256_file(args.audit),
                          "reason": "clean collection passed replay, alignment and protocol-bound layout filtering"},
            "view_sha256": meta["view_sha256"], "dataset_sha256": e1["dataset_info_sha256"],
            "split_sha256": common.sha256_json(split), "train_stats_sha256": stats_sha,
            "totals": {"episodes": total, "train": len(train_ids), "validation": len(val_ids),
                       "clean": total, "recovery": 0},
        })
    if args.public_manifest:
        common.write_json(args.public_manifest, {
            "schema_version": 1, "phase": "E1->E2", "created": meta["created"],
            "source_sha256": source_sha, "expert_sha256": meta["expert_sha256"],
            "protocol_sha256": meta["protocol_sha256"],
            "dataset_info_sha256": meta["source_dataset_info_sha256"],
            "view_sha256": meta["view_sha256"], "view_files": meta["view_files"],
            "view_bytes": meta["view_bytes"], "norm_stats_sha256": stats_sha,
            "norm_stats_canonical_sha256": canonical_sha,
            "clean_profile": meta["clean_profile"],
            "episodes": public_episodes, "split": split,
            "train_stats": {key: train_stats[key] for key in
                            ("scope", "train_episodes", "validation_episodes",
                             "train_frames", "constant_dimensions")},
            "claim": meta["claim"],
        })
    print(json.dumps({"phase": "convert", "status": "complete", "episodes": total,
                      "train": len(train_ids), "validation": len(val_ids),
                      "view_sha256": meta["view_sha256"], "norm_stats_sha256": stats_sha,
                      "norm_stats_canonical_sha256": canonical_sha,
                      "constant_dimensions": constant_dims}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
