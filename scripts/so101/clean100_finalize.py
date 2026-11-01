#!/usr/bin/env python3
"""Seal the clean100 preparation round with a manifest, SHA256 table and result.json."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import clean60_contracts as clean60  # noqa: E402
import clean100_checks  # noqa: E402
import so101_common as common  # noqa: E402

CAMERAS = ("central", "left_wrist", "right_wrist")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rate_summary(path: Path) -> dict:
    if not path.is_file():
        return {"samples": 0}
    rates, free = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        for token in line.replace(";", " ").split():
            if token.startswith("rate:") and token.endswith("s/it"):
                try:
                    rates.append(float(token[5:-4]))
                except ValueError:
                    pass
    return {
        "samples": len(rates),
        "min": min(rates) if rates else None,
        "median": statistics.median(rates) if rates else None,
        "max": max(rates) if rates else None,
        "raw_file": path.name,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--old-view", type=Path, required=True)
    parser.add_argument("--new-collection", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--registered-training-config", type=Path, default=None,
                        help="registered runtime bindings for the shared pi05 base checkpoint")
    args = parser.parse_args()
    view = args.view.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    old_view = args.old_view.expanduser().resolve()
    collection = args.new_collection.expanduser().resolve()
    protocol = common.load_json(args.protocol)
    protocol_sha = sha256_file(args.protocol)
    meta = common.load_json(view / "meta.json")
    binding = common.load_json(view / "protocol_binding.json")
    splits = common.load_json(view / "splits.json")
    admission = common.load_json(view / "e4t_admission.json")
    train_stats = common.load_json(view / "train_stats.json")
    scene_audit = common.load_json(view / "scene_audit.json")
    data_audit = common.load_json(view / "data_audit.json")
    stats_recompute = common.load_json(view / "stats_recompute.json")
    norm_audit = common.load_json(view / "action_norm_audit.json")
    metadata_audit = common.load_json(view / "metadata_audit.json")
    merge_report = common.load_json(view / "merge_report.json")
    merge_lineage = common.load_json(view / "merge_lineage.json")
    old_meta = common.load_json(old_view / "meta.json")
    norm_file_sha = sha256_file(view / "assets/so101/norm_stats.json")
    train_ids = sorted(int(row["episode_index"]) for row in meta["episodes"]
                       if row["split"] == "train")
    val_ids = sorted(int(row["episode_index"]) for row in meta["episodes"]
                     if row["split"] == "validation")
    attempts = data_audit.get("attempts") or {}
    scene_summary = scene_audit.get("summary") or {}
    expected_attempts = (int(attempts.get("total", -1)),
                         int(attempts.get("successes", -1)),
                         int(attempts.get("failures", -1)))
    problems = []
    problems += clean60.view_contract_problems(
        meta, protocol_binding=binding, expected_counts=(100, 80, 20), require_clean=True)
    problems += clean60.admission_contract_problems(
        admission, meta=meta, split=splits,
        train_stats_sha256=sha256_file(view / "train_stats.json"))
    problems += clean60.scene_audit_problems(
        scene_audit, view_sha256=meta["view_sha256"], expected_counts=(100, 80, 20),
        expected_attempts=expected_attempts,
        scene_key="all_100_scenes_reconstructed_hash_exact")
    problems += clean60.data_audit_problems(
        data_audit, view_sha256=meta["view_sha256"], expected_counts=(100, 80, 20),
        expected_attempts=expected_attempts)
    problems += clean60.stats_recompute_problems(
        stats_recompute, norm_stats_sha256=norm_file_sha,
        view_sha256=meta["view_sha256"], expected_train_ids=train_ids,
        expected_validation_ids=val_ids)
    problems += clean60.norm_audit_problems(
        norm_audit, view_sha256=meta["view_sha256"],
        norm_stats_sha256=norm_file_sha,
        base_checkpoint_sha256=norm_audit.get("base_checkpoint_sha256"),
        training_config_sha256=protocol_sha,
        expected_train=80, expected_validation=20)
    if args.registered_training_config is not None:
        registered = common.load_json(args.registered_training_config)
        expected_base = (registered.get("runtime_binding") or {}).get("base_checkpoint_sha256")
        if expected_base != norm_audit.get("base_checkpoint_sha256"):
            problems.append("normalization audit did not hash the registered base checkpoint")
    if metadata_audit.get("passed") is not True:
        problems.append("service metadata audit did not pass")
    if merge_report.get("passed") is not True:
        problems.append("merge report did not pass")
    if merge_report.get("unique_layouts") != 100:
        problems.append("the four-cube layouts are not 100 fully distinct scenes")
    if len(scene_summary.get("near_duplicate_families_crossing_split") or []) != 0:
        problems.append("scene families cross the split")
    if scene_summary.get("unique_scene_hashes") != 100:
        problems.append("scene hashes are not 100 unique identities")
    if merge_lineage.get("preserved_view_sha256") != old_meta.get("view_sha256"):
        problems.append("preserved view lineage hash mismatch")
    if splits.get("preserved", {}).get("train") != list(range(48)) or \
            splits.get("preserved", {}).get("validation") != list(range(48, 60)):
        problems.append("the preserved 48/12 split is not the original identity set")
    if [int(row["seed"]) for row in meta["episodes"][:60]] != [
            int(row["seed"]) for row in old_meta["episodes"]]:
        problems.append("the preserved episodes were reordered")
    if meta["episodes"][0]["scene_sha256"] != old_meta["episodes"][0]["scene_sha256"]:
        problems.append("the preserved scene identities changed")
    start, stop = (int(value) for value in protocol["collection"]["direct_seed_range"])
    new_seeds = [int(row["seed"]) for row in meta["episodes"][60:]]
    if not all(start <= seed <= stop for seed in new_seeds):
        problems.append("new seeds fall outside the registered block")
    if len(set(new_seeds)) != 40:
        problems.append("new seeds are not 40 unique values")

    scene_records = {int(row["episode_index"]): row for row in scene_audit.get("records", [])}
    manifest_rows = []
    for row in sorted(meta["episodes"], key=lambda item: item["episode_index"]):
        episode_id = int(row["episode_index"])
        directory = view / "episodes" / f"episode_{episode_id:03d}"
        with np.load(directory / "episode.npz", allow_pickle=False) as data:
            state = np.asarray(data["state"])
            action = np.asarray(data["action"])
            timestamp = np.asarray(data["timestamp"])
            frame_index = np.asarray(data["frame_index"])
            source_index = int(data["source_episode_index"])
            task = str(data["task"])
            npz_seed = int(data["seed"])
            npz_scene = str(data["scene_sha256"])
        frames = int(row["frames"])
        camera_counts = {
            camera: sum(1 for path in (directory / camera).iterdir() if path.is_file())
            for camera in CAMERAS}
        scene = scene_records.get(episode_id, {})
        entry_problems = []
        if state.shape != action.shape or state.ndim != 2 or state.shape[1] != 12:
            entry_problems.append("state/action shape")
        if not clean100_checks.all_finite(state.tolist()) or \
                not clean100_checks.all_finite(action.tolist()):
            entry_problems.append("non-finite state/action")
        if not clean100_checks.timestamps_nominal(timestamp, frames):
            entry_problems.append("timestamp grid")
        if not clean100_checks.frame_index_continuous(frame_index, frames):
            entry_problems.append("frame index")
        if any(count != frames for count in camera_counts.values()):
            entry_problems.append("camera file coverage")
        if task != protocol["task"]["instruction"]:
            entry_problems.append("task instruction")
        if npz_seed != int(row["seed"]):
            entry_problems.append("npz seed")
        if npz_scene != str(row["scene_sha256"]):
            entry_problems.append("npz scene hash")
        manifest_rows.append({
            "episode_index": episode_id,
            "seed": int(row["seed"]),
            "split": row["split"],
            "scene_sha256": row["scene_sha256"],
            "frames": frames,
            "source_collection": row.get("source_collection"),
            "source_episode_index": source_index,
            "npz_sha256": sha256_file(directory / "episode.npz"),
            "state_shape": list(state.shape),
            "action_shape": list(action.shape),
            "timestamp_first": float(timestamp[0]) if frames else None,
            "timestamp_last": float(timestamp[-1]) if frames else None,
            "frame_index_first": int(frame_index[0]) if frames else None,
            "frame_index_last": int(frame_index[-1]) if frames else None,
            "camera_file_counts": camera_counts,
            "object_positions_xy_m": scene.get("positions_xy_m"),
            "pairwise_distances_m": scene.get("pairwise_distances_m"),
            "minimum_pairwise_distance_m": scene.get("minimum_pairwise_distance_m"),
            "problems": entry_problems,
        })
        if entry_problems:
            problems.append(f"episode {episode_id}: {entry_problems}")
    manifest = {
        "schema_version": 1,
        "view": view.name,
        "view_sha256": meta["view_sha256"],
        "episodes": manifest_rows,
        "counts": {"episodes": 100, "train": 80, "validation": 20,
                   "preserved": 60, "new": 40},
        "seed_block": [start, stop],
    }
    common.write_json(view / "episode_manifest.json", manifest)

    files = {
        "meta": view / "meta.json",
        "splits": view / "splits.json",
        "protocol_binding": view / "protocol_binding.json",
        "admission": view / "e4t_admission.json",
        "train_stats": view / "train_stats.json",
        "norm_stats": view / "assets/so101/norm_stats.json",
        "scene_audit": view / "scene_audit.json",
        "data_audit": view / "data_audit.json",
        "stats_recompute": view / "stats_recompute.json",
        "action_norm_audit": view / "action_norm_audit.json",
        "metadata_audit": view / "metadata_audit.json",
        "merge_report": view / "merge_report.json",
        "merge_lineage": view / "merge_lineage.json",
        "episode_manifest": view / "episode_manifest.json",
        "protocol": args.protocol,
    }
    hashes = {name: sha256_file(path) for name, path in files.items()}
    hashes["view_tree"] = meta["view_sha256"]
    resource = {
        "collection_egl_device": 2,
        "gpu_compute_used": False,
        "cuda_memory_allocated": False,
        "training_container_observed": {
            "container": "so101-e4t-resume-clean60-base-25k-save3000-20261028T193857Z-791dc957",
            "present_and_running_at_collection_start": True,
            "step_at_collection_start": 5904,
            "rate_at_collection_start_s_per_it": 5.5,
            "disappeared_utc": "2026-10-29T08:18:23Z",
            "removal": "external to this round: this round only read docker logs and nvidia-smi; "
                       "it never issued docker stop/rm/kill or a signal to the training process",
        },
        "training_rate_baseline_before_collection": {
            "median": 5.5, "observed_range": [4.9, 5.8],
            "samples": "six 60-second samples before the collection",
            "note": "natural variation already spanned 4.9-5.8 s/it; no throttling or memory pressure",
        },
        "training_rate_during_collection": rate_summary(
            Path(args.log_dir) / "../collection_monitor.log"),
        "training_rate_during_pilot": rate_summary(
            Path(args.log_dir) / "../pilot_monitor/training_rate.log"),
        "note": "collection renders offscreen on EGL device 2 and never allocates CUDA compute; "
                "the pre-existing training container was running at collection start and was "
                "removed externally at 2026-10-29T08:18:23Z, after which the GPUs showed no "
                "training process; this round issued no docker stop/rm or process signal",
    }
    result = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "READY" if not problems else "BLOCKED",
        "mode": "PREPARE_ONLY",
        "formal_training_performed": False,
        "clean100_training_started": False,
        "blockers": problems,
        "data": {
            "view": view.name,
            "view_sha256": meta["view_sha256"],
            "episodes": 100, "train": 80, "validation": 20,
            "preserved_episodes": 60, "preserved_train": 48,
            "preserved_validation": 12,
            "new_episodes": 40, "new_train": 32, "new_validation": 8,
            "seed_block": [start, stop],
            "scene_hashes_unique": scene_summary.get("unique_scene_hashes"),
            "families": scene_summary.get("distinct_scene_families"),
            "families_crossing_split": scene_summary.get(
                "near_duplicate_families_crossing_split"),
            "attempts": {"total": expected_attempts[0],
                         "successes": expected_attempts[1],
                         "failures": expected_attempts[2],
                         "by_collection": attempts.get("by_collection"),
                         "failure_reasons": attempts.get("failure_reasons")},
            "minimum_pairwise_distance_m": min(
                row["minimum_pairwise_distance_m"] for row in manifest_rows
                if row.get("minimum_pairwise_distance_m") is not None),
        },
        "bindings": {
            "sha256": hashes,
            "protocol_sha256": protocol_sha,
            "base_checkpoint_sha256": norm_audit.get("base_checkpoint_sha256"),
            "norm_stats_sha256": norm_file_sha,
            "norm_stats_canonical_sha256": meta.get("norm_stats_canonical_sha256"),
            "train_stats_sha256": sha256_file(view / "train_stats.json"),
            "split_sha256": clean60.sha256_json(splits),
        },
        "audits": {
            "view_contract_passed": not clean60.view_contract_problems(
                meta, protocol_binding=binding, expected_counts=(100, 80, 20),
                require_clean=True),
            "admission_passed": not clean60.admission_contract_problems(
                admission, meta=meta, split=splits,
                train_stats_sha256=sha256_file(view / "train_stats.json")),
            "scene_audit_passed": scene_audit.get("passed"),
            "data_audit_passed": data_audit.get("passed"),
            "stats_recompute_passed": stats_recompute.get("passed"),
            "norm_audit_passed": norm_audit.get("passed"),
            "metadata_audit_passed": metadata_audit.get("passed"),
            "images_decoded": data_audit.get("images_decoded"),
            "actions_reapplied": data_audit.get("actions_reapplied"),
            "strict_predicate_all_100": all(
                row.get("strict_predicate_passed") for row in data_audit.get("episodes", [])),
            "replay_all_100": all(
                value.get("replay_passed") == value.get("replay_total")
                for value in (data_audit.get("verification") or {}).values()),
            "loader_episodes_covered": (norm_audit.get("loader_consistency") or {}
                                        ).get("episodes_covered"),
            "norm_roundtrip_worst_error": (norm_audit.get("transform") or {}
                                           ).get("worst_recovery_error"),
            "metadata_worst_per_dim_error": metadata_audit.get("worst_max_abs_error"),
        },
        "resource": resource,
        "pilot": {
            "directories": sorted(path.name for path in run_dir.glob("pilot_2026*")
                                  if path.is_dir()),
            "included_in_dataset": False,
            "note": "the one-episode pilot reused the already-consumed clean60 seed 12002 to exercise "
                    "the recording/rendering path; it is excluded from the 40 new episodes",
        },
        "lineage": {
            "preserved_view": old_view.name,
            "preserved_view_sha256": old_meta["view_sha256"],
            "preserved_protocol_sha256": old_meta.get("protocol_sha256"),
            "new_collection": collection.name,
            "new_collection_summary_sha256": sha256_file(collection / "summary.json"),
            "new_collection_manifest_sha256": sha256_file(collection / "e1_manifest.json"),
            "merge": merge_lineage,
        },
        "commands": [
            "python scripts/so101/verify_demos.py --protocol configs/so101/clean100_protocol.json "
            "--collection <collection> --report-output <collection>/verification.json",
            "python scripts/so101/export_training_view.py --collection <collection> "
            "--output <staging-view> --protocol configs/so101/clean100_protocol.json "
            "--audit <collection>/verification.json --workers 6",
            "python scripts/so101/clean100_merge_view.py --old-view <clean60-view> "
            "--old-collection <clean60-collection> --new-view <staging-view> "
            "--new-collection <collection> --protocol configs/so101/clean100_protocol.json "
            "--output <clean100-view>",
            "python scripts/so101/clean60_scene_audit.py --protocol "
            "configs/so101/clean100_protocol.json --view <clean100-view> "
            "--collection <clean60-collection> --collection <collection> "
            "--output <clean100-view>/scene_audit.json",
            "python scripts/so101/clean60_data_audit.py --view <clean100-view> "
            "--collection <clean60-collection> --collection <collection> --protocol "
            "configs/so101/clean100_protocol.json --external-root <external-source> "
            "--verify-images --output <clean100-view>/data_audit.json",
            "python scripts/so101/clean60_recompute_stats.py --view <clean100-view> "
            "--expected-episodes 100 --expected-train 80 --expected-validation 20 "
            "--dataset-label clean100",
            "python scripts/so101/clean60_container_exec.py --label normaudit --image "
            "openpi-server@sha256:405af52051682d5e13169d1de1523fd0adc116eceae4fc314cadfacd45b1ad52 "
            "--mount <checkpoint>:/base:ro --mount <clean100-view>:/view "
            "--container-python-file /work/scripts/so101/clean60_norm_audit.py "
            "--python-arg=--view --python-arg=/view --python-arg=--expected-episodes "
            "--python-arg=100 --python-arg=--expected-train --python-arg=80 "
            "--python-arg=--expected-validation --python-arg=20",
        ],
        "external_events": [
            {
                "utc": "2026-10-29T08:18:23Z",
                "event": "the pre-existing clean60 training container "
                         "so101-e4t-resume-clean60-base-25k-save3000-20261028T193857Z-791dc957 "
                         "disappeared from docker and its CUDA memory was released",
                "attribution": "external to this round: only docker logs and nvidia-smi reads were "
                               "issued; no docker stop/rm/kill or process signal was sent",
                "evidence": "collection_monitor.log: last rate sample 08:17:53, first sample "
                            "without the training container 08:18:23",
            },
        ],
        "remaining_risks": [
            "the 40 new episodes are a separate collection; the expert, thresholds and frozen "
            "source are byte-identical, but the first ~22 minutes ran alongside the pre-existing "
            "training container before it was removed externally",
            "EGL rendering used GPU 2; no CUDA memory was allocated, and the measured training "
            "rate during the overlap was 4.1-5.9 s/it (recorded in collection_monitor.log)",
            "the clean100 view is hardlinked to the preserved bytes; deleting the preserved view "
            "would break the new view's content identity",
            "10 of 12 normalized action dimensions contain values outside [-1, 1]; this is "
            "reported and not clipped, matching the preserved clean60 audit",
            "clean100 training is not authorized or started by this round",
        ],
    }
    common.write_json(run_dir / "result.json", result)
    print(json.dumps({"status": result["status"], "blockers": problems,
                      "view": view.name, "view_sha256": meta["view_sha256"],
                      "episode_manifest": str(view / "episode_manifest.json"),
                      "result": str(run_dir / "result.json")}, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
