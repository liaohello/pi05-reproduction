#!/usr/bin/env python3
"""Assemble the scene1 experiment records from the real run artifacts.

Only files produced by this line are read.  The output keeps relative paths and
hashes so the reports and the shared ledger can cite one stable document.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def _now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--training-summary", type=Path, default=None)
    parser.add_argument("--eval-summary", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("experiments/so101"))
    args = parser.parse_args()
    collection = args.collection.expanduser().resolve()
    view = args.view.expanduser().resolve()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load(collection / "collection_manifest.json")
    view_meta = _load(view / "meta.json")
    audit = _load(view / "data_audit.json")
    train_stats = _load(view / "train_stats.json")
    summary = _load(view / "_summary.json") if (view / "_summary.json").is_file() else None
    attempts = [json.loads(line) for line in (collection / "attempts.jsonl").read_text().splitlines()
                if line.strip()]
    failures = [row for row in attempts if row.get("kind") == "episode" and not row.get("success")]
    data = {
        "schema_version": 1, "branch": "so101", "phase": "scene1", "substage": "data",
        "created": _now(),
        "protocol": "configs/so101/scene1_single_random_protocol.json",
        "protocol_sha256": manifest["protocol_sha256"],
        "certificate_sha256": manifest["certificate_sha256"],
        "collection_run": manifest.get("seed_range"),
        "source": {
            "base_source_sha256": manifest["base_source_sha256"],
            "patched_source_sha256": manifest["current_source_fingerprint"]["source_sha256"],
            "expert_sha256": manifest["current_source_fingerprint"]["expert_sha256"],
            "patch_sha256": manifest.get("patch_sha256"),
        },
        "attempts": manifest["attempts"], "layout_rejects": manifest["layout_rejects"],
        "successes": manifest["successes"], "target": manifest["target"],
        "successes_by_color": {"red": manifest["red_successes"], "blue": manifest["blue_successes"]},
        "failure_reasons": {row.get("failure_reason"): sum(
            1 for other in failures if other.get("failure_reason") == row.get("failure_reason"))
            for row in failures},
        "failures": [{key: row.get(key) for key in
                      ("attempt_id", "seed", "color", "failure_reason", "failure_stage",
                       "failure_message", "frames")}
                     | {"yaw_degrees": (row.get("scene") or {}).get("yaw_degrees"),
                        "block_xy": (row.get("scene") or {}).get("block_xy"),
                        "trays": (row.get("scene") or {}).get("trays")}
                     for row in failures],
        "split": {key: manifest["split"][key] for key in
                  ("train", "validation", "train_red", "train_blue",
                   "validation_red", "validation_blue", "family_crosses")},
        "view": {
            "view_sha256": view_meta["view_sha256"],
            "view_files": view_meta["view_files"], "view_bytes": view_meta["view_bytes"],
            "norm_stats_sha256": train_stats["norm_stats_sha256"],
            "norm_stats_canonical_sha256": train_stats["norm_stats_canonical_sha256"],
            "train_frames": train_stats["train_frames"],
            "constant_dimensions": train_stats["constant_dimensions"],
        },
        "data_quality": {
            "frames_total": audit["frames_total"],
            "frames_min": audit["frames_min"], "frames_max": audit["frames_max"],
            "control_tick_s": audit["control_tick_s"],
            "substeps_all_full": audit["substeps_all_full"],
            "incomplete_frames": audit["incomplete_frames"],
            "max_state_error_vs_trace": audit["max_state_error_vs_trace"],
            "max_action_error_vs_trace": audit["max_action_error_vs_trace"],
            "task_per_episode_matches_scene": audit["task_per_episode_matches_scene"],
            "video_timestamps_aligned": audit["video_timestamps_aligned_across_cameras"],
            "camera_sync_passed": audit["camera_sync"].get("passed"),
            "contact_monitor_errors": audit["contact_monitor_errors"],
            "soft_contacts": audit["soft_contacts"],
        },
        "claim": "64 successful single-block scripted-expert episodes; not a benchmark success rate",
        "boundaries": {"perturbations": False, "visual_randomization": False,
                       "policy_actions_executed_in_simulation": False,
                       "real_robot_touched": False, "public_push": False},
    }
    (out_dir / "scene1-data.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    if args.training_summary:
        training = _load(args.training_summary)
        record = {
            "schema_version": 1, "branch": "so101", "phase": "scene1",
            "substage": "training", "created": _now(),
            "training_config": "configs/so101/scene1_training.json",
            "data": str(out_dir / "scene1-data.json"),
            "run": training,
            "boundaries": {"gate_thresholds_changed_after_training": False,
                           "policy_actions_executed_in_simulation": False,
                           "real_robot_touched": False, "public_push": False},
        }
        if args.eval_summary:
            record["evaluation"] = _load(args.eval_summary)
        (out_dir / "scene1-training.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"data": str(out_dir / "scene1-data.json"),
                      "training": bool(args.training_summary)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
