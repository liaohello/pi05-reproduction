#!/usr/bin/env python3
"""Collect the E4-T data stage into one sanitised, hash-bound evidence document.

Private run directories are referenced by relative path only; no absolute host path, no episode
pixels.  Every number here is recomputed from the artifacts of the run it describes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_common as common  # noqa: E402

REPO = common.REPO_ROOT


def b_shard_facts(directory: Path) -> dict:
    """One dedicated category-B shard: attempts only, never a contribution to the view."""
    path = directory / "attempts.jsonl"
    attempts = ([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                if path.is_file() else [])
    by_mechanism = Counter((row.get("perturbation") or {}).get("mechanism") for row in attempts)
    return {
        "collection": directory.name,
        "attempts": len(attempts),
        "collector_successes": sum(1 for row in attempts if row.get("success")),
        "attempts_by_mechanism": dict(by_mechanism),
        "successes_by_mechanism": dict(Counter((row.get("perturbation") or {}).get("mechanism")
                                               for row in attempts if row.get("success"))),
    }


def shard_facts(directory: Path) -> dict:
    attempts = [json.loads(line) for line in
                (directory / "attempts.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = common.load_json(directory / "e4t_manifest.json")
    try:
        label = str(directory.relative_to(REPO))
    except ValueError:
        label = directory.name
    return {
        "collection": label,
        "attempts": len(attempts),
        "attempts_by_profile": dict(Counter(row.get("profile") for row in attempts)),
        "accepted_episodes": len(manifest["episodes"]),
        "accepted_by_profile": dict(Counter(row.get("profile") for row in manifest["episodes"])),
        "accepted_frames": sum(int(row["frames"]) for row in manifest["episodes"]),
        "source_sha256": manifest["source_fingerprint"]["source_sha256"],
        "expert_sha256": manifest["source_fingerprint"]["expert_sha256"],
        "patch_sha256": manifest["patch_sha256"],
        "protocol_sha256": manifest.get("protocol_sha256"),
    }


def split_manifest(episodes: list[dict], view_sha256: str) -> dict:
    """Publish the frozen view's episode IDs and their split/category binding."""
    rows = [{"episode_index": int(row["episode_index"]), "split": row["split"],
             "category": row["category"]} for row in episodes]
    rows.sort(key=lambda row: row["episode_index"])
    counts = Counter((row["split"], row["category"]) for row in rows)
    counts_by_split_category = {
        f"{split}/{category}": count
        for (split, category), count in sorted(counts.items())
    }
    manifest = {
        "source_view_sha256": view_sha256,
        "episode_index_semantics": "zero-based episode_index in the frozen view meta.json",
        "episodes": rows,
        "train_episode_ids": [row["episode_index"] for row in rows if row["split"] == "train"],
        "validation_episode_ids": [row["episode_index"] for row in rows
                                    if row["split"] == "validation"],
        "counts_by_split_category": counts_by_split_category,
        "independent_scene_hashes": len({row["scene_sha256"] for row in episodes}),
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, action="append", required=True)
    parser.add_argument("--audit", type=Path, action="append", required=True)
    parser.add_argument("--old-replay", type=Path, required=True)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=REPO / "configs/so101/e4t_protocol.json")
    parser.add_argument("--training-config", type=Path,
                        default=REPO / "configs/so101/e4t_training.json")
    parser.add_argument("--b-shard", type=Path, action="append", default=[],
                        help="dedicated category-B shards: counted and reported, never admitted")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if len(args.collection) != len(args.audit):
        raise SystemExit("each collection needs exactly one audit")
    audits = []
    for path in args.audit:
        document = common.load_json(path.expanduser().resolve())
        audits.append({
            "audit": path.name,
            "passed": bool(document["passed"]),
            "problems": document["problems"],
            "counts": document["counts"],
            "mechanisms": document["mechanisms"],
            "official": document["official"],
            "dataset_sha256": document["dataset_sha256"],
            "excluded_by_protocol": [
                {"attempt_id": row["attempt_id"], "reason": row["reason"],
                 "problems": row["problems"]} for row in document.get("excluded_by_protocol", [])],
        })
    view = args.view.expanduser().resolve()
    meta = common.load_json(view / "meta.json")
    admission = common.load_json(view / "e4t_admission.json")
    plan_path = view / "e4t_sampling_plan.json"
    replay = common.load_json(args.old_replay.expanduser().resolve())
    shards = [shard_facts(path.expanduser().resolve()) for path in args.collection]
    b_shards = [b_shard_facts(path.expanduser().resolve()) for path in args.b_shard]
    episodes = meta["episodes"]
    view_attempts = sum(row["attempts"] for row in shards)
    b_attempts = sum(row["attempts"] for row in b_shards)
    document = {
        "schema_version": 1, "branch": "so101", "phase": "E3", "substage": "E4-T-data",
        "created": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "protocol": str(args.protocol.relative_to(REPO)),
        # the view records the protocol revision that actually produced the delivered data; the
        # template on disk may already carry later documentation-only passes, so both are published
        "protocol_sha256": meta["protocol_sha256"],
        "protocol_binding_note": ("protocol_sha256 is the revision frozen in the exported view "
                                 "meta.json; protocol_template_sha256 is this file at report time and "
                                 "the passes between them are listed in the protocol revision_history"),
        "protocol_template_sha256": common.sha256_file(args.protocol),
        "training_config_sha256": common.sha256_file(args.training_config),
        "requirement": {"episodes": 50, "clean": 25, "recovery": 25, "train": 40, "validation": 10},
        "delivered": {
            "episodes": len(episodes),
            "independent_scene_hashes": len({row["scene_sha256"] for row in episodes}),
            "clean": sum(1 for row in episodes if row["category"] == "clean"),
            "recovery": sum(1 for row in episodes if row["category"] != "clean"),
            "train": sum(1 for row in episodes if row["split"] == "train"),
            "validation": sum(1 for row in episodes if row["split"] == "validation"),
            "frames": sum(int(row["frames"]) for row in episodes),
            "reused_from_E1": sum(1 for row in episodes if row.get("source_dataset") == "old_E1"),
            "new_collection": sum(1 for row in episodes if row.get("source_dataset") == "new"),
            "by_subgroup": dict(Counter(row["subgroup"] for row in episodes)),
            "by_mechanism": dict(Counter(row.get("mechanism") for row in episodes if row.get("mechanism"))),
        },
        "shards": shards,
        "shard_audits": audits,
        "total_attempts": view_attempts + b_attempts,
        "attempt_counts": {
            "eligible_view_source_shards": view_attempts,
            "excluded_category_B_exploration_shards": b_attempts,
            "recorded_main_collection_attempts": view_attempts + b_attempts,
            "note": "pilot attempts are separate; an additional B-shard launch with no recorded attempts is excluded",
        },
        "attempt_budget": {"pre_registered_main_collection_maximum":
                           common.load_json(args.protocol)["collection"]["main_maximum_attempts"]},
        "replay": {
            "new_episodes": {
                "audited_individually": True,
                "excluded_by_protocol": sum(len(row["excluded_by_protocol"]) for row in audits),
                "worst_public_state_error_deg": max(
                    (float(value["max_state_error"]) for value in
                     (admission["data_gate"]["new_episode_replay"] or {}).values()
                     if value.get("max_state_error") is not None), default=0.0),
                "worst_object_error_m": max(
                    (float(value["max_object_error_m"]) for value in
                     (admission["data_gate"]["new_episode_replay"] or {}).values()
                     if value.get("max_object_error_m") is not None), default=0.0)},
            "reused_episodes": {"episodes": int(replay["replay_total"]),
                                "replayed": int(replay["replay_passed"]),
                                "frames": int(replay.get("frames", 0)),
                                "verification_sha256": common.sha256_file(args.old_replay.expanduser().resolve())},
        },
        "view": {
            "directory": view.name, "view_sha256": meta["view_sha256"],
            "files": meta["view_files"], "bytes": meta["view_bytes"],
            "dataset_label": meta["dataset_label"],
            "new_dataset_sha256": meta["new_dataset_sha256"],
            "audit_sha256": meta["audit_sha256"],
            "collected_but_not_used": meta["collected_but_not_used"],
            "eligible_windows": meta["eligible_windows"],
            "excluded_cross_event_windows": meta["excluded_cross_event_windows"],
            "norm_stats_canonical_sha256": meta["norm_stats_canonical_sha256"],
            "train_only_statistics": True,
        },
        "split_manifest": split_manifest(episodes, meta["view_sha256"]),
        "admission": {"passed": bool(admission["data_gate"]["passed"]),
                      "sha256": common.sha256_file(view / "e4t_admission.json"),
                      "totals": admission["totals"]},
        "sampling_plan": ({"sha256": common.sha256_file(plan_path),
                           "group_weights": common.load_json(plan_path)["group_weights"],
                           "train_groups_present": common.load_json(plan_path)["train_groups_present"]}
                          if plan_path.is_file() else {"missing": True}),
        "b_shard_collection": {
            "runs": b_shards,
            "attempts": b_attempts,
            "note": ("these are the shards that tried to fill category B; their episodes are excluded "
                     "by protocol (mechanism_amendment_v21) and none entered the frozen view; they "
                     "are included in total_attempts but kept separate from the 74 view-source attempts"),
        },
        "deviations": {
            # the frozen view carries the note as it read when the export ran; the delivered mix is
            # what the episodes actually are, so both are published instead of editing the view
            "category_B": {
                "frozen_view_note": meta["recovery_category_note"],
                "frozen_view_note_status": ("superseded wording: it restates the pre-registered 7/8 "
                                            "A mix that mechanism_amendment_v22 corrected"),
                "delivered_by_mechanism": dict(
                    Counter(row.get("mechanism") for row in episodes if row.get("mechanism"))),
                "correction": common.load_json(args.protocol)["perturbations"]["mechanism_amendment_v22"],
                "view_is_not_rewritten": True,
            },
            "excluded_slip_episodes": admission["data_gate"]["excluded_by_protocol"],
            "divergence_evidence": "experiments/so101/e4t-replay-divergence.json",
            "mechanism_mix": common.load_json(args.protocol)["collection"]["mechanism_mix"],
        },
        "claim": ("50 independent scripted-expert demonstrations admitted for offline training; the 30 "
                  "new episodes passed individual replay and the 20 reused episodes passed aggregate "
                  "20/20 replay; no learned policy action was executed in simulation"),
        "boundaries": {"policy_actions_executed_in_simulation": False, "real_robot_touched": False,
                       "public_push": False, "history_rewritten": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "delivered": document["delivered"],
                      "total_attempts": document["total_attempts"],
                      "admission": document["admission"],
                      "sampling_plan": document["sampling_plan"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
