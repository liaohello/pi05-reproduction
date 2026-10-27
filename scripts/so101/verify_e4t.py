#!/usr/bin/env python3
"""Audit E4-T demonstrations and replay every episode physically."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_common as common  # noqa: E402


def admitted_mechanisms(protocol: dict) -> dict[str, set[str]]:
    """Per category, the mechanisms with a positive pre-registered quota.

    A zero quota is a withdrawal, not a registration: episodes collected under a withdrawn
    mechanism are excluded from the dataset and reported, never silently admitted.
    """
    return {name: {key for key, quota in (settings.get("mechanisms") or {}).items() if int(quota) > 0}
            for name, settings in protocol["perturbations"]["categories"].items()}


def replay_with_event(directory: Path) -> dict:
    """Invoke the patched external replay and return its report."""
    import importlib

    import os
    common.setup_external(os.environ["BIMANUAL_ROOT"])
    module = importlib.import_module("bimanual.recording.task_replay")
    try:
        result = module.replay_episode(directory)
        return {"passed": bool(result.get("passed")), "replay": result}
    except (AssertionError, ValueError) as error:
        return {"passed": False, "error": str(error), "replay": None}


def event_window(events: list[dict], frame: int) -> tuple[dict | None, dict | None]:
    stage_events = [row for row in events if row.get("type") == "stage"]
    return None, None


def episode_audit(entry: dict, directory: Path, config: dict) -> dict:
    events = []
    event_path = directory / "events.jsonl"
    if event_path.is_file():
        events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    trace_path = directory / "trace.npz"
    problems = []
    if not trace_path.is_file():
        return {"passed": False, "problems": ["missing trace"], "category": entry.get("category") or str(entry.get("profile", "clean_new"))[-1]}
    with np.load(trace_path, allow_pickle=False) as trace:
        positions = np.asarray(trace["objects_pos"], dtype=float)
        actions = np.asarray(trace["action"], dtype=float)
        states = np.asarray(trace["state"], dtype=float)
        substeps = np.asarray(trace["substeps"], dtype=int) if "substeps" in trace.files else None
    if actions.shape != states.shape or actions.shape[1] != 12:
        problems.append("state/action shape")
    if not np.isfinite(actions).all() or not np.isfinite(states).all():
        problems.append("non-finite state/action")
    if not summary.get("success") or int(summary.get("sorted_count", 0)) != 4:
        problems.append("episode did not satisfy the original physical success predicate")

    perturbation = next((row for row in events if row.get("type") == "perturbation"), None)
    replan = next((row for row in events if row.get("type") == "expert_replan"), None)
    frame_displacement = None
    correction = None
    if entry.get("perturbation"):
        if perturbation is None:
            problems.append("missing typed perturbation event")
        if replan is None:
            problems.append("missing expert replan")
        else:
            correction = {
                "pending_after_replan": replan.get("pending_objects"),
                "plan_start_after_replan": bool(
                    any(row.get("type") == "plan_start" and row.get("object") == perturbation.get("object")
                        and float(row.get("sim_time", 0)) >= float(perturbation.get("sim_time_s", 0))
                        for row in events))}
            if not correction["plan_start_after_replan"]:
                problems.append("no restarted expert plan for the perturbed object")
        mechanism = (perturbation or {}).get("mechanism")
        windowed = mechanism in ("gripper_release_slip", "arm_bump_impulse")
        span = (int(config["perturbations"].get("slip_effect_window_frames", 50))
                if windowed else 1)
        if perturbation and substeps is not None:
            frame = int(perturbation["control_frame"])
            object_index = int(str(perturbation["object"]).rsplit("_", 1)[-1])
            if frame + span >= positions.shape[0]:
                problems.append("event in terminal frame")
            else:
                window = positions[frame:frame + span + 1, object_index, :2]
                frame_displacement = float(np.max(
                    np.linalg.norm(window - positions[frame, object_index, :2], axis=1)))
                limits = config["perturbations"]
                if mechanism == "gripper_release_slip":
                    # The slip's signature is the grasp coming apart, not a travel
                    # distance; the direct evidence is the settle-and-regrasp events.
                    separated = any(row.get("type") == "expert_recovery_settle"
                                    and row.get("object") == perturbation.get("object")
                                    and float(row.get("sim_time", 0))
                                    >= float(perturbation.get("sim_time_s", 0))
                                    for row in events)
                    if not separated:
                        problems.append("slip grasp never separated")
                    if frame_displacement > float(limits["maximum_frame_xy_displacement_m"]):
                        problems.append("slip motion exceeds bounded range")
                else:
                    if frame_displacement < float(limits["minimum_frame_xy_displacement_m"]):
                        problems.append("frame displacement below bound")
                    if frame_displacement > float(limits["maximum_frame_xy_displacement_m"]):
                        problems.append("frame displacement exceeds bound")
        if perturbation and mechanism not in (
                "bounded_velocity_impulse", "explicit_state_displacement",
                "gripper_release_slip", "arm_bump_impulse"):
            problems.append("unknown mechanism")
        if entry.get("mechanism") and perturbation and entry["mechanism"] != perturbation.get("mechanism"):
            problems.append("planned mechanism differs from applied mechanism")
        if entry.get("category") and perturbation and entry["category"] != perturbation.get("category"):
            problems.append("planned category differs from applied category")
    else:
        if perturbation or replan:
            problems.append("clean episode contains a perturbation event")

    replay = replay_with_event(directory)
    replay_ok = bool(replay.get("passed"))
    if not replay_ok:
        problems.append("physical replay failed")
    if substeps is not None and not np.isin(substeps, range(1, 11)).all():
        problems.append("physics substeps outside 1..10")
    replay_report = replay.get("replay") or {}
    return {
        "passed": not problems,
        "problems": problems,
        "replay_error": replay.get("error"),
        "episode_index": entry.get("source_episode_index"),
        "category": entry.get("category")
                    or ("clean" if entry.get("profile") == "clean_new"
                        else str(entry.get("profile", ""))[-1]),
        "mechanism": (perturbation or {}).get("mechanism"),
        "seed": entry.get("seed"),
        "scene_sha256": entry.get("scene_sha256"),
        "frames": int(actions.shape[0]),
        "success": bool(summary.get("success")),
        "sorted_count": int(summary.get("sorted_count", 0)),
        "perturbation_event": perturbation,
        "expert_replan_event": replan,
        "frame_xy_displacement_m": frame_displacement,
        "correction": correction,
        "replay_max_state_error": replay_report.get("max_public_state_error"),
        "replay_max_object_error_m": replay_report.get("max_object_position_error_m"),
        "replay_max_clock_error_s": replay_report.get("max_clock_error_s"),
        "substep_replay_count": (replay_report.get("perturbation") or {}).get("applied_at"),
        "perturbation_replay_error": (replay_report.get("perturbation") or {}).get("delta_error"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--protocol", type=Path, default=common.REPO_ROOT / "configs/so101/e4t_protocol.json")
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, default=None)
    parser.add_argument("--partial", action="store_true",
                        help="audit one shard of a parallel collection: skip whole-set split and quota asserts")
    args = parser.parse_args()

    root = common.setup_external(args.root or sys.modules["os"].environ.get("SO101_SOURCE_ROOT"))
    protocol = common.load_json(args.protocol)
    config = common.load_external_config(root, {**protocol, "collection": {
        "direct_seed_range": protocol["seeds"]["recovery_candidates"], "target_successes": 1,
        "maximum_attempts": 1, "recording_fps": protocol["control"]["recording_fps"],
        "control_hz": protocol["control"]["control_hz"], "diagnostics": True,
        "split": {"train": 1, "validation": 0, "rule": "audit only"}}})
    config["sorting"]["timeout_s"] = float(protocol["control"]["timeout_s"])
    import os
    if args.root:
        os.environ["SO101_SOURCE_ROOT"] = str(Path(str(args.root)).expanduser().resolve())
    collection = args.collection.expanduser().resolve()
    manifest = common.load_json(collection / "e4t_manifest.json")
    split_path = collection / "split.json"
    split = common.load_json(split_path) if split_path.is_file() else {"train": [], "validation": []}
    episodes = manifest["episodes"]
    audits = []
    # Only mechanisms with a positive pre-registered quota are admitted. The slip episodes stay
    # failed (their replay is not exact) but are reported as excluded by protocol rather than
    # being silently repaired or counted as a pass.
    registered = admitted_mechanisms(protocol)
    for entry in episodes:
        directory = Path(entry["diagnostics"])
        audit = episode_audit(entry, directory, protocol)
        audit["attempt_id"] = entry["attempt_id"]
        event = (audit.get("perturbation_event") or {})
        category = event.get("category") or audit.get("category")
        mechanism = event.get("mechanism") or audit.get("mechanism")
        audit["excluded_by_protocol"] = bool(
            mechanism and category in registered and mechanism not in registered[category])
        if audit["excluded_by_protocol"]:
            audit["exclusion_reason"] = (f"mechanism '{mechanism}' has a zero quota for "
                                         f"category {category} in the registered protocol")
        audits.append(audit)

    from bimanual.integrations.lerobot.dataset import verify_dataset
    official = verify_dataset(collection / "dataset", manifest["repo_id"], config["recording"]["fps"])
    dataset_lengths = [int(row["frames"]) for row in episodes]
    problems = []
    if official.get("episodes") != len(episodes):
        problems.append("new dataset episode count differs from the collection manifest")
    if official.get("frames") != sum(dataset_lengths):
        problems.append("dataset frame total differs from audited episode totals")
    if official.get("state_shape") != [12] or official.get("action_chunk_shape") != [3, 12]:
        problems.append("official loader shape is not 12-D / 3-step chunk")
    if sorted(official.get("image_shapes", {})) != sorted(protocol["observation"]["cameras"]):
        problems.append("camera set differs from protocol")
    if len({row["scene_sha256"] for row in episodes}) != len(episodes):
        problems.append("duplicate scene identity among new episodes")
    counts = {}
    mechanisms = {}
    for row in episodes:
        if row["attempt_id"] in {a["attempt_id"] for a in audits if a.get("excluded_by_protocol")}:
            continue
        category = row.get("profile", "clean_new")
        counts[category] = counts.get(category, 0) + 1
        mechanism = (row.get("recovery_audit", {}).get("perturbation_event") or {}).get("mechanism")
        if mechanism:
            mechanisms[mechanism] = mechanisms.get(mechanism, 0) + 1
    if not args.partial and counts != protocol["collection"]["target_successes_per_profile"]:
        problems.append(f"profile quota mismatch {counts}")
    expected_mechanisms = {}
    for settings in protocol["perturbations"]["categories"].values():
        for mechanism, quota in (settings.get("mechanisms") or {}).items():
            expected_mechanisms[mechanism] = expected_mechanisms.get(mechanism, 0) + int(quota)
    expected_mechanisms = {key: value for key, value in expected_mechanisms.items() if value}
    train_ids = set(split["train"])
    val_ids = set(split["validation"])
    eligible_audits = [row for row in audits if not row.get("excluded_by_protocol")]
    if not args.partial:
        if len(train_ids) != 24 or len(val_ids) != 6 or train_ids & val_ids:
            problems.append("new-run split size is invalid")
        if mechanisms != expected_mechanisms:
            problems.append(f"mechanism quota mismatch {mechanisms} expected {expected_mechanisms}")
        if sum(1 for row in eligible_audits if row["category"] in ("A", "B", "C")) != 25:
            problems.append("recovery count is not 25")
    failures = [row for row in eligible_audits if not row["passed"]]
    if failures:
        problems.append(f"episode-level audit failures: {[row['attempt_id'] for row in failures]}")

    excluded = [{"attempt_id": row["attempt_id"], "reason": row.get("exclusion_reason"),
                 "problems": row["problems"], "replay_max_state_error": row.get("replay_max_state_error")}
                for row in audits if row.get("excluded_by_protocol")]
    output = args.output.expanduser().resolve()
    payload = {
        "schema_version": 1, "phase": "E4-T-data-audit",
        "created": common.REPO_ROOT and str(__import__("datetime").datetime.now().astimezone()),
        "passed": not problems,
        "problems": problems,
        "protocol_sha256": common.sha256_file(args.protocol),
        "patch_sha256": manifest["patch_sha256"],
        "source_sha256": manifest["source_fingerprint"]["source_sha256"],
        "expert_sha256": manifest["source_fingerprint"]["expert_sha256"],
        "official": official,
        "dataset_sha256": common.sha256_tree(collection / "dataset")["sha256"],
        "dataset_info_sha256": common.sha256_file(collection / "dataset" / "meta" / "info.json"),
        "split_sha256": (common.sha256_file(split_path) if split_path.is_file() else None),
        "partial_shard": bool(args.partial),
        "split_assigned_at_export": bool(args.partial),
        "episodes": audits,
        "counts": counts,
        "mechanisms": mechanisms,
        "excluded_by_protocol": excluded,
    }
    common.write_json(output, payload)
    if args.public_output:
        common.write_json(args.public_output, {
            "schema_version": 1, "phase": "E4-T-data-audit", "passed": not problems,
            "problem_count": len(problems), "episodes": len(eligible_audits),
            "frames": official["frames"],
            "counts": counts, "mechanisms": mechanisms,
            "excluded_by_protocol": excluded,
            "episode_summaries": [{key: row.get(key) for key in ("attempt_id", "episode_index", "seed",
                                                                   "scene_sha256", "category", "mechanism",
                                                                   "frames", "success", "sorted_count",
                                                                   "frame_xy_displacement_m", "passed",
                                                                   "excluded_by_protocol")}
                                    for row in audits],
            "claim": "new E4-T collection audited and physically replayed; not a learned-policy result",
        })
    print(json.dumps({"passed": not problems, "problems": problems,
                      "counts": counts, "mechanisms": mechanisms,
                      "excluded_by_protocol": [row["attempt_id"] for row in excluded],
                      "official_frames": official["frames"]}, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
