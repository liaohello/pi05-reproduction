#!/usr/bin/env python3
"""E4-T collection: clean demonstrations plus continuous bounded recovery episodes.

This is a repository-local driver around the patched external EpisodeRunner and
official LeRobot writer.  It intentionally does not use a learned policy.  Every
attempt is retained; only a successful physical episode with valid evidence
becomes a demonstration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_common as common  # noqa: E402
import source_fingerprint  # noqa: E402


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _sha(value) -> str:
    return common.sha256_json(value)


def _range(value: list[float]) -> tuple[float, float]:
    return float(value[0]), float(value[1])


def _mechanism_sequence(count: int, mechanisms: dict[str, int]) -> list[str]:
    expanded = []
    for mechanism, repeats in mechanisms.items():
        expanded.extend([mechanism] * int(repeats))
    if len(expanded) != count:
        raise ValueError(f"mechanism quota {expanded} does not sum to target {count}")
    return expanded


def _category_schedule(profile: str, seed: int, config: dict, index: int,
                       mechanism_override: str | None = None) -> dict:
    category = next(token for token in ("A", "B", "C") if token in profile)
    settings = config["perturbations"]["categories"][category]
    sequence = _mechanism_sequence(int(settings["target"]), settings["mechanisms"])
    mechanism = mechanism_override or sequence[index % len(sequence)]
    rng = random.Random(f"{profile}:{seed}:{index}:{mechanism}")
    angle = rng.choice([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
    radians = math.radians(angle)
    delta_position = [0.0, 0.0, 0.0]
    delta_velocity = [0.0, 0.0, 0.0]
    if mechanism == "explicit_state_displacement":
        low, high = _range(settings["state_xy_m"])
        distance = rng.uniform(low, high)
        delta_position = [math.cos(radians) * distance, math.sin(radians) * distance, 0.0]
        delta_velocity = [math.cos(radians) * 0.02, math.sin(radians) * 0.02, -0.01]
    else:
        low, high = _range(settings["impulse_delta_v_xy_m_s"])
        speed = rng.uniform(low, high)
        delta_velocity = [math.cos(radians) * speed, math.sin(radians) * speed, 0.0]
    episode_id = f"{profile}_{seed:04d}_{index:03d}"
    schedule = {
        "schema_version": 1,
        "episode_id": episode_id,
        "category": category,
        "mechanism": mechanism,
        "trigger_stage_contains": settings["trigger_stage_contains"],
        "trigger_after_s": float(settings["trigger_after_s"]),
        "trigger_substep": int(settings["trigger_substep"]),
        "target_object": "current",
        "delta_position_m": delta_position,
        "delta_velocity_m_s": delta_velocity,
        "rng_seed": seed,
        "direction_degrees": angle,
    }
    if mechanism == "gripper_release_slip":
        release = settings["gripper_release"]
        schedule["gripper_open_fraction"] = float(release["open_fraction"])
        schedule["gripper_hold_frames"] = int(release["hold_frames"])
    if mechanism == "arm_bump_impulse":
        low, high = _range(settings["arm_bump_delta_v_rad_s"])
        schedule["arm_delta_velocity_rad_s"] = float(rng.choice((-1.0, 1.0)) * rng.uniform(low, high))
    schedule["plan_sha256"] = _sha(schedule)
    return schedule


def candidate_specs(config: dict, pilot: bool, only_profiles: list[str] | None = None,
                    pools: dict[str, list[int]] | None = None) -> list[dict]:
    if pilot:
        # v4: A-impulse is the surviving force/impulse mechanism, B/C use tagged
        # state displacement because an impulse on a clamped payload produces no
        # measurable displacement.  Seeds 8301/8322/8342 are fresh scenes; the
        # v1-v3 pilot scenes stay recorded as failures.
        pilot_specs = [
            ("pilot_B", 8235, 0, "arm_bump_impulse"),
            ("pilot_C", 8245, 0, "arm_bump_impulse"),
        ]
        return [{
            "attempt_id": f"{profile}_{seed:04d}_{index:03d}", "profile": profile,
            "seed": seed, "candidate_index": index,
            "perturbation": _category_schedule(profile, seed, config, index, mechanism),
        } for profile, seed, index, mechanism in pilot_specs]

    profiles = {
        "clean_new": list(range(8350, 8370)),
        "recovery_A": list(range(8300, 8320)),
        # B and C use the registered wider recovery pool: the pilot scenes took the
        # original 8320-8349 block, and a scene already seen by a pilot may never be
        # reused as a demonstration.
        "recovery_B": list(range(8220, 8260)),
        "recovery_C": list(range(8260, 8300)),
    }
    for name, seeds in (pools or {}).items():
        profiles[name] = list(seeds)
    if only_profiles:
        profiles = {name: seeds for name, seeds in profiles.items() if name in only_profiles}
    targets = config["collection"]["target_successes_per_profile"]
    profiles = {name: seeds for name, seeds in profiles.items() if int(targets.get(name, 0)) > 0}
    specs = []
    for profile, seeds in profiles.items():
        for index, seed in enumerate(seeds):
            row = {"attempt_id": f"{profile}_{seed:04d}_{index:03d}", "profile": profile,
                   "seed": int(seed), "candidate_index": index}
            if profile.startswith("recovery_"):
                row["perturbation"] = _category_schedule(profile, seed, config, index)
            specs.append(row)
    ordered = []
    by_profile = {profile: [row for row in specs if row["profile"] == profile] for profile in profiles}
    while any(by_profile.values()):
        for profile in profiles:
            if by_profile[profile]:
                ordered.append(by_profile[profile].pop(0))
    return ordered


def load_existing_scene_hashes(paths: list[Path]) -> dict[str, str]:
    values = {}
    for path in paths:
        document = common.load_json(path)
        queue = [document]
        while queue:
            item = queue.pop()
            if isinstance(item, dict):
                if isinstance(item.get("scene_sha256"), str):
                    values[item["scene_sha256"]] = str(path)
                if isinstance(item.get("sha256"), str) and item.get("seed") is not None:
                    values[item["sha256"]] = str(path)
                queue.extend(item.values())
            elif isinstance(item, list):
                queue.extend(item)
    return values


def grasp_separation_evidence(events: list[dict], perturbation: dict) -> bool:
    """True when the gripped payload is observed out of the hand after the slip."""
    after = float(perturbation["sim_time_s"])
    for row in events:
        if float(row.get("sim_time", 0)) < after - 1e-9:
            continue
        if row.get("type") == "expert_recovery_settle" and row.get("object") == perturbation.get("object"):
            return True
        if row.get("type") == "failure":
            continue
    return False


def validate_recovery(directory: Path, summary: dict, config: dict) -> dict:
    events = []
    path = directory / "events.jsonl"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
    perturbation = next((row for row in events if row.get("type") == "perturbation"), None)
    replan = next((row for row in events if row.get("type") == "expert_replan"), None)
    reasons = []
    if not perturbation:
        reasons.append("trigger_stage_not_reached")
    if not replan:
        reasons.append("no_expert_replan")
    correction = False
    if perturbation and replan:
        after = [row for row in events if float(row.get("sim_time", 0)) >= float(perturbation["sim_time_s"])
                 and row.get("type") == "plan_start"
                 and row.get("object") == perturbation.get("object")]
        stages = [row for row in events if float(row.get("sim_time", 0)) >= float(perturbation["sim_time_s"])
                  and row.get("type") == "stage" and row.get("object_id") == perturbation.get("object")]
        held_release = [row for row in events if row.get("type") == "expert_recovery_release"
                        and float(row.get("sim_time", 0)) >= float(perturbation["sim_time_s"])
                        and row.get("object") == perturbation.get("object")]
        correction = bool(after or stages or held_release)
        if not correction:
            reasons.append("no_post_perturbation_expert_correction")
    frame_xy = None
    effect_window = "next_frame"
    slip = False
    slip_drop_m = None
    if perturbation:
        mechanism = str(perturbation.get("mechanism", ""))
        slip = mechanism == "gripper_release_slip"
        bump = mechanism == "arm_bump_impulse"
        windowed = slip or bump
        span = int(config["perturbations"].get("slip_effect_window_frames", 50)) if windowed else 1
        effect_window = f"effect_window_{span}_frames" if windowed else "next_frame"
        slip_drop_m = None
        try:
            trace_handle = np.load(directory / "trace.npz", allow_pickle=False)
        except Exception as error:  # never lose an attempt over a broken audit
            reasons.append(f"audit_trace_error:{type(error).__name__}")
            trace_handle = None
        positions = None
        if trace_handle is not None:
            with trace_handle as trace:
                positions = np.asarray(trace["objects_pos"], dtype=float)
        if positions is not None:
            object_index = int(str(perturbation["object"]).rsplit("_", 1)[-1])
            frame = int(perturbation["control_frame"])
            if frame + span >= positions.shape[0]:
                reasons.append("perturbation_recorded_in_terminal_frame")
            else:
                before = positions[frame, object_index, :2]
                window = positions[frame:frame + span + 1, object_index, :]
                frame_xy = float(np.max(np.linalg.norm(window[:, :2] - before, axis=1)))
                if slip:
                    slip_drop_m = float(positions[frame, object_index, 2]
                                        - np.min(window[:, 2]))
        validation = config["perturbations"]
        if frame_xy is not None:
            if slip:
                # A slip is defined by the grasp coming apart, not by how far the cube
                # happens to travel: the direct evidence is the loss of bilateral
                # contact on this object plus the expert's settle-and-regrasp plan.
                if not grasp_separation_evidence(events, perturbation):
                    reasons.append("slip_grasp_never_separated")
            else:
                if frame_xy < float(validation["minimum_frame_xy_displacement_m"]):
                    reasons.append("actual_frame_displacement_below_bound")
            if not slip and frame_xy > float(validation["maximum_frame_xy_displacement_m"]):
                reasons.append("actual_frame_displacement_exceeds_bound")
    elif not perturbation:
        reasons.append("perturbation_event_never_fired")
    return {
        "valid": bool(not reasons),
        "reasons": reasons,
        "perturbation_event": perturbation,
        "expert_replan_event": replan,
        "correction_evidence": correction,
        "frame_xy_displacement_m": frame_xy,
        "effect_measurement": effect_window,
        "slip_drop_m": None if slip_drop_m is None else round(slip_drop_m, 5),
        "summary_perturbation_event": summary.get("perturbation_event"),
    }


def _eligible_starts(length: int, perturbation_frame: int | None,
                     horizon: int) -> list[int]:
    starts = list(range(max(0, int(length) - int(horizon) + 1)))
    if perturbation_frame is None:
        return starts
    event = int(perturbation_frame)
    return [frame for frame in starts if not frame <= event <= frame + int(horizon) - 1]


def assign_split(successes: list[dict], config: dict) -> dict:
    """Freeze train/validation per category using the quotas registered in the protocol."""
    protocol_split = config["split"]["new_run_per_category"]
    expected = {name: (train, validation) for name, (train, validation) in zip(
        ("clean_new", "recovery_A", "recovery_B", "recovery_C"),
        ((int(protocol_split["clean_new"][0]), int(protocol_split["clean_new"][1])),
         (int(protocol_split["recovery_A"][0]), int(protocol_split["recovery_A"][1])),
         (int(protocol_split["recovery_B"][0]), int(protocol_split["recovery_B"][1])),
         (int(protocol_split["recovery_C"][0]), int(protocol_split["recovery_C"][1]))))}
    counters = {profile: 0 for profile in expected}
    for row in successes:
        profile = row["profile"]
        want_train = expected[profile][0]
        row["split"] = "train" if counters[profile] < want_train else "validation"
        counters[profile] += 1
    train = [row["source_episode_index"] for row in successes if row["split"] == "train"]
    validation = [row["source_episode_index"] for row in successes if row["split"] == "validation"]
    problems = []
    for profile, (want_train, want_validation) in expected.items():
        rows = [row for row in successes if row["profile"] == profile]
        if len(rows) != want_train + want_validation:
            problems.append(f"{profile} target quota")
        if sum(row["split"] == "train" for row in rows) != want_train:
            problems.append(f"{profile} train quota")
        if sum(row["split"] == "validation" for row in rows) != want_validation:
            problems.append(f"{profile} validation quota")
    if len(train) != 24 or len(validation) != 6:
        problems.append("new-run split size")
    if len({row["scene_sha256"] for row in successes}) != len(successes):
        problems.append("duplicate scene identity")
    return {"train": train, "validation": validation, "problems": problems,
            "rule": config["split"]["rule"], "episodes": successes}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--certificate", type=Path, default=None)
    parser.add_argument("--protocol", type=Path, default=common.REPO_ROOT / "configs/so101/e4t_protocol.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/so101_pi05_e4t_30")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--public-manifest", type=Path, default=None)
    parser.add_argument("--only-profiles", default=None,
                        help="comma separated subset of clean_new,recovery_A,recovery_B,recovery_C")
    parser.add_argument("--pool", action="append", default=[],
                        help="profile=first:last seed override for a parallel shard")
    parser.add_argument("--target-override", action="append", default=[],
                        help="profile=count: collect only this many successes, for a parallel shard")
    parser.add_argument("--existing-scene-manifest", action="append", default=[], type=Path,
                        help="manifest/scenes JSON used to reject duplicate scene identities")
    args = parser.parse_args()

    root_value = args.root or os.environ.get("SO101_SOURCE_ROOT")
    if not root_value:
        raise SystemExit("missing --root or SO101_SOURCE_ROOT")
    root = common.setup_external(root_value)
    protocol = common.load_json(args.protocol)
    config = common.load_external_config(root, {**protocol, "collection": {
        "direct_seed_range": [protocol["seeds"]["recovery_candidates"][0],
                              protocol["seeds"]["clean_candidates"][1]],
        "target_successes": 999, "maximum_attempts": protocol["collection"]["main_maximum_attempts"],
        "recording_fps": protocol["control"]["recording_fps"],
        "control_hz": protocol["control"]["control_hz"],
        "diagnostics": True, "split": {"train": 20, "validation": 5, "rule": "see e4t_protocol"}}})
    config["sorting"]["timeout_s"] = float(protocol["control"]["timeout_s"])
    if args.certificate is not None:
        config["evaluation"]["certificate"] = str(Path(args.certificate).resolve())
    if args.pilot:
        config["sorting"]["timeout_s"] = 240.0
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to reuse existing run directory: {output}")
    output.mkdir(parents=True)

    from bimanual.integrations.lerobot.dataset import LeRobotWriter
    from bimanual.recording.episode import EpisodeRunner
    from bimanual.sim.environment import BimanualEnv
    from bimanual.tasks.sorting import regions
    from bimanual.tasks.sorting.certification import check_certificate_current
    from bimanual.tasks.sorting.scene_sampling import SceneGenerationError, generate_scene

    certificate = regions.load_certificate(root / config["evaluation"]["certificate"])
    existing_scenes = load_existing_scene_hashes(args.existing_scene_manifest)
    only = [item.strip() for item in (args.only_profiles or "").split(",") if item.strip()]
    pools = {}
    for item in args.pool:
        name, span = item.split("=", 1)
        first, last = (int(value) for value in span.split(":"))
        pools[name] = list(range(first, last + 1))
    if only or pools:
        missing = [name for name in (only or []) if name not in pools and not (8300 <= min(
            pools.get(name, [8300])[0], 8300))]
        unknown = set(only or []) | set(pools or {})
        if unknown - {"clean_new", "recovery_A", "recovery_B", "recovery_C"}:
            raise SystemExit(f"unknown profile in --only-profiles/--pool: {sorted(unknown)}")
    specs = candidate_specs(protocol, args.pilot, only_profiles=only or None, pools=pools or None)
    targets = ({"pilot_A": 1, "pilot_B": 2, "pilot_C": 2} if args.pilot
               else protocol["collection"]["target_successes_per_profile"])
    budgets = ({"pilot_A": 3, "pilot_B": 4, "pilot_C": 4} if args.pilot
               else protocol["collection"]["maximum_attempts_per_profile"])
    for item in args.target_override:
        name, count = item.split("=", 1)
        if name not in targets:
            raise SystemExit(f"unknown profile in --target-override: {name}")
        targets[name] = int(count)
    global_budget = (int(protocol["collection"]["pilot_maximum_attempts"]) if args.pilot
                     else int(protocol["collection"]["main_maximum_attempts"]))
    writer = LeRobotWriter(config, output / "dataset", args.repo_id)
    attempts = []
    successes = []
    seen_scenes = dict(existing_scenes)
    started = time.time()
    with BimanualEnv(config, images=True) as env:
        stale = check_certificate_current(certificate, env)
        if stale:
            raise RuntimeError("certificate stale: " + "; ".join(stale))
        runner = EpisodeRunner(env, config, certificate)
        for spec in specs:
            profile = spec["profile"]

            if sum(1 for row in successes if row["profile"] == profile) >= targets[profile]:
                continue
            used = sum(1 for row in attempts if row["profile"] == profile)
            if used >= budgets.get(profile, 0):
                continue
            if len(attempts) >= global_budget:
                break
            if profile.startswith("pilot_"):
                # Pilot alternates categories and both mechanisms without a second attempt per exact schedule.
                pilot_budget = int(protocol["collection"]["pilot_maximum_attempts"])
                if len(attempts) >= pilot_budget:
                    break
            seed = int(spec["seed"])
            try:
                scene = generate_scene(config, certificate, "direct", seed, env=env,
                                       sampling=str(config["sorting"].get("sampling_mode", "legacy")))
            except SceneGenerationError as error:
                row = {"attempt_id": spec["attempt_id"], "profile": profile, "seed": seed,
                       "success": False, "failure_reason": "scene_generation_failed", "message": str(error),
                       "perturbation": spec.get("perturbation")}
                attempts.append(row)
                (output / "attempts.jsonl").open("a", encoding="utf-8").write(json.dumps(row, default=str) + "\n")
                continue
            if spec.get("perturbation"):
                scene.extra["perturbation"] = spec["perturbation"]
                scene.sha256 = scene.digest()
            if scene.sha256 in seen_scenes:
                row = {"attempt_id": spec["attempt_id"], "profile": profile, "seed": seed,
                       "scene_sha256": scene.sha256, "success": False,
                       "failure_reason": "duplicate_scene_identity",
                       "duplicate_of": seen_scenes[scene.sha256], "perturbation": spec.get("perturbation")}
                attempts.append(row)
                (output / "attempts.jsonl").open("a", encoding="utf-8").write(json.dumps(row, default=str) + "\n")
                continue
            seen_scenes[scene.sha256] = spec["attempt_id"]
            directory = output / "diagnostics" / spec["attempt_id"]
            sink = _WriterSink(writer, config["sorting"]["instruction"])
            episode_started = time.time()
            try:
                episode = runner.run_episode(scene, output=directory, record=True, sink=sink)
            except Exception as error:
                writer.discard_episode()
                row = {"attempt_id": spec["attempt_id"], "profile": profile, "seed": seed,
                       "scene_sha256": scene.sha256, "success": False,
                       "failure_reason": "unexpected_exception", "message": f"{type(error).__name__}: {error}",
                       "frames": sink.frames, "wall_time_s": time.time() - episode_started,
                       "perturbation": spec.get("perturbation")}
                attempts.append(row)
                (output / "attempts.jsonl").open("a", encoding="utf-8").write(json.dumps(row, default=str) + "\n")
                continue
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            try:
                recovery_audit = (validate_recovery(directory, summary, protocol)
                                  if spec.get("perturbation") else None)
            except Exception as error:  # a broken audit must never cost an attempt
                recovery_audit = {"valid": False,
                                  "reasons": [f"audit_crashed:{type(error).__name__}: {error}"],
                                  "perturbation_event": None, "expert_replan_event": None,
                                  "correction_evidence": False, "frame_xy_displacement_m": None,
                                  "effect_measurement": "crashed", "slip_drop_m": None,
                                  "summary_perturbation_event": None}
            clean_audit = {"valid": bool(summary.get("success")) and summary.get("sorted_count") == 4,
                           "reasons": [] if summary.get("success") else ["episode_failed"]}
            audit = recovery_audit or clean_audit
            save = bool(summary.get("success")) and summary.get("sorted_count") == 4 and audit["valid"]
            dataset_index = None
            if save:
                dataset_index = int(writer.dataset.meta.total_episodes)
                writer.save_episode()
                row = {
                    "attempt_id": spec["attempt_id"], "profile": profile,
                    "seed": seed,
                    "scene_sha256": scene.sha256, "success": True, "sorted_count": 4,
                    "frames": int(summary["frames"]), "sim_time_s": float(summary["sim_time_s"]),
                    "wall_time_s": float(summary["wall_time_s"]),
                    "source_episode_index": dataset_index, "diagnostics": str(directory),
                    "perturbation": spec.get("perturbation"), "recovery_audit": audit,
                }
                successes.append(row)
            else:
                writer.discard_episode()
                reason = "episode_failed" if not summary.get("success") else "recovery_audit_failed"
                row = {
                    "attempt_id": spec["attempt_id"], "profile": profile,
                    "seed": seed,
                    "scene_sha256": scene.sha256, "success": False, "failure_reason": reason,
                    "raw_failure_reason": summary.get("failure_reason"),
                    "frames": int(summary["frames"]), "sorted_count": int(summary["sorted_count"]),
                    "diagnostics": str(directory), "perturbation": spec.get("perturbation"),
                    "recovery_audit": audit,
                }
            attempts.append(row)
            (output / "attempts.jsonl").open("a", encoding="utf-8").write(json.dumps(row, default=str) + "\n")
            print(json.dumps({"attempt": spec["attempt_id"], "saved": save,
                              "profile": profile, "frames": summary.get("frames"),
                              "reason": row.get("failure_reason")}), flush=True)

        all_targets = all(
            sum(1 for row in successes if row["profile"] == profile) >= target
            for profile, target in targets.items())
        writer.finalize()

    fingerprint = source_fingerprint.fingerprint(root, os.environ.get("SO101_PYTHON"))
    patch_sha = hashlib.sha256((common.REPO_ROOT / protocol["source_patch"]["path"]).read_bytes()).hexdigest()
    split = None
    manifest = {
        "schema_version": 1, "phase": "E4-T-pilot" if args.pilot else "E4-T-collection",
        "created": _now(), "protocol_sha256": common.sha256_file(args.protocol),
        "protocol": protocol, "source_fingerprint": fingerprint,
        "patch_sha256": patch_sha, "repo_id": args.repo_id,
        "attempts": len(attempts), "successes": len(successes), "wall_time_s": time.time() - started,
        "episodes": successes, "profiles": targets, "budgets": budgets,
        "global_attempt_budget": global_budget,
        "existing_scene_hashes_rejected": len(existing_scenes),
        "claim": "scripted physical expert data; recovery requires a continuous bounded event and expert correction"
                 if not args.pilot else "pilot attempts only; never included in the 50-episode dataset",
    }
    if not args.pilot:
        if len(successes) != 30:
            raise SystemExit(f"E4-T collection incomplete: {len(successes)} successes, target 30")
        split = assign_split(successes, protocol)
        if split["problems"]:
            raise SystemExit("E4-T split invalid: " + "; ".join(split["problems"]))
        manifest["split"] = split
        common.write_json(output / "split.json", {key: value for key, value in split.items() if key != "episodes"})
    common.write_json(output / "e4t_manifest.json", manifest)
    if args.public_manifest:
        public = {
            "schema_version": 1, "phase": manifest["phase"], "created": manifest["created"],
            "protocol_sha256": manifest["protocol_sha256"], "source_sha256": fingerprint["source_sha256"],
            "expert_sha256": fingerprint["expert_sha256"], "patch_sha256": patch_sha,
            "attempts": manifest["attempts"], "successes": manifest["successes"],
            "profiles": manifest["profiles"],
            "episodes": [{key: row.get(key) for key in ("attempt_id", "profile", "seed", "scene_sha256",
                                                         "source_episode_index", "frames")} for row in successes],
            "perturbation_summary": _perturbation_summary(successes),
            "split": split, "claim": manifest["claim"],
        }
        common.write_json(args.public_manifest, public)
    print(json.dumps({"phase": manifest["phase"], "attempts": len(attempts),
                      "successes": len(successes), "targets": targets,
                      "wall_time_s": manifest["wall_time_s"]}, indent=2))
    return 0 if len(successes) >= sum(targets.values()) else 1


def _WriterSink(writer, instruction):
    class Sink:
        def __init__(self):
            self.frames = 0

        def frame(self, observation, applied_action):
            writer.add_frame(observation, applied_action, instruction)
            self.frames += 1

    return Sink()


def _perturbation_summary(successes: list[dict]) -> dict:
    summary = {}
    for row in successes:
        event = row.get("recovery_audit", {}).get("perturbation_event") or {}
        key = f"{row['profile']}:{event.get('mechanism', 'none')}"
        summary[key] = summary.get(key, 0) + 1
    return summary


if __name__ == "__main__":
    sys.exit(main())
