#!/usr/bin/env python3
"""Physical pilot and formal collection for single-block random scenes.

Only attempts that pass the registered physical predicate are saved.  Every
attempt, including layout rejections, duplicate scenes and expert failures, is
written to ``attempts.jsonl`` with its SceneSpec.  The pilot never enters the
dataset; the formal run targets the registered colour-balanced count.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_common as common  # noqa: E402
import source_fingerprint  # noqa: E402


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _color_for(red: int, blue: int) -> str:
    return "red" if red <= blue else "blue"


class RenderSink:
    """Render the real three-camera observation without writing a dataset."""

    def __init__(self):
        self.frames = 0

    def frame(self, observation, applied_action):
        self.frames += 1


class LeRobotSink:
    def __init__(self, writer, instruction):
        self.writer = writer
        self.instruction = instruction
        self.frames = 0

    def frame(self, observation, applied_action):
        self.writer.add_frame(observation, applied_action, self.instruction)
        self.frames += 1


def family_key(row):
    scene = row["scene"]
    extra = scene.get("extra") or scene
    block = extra["block_xy"]
    tray = extra["trays"][extra["color"]]
    yaw = float(extra["yaw_degrees"])
    return (extra["color"],
            int(math.floor(float(block[0]) / 0.0125)),
            int(math.floor(float(block[1]) / 0.0125)),
            int(math.floor(float(tray[0]) / 0.0125)),
            int(math.floor(float(tray[1]) / 0.0125)),
            int(math.floor(yaw / 15.0)))


def _coverage(rows):
    bins = set()
    for row in rows:
        extra = row["scene"].get("extra") or row["scene"]
        block = extra["block_xy"]
        tray = extra["trays"][extra["color"]]
        bins.add(("bx", int(float(block[0]) / 0.03)))
        bins.add(("by", int(float(block[1]) / 0.03)))
        bins.add(("tx", int(float(tray[0]) / 0.02)))
        bins.add(("ty", int(float(tray[1]) / 0.02)))
        bins.add(("yaw", int(float(extra["yaw_degrees"]) / 15.0)))
    return len(bins)


def split_by_family(successes):
    """Select exactly 6 validation episodes per colour with whole families."""
    validation = []
    for color in ("red", "blue"):
        rows = [row for row in successes
                if (row["scene"].get("extra") or row["scene"])["color"] == color]
        groups = {}
        for row in rows:
            groups.setdefault(family_key(row), []).append(row)
        items = list(groups.items())
        best = {0: (0, 0, ())}

        def better(first, second):
            return first[0] > second[0] or (first[0] == second[0] and first[1] < second[1])

        for index, (key, members) in enumerate(items):
            size = len(members)
            for total in sorted(list(best), reverse=True):
                if total + size > 6:
                    continue
                coverage, count, chosen = best[total]
                candidate = (coverage + _coverage(members), count + 1,
                             chosen + (index,))
                current = best.get(total + size)
                if current is None or better(candidate, current):
                    best[total + size] = candidate
        if 6 not in best:
            raise SystemExit(f"no whole-family validation subset of size 6 for {color}")
        chosen = best[6][2]
        for index in chosen:
            validation.extend(items[index][1])
    validation_ids = {id(row) for row in validation}
    train = [row for row in successes if id(row) not in validation_ids]
    return train, validation


def _load_config(root, protocol):
    from bimanual.config import load_config

    config = load_config(root=root, profile=protocol["source_binding"]["profile"])
    config["runtime"]["control_hz"] = int(protocol["control"]["control_hz"])
    config["recording"]["fps"] = int(protocol["control"]["recording_fps"])
    config["sorting"]["timeout_s"] = float(protocol["control"]["timeout_s"])
    config["domain_randomization"]["enabled"] = False
    config["collection"]["recording_type"] = "scripted_single_block_place_demonstration"
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--protocol", type=Path,
                        default=common.REPO_ROOT / "configs/so101/scene1_single_random_protocol.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--pilot-attempts", type=int, default=12)
    parser.add_argument("--target", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--seed-end", type=int, default=None)
    parser.add_argument("--max-layout-attempts", type=int, default=400)
    parser.add_argument("--repo-id", default=None)
    args = parser.parse_args()

    root = common.setup_external(args.root)
    protocol = common.load_json(args.protocol)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to reuse existing run directory: {output}")
    output.mkdir(parents=True)
    config = _load_config(root, protocol)
    from bimanual.sim.environment import BimanualEnv
    from bimanual.tasks.sorting import regions
    from bimanual.tasks.sorting.certification import check_certificate_current
    from bimanual.tasks.single_block.scene_sampling import (SceneGenerationError,
                                                            generate_single_scene)
    from bimanual.recording.episode import EpisodeRunner
    from bimanual.integrations.lerobot.dataset import LeRobotWriter, verify_dataset

    certificate_path = root / config["evaluation"]["certificate"]
    certificate = regions.load_certificate(certificate_path)
    collection = protocol["collection"]
    formal = collection["formal"]
    if args.pilot:
        target = int(args.target or collection["pilot"]["physical"]["target_successes"])
        maximum_attempts = int(args.max_attempts or args.pilot_attempts)
        seed_start = int(args.seed_start or collection["seed_registry"]["pilot"][0])
        seed_end = int(args.seed_end or collection["seed_registry"]["pilot"][1])
        repo_id = None
    else:
        target = int(args.target or formal["target_successes"])
        if args.max_attempts is None:
            if formal.get("maximum_attempts") is None:
                raise SystemExit("formal maximum_attempts is not frozen; run the pilot and register it")
            maximum_attempts = int(formal["maximum_attempts"])
        else:
            maximum_attempts = int(args.max_attempts)
        seed_start = int(args.seed_start or collection["seed_registry"]["collection"][0])
        seed_end = int(args.seed_end or collection["seed_registry"]["collection"][1])
        repo_id = args.repo_id or formal["repo_id"]

    attempts_path = output / "attempts.jsonl"
    successes = []
    attempts = 0
    layout_rejects = 0
    seen_scenes = {}
    red_success = blue_success = 0
    started = time.time()
    writer = None
    with BimanualEnv(config, images=not args.pilot) as env:
        stale = check_certificate_current(certificate, env)
        if stale:
            raise SystemExit("certificate stale: " + "; ".join(stale))
        if not args.pilot:
            writer = LeRobotWriter(config, output / "dataset", repo_id)
        runner = EpisodeRunner(env, config, certificate)
        seed = seed_start
        while attempts < maximum_attempts and seed <= seed_end:
            if red_success + blue_success >= target:
                break
            color = _color_for(red_success, blue_success) if not args.pilot else (
                "red" if (seed - seed_start) % 2 == 0 else "blue")
            attempt_id = f"{color}_{seed:05d}"
            trace = []
            try:
                scene = generate_single_scene(config, certificate, seed, color, env=env,
                                              max_attempts=args.max_layout_attempts, trace=trace)
            except SceneGenerationError as error:
                layout_rejects += 1
                row = {"attempt_id": attempt_id, "kind": "layout_reject", "seed": seed,
                       "color": color, "success": False, "failure_reason": "scene_generation_failed",
                       "message": str(error)[:500], "sampler_trace": trace}
                with attempts_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, default=str) + "\n")
                seed += 1
                continue
            if scene.sha256 in seen_scenes:
                row = {"attempt_id": attempt_id, "kind": "duplicate_scene", "seed": seed,
                       "color": color, "scene_sha256": scene.sha256, "success": False,
                       "failure_reason": "duplicate_scene_identity",
                       "duplicate_of": seen_scenes[scene.sha256]}
                with attempts_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, default=str) + "\n")
                attempts += 1
                seed += 1
                continue
            seen_scenes[scene.sha256] = attempt_id
            attempts += 1
            directory = output / "diagnostics" / attempt_id
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "scene_generation.json").write_text(
                json.dumps({"seed": seed, "color": color, "trace": trace}, indent=2) + "\n",
                encoding="utf-8")
            sink = (None if args.pilot
                    else LeRobotSink(writer, scene.extra["instruction"]))
            episode_started = time.time()
            try:
                episode = runner.run_episode(scene, output=directory, record=True, sink=sink)
            except Exception as error:
                if writer is not None:
                    writer.discard_episode()
                row = {"attempt_id": attempt_id, "kind": "exception", "seed": seed,
                       "color": color, "scene_sha256": scene.sha256, "success": False,
                       "failure_reason": "unexpected_exception",
                       "message": f"{type(error).__name__}: {error}",
                       "scene": scene.as_dict(), "frames": 0,
                       "wall_time_s": time.time() - episode_started}
                with attempts_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, default=str) + "\n")
                seed += 1
                continue
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            guard_errors = int(summary.get("contact_monitor_error_count", 0))
            accepted = bool(summary.get("success")) and summary.get("sorted_count") == 1 \
                and guard_errors == 0
            row = {"attempt_id": attempt_id, "kind": "episode", "seed": seed, "color": color,
                   "scene_sha256": scene.sha256, "scene": scene.as_dict(),
                   "success": accepted, "raw_success": bool(summary.get("success")),
                   "sorted_count": int(summary.get("sorted_count", 0)),
                   "failure_reason": (None if accepted else
                                      (summary.get("failure_reason") or "predicate_not_met")),
                   "failure_stage": summary.get("failure_stage"),
                   "failure_message": summary.get("failure_message"),
                   "frames": int(summary.get("frames", 0)),
                   "guard_frames": (sink.frames if sink is not None else 0),
                   "contact_monitor_error_count": guard_errors,
                   "soft_contact_count": int(summary.get("soft_contact_count", 0)),
                   "sim_time_s": float(summary.get("sim_time_s", 0.0)),
                   "wall_time_s": float(summary.get("wall_time_s", 0.0)),
                   "instruction": scene.extra["instruction"],
                   "diagnostics": str(directory),
                   "attempt": attempts,
                   "sampler_attempts": scene.extra["checks"]["sampler_attempts"],
                   "sampler_first_draw": bool(len(trace) == 1),
                   "source_episode_index": None}
            if accepted:
                if writer is not None:
                    index = int(writer.dataset.meta.total_episodes)
                    writer.save_episode()
                    row["source_episode_index"] = index
                successes.append(row)
                if color == "red":
                    red_success += 1
                else:
                    blue_success += 1
            elif writer is not None:
                writer.discard_episode()
            with attempts_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, default=str) + "\n")
            print(json.dumps({"attempt": attempts, "seed": seed, "color": color,
                              "accepted": accepted,
                              "reason": row["failure_reason"],
                              "yaw": scene.extra["yaw_degrees"],
                              "tray": scene.extra["trays"][color],
                              "wall_s": round(row["wall_time_s"], 2)}), flush=True)
            seed += 1
        if writer is not None:
            writer.finalize()
            verify = verify_dataset(output / "dataset", repo_id, config["recording"]["fps"])
        else:
            verify = None
    fingerprint = source_fingerprint.fingerprint(root, os.environ.get("SO101_PYTHON", sys.executable))
    patch_path = common.REPO_ROOT / protocol["source_binding"]["source_patch"]["path"]
    patch_sha = common.sha256_file(patch_path) if patch_path.is_file() else None
    manifest = {
        "schema_version": 1, "created": _now(), "phase": "pilot" if args.pilot else "formal",
        "protocol_id": protocol["protocol_id"], "protocol_sha256": common.sha256_file(args.protocol),
        "certificate_sha256": common.sha256_file(certificate_path),
        "base_source_sha256": protocol["source_binding"]["source_sha256"],
        "current_source_fingerprint": fingerprint, "patch_sha256": patch_sha,
        "repo_id": repo_id, "seed_range": [seed_start, seed_end],
        "attempts": attempts, "layout_rejects": layout_rejects,
        "successes": len(successes), "target": target,
        "red_successes": red_success, "blue_successes": blue_success,
        "wall_time_s": time.time() - started,
        "successes_rows": successes, "verify": verify,
        "attempts_file": str(attempts_path),
        "claim": ("physical pilot only; never included in the training dataset" if args.pilot
                  else "single-block scripted-expert demonstrations, not a benchmark success rate"),
    }
    if not args.pilot:
        train, validation = split_by_family(successes)
        manifest["split"] = {
            "train": len(train), "validation": len(validation),
            "train_red": sum(1 for row in train if row["color"] == "red"),
            "train_blue": sum(1 for row in train if row["color"] == "blue"),
            "validation_red": sum(1 for row in validation if row["color"] == "red"),
            "validation_blue": sum(1 for row in validation if row["color"] == "blue"),
            "family_crosses": len({family_key(row) for row in train} & {family_key(row) for row in validation}),
            "train_indices": [row["source_episode_index"] for row in train],
            "validation_indices": [row["source_episode_index"] for row in validation],
            "train_seeds": [row["seed"] for row in train],
            "validation_seeds": [row["seed"] for row in validation],
            "rule": protocol["scene_family"]["rule"],
        }
        if len(successes) != target or red_success != 32 or blue_success != 32:
            manifest["complete"] = False
            common.write_json(output / "collection_manifest.json", manifest)
            raise SystemExit(f"formal collection incomplete: {len(successes)}/{target}, "
                             f"red {red_success}, blue {blue_success}")
        if (len(train) != 52 or len(validation) != 12
                or manifest["split"]["family_crosses"] != 0
                or manifest["split"]["train_red"] != 26 or manifest["split"]["train_blue"] != 26
                or manifest["split"]["validation_red"] != 6 or manifest["split"]["validation_blue"] != 6):
            manifest["complete"] = False
            common.write_json(output / "collection_manifest.json", manifest)
            raise SystemExit("formal split invalid: " + json.dumps(manifest["split"]))
    common.write_json(output / ("pilot_manifest.json" if args.pilot else "collection_manifest.json"),
                      manifest)
    print(json.dumps({key: value for key, value in manifest.items()
                      if key not in ("successes_rows", "current_source_fingerprint")}, indent=2))
    return 0 if len(successes) >= target else 1


if __name__ == "__main__":
    sys.exit(main())
