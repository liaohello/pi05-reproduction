#!/usr/bin/env python3
"""Two-episode dataset smoke for the single-block scene (one red, one blue).

Checks the real three-camera render path, LeRobot v3 writing, per-episode task
strings and frame counts before the 64-episode formal collection.  This is a
smoke test, not the formal dataset.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_common as common  # noqa: E402
import source_fingerprint  # noqa: E402
from scene1_collect import LeRobotSink  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--protocol", type=Path,
                        default=common.REPO_ROOT / "configs/so101/scene1_single_random_protocol.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default="15050:red,15051:blue")
    parser.add_argument("--repo-id", default="local/so101_scene1_dataset_smoke")
    args = parser.parse_args()

    root = common.setup_external(args.root)
    protocol = common.load_json(args.protocol)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to reuse existing smoke directory: {output}")
    output.mkdir(parents=True)
    from bimanual.config import load_config
    from bimanual.sim.environment import BimanualEnv
    from bimanual.tasks.sorting import regions
    from bimanual.tasks.sorting.certification import check_certificate_current
    from bimanual.tasks.single_block.scene_sampling import generate_single_scene
    from bimanual.recording.episode import EpisodeRunner
    from bimanual.integrations.lerobot.dataset import LeRobotWriter, verify_dataset

    config = load_config(root=root, profile=protocol["source_binding"]["profile"])
    config["runtime"]["control_hz"] = int(protocol["control"]["control_hz"])
    config["recording"]["fps"] = int(protocol["control"]["recording_fps"])
    config["sorting"]["timeout_s"] = float(protocol["control"]["timeout_s"])
    config["domain_randomization"]["enabled"] = False
    config["collection"]["recording_type"] = "scripted_single_block_place_demonstration"
    certificate = regions.load_certificate(root / config["evaluation"]["certificate"])
    pairs = []
    for item in args.seeds.split(","):
        seed, color = item.split(":")
        pairs.append((int(seed), color.strip()))
    rows = []
    with BimanualEnv(config, images=True) as env:
        stale = check_certificate_current(certificate, env)
        if stale:
            raise SystemExit("certificate stale: " + "; ".join(stale))
        writer = LeRobotWriter(config, output / "dataset", args.repo_id)
        runner = EpisodeRunner(env, config, certificate)
        for seed, color in pairs:
            scene = generate_single_scene(config, certificate, seed, color, env=env,
                                          max_attempts=400)
            sink = LeRobotSink(writer, scene.extra["instruction"])
            started = time.time()
            episode = runner.run_episode(scene, output=output / "diagnostics" / scene.extra["instruction"].split()[1],
                                         record=True, sink=sink)
            row = {"seed": seed, "color": color, "success": episode.success,
                   "failure_reason": episode.failure_reason,
                   "frames": sink.frames, "wall_time_s": time.time() - started,
                   "scene_sha256": scene.sha256,
                   "instruction": scene.extra["instruction"],
                   "yaw_degrees": scene.extra["yaw_degrees"],
                   "trays": scene.extra["trays"]}
            if episode.success:
                writer.save_episode()
                row["saved"] = True
            else:
                writer.discard_episode()
                row["saved"] = False
            rows.append(row)
            print(json.dumps(row), flush=True)
        writer.finalize()
        verify = verify_dataset(output / "dataset", args.repo_id, config["recording"]["fps"])
        meta = json.loads((output / "dataset/meta/info.json").read_text())
        tasks = []
        tasks_path = output / "dataset/meta/tasks.jsonl"
        if tasks_path.is_file():
            tasks = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]
        episodes_path = output / "dataset/meta/episodes.jsonl"
        episodes = ([json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
                    if episodes_path.is_file() else [])
    payload = {
        "schema_version": 1, "phase": "dataset_smoke",
        "protocol_sha256": common.sha256_file(args.protocol),
        "source_fingerprint": source_fingerprint.fingerprint(
            root, os.environ.get("SO101_PYTHON", sys.executable)),
        "certificate_sha256": common.sha256_file(root / config["evaluation"]["certificate"]),
        "repo_id": args.repo_id, "episodes": rows,
        "verify": verify, "tasks": tasks, "episode_meta": episodes,
        "info": {key: meta.get(key) for key in ("codebase_version", "robot_type", "fps",
                                                "total_episodes", "total_frames")},
        "claim": "two-episode dataset smoke; not the formal collection",
    }
    common.write_json(output / "smoke_manifest.json", payload)
    print(json.dumps(payload, indent=2)[:6000])
    return 0 if sum(1 for row in rows if row["saved"]) == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
