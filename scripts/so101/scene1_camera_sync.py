#!/usr/bin/env python3
"""Verify the three cameras are rendered from one pre-action MuJoCo state.

The check wraps ``BimanualEnv.render`` and records the simulator time at each
camera call inside one ``observe``.  It also steps once and confirms the time
advances by exactly one control period.  No episode is saved.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_common as common  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--protocol", type=Path,
                        default=common.REPO_ROOT / "configs/so101/scene1_single_random_protocol.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = common.setup_external(args.root)
    protocol = common.load_json(args.protocol)
    if args.output.exists():
        raise SystemExit(f"refusing to reuse existing output: {args.output}")
    args.output.mkdir(parents=True)
    from bimanual.config import load_config
    from bimanual.sim.environment import BimanualEnv
    from bimanual.tasks.sorting import regions
    from bimanual.tasks.single_block.scene_sampling import generate_single_scene

    config = load_config(root=root, profile=protocol["source_binding"]["profile"])
    certificate = regions.load_certificate(root / config["evaluation"]["certificate"])
    observations = []
    with BimanualEnv(config, images=True) as env:
        scene = generate_single_scene(config, certificate, 15050, "red", env=env, max_attempts=400)
        env.reset(seed=scene.seed, scene=scene)
        original = env.render
        calls = []

        def logged(camera="central", depth=False):
            calls.append({"camera": camera, "sim_time": float(env.data.time),
                          "elapsed_steps": int(env.elapsed_steps)})
            return original(camera=camera, depth=depth)

        env.render = logged
        for frame in range(10):
            before = len(calls)
            observation = env.observe()
            group = calls[before:]
            observations.append({
                "frame": frame,
                "calls": group,
                "cameras": [entry["camera"] for entry in group],
                "same_sim_time": len({entry["sim_time"] for entry in group}) == 1,
                "same_elapsed_steps": len({entry["elapsed_steps"] for entry in group}) == 1,
                "image_shapes": {key: list(value.shape) for key, value in observation.items()
                                 if key.startswith("observation.images.")},
            })
        env.step(env.last_action.copy(), observe=False)
        observations.append({"frame": 10, "calls": [], "step_advanced_s": float(env.data.time)})
    expected = [k for k in ("observation.images.central", "observation.images.left_wrist",
                            "observation.images.right_wrist")]
    passed = all(row["same_sim_time"] and row["same_elapsed_steps"]
                 and row["cameras"] == ["central", "left_wrist", "right_wrist"]
                 and set(row["image_shapes"]) == set(expected)
                 for row in observations[:-1])
    payload = {"schema_version": 1, "protocol_sha256": common.sha256_file(args.protocol),
               "camera_order": ["central", "left_wrist", "right_wrist"],
               "frames": len(observations) - 1, "passed": bool(passed),
               "note": "one observe() call renders all three cameras before any physics step",
               "observations": observations}
    common.write_json(args.output / "camera_sync.json", payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "observations"}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
