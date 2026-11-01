#!/usr/bin/env python3
"""Post-hoc carry-path IK audit for collected single-block scenes.

The scene sampler screens pick, hover and place poses, but the reactive carry
between them can still hit an IK local minimum.  This read-only audit samples
the straight carry line for every attempt scene and records failures without
changing the collection.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_common as common  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--protocol", type=Path,
                        default=common.REPO_ROOT / "configs/so101/scene1_single_random_protocol.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--waypoints", type=int, default=20)
    args = parser.parse_args()
    root = common.setup_external(args.root)
    protocol = common.load_json(args.protocol)
    from bimanual.config import load_config
    from bimanual.sim.environment import BimanualEnv
    from bimanual.tasks.sorting import regions
    from bimanual.tasks.single_block.scene_sampling import trays_at
    from bimanual.control.ik import PoseIK
    from bimanual.control.motion import rotation_z
    import mujoco

    config = load_config(root=root, profile=protocol["source_binding"]["profile"])
    rows = [json.loads(line) for line in (args.collection / "attempts.jsonl").read_text().splitlines()
            if line.strip()]
    results = []
    with BimanualEnv(config, images=False) as env:
        env.data.qpos[env.robot.qpos_ids] = env.robot.to_radians(env.robot.home())
        mujoco.mj_forward(env.model, env.data)
        for row in rows:
            scene = row.get("scene")
            if not scene:
                continue
            extra = scene.get("extra") or scene
            arm = extra["active_arm"]
            offset = 0 if arm == "left" else 6
            trays = {color: extra["trays"][color] for color in ("red", "blue")}
            with trays_at(env, trays):
                table = float(config["scene"]["table"]["surface_z"])
                half = regions.cube_half(config)
                block = np.asarray(extra["block_xy"], dtype=float)
                slot = np.asarray(extra["slot_xy"], dtype=float)
                start = np.array([block[0], block[1], table + half + config["grasp"]["hover_height_m"]])
                end = np.array([slot[0], slot[1],
                                extra["support_z"] + half + config["grasp"]["release_clearance_m"]
                                + config["grasp"].get("place_hover_height_m", config["grasp"]["hover_height_m"])])
                rotation = rotation_z(float(extra["yaw_degrees"]))
                ik = PoseIK(env)
                seed = env.data.qpos[env.robot.qpos_ids[offset:offset + 5]].copy()
                failed = None
                for index, fraction in enumerate(np.linspace(0, 1, args.waypoints + 1)[1:]):
                    weight = fraction * fraction * (3 - 2 * fraction)
                    point = start + (end - start) * weight
                    pose = ik.solve(arm, point, rotation, seed_rad=seed,
                                    tolerance=0.0015, rotation_tolerance=0.035)
                    if not pose.converged:
                        failed = {"waypoint": index + 1, "fraction": round(float(fraction), 4),
                                  "point": point.round(5).tolist(),
                                  "position_error_m": pose.position_error_m,
                                  "rotation_error_rad": pose.rotation_error_rad}
                        break
                    seed = pose.joints_rad
            results.append({"attempt_id": row.get("attempt_id"), "seed": row.get("seed"),
                            "color": extra["color"], "recorded_success": row.get("success"),
                            "yaw_degrees": extra["yaw_degrees"],
                            "block_xy": extra["block_xy"], "slot_xy": extra["slot_xy"],
                            "carry_ik_passed": failed is None, "failure": failed})
    payload = {"schema_version": 1, "waypoints": args.waypoints,
               "scenes_checked": len(results),
               "carry_ik_failures": sum(1 for row in results if not row["carry_ik_passed"]),
               "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "results"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
