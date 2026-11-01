#!/usr/bin/env python3
"""Build the single-block pick/place certificate for workstream B.

The certificate binds the single-cube profile and the nominal tray layout.  Its
pick-point envelope is reused by the scene sampler; every sampled scene is then
screened again at its own yaw and tray positions.  The shared handoff area is
not part of this task and is recorded empty.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_common as common  # noqa: E402


def _now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--protocol", type=Path,
                        default=common.REPO_ROOT / "configs/so101/scene1_single_random_protocol.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid", type=float, default=0.01)
    args = parser.parse_args()

    root = common.setup_external(args.root)
    protocol = common.load_json(args.protocol)
    from bimanual.config import load_config
    from bimanual.sim.environment import BimanualEnv
    from bimanual.tasks.sorting import certification, regions

    config = load_config(root=root, profile=protocol["source_binding"]["profile"])
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing certificate: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with BimanualEnv(config, images=False) as env:
        env.data.qpos[env.robot.qpos_ids] = env.robot.to_radians(env.robot.home())
        import mujoco
        mujoco.mj_forward(env.model, env.data)
        profile = config["grasp"]
        hover_height = profile["hover_height_m"]
        tolerance = profile["robot_penetration_tolerance_m"]
        certificate = {
            "schema_version": regions.SCHEMA_VERSION,
            "created": _now(),
            "grid_m": args.grid,
            "hover_height_m": hover_height,
            "release_clearance_m": profile["release_clearance_m"],
            "retreat_clearance_m": profile["retreat_clearance_x_m"],
            "robot_penetration_tolerance_m": tolerance,
            "orientation_candidates_degrees": [0.0, 90.0],
            "model_sha256": regions.model_fingerprint(env),
            "scene_sha256": regions.config_fingerprint(config),
            "config_sha256": regions.config_fingerprint(config, names=("runtime", "camera")),
            "grasp_sha256": regions.config_fingerprint(config, names=("grasp",)),
            "pick_region": {arm: regions.pick_region(config, arm).as_dict()
                            for arm in ("left", "right")},
            "pick_points": {},
            "pick_rejected": {},
            "place_slots": {},
            "shared": {"region": regions.shared_region(config).as_dict(),
                       "grid_m": None, "certified_points": [],
                       "note": "not used by the single-block place task"},
            "park": {},
            "task": "single_block_place",
            "tray_sampling": config["sorting"]["tray_sampling"],
        }
        for arm in ("left", "right"):
            entry = certification.certify_pick_grid(env, arm, args.grid, (0.0, 90.0),
                                                    hover_height, tolerance)
            certificate["pick_points"][arm] = entry["points"]
            certificate["pick_rejected"][arm] = entry["rejected"]
            print(json.dumps({"arm": arm, "points": len(entry["points"]),
                              "rejected": len(entry["rejected"])}), flush=True)
        for color in ("red", "blue"):
            for slot in regions.tray_slots(config, color):
                certificate["place_slots"][slot.name] = certification.certify_place_slot(
                    env, color, slot)
        certificate["park"] = certification.certify_park(env)
        certificate["summary"] = {
            "pick_points": {arm: len(certificate["pick_points"][arm]) for arm in ("left", "right")},
            "place_slots": {name: value["certified"] for name, value in certificate["place_slots"].items()},
            "shared_points": 0,
            "park": {arm: value["certified"] for arm, value in certificate["park"].items()},
            "note": "pick envelope + nominal place references; per-scene place screening is mandatory",
        }
    regions.save_certificate(certificate, output)
    print(json.dumps({"output": str(output), **certificate["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
