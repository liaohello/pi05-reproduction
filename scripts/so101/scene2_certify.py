#!/usr/bin/env python3
"""Build the scene2 geometric certificate on the independent source snapshot.

The certificate certifies the pick envelope for both arms and the nominal place
slots for both tray arrangements.  It is an envelope, not a per-scene place
guarantee: the sampler re-screens every accepted layout at its own yaw and tray
pose.  The shared handoff area is not used by this task and is recorded empty.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene2_common as s2  # noqa: E402


def _now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def _certify_place_slot_yaws(env, color, slot, yaws, domain_yaws) -> dict:
    """Nominal place reference at an explicit yaw list.

    The external helper only tests 0 and 90 degrees.  With the registered 70 %
    opening the withdrawn fingers can touch a tray wall at those yaws, so the
    reference is evaluated over the actual candidate domain and the excluded
    yaws are kept in the record instead of being dropped.
    """
    certification = s2._b("certification")
    regions = s2._b("regions")
    PoseIK = s2._b("PoseIK")
    profile = env.config["grasp"]
    arm = regions.destination_arm(env.config, color)
    ik = PoseIK(env)
    support_z = float(slot.support_z)
    release_z = support_z + regions.cube_half(env.config) + profile["release_clearance_m"]
    hover_height = profile.get("place_hover_height_m", profile["hover_height_m"])
    duration = profile["stage_durations_s"]
    results = {}
    for degrees in yaws:
        rotation = certification._rotation(degrees)
        release = ik.solve(arm, [slot.xy[0], slot.xy[1], release_z], rotation,
                           tolerance=0.0002, rotation_tolerance=0.01)
        if not release.converged:
            results[str(int(degrees))] = {"certified": False, "reason": "release_ik"}
            continue
        hover = ik.solve(arm, [slot.xy[0], slot.xy[1], release_z + hover_height], rotation,
                         seed_rad=release.joints_rad, tolerance=0.0002,
                         rotation_tolerance=0.01)
        if not hover.converged:
            results[str(int(degrees))] = {"certified": False, "reason": "place_hover_ik"}
            continue
        base = env.robot.home()
        low = certification._arm_target(base, arm, joints=release.joints_rad,
                                        gripper=profile["closed_percent"])
        high = certification._arm_target(base, arm, joints=hover.joints_rad,
                                         gripper=profile["closed_percent"])
        descent, _ = certification._interpolate(env, high, low, duration["lower"])
        descent_screen = certification._screen(env, arm, descent, 0.0005)
        if not descent_screen["passed"]:
            results[str(int(degrees))] = {"certified": False, "reason": "descent_path",
                                          "contacts": descent_screen.get("contacts", [])[:3]}
            continue
        withdraw = None
        for sign in (-1.0, 1.0):
            sideways_xy = np.asarray(slot.xy, dtype=float) + rotation[:, 0][:2] * \
                profile["retreat_clearance_x_m"] * sign
            sideways = ik.solve(arm, [sideways_xy[0], sideways_xy[1], release_z], rotation,
                                seed_rad=release.joints_rad, tolerance=0.0005,
                                rotation_tolerance=0.02)
            if not sideways.converged:
                continue
            lifted = ik.solve(arm, [sideways_xy[0], sideways_xy[1], release_z + hover_height],
                              rotation, seed_rad=sideways.joints_rad, tolerance=0.0008,
                              rotation_tolerance=0.03)
            if not lifted.converged:
                continue
            out = certification._arm_target(base, arm, joints=sideways.joints_rad,
                                            gripper=profile["open_percent"])
            up = certification._arm_target(base, arm, joints=lifted.joints_rad,
                                           gripper=profile["open_percent"])
            release_path, _ = certification._interpolate(env, low, out, duration["release"] / 2)
            retreat_path, _ = certification._interpolate(env, out, up, duration["retreat"] / 2)
            release_screen = certification._screen(env, arm, release_path, 0.0005)
            retreat_screen = certification._screen(env, arm, retreat_path, 0.0005)
            if release_screen["passed"] and retreat_screen["passed"]:
                withdraw = {"sign": sign, "release": True, "retreat": True}
                break
        if withdraw is None:
            results[str(int(degrees))] = {"certified": False, "reason": "withdraw_path"}
        else:
            results[str(int(degrees))] = {"certified": True, "release_z_m": release_z,
                                          "hover_z_m": release_z + hover_height,
                                          "withdraw_sign": withdraw["sign"]}
    domain = {str(int(value)) for value in domain_yaws}
    return {"name": slot.name, "color": color, "arm": arm, "support_z": support_z,
            "yaws": results,
            "certified": any(value["certified"] for value in results.values()),
            "certified_in_domain": any(results.get(key, {}).get("certified", False)
                                       for key in domain),
            "domain_yaws": sorted(int(value) for value in domain_yaws)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--protocol", type=Path,
                        default=s2.REPO_ROOT / "configs/so101/scene2_dual_random_protocol.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid", type=float, default=0.01)
    args = parser.parse_args()

    root = s2.configure(args.root)
    protocol = s2.load_protocol(args.protocol)
    s2.set_protocol_id(protocol["protocol_id"])
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing certificate: {output}")
    config = s2.load_config(root, protocol, certificate=output)
    BimanualEnv = s2._b("BimanualEnv")
    certification = s2._b("certification")
    regions = s2._b("regions")
    mujoco = s2._b("mujoco")

    output.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    with BimanualEnv(config, images=False) as env:
        env.data.qpos[env.robot.qpos_ids] = env.robot.to_radians(env.robot.home())
        mujoco.mj_forward(env.model, env.data)
        profile = config["grasp"]
        certificate = {
            "schema_version": regions.SCHEMA_VERSION,
            "created": _now(),
            "grid_m": args.grid,
            "hover_height_m": profile["hover_height_m"],
            "release_clearance_m": profile["release_clearance_m"],
            "retreat_clearance_m": profile["retreat_clearance_x_m"],
            "robot_penetration_tolerance_m": profile["robot_penetration_tolerance_m"],
            "open_percent": profile["open_percent"],
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
                       "note": "not used by the dual-block select-place task"},
            "park": {},
            "task": "dual_block_select_place",
            "tray_sampling": {
                "positive_y": config["sorting"]["tray_sampling_positive"],
                "negative_y": config["sorting"]["tray_sampling_negative"],
            },
            "arrangements": ["red_positive_y", "red_negative_y"],
        }
        for arm in ("left", "right"):
            entry = certification.certify_pick_grid(
                env, arm, args.grid, (0.0, 90.0), profile["hover_height_m"],
                profile["robot_penetration_tolerance_m"])
            certificate["pick_points"][arm] = entry["points"]
            certificate["pick_rejected"][arm] = entry["rejected"]
            print(json.dumps({"arm": arm, "points": len(entry["points"]),
                              "rejected": len(entry["rejected"])}), flush=True)

        base_trays = {color: list(config["scene"]["trays"][color]) for color in ("red", "blue")}
        base_destination = config["sorting"].get("destination_arm")
        yaws = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]
        domain_yaws = list(range(int(config["sorting"]["yaw_domain_degrees"][0]),
                                 int(config["sorting"]["yaw_domain_degrees"][1]) + 1, 15))
        for color in ("red", "blue"):
            for slot in regions.tray_slots(config, color):
                certificate["place_slots"][slot.name] = _certify_place_slot_yaws(
                    env, color, slot, yaws, domain_yaws)
        print(json.dumps({"place_slots": len(certificate["place_slots"])}), flush=True)

        swapped = {"red": [0.24, -0.26], "blue": [0.24, 0.26]}
        with s2.trays_at(env, swapped):
            config["scene"]["trays"]["red"] = list(swapped["red"])
            config["scene"]["trays"]["blue"] = list(swapped["blue"])
            config["sorting"]["destination_arm"] = {"red": "right", "blue": "left"}
            for color in ("red", "blue"):
                for slot in regions.tray_slots(config, color):
                    certificate["place_slots"][f"swapped_{slot.name}"] = \
                        _certify_place_slot_yaws(env, color, slot, yaws, domain_yaws)
        config["scene"]["trays"]["red"] = base_trays["red"]
        config["scene"]["trays"]["blue"] = base_trays["blue"]
        if base_destination is None:
            config["sorting"].pop("destination_arm", None)
        else:
            config["sorting"]["destination_arm"] = base_destination
        print(json.dumps({"place_slots_all": len(certificate["place_slots"])}), flush=True)

        certificate["park"] = certification.certify_park(env)
        certificate["summary"] = {
            "pick_points": {arm: len(points) for arm, points in certificate["pick_points"].items()},
            "place_slots": {name: bool(value["certified"])
                            for name, value in certificate["place_slots"].items()},
            "place_slots_in_yaw_domain": {name: bool(value["certified_in_domain"])
                                          for name, value in certificate["place_slots"].items()},
            "shared_points": 0,
            "park": {arm: bool(value.get("certified"))
                     for arm, value in certificate["park"].items()},
            "open_percent": profile["open_percent"],
            "note": "pick envelope + nominal place references for both arrangements; "
                    "per-scene place screening is mandatory",
        }
        stale = certification.check_certificate_current(regions.Certificate(certificate), env)
        if stale:
            raise SystemExit("freshly built certificate is stale: " + "; ".join(stale))
    regions.save_certificate(certificate, output)
    print(json.dumps({"output": str(output), **certificate["summary"],
                      "wall_time_s": (datetime.now(timezone.utc) - started).total_seconds()},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
