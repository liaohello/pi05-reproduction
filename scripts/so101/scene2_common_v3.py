"""V3 per-scene release/withdraw search for the scene2 dual-block family.

v3 builds on v2 and keeps every v1/v2 file untouched.  The only behavioural
change is that release opening and withdraw direction are searched at the
actual release pose instead of being fixed by the profile.  The success
predicate and the 0.5 mm contact guard are unchanged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import scene2_common as _v1
import scene2_common_v2 as _v2
from scene2_common_v2 import *  # noqa: F401,F403

_b = _v1._b
_OPEN_CANDIDATES = (40.0, 50.0, 30.0)
_DIRECTIONS = ("gripper_x_neg", "gripper_x_pos", "world_x_neg", "world_x_pos")
_RETREAT_M = 0.06


def configure(root, *, python=None):
    root = _v2.configure(root, python=python)
    _v1.place_verdict = place_verdict
    _v1.dual_block_expert_class = dual_block_expert_class
    return root


def provenance(root, protocol_path, config, certificate_path, adapter_path=None):
    result = _v1.provenance(root, protocol_path, config, certificate_path,
                            adapter_path=Path(__file__).resolve())
    protocol = _v1.load_protocol(protocol_path)
    profile = _v1.REPO_ROOT / protocol["source_binding"]["profile"]
    result["profile_sha256"] = _v1.common.sha256_file(profile)
    result["adapter_sha256"] = _v1.common.sha256_file(Path(__file__).resolve())
    return result


def _direction_vector(name, rotation):
    if name == "gripper_x_neg":
        return -rotation[:, 0][:2]
    if name == "gripper_x_pos":
        return rotation[:, 0][:2]
    if name == "world_x_neg":
        return np.array([-1.0, 0.0])
    if name == "world_x_pos":
        return np.array([1.0, 0.0])
    if name == "world_y_neg":
        return np.array([0.0, -1.0])
    if name == "world_y_pos":
        return np.array([0.0, 1.0])
    raise ValueError(f"unknown withdraw direction {name!r}")


def place_verdict(env, config, arm, target_name, slot, yaw):
    profile = config["grasp"]
    regions = _v1._b("regions")
    cube_half = regions.cube_half(config)
    support_z = float(slot.support_z)
    release_z = support_z + cube_half + profile["release_clearance_m"]
    place_hover_height = profile.get("place_hover_height_m", profile["hover_height_m"])
    lift_height = profile["hover_height_m"]
    opens = list(profile.get("scene2_release_open_candidates", _OPEN_CANDIDATES))
    directions = list(profile.get("scene2_withdraw_directions", _DIRECTIONS))
    retreat_m = float(profile.get("scene2_withdraw_retreat_m", _RETREAT_M))
    rotation = _v1._b("rotation_z")(yaw)
    offset = 0 if arm == "left" else 6
    seed = env.data.qpos[env.robot.qpos_ids[offset:offset + 5]].copy()
    ik = _v1._b("PoseIK")(env)
    release = ik.solve(arm, [slot.xy[0], slot.xy[1], release_z], rotation,
                       seed_rad=seed, tolerance=0.0002, rotation_tolerance=0.01)
    if not release.converged:
        return {"passed": False, "reason": "release_ik",
                "position_error_m": release.position_error_m}
    place_hover = ik.solve(arm, [slot.xy[0], slot.xy[1], release_z + place_hover_height],
                           rotation, seed_rad=release.joints_rad, tolerance=0.0002,
                           rotation_tolerance=0.01)
    lift_hover = ik.solve(arm, [slot.xy[0], slot.xy[1], release_z + lift_height], rotation,
                          seed_rad=release.joints_rad, tolerance=0.0002,
                          rotation_tolerance=0.01)
    if not (place_hover.converged and lift_hover.converged):
        return {"passed": False, "reason": "place_hover_ik"}
    center = np.asarray(config["scene"]["trays"][slot.color], dtype=float)
    inner = regions.tray_inner_half(config)
    extents = np.abs(rotation) @ np.full(3, cube_half)
    contained = bool(np.all(np.abs(np.asarray(slot.xy) - center) + extents[:2] <= inner))
    if not contained:
        return {"passed": False, "reason": "slot_containment", "extents": extents.tolist()}
    home = env.robot.home()
    closed = profile["closed_percent"]
    low = _v1._arm_target(home, arm, release.joints_rad, closed)
    durations = profile["stage_durations_s"]
    descent = _v1._path(env, _v1._arm_target(home, arm, place_hover.joints_rad, closed),
                        low, durations["lower"])
    descent_screen = _v1._screen_joint(env, arm, descent, target_name)
    release_checks = []
    chosen_open = None
    for open_percent in opens:
        action = _v1._arm_target(home, arm, release.joints_rad, float(open_percent))
        screen = _v1._screen_joint(env, arm, [action], target_name)
        release_checks.append({"open_percent": float(open_percent),
                               "passed": bool(screen["passed"]),
                               "screen": _v1._compact_screen(screen)})
        if screen["passed"]:
            chosen_open = float(open_percent)
            break
    if chosen_open is None:
        return {"passed": False, "reason": "release_open_contact",
                "descent": _v1._compact_screen(descent_screen),
                "release": release_checks}
    high = _v1._arm_target(home, arm, lift_hover.joints_rad, chosen_open)
    lift = _v1._path(env, low, high, durations["release"] / 2)
    lift_screen = _v1._screen_joint(env, arm, lift, target_name)
    if not lift_screen["passed"]:
        return {"passed": False, "reason": "lift_path",
                "descent": _v1._compact_screen(descent_screen),
                "release": release_checks,
                "lift": _v1._compact_screen(lift_screen)}
    direction_checks = []
    chosen_direction = None
    lift_z = release_z + lift_height
    for name in directions:
        vector = _direction_vector(name, rotation)
        retreat_xy = (float(slot.xy[0]) + float(vector[0]) * retreat_m,
                      float(slot.xy[1]) + float(vector[1]) * retreat_m)
        retreat = ik.solve(arm, [retreat_xy[0], retreat_xy[1], lift_z], rotation,
                           seed_rad=lift_hover.joints_rad, tolerance=0.0005,
                           rotation_tolerance=0.02)
        if not retreat.converged:
            direction_checks.append({"direction": name, "passed": False,
                                     "reason": "retreat_ik"})
            continue
        out = _v1._arm_target(home, arm, retreat.joints_rad, chosen_open)
        path = _v1._path(env, high, out,
                         durations["release"] / 2 + durations["retreat"] / 2)
        screen = _v1._screen_joint(env, arm, path, target_name)
        direction_checks.append({"direction": name, "passed": bool(screen["passed"]),
                                 "retreat_xy": [round(retreat_xy[0], 6), round(retreat_xy[1], 6)],
                                 "screen": _v1._compact_screen(screen)})
        if screen["passed"]:
            chosen_direction = name
            break
    passed = bool(descent_screen["passed"] and chosen_direction is not None)
    return {"passed": passed,
            "reason": None if passed else ("place_path" if not descent_screen["passed"]
                                            else "vertical_withdraw"),
            "descent": _v1._compact_screen(descent_screen),
            "release": release_checks,
            "chosen_open_percent": chosen_open,
            "lift": _v1._compact_screen(lift_screen),
            "directions": direction_checks,
            "chosen_direction": chosen_direction,
            "release_z_m": release_z, "lift_z_m": lift_z,
            "contained": contained, "extents": extents.tolist()}


def dual_block_expert_class():
    base = _v1._b("SingleBlockExpert")
    regions = _v1._b("regions")

    class DualBlockExpert(base):
        def _slot(self, name):
            return regions.TraySlot(name, self.scene.extra["target_color"],
                                    tuple(self.scene.extra["slot_xy"]),
                                    float(self.scene.extra["support_z"]))

        def direct_episode(self, env, assignment):
            from bimanual.control.errors import MotionFailure
            from bimanual.control.motion import (MotionPlan, cartesian_actions,
                                                 object_follow_stage, park_stages,
                                                 pick_stages, place_stages)

            entry = self._object(assignment.object_id)
            slot = self._slot(assignment.slot_name)
            profile = self.config["grasp"]
            approach = profile.get("place_hover_height_m", profile["hover_height_m"])
            release_z = (slot.support_z + self.config["sorting"]["cube_size_m"] / 2
                         + profile["release_clearance_m"])
            carry_target = [slot.xy[0], slot.xy[1], release_z + approach]
            stages = pick_stages(env, assignment.supply_arm, entry.name,
                                 entry.orientation_degrees)
            for stage in stages:
                if stage.name == "pick_descend":
                    stage.metadata["allowed_object"] = entry.name
                    pick_arm = stage.arm
                    target = entry.name
                    stage.screen = (
                        lambda live, actions, _arm=pick_arm, _target=target:
                        _v1._b("payload_path").screen_joint_path(
                            live, _arm, actions, allowed_object=_target))
            yield MotionPlan(arm=assignment.supply_arm, object_id=entry.name,
                             stages=stages, label=f"direct_pick_{entry.name}")
            yield MotionPlan(arm=assignment.supply_arm, object_id=entry.name,
                             stages=object_follow_stage(
                                 env, assignment.supply_arm, entry.name, carry_target,
                                 label=f"direct_carry_{entry.name}"),
                             label=f"direct_carry_{entry.name}")
            place = place_stages(env, assignment.destination_arm, entry.name,
                                 slot.xy, slot.support_z,
                                 support_geoms={f"{slot.color}_tray_0", "table"})
            opens = list(profile.get("scene2_release_open_candidates", _OPEN_CANDIDATES))
            directions = list(profile.get("scene2_withdraw_directions", _DIRECTIONS))
            retreat_m = float(profile.get("scene2_withdraw_retreat_m", _RETREAT_M))
            lift_height = profile["hover_height_m"]
            state = {"open": None, "direction": None}

            def build_release(live, _arm=assignment.destination_arm, _opens=opens):
                site = live.robot.grasp_ids[_arm]
                target = live.data.site_xpos[site].copy()
                checks = []
                for open_percent in _opens:
                    actions = cartesian_actions(
                        live, _arm, [target], rotation=None,
                        seconds=self.config["grasp"]["stage_durations_s"]["release"] / 2,
                        gripper=float(open_percent), name="place_release")
                    screen = _v1._b("payload_path").screen_joint_path(
                        live, _arm, actions[0], allowed_object=entry.name)
                    checks.append({"open_percent": float(open_percent),
                                   "passed": bool(screen.get("passed")),
                                   "screen": screen})
                    if screen.get("passed"):
                        state["open"] = float(open_percent)
                        return actions[0]
                raise MotionFailure("placement_failed",
                                    "no release opening clears the tray wall",
                                    stage="place_release", evidence={"release": checks})

            def build_vertical(live, _arm=assignment.destination_arm,
                               _directions=directions, _retreat=retreat_m):
                if state["open"] is None:
                    raise MotionFailure("placement_failed",
                                        "withdraw ran before a release opening was chosen",
                                        stage="place_withdraw")
                site = live.robot.grasp_ids[_arm]
                position = live.data.site_xpos[site].copy()
                rotation = live.data.site_xmat[site].reshape(3, 3).copy()
                high = position + np.array([0.0, 0.0, lift_height])
                checks = []
                for name in _directions:
                    vector = _direction_vector(name, rotation)
                    retreat = high + np.array([float(vector[0]) * _retreat,
                                               float(vector[1]) * _retreat, 0.0])
                    try:
                        segments = cartesian_actions(
                            live, _arm, [high, retreat], rotation=rotation, seconds=1.0,
                            gripper=state["open"], name="place_withdraw")
                    except MotionFailure as error:
                        checks.append({"direction": name, "passed": False,
                                       "error": str(error)[:160]})
                        continue
                    actions = segments[0] + segments[1]
                    screen = _v1._b("payload_path").screen_joint_path(
                        live, _arm, actions, allowed_object=entry.name)
                    checks.append({"direction": name,
                                   "passed": bool(screen.get("passed")),
                                   "screen": screen})
                    if screen.get("passed"):
                        state["direction"] = name
                        return actions
                raise MotionFailure("ik_unreachable",
                                    "no vertical withdraw direction clears the tray",
                                    stage="place_withdraw", evidence={"withdraw": checks})

            for stage in place:
                if stage.name == "place_release":
                    stage.build = build_release
                if stage.name == "place_withdraw":
                    stage.build = build_vertical
            yield MotionPlan(arm=assignment.destination_arm, object_id=entry.name,
                             stages=place, label=f"direct_place_{entry.name}")
            yield MotionPlan(arm=assignment.destination_arm, object_id=entry.name,
                             stages=park_stages(env, assignment.destination_arm),
                             label=f"direct_park_{entry.name}")

    return DualBlockExpert
