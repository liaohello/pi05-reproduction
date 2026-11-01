"""V2 clearance probe for the scene2 dual-block family.

This module wraps :mod:`scene2_common` and leaves every v1 file untouched.  The
v2 profile opts into four measured clearance fixes:

* a minimum block-centre distance to each tray's outer rectangle (0.07 m),
* one central slot per tray instead of the two offset slots,
* a 50 percent release opening,
* a vertical-first withdraw, plus static close/release screening.

Nothing here relaxes the registered success predicate or the 0.5 mm contact
guard.  The v1 protocol, scenes and reports remain the v1 record.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import scene2_common as _v1
from scene2_common import *  # noqa: F401,F403

_b = _v1._b

_ORIGINAL_BLOCK_POSITION = _v1.block_position_valid
_ORIGINAL_PICK_VERDICT = _v1.pick_verdict
_ORIGINAL_PLACE_VERDICT = _v1.place_verdict


def configure(root, *, python=None):
    root = _v1.configure(root, python=python)
    _patch()
    return root


def _tray_slots(config, color):
    if (config.get("sorting") or {}).get("scene2_slot_mode") == "center":
        regions = _v1._b("regions")
        center = regions.tray_center(config, color)
        return [regions.TraySlot(f"{color}_center", color,
                                 (float(center[0]), float(center[1])),
                                 float(regions.tray_floor_top_z(config)))]
    return _v1._original_tray_slots(config, color)


def _patch():
    regions = _v1._b("regions")
    if not getattr(regions, "_scene2_v2_patched", False):
        _v1._original_tray_slots = regions.tray_slots
        regions.tray_slots = _tray_slots
        regions._scene2_v2_patched = True
    _v1.block_position_valid = block_position_valid
    _v1.pick_verdict = pick_verdict
    _v1.place_verdict = place_verdict
    _v1.dual_block_expert_class = dual_block_expert_class


def provenance(root, protocol_path, config, certificate_path, adapter_path=None):
    result = _v1.provenance(root, protocol_path, config, certificate_path,
                            adapter_path=Path(__file__).resolve())
    protocol = _v1.load_protocol(protocol_path)
    profile = _v1.REPO_ROOT / protocol["source_binding"]["profile"]
    result["profile_sha256"] = _v1.common.sha256_file(profile)
    result["adapter_sha256"] = _v1.common.sha256_file(Path(__file__).resolve())
    return result


def block_position_valid(config, arm, xy, trays):
    valid, reason = _ORIGINAL_BLOCK_POSITION(config, arm, xy, trays)
    if not valid:
        return valid, reason
    clearance = (config.get("sorting") or {}).get("scene2_block_tray_outer_clearance_m")
    if clearance is None:
        return True, None
    regions = _v1._b("regions")
    outer = regions.tray_inner_half(config) + 0.006
    for color in ("red", "blue"):
        center = np.asarray(trays[color], dtype=float)
        dx = max(abs(float(xy[0]) - float(center[0])) - float(outer[0]), 0.0)
        dy = max(abs(float(xy[1]) - float(center[1])) - float(outer[1]), 0.0)
        if math.hypot(dx, dy) < float(clearance):
            return False, "tray_outer_clearance"
    return True, None


def pick_verdict(env, config, arm, target_name, xy, yaw):
    result = _ORIGINAL_PICK_VERDICT(env, config, arm, target_name, xy, yaw)
    if not (config.get("grasp") or {}).get("scene2_close_screen", False):
        return result
    if not result.get("passed"):
        return result
    profile = config["grasp"]
    regions = _v1._b("regions")
    z = float(config["scene"]["table"]["surface_z"]) + regions.cube_half(config)
    rotation = _v1._b("rotation_z")(yaw)
    offset = 0 if arm == "left" else 6
    seed = env.data.qpos[env.robot.qpos_ids[offset:offset + 5]].copy()
    grasp = _v1._b("PoseIK")(env).solve(
        arm, [xy[0], xy[1], z], rotation, seed_rad=seed,
        tolerance=0.0002, rotation_tolerance=0.01)
    if not grasp.converged:
        updated = dict(result)
        updated.update({"passed": False, "reason": "close_grasp_ik"})
        return updated
    closed = _v1._arm_target(env.robot.home(), arm, grasp.joints_rad,
                             profile["closed_percent"])
    screen = _v1._screen_joint(env, arm, [closed], target_name)
    updated = dict(result)
    updated["close"] = _v1._compact_screen(screen)
    if not screen["passed"]:
        updated.update({"passed": False, "reason": "close_contact"})
    return updated


def place_verdict(env, config, arm, target_name, slot, yaw):
    if not (config.get("grasp") or {}).get("scene2_vertical_withdraw", False):
        return _ORIGINAL_PLACE_VERDICT(env, config, arm, target_name, slot, yaw)
    profile = config["grasp"]
    regions = _v1._b("regions")
    cube_half = regions.cube_half(config)
    support_z = float(slot.support_z)
    release_z = support_z + cube_half + profile["release_clearance_m"]
    lift_height = profile["hover_height_m"]
    release_open = float(profile.get("scene2_release_open_percent",
                                     profile["open_percent"]))
    rotation = _v1._b("rotation_z")(yaw)
    offset = 0 if arm == "left" else 6
    seed = env.data.qpos[env.robot.qpos_ids[offset:offset + 5]].copy()
    ik = _v1._b("PoseIK")(env)
    release = ik.solve(arm, [slot.xy[0], slot.xy[1], release_z], rotation,
                       seed_rad=seed, tolerance=0.0002, rotation_tolerance=0.01)
    if not release.converged:
        return {"passed": False, "reason": "release_ik",
                "position_error_m": release.position_error_m}
    place_hover_height = profile.get("place_hover_height_m", profile["hover_height_m"])
    place_hover = ik.solve(arm, [slot.xy[0], slot.xy[1], release_z + place_hover_height],
                           rotation, seed_rad=release.joints_rad, tolerance=0.0002,
                           rotation_tolerance=0.01)
    if not place_hover.converged:
        return {"passed": False, "reason": "place_hover_ik",
                "position_error_m": place_hover.position_error_m}
    lift_hover = ik.solve(arm, [slot.xy[0], slot.xy[1], release_z + lift_height], rotation,
                          seed_rad=release.joints_rad, tolerance=0.0002,
                          rotation_tolerance=0.01)
    if not lift_hover.converged:
        return {"passed": False, "reason": "place_hover_ik",
                "position_error_m": lift_hover.position_error_m}
    center = np.asarray(config["scene"]["trays"][slot.color], dtype=float)
    inner = regions.tray_inner_half(config)
    extents = np.abs(rotation) @ np.full(3, cube_half)
    contained = bool(np.all(np.abs(np.asarray(slot.xy) - center) + extents[:2] <= inner))
    if not contained:
        return {"passed": False, "reason": "slot_containment", "extents": extents.tolist()}
    retreat_xy = (float(slot.xy[0]) - 0.08, float(slot.xy[1]))
    retreat = ik.solve(arm, [retreat_xy[0], retreat_xy[1], release_z + lift_height],
                       rotation, seed_rad=lift_hover.joints_rad, tolerance=0.0005,
                       rotation_tolerance=0.02)
    if not retreat.converged:
        return {"passed": False, "reason": "vertical_retreat_ik",
                "position_error_m": retreat.position_error_m}
    home = env.robot.home()
    closed = profile["closed_percent"]
    low = _v1._arm_target(home, arm, release.joints_rad, closed)
    high = _v1._arm_target(home, arm, lift_hover.joints_rad, release_open)
    out = _v1._arm_target(home, arm, retreat.joints_rad, release_open)
    durations = profile["stage_durations_s"]
    descent = _v1._path(env, _v1._arm_target(home, arm, place_hover.joints_rad, closed),
                        low, durations["lower"])
    descent_screen = _v1._screen_joint(env, arm, descent, target_name)
    lift = _v1._path(env, low, high, durations["release"] / 2)
    retreat_path = _v1._path(env, high, out,
                             durations["release"] / 2 + durations["retreat"] / 2)
    lift_screen = _v1._screen_joint(env, arm, lift, target_name)
    retreat_screen = _v1._screen_joint(env, arm, retreat_path, None)
    passed = bool(descent_screen["passed"] and lift_screen["passed"]
                  and retreat_screen["passed"])
    return {"passed": passed,
            "reason": None if passed else (
                "place_path" if not descent_screen["passed"] else "vertical_withdraw"),
            "descent": _v1._compact_screen(descent_screen),
            "vertical": {"lift": _v1._compact_screen(lift_screen),
                         "retreat": _v1._compact_screen(retreat_screen)},
            "release_z_m": release_z, "lift_z_m": release_z + lift_height,
            "retreat_xy": list(retreat_xy), "contained": contained,
            "extents": extents.tolist(),
            "release_position_error_m": release.position_error_m,
            "retreat_position_error_m": retreat.position_error_m}


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
                    arm = stage.arm
                    target = entry.name
                    stage.screen = (
                        lambda live, actions, _arm=arm, _target=target:
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
            release_open = profile.get("scene2_release_open_percent")
            vertical = bool(profile.get("scene2_vertical_withdraw", False))
            release_seconds = profile["stage_durations_s"]["release"] / 2

            def build_release(live, _arm=assignment.destination_arm,
                              _open=release_open, _seconds=release_seconds):
                target = live.data.site_xpos[live.robot.grasp_ids[_arm]].copy()
                return cartesian_actions(live, _arm, [target], seconds=_seconds,
                                         gripper=_open, name="place_release")[0]

            def build_vertical(live, _arm=assignment.destination_arm,
                               _open=release_open,
                               _height=profile["hover_height_m"]):
                site = live.robot.grasp_ids[_arm]
                position = live.data.site_xpos[site].copy()
                rotation = live.data.site_xmat[site].reshape(3, 3).copy()
                high = position + np.array([0.0, 0.0, _height])
                retreat = high + np.array([-0.08, 0.0, 0.0])
                segments = cartesian_actions(live, _arm, [high, retreat],
                                             rotation=rotation, seconds=1.0,
                                             gripper=_open, name="place_withdraw")
                actions = segments[0] + segments[1]
                screening = _v1._b("payload_path").screen_joint_path(
                    live, _arm, actions, allowed_object=None)
                if not screening.get("passed", False):
                    raise MotionFailure("ik_unreachable",
                                        "place: vertical withdraw rejected",
                                        stage="place_withdraw", evidence=screening)
                return actions

            for stage in place:
                if stage.name == "place_release" and release_open is not None:
                    stage.build = build_release
                if stage.name == "place_withdraw" and vertical:
                    stage.build = build_vertical
            yield MotionPlan(arm=assignment.destination_arm, object_id=entry.name,
                             stages=place, label=f"direct_place_{entry.name}")
            yield MotionPlan(arm=assignment.destination_arm, object_id=entry.name,
                             stages=park_stages(env, assignment.destination_arm),
                             label=f"direct_park_{entry.name}")

    return DualBlockExpert
