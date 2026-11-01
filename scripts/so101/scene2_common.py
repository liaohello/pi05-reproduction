"""Workstream C adapter: two blocks, two colour-randomised trays, one target.

The external dual-SO101 checkout is imported, never edited.  This module adds
only the scene layer that the external stack does not provide:

* a two-block scene sampler with per-scene IK and collision screening,
* a task whose success rule checks the target and the distractor separately,
* an expert wrapper that places into the sampled slot rather than a nominal one,
* scene/condition hashes and a post-settle initial-state fingerprint.

The external ``SingleBlockPlaceTask`` is deliberately not used: it hard-codes
``cube_0`` and one tray pair.  The external expert primitives are reused.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import os
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_common as common  # noqa: E402

_BOUND: dict[str, object] = {}


def _b(name: str):
    if not _BOUND:
        raise RuntimeError("call configure(root) before importing external modules")
    return _BOUND[name]


def configure(root, *, python: str | None = None) -> Path:
    """Bind the external checkout once and cache the modules this adapter uses."""
    root = common.setup_external(root)
    _BOUND.clear()
    _BOUND.update({
        "root": root,
        "mujoco": importlib.import_module("mujoco"),
        "regions": importlib.import_module("bimanual.tasks.sorting.regions"),
        "certification": importlib.import_module("bimanual.tasks.sorting.certification"),
        "contacts": importlib.import_module("bimanual.control.contacts"),
        "payload_path": importlib.import_module("bimanual.control.payload_path"),
        "PoseIK": importlib.import_module("bimanual.control.ik").PoseIK,
        "rotation_z": importlib.import_module("bimanual.control.motion").rotation_z,
        "joint_trajectory": importlib.import_module("bimanual.control.trajectory").joint_trajectory,
        "ObjectSpec": importlib.import_module("bimanual.tasks.sorting.scene_sampling").ObjectSpec,
        "SceneSpec": importlib.import_module("bimanual.tasks.sorting.scene_sampling").SceneSpec,
        "SceneGenerationError": importlib.import_module(
            "bimanual.tasks.sorting.scene_sampling").SceneGenerationError,
        "BimanualEnv": importlib.import_module("bimanual.sim.environment").BimanualEnv,
        "EpisodeRunner": importlib.import_module("bimanual.recording.episode").EpisodeRunner,
        "episode_module": importlib.import_module("bimanual.recording.episode"),
        "SingleBlockExpert": importlib.import_module("bimanual.experts.single_block").SingleBlockExpert,
        "Assignment": importlib.import_module("bimanual.experts.allocation").Assignment,
        "source_fingerprint": importlib.import_module("source_fingerprint"),
        "python": python or os.environ.get("SO101_PYTHON") or sys.executable,
    })
    return root


def load_protocol(path: Path) -> dict:
    return common.load_json(path)


def load_config(root: Path, protocol: dict, *, certificate: Path | None = None) -> dict:
    from bimanual.config import load_config as external_load_config

    profile = REPO_ROOT / protocol["source_binding"]["profile"]
    config = external_load_config(root=root, profile=str(profile))
    config["runtime"]["physics_hz"] = int(protocol["control"]["physics_hz"])
    config["runtime"]["control_hz"] = int(protocol["control"]["control_hz"])
    config["recording"]["fps"] = int(protocol["control"]["recording_fps"])
    config["runtime"]["settle_s"] = float(protocol["control"]["settle_s"])
    config["sorting"]["timeout_s"] = float(protocol["control"]["timeout_s"])
    config.setdefault("domain_randomization", {})["enabled"] = False
    config["scene"]["transfer"]["marker"] = False
    if certificate is not None:
        config["evaluation"]["certificate"] = str(Path(certificate).expanduser().resolve())
    return config


def load_certificate(config: dict):
    regions = _b("regions")
    path = Path(config["evaluation"]["certificate"])
    if not path.is_absolute():
        path = Path(config["root"]) / path
    if not path.is_file():
        raise FileNotFoundError(f"scene2 certificate not found: {path}")
    return regions.load_certificate(path), path


def provenance(root: Path, protocol_path: Path, config: dict, certificate_path: Path,
               adapter_path: Path | None = None) -> dict:
    """Hashes that bind a generated scene to the code, config and assets used."""
    fp = _b("source_fingerprint").fingerprint(root, _b("python"))
    adapter = adapter_path or Path(__file__).resolve()
    payload = {
        "protocol_id": _b("config_protocol_id") if "config_protocol_id" in _BOUND else None,
        "protocol_sha256": common.sha256_file(protocol_path),
        "profile_sha256": common.sha256_file(REPO_ROOT / "configs/so101/scene2_dual_random_profile.yaml"),
        "adapter_sha256": common.sha256_file(adapter),
        "certificate_sha256": common.sha256_file(certificate_path),
        "source_sha256": fp["source_sha256"],
        "source_files": fp["source_files"],
        "expert_sha256": fp["expert_sha256"],
        "expert_files": fp["expert_files"],
        "model_sha256": fp["model_sha256"],
        "config_sha256": fp["config_sha256"],
        "config_files": fp["config_files"],
        "packages": fp["packages"],
    }
    return payload


def set_protocol_id(protocol_id: str) -> None:
    _BOUND["config_protocol_id"] = protocol_id


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def other_arm(arm: str) -> str:
    return "right" if arm == "left" else "left"


def arm_for_y(y: float) -> str:
    return "left" if float(y) > 0.0 else "right"


def arrangement_sides(arrangement: str) -> dict:
    if arrangement == "red_positive_y":
        return {"red": 1.0, "blue": -1.0}
    if arrangement == "red_negative_y":
        return {"red": -1.0, "blue": 1.0}
    raise ValueError(f"unknown arrangement {arrangement!r}")


def _tray_outer_half(config: dict) -> np.ndarray:
    inner = np.asarray(config["scene"]["trays"]["inner_half_size"], dtype=float)
    return inner + 0.006


def _tray_fits_table(config: dict, xy) -> bool:
    table = config["scene"]["table"]
    center = np.asarray(table["center_xy"], dtype=float)
    half = np.asarray(table["half_size_xy"], dtype=float)
    outer = _tray_outer_half(config)
    margin = float(config["sorting"].get("tray_table_margin_m", 0.02))
    delta = np.abs(np.asarray(xy, dtype=float) - center)
    return bool(np.all(delta + outer <= half - margin))


def _tray_gap(config: dict, first, second) -> float:
    outer = _tray_outer_half(config)
    dx = max(0.0, abs(float(first[0]) - float(second[0])) - 2.0 * outer[0])
    dy = max(0.0, abs(float(first[1]) - float(second[1])) - 2.0 * outer[1])
    return float(math.hypot(dx, dy))


def tray_keepout_boxes(config: dict, trays: dict):
    regions = _b("regions")
    clearance = float(config["sorting"].get("block_tray_clearance_m", 0.01))
    half = regions.tray_inner_half(config) + 0.02 + clearance
    boxes = []
    for color in ("red", "blue"):
        center = np.asarray(trays[color], dtype=float)
        boxes.append(regions.RegionSpec(f"{color}_tray_keepout",
                                        float(center[0] - half[0]), float(center[0] + half[0]),
                                        float(center[1] - half[1]), float(center[1] + half[1])))
    return boxes


def block_position_valid(config: dict, arm: str, xy, trays: dict) -> tuple[bool, str | None]:
    regions = _b("regions")
    table = config["scene"]["table"]
    center = np.asarray(table["center_xy"], dtype=float)
    half = np.asarray(table["half_size_xy"], dtype=float)
    if np.any(np.abs(np.asarray(xy, dtype=float) - center) > half - 0.06):
        return False, "table_bounds"
    if not regions.pick_region(config, arm).contains(xy):
        return False, "pick_region"
    for box in tray_keepout_boxes(config, trays):
        if box.contains(xy):
            return False, "tray_keepout"
    return True, None


def _color_rgba(color: str):
    return [0.85, 0.08, 0.06, 1.0] if color == "red" else [0.08, 0.25, 0.90, 1.0]


def sample_trays(config: dict, rng, arrangement: str) -> dict:
    SceneGenerationError = _b("SceneGenerationError")
    sides = arrangement_sides(arrangement)
    trays = {}
    for color in ("red", "blue"):
        entry = config["sorting"]["tray_sampling_positive" if sides[color] > 0
                                 else "tray_sampling_negative"]
        chosen = None
        for _ in range(2000):
            xy = (float(rng.uniform(*entry["x_m"])), float(rng.uniform(*entry["y_m"])))
            if _tray_fits_table(config, xy):
                chosen = xy
                break
        if chosen is None:
            raise SceneGenerationError(f"no table-legal position for the {color} tray")
        trays[color] = chosen
    gap = _tray_gap(config, trays["red"], trays["blue"])
    minimum = float(config["sorting"].get("tray_tray_gap_m", 0.03))
    if gap < minimum:
        raise SceneGenerationError(f"tray gap {gap:.4f} m below {minimum:.4f} m")
    return trays


@contextmanager
def trays_at(env, trays: dict):
    """Move the two tray bodies for screening, then restore them."""
    mujoco = _b("mujoco")
    bodies = {color: env.model.body(f"{color}_tray").id for color in ("red", "blue")}
    previous = {color: env.model.body_pos[index].copy() for color, index in bodies.items()}
    z = float(env.config["scene"]["table"]["surface_z"])
    try:
        for color, index in bodies.items():
            env.model.body_pos[index] = [float(trays[color][0]), float(trays[color][1]), z]
        mujoco.mj_forward(env.model, env.data)
        yield
    finally:
        for color, index in bodies.items():
            env.model.body_pos[index] = previous[color]
        mujoco.mj_forward(env.model, env.data)


def set_scene_state(env, config: dict, trays: dict, blocks: list[dict], *,
                    settle_s: float = 0.0) -> None:
    """Install a candidate initial state and optionally settle it."""
    mujoco = _b("mujoco")
    mujoco.mj_resetData(env.model, env.data)
    table_z = float(config["scene"]["table"]["surface_z"])
    half = float(config["sorting"]["cube_size_m"]) / 2.0
    for color, xy in trays.items():
        env.model.body_pos[env.model.body(f"{color}_tray").id] = [float(xy[0]), float(xy[1]), table_z]
    for block in blocks:
        name = block["name"]
        address = env.model.jnt_qposadr[env.model.joint(f"{name}_free").id]
        yaw = math.radians(float(block["yaw_degrees"])) / 2.0
        env.data.qpos[address:address + 3] = [
            float(block["xy"][0]), float(block["xy"][1]), table_z + half + 0.001]
        env.data.qpos[address + 3:address + 7] = [math.cos(yaw), 0.0, 0.0, math.sin(yaw)]
        env.data.qvel[address:address + 6] = 0.0
        env.model.geom_rgba[env.model.geom(f"{name}_geom").id] = _color_rgba(block["color"])
    env.data.qpos[env.robot.qpos_ids] = env.robot.to_radians(env.robot.home())
    env.data.qvel[env.robot.dof_ids] = 0.0
    # mj_resetData clears ctrl; without this the position servos would drive the
    # arms away from home during the settle and corrupt the initial-state screen.
    env.last_action = env.robot.apply_action(env.data, env.robot.home())
    mujoco.mj_forward(env.model, env.data)
    for _ in range(round(float(settle_s) * float(config["runtime"]["physics_hz"]))):
        mujoco.mj_step(env.model, env.data)
    mujoco.mj_forward(env.model, env.data)


def _arm_public(model, data, arm: str, config: dict) -> np.ndarray:
    from bimanual.types import JOINT_NAMES

    values = []
    for joint in JOINT_NAMES:
        index = model.jnt_qposadr[model.joint(f"{arm}_{joint}").id]
        values.append(float(data.qpos[index]))
    degrees = np.rad2deg(np.asarray(values[:5], dtype=float))
    closed = float(config["robot"]["gripper_closed_rad"])
    opened = float(config["robot"]["gripper_open_rad"])
    gripper = (values[5] - closed) / (opened - closed) * 100.0
    return np.concatenate([degrees, [gripper]])


def _tray_contact_count(model, data) -> list[dict]:
    found = []
    for contact in data.contact:
        names = [model.geom(int(index)).name for index in (contact.geom1, contact.geom2)]
        robot = any(name.startswith(("left_", "right_")) for name in names)
        tray = any(name.startswith(("red_tray_", "blue_tray_")) for name in names)
        if robot and tray:
            found.append({"geoms": names, "penetration_m": float(-contact.dist)})
    return found


def _cube_cube_contacts(model, data, names: list[str]) -> list[dict]:
    found = []
    for contact in data.contact:
        geoms = [model.geom(int(index)).name for index in (contact.geom1, contact.geom2)]
        blocks = [name for name in names if any(g.startswith(name + "_") for g in geoms)]
        if len(blocks) == 2:
            found.append({"geoms": geoms, "penetration_m": float(-contact.dist)})
    return found


def measure_initial(env, config: dict, blocks: list[dict]) -> dict:
    contacts = _b("contacts")
    regions = _b("regions")
    table = config["scene"]["table"]
    records = {}
    for block in blocks:
        name = block["name"]
        position, rotation = contacts.object_pose(env.model, env.data, name)
        linear, angular, _, _ = contacts.object_speed(env.model, env.data, name)
        supported, supports = contacts.object_support(env.model, env.data, name, {"table"})
        yaw = float(np.rad2deg(np.arctan2(rotation[1, 0], rotation[0, 0])))
        out_of_bounds = bool(
            position[2] < table["surface_z"] - 0.10
            or np.any(np.abs(position[:2] - np.asarray(table["center_xy"]))
                      > np.asarray(table["half_size_xy"]) + 0.1))
        records[name] = {
            "color": block["color"],
            "position_xy": [float(position[0]), float(position[1])],
            "position_z": float(position[2]),
            "yaw_degrees": yaw,
            "supported": bool(supported),
            "support_geoms": supports,
            "robot_contacts": contacts.robot_contacts_with(env.model, env.data, name),
            "linear_speed_m_s": linear,
            "angular_speed_rad_s": angular,
            "out_of_bounds": out_of_bounds,
            "commanded_xy_error_m": float(np.linalg.norm(
                np.asarray(position[:2]) - np.asarray(block["xy"], dtype=float))),
            "commanded_yaw_error_degrees": abs(yaw - float(block["yaw_degrees"])),
        }
    parked = {}
    home_deg = np.asarray(config["robot"]["home_degrees"], dtype=float)
    home_grip = float(config["robot"]["home_gripper_percent"])
    for arm in ("left", "right"):
        state = _arm_public(env.model, env.data, arm, config)
        parked[arm] = {
            "joint_error_deg": float(np.max(np.abs(state[:5] - home_deg))),
            "gripper_error_percent": float(abs(state[5] - home_grip)),
        }
    limits = float(config["sorting"].get("linear_speed_m_s", 0.02))
    angular_limits = float(config["sorting"].get("angular_speed_rad_s", 0.2))
    checks = {
        "all_on_table": all(row["supported"] for row in records.values()),
        "no_robot_block_contact": all(not row["robot_contacts"] for row in records.values()),
        "no_block_block_contact": not _cube_cube_contacts(env.model, env.data, list(records)),
        "no_robot_tray_contact": not _tray_contact_count(env.model, env.data),
        "all_slow": all(row["linear_speed_m_s"] <= limits
                        and row["angular_speed_rad_s"] <= angular_limits
                        for row in records.values()),
        "all_in_bounds": all(not row["out_of_bounds"] for row in records.values()),
        "xy_error_within_5mm": all(row["commanded_xy_error_m"] <= 0.005
                                   for row in records.values()),
        "yaw_error_within_2deg": all(row["commanded_yaw_error_degrees"] <= 2.0
                                      for row in records.values()),
        "arms_at_home": all(row["joint_error_deg"] <= 2.0
                            and row["gripper_error_percent"] <= 5.0
                            for row in parked.values()),
    }
    return {"objects": records, "arms": parked, "checks": checks,
            "passed": all(checks.values())}


def initial_fingerprint(env, scene) -> dict:
    """Hash the post-settle integration state plus object and tray poses."""
    contacts = _b("contacts")
    snapshot = env.snapshot()
    payload = {
        "snapshot_sha256": hashlib.sha256(np.asarray(snapshot, dtype=np.float64).tobytes()).hexdigest(),
        "objects": {},
        "trays": {},
    }
    for entry in scene.objects:
        position, rotation = contacts.object_pose(env.model, env.data, entry.name)
        payload["objects"][entry.name] = {
            "position": [float(value) for value in position],
            "rotation": [[float(value) for value in row] for row in rotation],
        }
    for color in ("red", "blue"):
        body = env.model.body(f"{color}_tray").id
        payload["trays"][color] = [float(value) for value in env.model.body_pos[body][:3]]
    payload["fingerprint_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return payload


# ---------------------------------------------------------------------------
# screening
# ---------------------------------------------------------------------------
def _arm_target(base, arm: str, joints=None, gripper=None):
    result = np.asarray(base, dtype=float).copy()
    offset = 0 if arm == "left" else 6
    if joints is not None:
        result[offset:offset + 5] = np.rad2deg(joints)
    if gripper is not None:
        result[offset + 5] = gripper
    return result


def _path(env, start, target, seconds: float) -> list:
    frames = max(2, round(seconds * env.config["runtime"]["control_hz"]))
    return list(_b("joint_trajectory")(start, target, seconds,
                                       env.config["runtime"]["control_hz"]))[:frames]


def _screen_joint(env, arm: str, actions: list, allowed_object: str | None) -> dict:
    return _b("payload_path").screen_joint_path(env, arm, actions,
                                                 allowed_object=allowed_object)


def pick_verdict(env, config: dict, arm: str, target_name: str, xy, yaw: float) -> dict:
    profile = config["grasp"]
    regions = _b("regions")
    # A free block rests on the table, not on the 4 mm tray floor.
    z = float(config["scene"]["table"]["surface_z"]) + regions.cube_half(config)
    rotation = _b("rotation_z")(yaw)
    offset = 0 if arm == "left" else 6
    seed = env.data.qpos[env.robot.qpos_ids[offset:offset + 5]].copy()
    ik = _b("PoseIK")(env)
    grasp = ik.solve(arm, [xy[0], xy[1], z], rotation, seed_rad=seed,
                     tolerance=0.0002, rotation_tolerance=0.01)
    if not grasp.converged:
        return {"passed": False, "reason": "grasp_ik",
                "position_error_m": grasp.position_error_m}
    hover = ik.solve(arm, [xy[0], xy[1], z + profile["hover_height_m"]], rotation,
                     seed_rad=grasp.joints_rad, tolerance=0.0002, rotation_tolerance=0.01)
    if not hover.converged:
        return {"passed": False, "reason": "hover_ik",
                "position_error_m": hover.position_error_m}
    home = env.robot.home()
    opened = profile["open_percent"]
    low = _arm_target(home, arm, grasp.joints_rad, opened)
    high = _arm_target(home, arm, hover.joints_rad, opened)
    durations = profile["stage_durations_s"]
    approach = _path(env, home, high, durations["approach"])
    descend = _path(env, high, low, durations["descend"])
    lift = _path(env, low, high, durations["lift"])
    approach_screen = _screen_joint(env, arm, approach, None)
    # The descend ends at the grasp pose.  A CAD band touching the *target* there
    # is the intended grasp; non-band geoms, the distractor, trays and the table
    # remain hard.  This mirrors the runtime descend-stage adapter below.
    descend_screen = _screen_joint(env, arm, descend, target_name)
    lift_screen = _screen_joint(env, arm, lift, target_name)
    passed = bool(approach_screen["passed"] and descend_screen["passed"]
                  and lift_screen["passed"])
    return {"passed": passed,
            "reason": None if passed else "pick_path",
            "approach": _compact_screen(approach_screen),
            "descend": _compact_screen(descend_screen),
            "lift": _compact_screen(lift_screen),
            "grasp_position_error_m": grasp.position_error_m,
            "hover_position_error_m": hover.position_error_m}


def place_verdict(env, config: dict, arm: str, target_name: str, slot, yaw: float) -> dict:
    profile = config["grasp"]
    regions = _b("regions")
    cube_half = regions.cube_half(config)
    support_z = float(slot.support_z)
    release_z = support_z + cube_half + profile["release_clearance_m"]
    hover_height = profile.get("place_hover_height_m", profile["hover_height_m"])
    rotation = _b("rotation_z")(yaw)
    offset = 0 if arm == "left" else 6
    seed = env.data.qpos[env.robot.qpos_ids[offset:offset + 5]].copy()
    ik = _b("PoseIK")(env)
    release = ik.solve(arm, [slot.xy[0], slot.xy[1], release_z], rotation, seed_rad=seed,
                       tolerance=0.0002, rotation_tolerance=0.01)
    if not release.converged:
        return {"passed": False, "reason": "release_ik",
                "position_error_m": release.position_error_m}
    hover = ik.solve(arm, [slot.xy[0], slot.xy[1], release_z + hover_height], rotation,
                     seed_rad=release.joints_rad, tolerance=0.0002, rotation_tolerance=0.01)
    if not hover.converged:
        return {"passed": False, "reason": "place_hover_ik",
                "position_error_m": hover.position_error_m}
    center = np.asarray(config["scene"]["trays"][slot.color], dtype=float)
    inner = regions.tray_inner_half(config)
    extents = np.abs(rotation) @ np.full(3, cube_half)
    contained = bool(np.all(np.abs(np.asarray(slot.xy) - center) + extents[:2] <= inner))
    if not contained:
        return {"passed": False, "reason": "slot_containment", "extents": extents.tolist()}
    home = env.robot.home()
    closed = profile["closed_percent"]
    opened = profile["open_percent"]
    low = _arm_target(home, arm, release.joints_rad, closed)
    high = _arm_target(home, arm, hover.joints_rad, closed)
    durations = profile["stage_durations_s"]
    descent = _path(env, high, low, durations["lower"])
    descent_screen = _screen_joint(env, arm, descent, target_name)
    withdraw = []
    for sign in (-1.0, 1.0):
        sideways_xy = np.asarray(slot.xy, dtype=float) + rotation[:, 0][:2] * profile["retreat_clearance_x_m"] * sign
        sideways = ik.solve(arm, [sideways_xy[0], sideways_xy[1], release_z], rotation,
                            seed_rad=release.joints_rad, tolerance=0.0005, rotation_tolerance=0.02)
        lifted = None
        if sideways.converged:
            lifted = ik.solve(arm, [sideways_xy[0], sideways_xy[1], release_z + hover_height], rotation,
                              seed_rad=sideways.joints_rad, tolerance=0.0008, rotation_tolerance=0.03)
        if not (sideways.converged and lifted.converged):
            withdraw.append({"sign": sign, "passed": False, "reason": "withdraw_ik"})
            continue
        out = _arm_target(home, arm, sideways.joints_rad, opened)
        up = _arm_target(home, arm, lifted.joints_rad, opened)
        release_path = _path(env, low, out, durations["release"] / 2)
        retreat_path = _path(env, out, up, durations["release"] / 2)
        release_screen = _screen_joint(env, arm, release_path, target_name)
        retreat_screen = _screen_joint(env, arm, retreat_path, target_name)
        passed = bool(release_screen["passed"] and retreat_screen["passed"])
        withdraw.append({"sign": sign, "passed": passed,
                         "release": _compact_screen(release_screen),
                         "retreat": _compact_screen(retreat_screen)})
    passed = bool(descent_screen["passed"] and any(entry["passed"] for entry in withdraw))
    return {"passed": passed,
            "reason": None if passed else ("place_path" if not descent_screen["passed"] else "withdraw_path"),
            "descent": _compact_screen(descent_screen), "withdraw": withdraw,
            "release_z_m": release_z, "hover_z_m": release_z + hover_height,
            "contained": contained, "extents": extents.tolist(),
            "release_position_error_m": release.position_error_m,
            "hover_position_error_m": hover.position_error_m}


def _compact_screen(screen: dict) -> dict:
    result = {"passed": bool(screen.get("passed")), "samples": screen.get("samples")}
    if screen.get("contacts"):
        result["contacts"] = screen["contacts"][:4]
    if screen.get("first_index") is not None:
        result["first_index"] = screen["first_index"]
    return result


# ---------------------------------------------------------------------------
# scene generation
# ---------------------------------------------------------------------------
def sample_candidate(config: dict, certificate, rng, arrangement: str,
                     trace: dict | None = None) -> dict:
    """Draw trays, both block poses and both yaws; no IK checks yet.

    ``trace`` is filled before any rejection so the pilot can report the yaw/XY
    distribution of *draws*, not only of accepted layouts.
    """
    SceneGenerationError = _b("SceneGenerationError")
    domains = config["sorting"]["yaw_domain_degrees"]
    trays = sample_trays(config, rng, arrangement)
    if trace is not None:
        trace["trays"] = {color: [round(float(xy[0]), 6), round(float(xy[1]), 6)]
                          for color, xy in trays.items()}
    sides = arrangement_sides(arrangement)
    blocks = []
    for name, color in (("cube_0", "red"), ("cube_1", "blue")):
        arm = arm_for_y(trays[color][1])
        points = certificate.points(arm)
        if not points:
            raise SceneGenerationError(f"certificate has no pick points for {arm}")
        point = points[int(rng.integers(0, len(points)))]
        jitter = float(config["sorting"].get("block_jitter_m", 0.004))
        xy = (float(point[0]) + float(rng.uniform(-jitter, jitter)),
              float(point[1]) + float(rng.uniform(-jitter, jitter)))
        yaw = float(rng.uniform(float(domains[0]), float(domains[1])))
        blocks.append({"name": name, "color": color, "xy": [round(xy[0], 6), round(xy[1], 6)],
                       "yaw_degrees": round(yaw, 6), "arm": arm,
                       "source_region": f"{arm}_source", "side": int(sides[color])})
        if trace is not None:
            trace.setdefault("blocks", []).append(
                {"name": name, "color": color, "xy": blocks[-1]["xy"],
                 "yaw_degrees": blocks[-1]["yaw_degrees"], "arm": arm})
    for block in blocks:
        valid, reason = block_position_valid(config, block["arm"], block["xy"], trays)
        if not valid:
            raise SceneGenerationError(f"{block['name']}_{reason}")
    minimum = float(config["sorting"].get("block_block_min_distance_m", 0.06))
    distance = math.dist(blocks[0]["xy"], blocks[1]["xy"])
    if trace is not None:
        trace["separation_m"] = round(distance, 6)
    if distance < minimum:
        raise SceneGenerationError(f"block separation {distance:.4f} m below {minimum:.4f} m")
    return {"arrangement": arrangement, "trays": trays, "blocks": blocks}


def generate_layout(config: dict, certificate, seed: int, arrangement: str, *, env,
                    provenance_hashes: dict, max_attempts: int = 400,
                    trace: list | None = None) -> dict:
    """Return a fully screened layout that supports both target instructions."""
    SceneGenerationError = _b("SceneGenerationError")
    regions = _b("regions")
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), 0x5CE2, 0 if arrangement == "red_positive_y" else 1]))
    rejected = Counter()
    local_trace = [] if trace is None else trace
    for attempt in range(1, int(max_attempts) + 1):
        entry = {"attempt": attempt, "result": None}
        try:
            candidate = sample_candidate(config, certificate, rng, arrangement, trace=entry)
        except SceneGenerationError as error:
            message = getattr(error, "message", str(error))
            reason = message.split(":")[0].strip() or "sample_reject"
            rejected[reason] += 1
            entry["result"] = reason
            entry["message"] = message[:200]
            local_trace.append(entry)
            continue
        trays = candidate["trays"]
        blocks = candidate["blocks"]
        effective = copy.deepcopy(config)
        effective["scene"]["trays"] = {"red": list(trays["red"]), "blue": list(trays["blue"]),
                                       "inner_half_size": list(config["scene"]["trays"]["inner_half_size"])}
        set_scene_state(env, config, trays, blocks, settle_s=0.0)
        condition_checks = {}
        failed = None
        slots = {}
        for target in blocks:
            arm = target["arm"]
            pick = pick_verdict(env, effective, arm, target["name"],
                                target["xy"], target["yaw_degrees"])
            condition_checks[target["color"]] = {"pick": pick, "slots": []}
            if not pick["passed"]:
                failed = f"pick_{target['color']}_{pick['reason']}"
                break
            chosen = None
            for slot in regions.tray_slots(effective, target["color"]):
                verdict = place_verdict(env, effective, arm, target["name"], slot,
                                        target["yaw_degrees"])
                condition_checks[target["color"]]["slots"].append(
                    {"slot": slot.name, **{key: value for key, value in verdict.items()
                                           if key not in ("descent", "withdraw")}})
                if verdict["passed"]:
                    chosen = {"name": slot.name, "xy": [round(float(slot.xy[0]), 6),
                                                        round(float(slot.xy[1]), 6)],
                              "support_z": float(slot.support_z)}
                    break
            if chosen is None:
                failed = f"place_{target['color']}_checks"
                break
            slots[target["color"]] = chosen
        if failed is not None:
            rejected[failed] += 1
            entry["result"] = failed
            local_trace.append(entry)
            continue
        set_scene_state(env, config, trays, blocks,
                        settle_s=float(config["runtime"]["settle_s"]))
        initial = measure_initial(env, config, blocks)
        if not initial["passed"]:
            failed = "initial_" + ",".join(name for name, ok in initial["checks"].items() if not ok)
            rejected[failed] += 1
            entry["result"] = failed
            local_trace.append(entry)
            continue
        entry["result"] = "accepted"
        entry["yaw"] = [block["yaw_degrees"] for block in blocks]
        local_trace.append(entry)
        if trace is not None:
            trace.extend(local_trace)
        layout = {
            "schema_version": 1,
            "protocol_id": _BOUND.get("config_protocol_id"),
            "seed": int(seed),
            "arrangement": arrangement,
            "trays": {color: [round(float(xy[0]), 6), round(float(xy[1]), 6)]
                      for color, xy in trays.items()},
            "blocks": blocks,
            "slots": slots,
            "checks": {
                "sampler_attempts": attempt,
                "rejected": dict(rejected),
                "tray_gap_m": round(_tray_gap(config, trays["red"], trays["blue"]), 6),
                "conditions": condition_checks,
                "initial": initial,
            },
        }
        physical = {
            "seed": layout["seed"],
            "arrangement": layout["arrangement"],
            "trays": layout["trays"],
            "blocks": [{key: block[key] for key in
                        ("name", "color", "xy", "yaw_degrees", "arm")}
                       for block in layout["blocks"]],
        }
        layout["physical_sha256"] = common.sha256_json(physical)
        layout["layout_sha256"] = common.sha256_json({"physical": physical,
                                                      "provenance": provenance_hashes})
        return layout
    if trace is not None:
        trace.extend(local_trace)
    raise SceneGenerationError(
        f"no legal dual-block layout for seed {seed} arrangement {arrangement} "
        f"after {max_attempts} attempts; rejections {dict(rejected)}")


def build_scene(config: dict, layout: dict, target_color: str) -> object:
    """Build the condition SceneSpec; both conditions share the layout identity."""
    if target_color not in ("red", "blue"):
        raise ValueError(target_color)
    SceneSpec = _b("SceneSpec")
    ObjectSpec = _b("ObjectSpec")
    target = next(block for block in layout["blocks"] if block["color"] == target_color)
    distractor = next(block for block in layout["blocks"] if block["color"] != target_color)
    target_arm = target["arm"]
    other = other_arm(target_arm)
    slot = layout["slots"][target_color]
    instruction = config["sorting"]["instruction_template"].format(color=target_color)
    target_entry = ObjectSpec(
        name=target["name"], color=target["color"], xy=tuple(target["xy"]),
        yaw_degrees=float(target["yaw_degrees"]), route="direct",
        supply_arm=target_arm, destination_arm=target_arm,
        orientation_degrees=float(target["yaw_degrees"]),
        source_region=f"{target_arm}_source")
    distractor_entry = ObjectSpec(
        name=distractor["name"], color=distractor["color"], xy=tuple(distractor["xy"]),
        yaw_degrees=float(distractor["yaw_degrees"]), route="direct",
        supply_arm=other, destination_arm=other,
        orientation_degrees=float(distractor["yaw_degrees"]),
        source_region=f"{other}_source")
    condition_payload = {
        "physical_sha256": layout["physical_sha256"],
        "layout_sha256": layout["layout_sha256"],
        "target_color": target_color,
        "target_name": target["name"],
        "distractor_name": distractor["name"],
        "active_arm": target_arm,
        "other_arm": other,
        "slot": slot,
        "instruction": instruction,
    }
    extra = {
        "task_type": "dual_block_select_place",
        "arrangement": layout["arrangement"],
        "layout_seed": layout["seed"],
        "physical_sha256": layout["physical_sha256"],
        "layout_sha256": layout["layout_sha256"],
        "condition_sha256": common.sha256_json(condition_payload),
        "trays": layout["trays"],
        "blocks": layout["blocks"],
        "target_name": target["name"],
        "target_color": target_color,
        "distractor_name": distractor["name"],
        "distractor_color": distractor["color"],
        "active_arm": target_arm,
        "other_arm": other,
        "instruction": instruction,
        "slot_name": slot["name"],
        "slot_xy": list(slot["xy"]),
        "support_z": float(slot["support_z"]),
        "checks": layout["checks"],
        "condition_checks": layout["checks"]["conditions"][target_color],
        "layout_spec": layout,
    }
    scene = SceneSpec(
        schema_version=1, profile="dual_block_select_place", seed=int(layout["seed"]),
        objects=[target_entry, distractor_entry], shared_xy=(0.0, 0.0),
        shared_orientation_degrees=0.0, expected_transfers=[], summary=instruction,
        extra=extra)
    scene.sha256 = layout["layout_sha256"]
    return scene


# ---------------------------------------------------------------------------
# task
# ---------------------------------------------------------------------------
class DualBlockSelectPlaceTask:
    """Target success and distractor integrity are judged from physics only."""

    def __init__(self, config: dict):
        self.config = config
        self.scene = None
        self.start_time = 0.0
        self.stable_since = None
        self.instruction = config["sorting"]["instruction"]
        self.names = ["cube_0", "cube_1"]
        self.target_name = None
        self.target_color = None
        self.distractor_name = None
        self.distractor_color = None
        self.active_arm = None
        self.target_initial = None
        self.distractor_initial = None
        self.initial_fingerprint = None
        self.initial_metrics = None

    def reset(self, model, data, rng, scene=None):
        if scene is None or scene.extra.get("task_type") != "dual_block_select_place":
            raise ValueError("DualBlockSelectPlaceTask requires a dual_block_select_place SceneSpec")
        self.scene = scene
        self.names = [entry.name for entry in scene.objects]
        self.target_name = scene.extra["target_name"]
        self.target_color = scene.extra["target_color"]
        self.distractor_name = scene.extra["distractor_name"]
        self.distractor_color = scene.extra["distractor_color"]
        self.active_arm = scene.extra["active_arm"]
        self.instruction = scene.extra["instruction"]
        self.config["sorting"]["instruction"] = self.instruction
        self.stable_since = None
        self.target_initial = None
        self.distractor_initial = None
        self.initial_fingerprint = None
        self.initial_metrics = None
        table_z = float(self.config["scene"]["table"]["surface_z"])
        for color in ("red", "blue"):
            x, y = scene.extra["trays"][color]
            body_id = model.body(f"{color}_tray").id
            model.body_pos[body_id] = np.array([float(x), float(y), table_z], dtype=float)
        for entry in scene.objects:
            address = model.jnt_qposadr[model.joint(f"{entry.name}_free").id]
            yaw = math.radians(float(entry.yaw_degrees)) / 2.0
            data.qpos[address:address + 3] = [
                float(entry.xy[0]), float(entry.xy[1]),
                table_z + _b("regions").cube_half(self.config) + 0.001]
            data.qpos[address + 3:address + 7] = [math.cos(yaw), 0.0, 0.0, math.sin(yaw)]
            data.qvel[address:address + 6] = 0.0
            model.geom_rgba[model.geom(f"{entry.name}_geom").id] = _color_rgba(entry.color)

    def set_initial_measured(self, model, data):
        contacts = _b("contacts")
        if self.scene is None:
            raise RuntimeError("task has no scene")
        for name in (self.target_name, self.distractor_name):
            position, rotation = contacts.object_pose(model, data, name)
            value = {"position": position.copy(), "rotation": rotation.copy()}
            if name == self.target_name:
                self.target_initial = value
            else:
                self.distractor_initial = value
        self.initial_metrics = measure_initial(
            _BOUND["env_proxy"], self.config, self.scene.extra["blocks"]) if "env_proxy" in _BOUND else None

    def _tray(self, color: str) -> np.ndarray:
        return np.asarray(self.scene.extra["trays"][color], dtype=float)

    def _status(self, model, data, name: str) -> dict:
        contacts = _b("contacts")
        regions = _b("regions")
        entry = next(item for item in self.scene.objects if item.name == name)
        color = entry.color
        position, rotation = contacts.object_pose(model, data, name)
        half = np.full(3, regions.cube_half(self.config))
        extents = np.abs(rotation) @ half
        center = self._tray(color)
        inner = regions.tray_inner_half(self.config)
        contained = bool(np.all(np.abs(position[:2] - center) + extents[:2] <= inner)
                         and self.config["scene"]["table"]["surface_z"] < position[2]
                         < self.config["scene"]["table"]["surface_z"] + 0.06)
        support_geoms = {f"{color}_tray_0"}
        supported, supports = contacts.object_support(model, data, name, support_geoms)
        table_supported, table_supports = contacts.object_support(model, data, name, {"table"})
        touching = contacts.robot_contacts_with(model, data, name)
        linear, angular, _, _ = contacts.object_speed(model, data, name)
        limits = float(self.config["sorting"].get("linear_speed_m_s", 0.02))
        angular_limit = float(self.config["sorting"].get("angular_speed_rad_s", 0.2))
        stable_in_tray = bool(contained and supported and not touching
                              and linear <= limits and angular <= angular_limit)
        stable_on_table = bool(table_supported and not touching
                               and linear <= limits and angular <= angular_limit)
        table = self.config["scene"]["table"]
        out_of_bounds = bool(position[2] < table["surface_z"] - 0.10
                             or np.any(np.abs(position[:2] - np.asarray(table["center_xy"]))
                                       > np.asarray(table["half_size_xy"]) + 0.1))
        grasped = any(contacts.bilateral_grasp(model, data, name, arm)[0]
                      for arm in ("left", "right"))
        displacement = None
        if name == self.target_name and self.target_initial is not None:
            displacement = float(np.linalg.norm(position - self.target_initial["position"]))
        if name == self.distractor_name and self.distractor_initial is not None:
            displacement = float(np.linalg.norm(position - self.distractor_initial["position"]))
        return {"color": color, "position_xy": [float(position[0]), float(position[1])],
                "position": position.tolist(), "contained": contained,
                "supported_geoms": supports, "supported": bool(supported),
                "table_supported": bool(table_supported),
                "table_support_geoms": table_supports,
                "robot_contacts": touching, "grasped": bool(grasped),
                "linear_speed_m_s": linear, "angular_speed_rad_s": angular,
                "stable_in_tray": stable_in_tray, "stable_on_table": stable_on_table,
                "displacement_from_initial_m": displacement,
                "out_of_bounds": out_of_bounds}

    def object_statuses(self, model, data) -> dict:
        return {name: self._status(model, data, name) for name in self.names}

    def arm_clearance(self, model, data) -> dict:
        contacts = _b("contacts")
        tolerance_deg = float(self.config["sorting"].get("arm_park_joint_tolerance_deg", 2.0))
        tolerance_grip = float(self.config["sorting"].get("arm_park_gripper_tolerance_percent", 5.0))
        home_deg = np.asarray(self.config["robot"]["home_degrees"], dtype=float)
        home_grip = float(self.config["robot"]["home_gripper_percent"])
        parked = {}
        for arm in ("left", "right"):
            state = _arm_public(model, data, arm, self.config)
            joint_error = float(np.max(np.abs(state[:5] - home_deg)))
            grip_error = float(abs(state[5] - home_grip))
            parked[arm] = {"joint_error_deg": joint_error, "gripper_error_percent": grip_error,
                           "parked": bool(joint_error <= tolerance_deg and grip_error <= tolerance_grip)}
        margin = float(self.config["sorting"].get("arm_clearance_m", 0.01))
        position = contacts.object_pose(model, data, self.target_name)[0]
        outer = _b("regions").tray_inner_half(self.config) + 0.006
        tray_contacts = _tray_contact_count(model, data)
        clearances = {}
        for arm in ("left", "right"):
            tcp = data.site_xpos[model.site(f"{arm}_tcp").id]
            block_distance = float(np.linalg.norm(tcp - position))
            tray_distance = float("inf")
            for color in ("red", "blue"):
                center = self._tray(color)
                dx = max(0.0, abs(float(tcp[0]) - center[0]) - outer[0])
                dy = max(0.0, abs(float(tcp[1]) - center[1]) - outer[1])
                tray_distance = min(tray_distance, math.hypot(dx, dy))
            clearances[arm] = {"block_m": block_distance, "tray_m": tray_distance,
                               "clear": bool(block_distance >= margin and tray_distance >= margin)}
        return {"parked": parked, "clearances": clearances, "tray_contacts": tray_contacts,
                "clear": bool(all(item["parked"] for item in parked.values())
                              and all(item["clear"] for item in clearances.values())
                              and not tray_contacts)}

    def evaluate(self, model, data):
        statuses = self.object_statuses(model, data)
        clearance = self.arm_clearance(model, data)
        target = statuses[self.target_name]
        distractor = statuses[self.distractor_name]
        threshold = self.config["sorting"].get("distractor_max_displacement_m")
        displacement_ok = bool(threshold is not None
                               and distractor["displacement_from_initial_m"] is not None
                               and distractor["displacement_from_initial_m"] <= float(threshold))
        distractor_ok = bool(distractor["stable_on_table"] and not distractor["robot_contacts"]
                             and not distractor["grasped"] and not distractor["out_of_bounds"]
                             and displacement_ok)
        target_ok = bool(target["stable_in_tray"] and clearance["clear"])
        stable_now = bool(target_ok and distractor_ok)
        if stable_now:
            if self.stable_since is None:
                self.stable_since = float(data.time)
        else:
            self.stable_since = None
        stable_seconds = float(self.config["sorting"]["stable_success_s"])
        stable = bool(self.stable_since is not None
                      and data.time - self.stable_since >= stable_seconds)
        timeout = bool(data.time - self.start_time >= float(self.config["sorting"]["timeout_s"]))
        success = bool(stable)
        reason = None
        if target["out_of_bounds"] or distractor["out_of_bounds"]:
            reason = "object_out_of_bounds"
        elif not target["contained"]:
            reason = "wrong_tray"
        elif not target["supported"]:
            reason = "not_supported"
        elif target["robot_contacts"]:
            reason = "robot_contact"
        elif (target["linear_speed_m_s"] > float(self.config["sorting"].get("linear_speed_m_s", 0.02))
              or target["angular_speed_rad_s"] > float(self.config["sorting"].get("angular_speed_rad_s", 0.2))):
            reason = "not_stable"
        elif not clearance["clear"]:
            reason = "arm_not_clear"
        elif distractor["grasped"]:
            reason = "distractor_grasped"
        elif distractor["robot_contacts"]:
            reason = "distractor_contact"
        elif not displacement_ok:
            reason = "distractor_displaced"
        elif not distractor["stable_on_table"]:
            reason = "distractor_unstable"
        elif target["stable_in_tray"] and not success:
            reason = "not_stable"
        if timeout and not success:
            reason = reason or "timeout"
        info = {"success": success, "sorted_count": int(target["stable_in_tray"]),
                "objects": statuses, "arm_clearance": clearance,
                "distractor": {**distractor, "max_displacement_m": threshold,
                               "displacement_ok": displacement_ok},
                "stable_seconds": (None if self.stable_since is None
                                   else round(float(data.time - self.stable_since), 6)),
                "failure_reason": reason, "timeout": timeout,
                "success_mode": "dual_block_physical_v1"}
        return float(target["stable_in_tray"]), success, bool(timeout and not success), info


# ---------------------------------------------------------------------------
# expert and runner wiring
# ---------------------------------------------------------------------------
def dual_block_expert_class():
    """Subclass the external expert so the sampled slot is authoritative."""
    base = _b("SingleBlockExpert")
    regions = _b("regions")

    class DualBlockExpert(base):
        def _slot(self, name):
            return regions.TraySlot(name, self.scene.extra["target_color"],
                                    tuple(self.scene.extra["slot_xy"]),
                                    float(self.scene.extra["support_z"]))

        def direct_episode(self, env, assignment):
            from bimanual.control.motion import (MotionPlan, object_follow_stage,
                                                 park_stages, pick_stages, place_stages)

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
                    # The descend ends at the target grasp pose.  Admit the
                    # target's own CAD-band contact there; every other contact
                    # stays hard.
                    stage.metadata["allowed_object"] = entry.name
                    arm = stage.arm
                    target = entry.name
                    stage.screen = (
                        lambda live, actions, _arm=arm, _target=target:
                        _b("payload_path").screen_joint_path(
                            live, _arm, actions, allowed_object=_target))
            yield MotionPlan(arm=assignment.supply_arm, object_id=entry.name,
                             stages=stages, label=f"direct_pick_{entry.name}")
            yield MotionPlan(arm=assignment.supply_arm, object_id=entry.name,
                             stages=object_follow_stage(
                                 env, assignment.supply_arm, entry.name, carry_target,
                                 label=f"direct_carry_{entry.name}"),
                             label=f"direct_carry_{entry.name}")
            yield MotionPlan(arm=assignment.destination_arm, object_id=entry.name,
                             stages=place_stages(
                                 env, assignment.destination_arm, entry.name,
                                 slot.xy, slot.support_z,
                                 support_geoms={f"{slot.color}_tray_0", "table"}),
                             label=f"direct_place_{entry.name}")
            yield MotionPlan(arm=assignment.destination_arm, object_id=entry.name,
                             stages=park_stages(env, assignment.destination_arm),
                             label=f"direct_park_{entry.name}")

    return DualBlockExpert


def install_runtime(config: dict, env) -> dict:
    """Install the dual task, reset hook and expert class for this process."""
    task = DualBlockSelectPlaceTask(config)
    env.task = task
    _BOUND["env_proxy"] = env
    original_reset = env.reset

    def reset_and_fingerprint(seed=0, scene=None):
        observation = original_reset(seed=seed, scene=scene)
        task.set_initial_measured(env.model, env.data)
        if scene is not None:
            task.initial_fingerprint = initial_fingerprint(env, scene)
        return observation

    env.reset = reset_and_fingerprint
    expert = dual_block_expert_class()
    _b("episode_module").SingleBlockExpert = expert
    runner_config = copy.deepcopy(config)
    runner_config["task"]["type"] = "single_block_place"
    return {"task": task, "runner_config": runner_config, "expert_class": expert}
