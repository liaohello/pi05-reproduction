#!/usr/bin/env python3
"""Reconstruct and audit all clean60 initial scenes from the pinned sampler.

Writes only to the ignored run directory. The external sampler and its certified
IK remain authoritative; this script records the exact SceneSpec, per-object
geometry/IK evidence, train/validation family separation, and match to the frozen
training view's scene hashes.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import so101_common as common  # noqa: E402


def _plain(value: Any, depth: int = 0) -> Any:
    if depth > 8:
        return f"<{type(value).__name__}>"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(item, depth + 1) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item, depth + 1) for key, item in value.items()
                if str(key) not in {"env", "environment", "root", "path"}}
    if hasattr(value, "tolist"):
        return _plain(value.tolist(), depth + 1)
    if hasattr(value, "as_dict"):
        return _plain(value.as_dict(), depth + 1)
    if dataclasses.is_dataclass(value):
        return _plain(dataclasses.asdict(value), depth + 1)
    if hasattr(value, "__dict__"):
        return _plain(vars(value), depth + 1)
    return {"type": type(value).__name__}


def _ik_status(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if type(value).__name__ == "bool_":
        return bool(value)
    for key in ("converged", "success", "is_success", "valid"):
        if hasattr(value, key):
            flag = getattr(value, key)
            if isinstance(flag, (bool, int)):
                return bool(flag)
    if isinstance(value, tuple):
        for child in value:
            status = _ik_status(child)
            if status is not None:
                return status
    return None


def _xy_from_call(arguments: dict) -> list[float] | None:
    for key, value in arguments.items():
        lowered = key.lower()
        if lowered in {"xy", "position", "pos", "target_xy", "object_xy"} \
                or "position" in lowered or lowered.endswith("_xy"):
            try:
                seq = list(value)
                if len(seq) >= 2 and all(math.isfinite(float(v)) for v in seq[:2]):
                    return [float(seq[0]), float(seq[1])]
            except (TypeError, ValueError):
                continue
    return None


def _object_arm(obj_dict: dict, feasible: dict | None) -> Any:
    candidates = [obj_dict, (feasible or {}).get("arguments", {})]
    for values in candidates:
        for key in ("supply_arm", "assigned_arm", "arm", "robot", "robot_name",
                    "side", "manipulator"):
            if key in values and isinstance(values[key], (str, int, float, bool)):
                return values[key]
    return None


def _role_signature(obj: dict) -> tuple:
    return tuple(obj.get(key) for key in
                 ("color", "supply_arm", "destination_arm", "route", "source_region"))


def _same_family(left: list[dict], right: list[dict],
                 threshold: float) -> tuple[bool, float | None]:
    """Match complete cube roles before comparing geometry.

    Object order is not semantic.  A family edge requires a one-to-one matching
    with equal colour/arm/route roles and every matched centre within half the
    registered cube width.  Union-find in the caller supplies transitivity.
    """
    if len(left) != len(right):
        return False, None
    candidates = []
    for order in itertools.permutations(range(len(right))):
        if any(_role_signature(left[i]) != _role_signature(right[j])
               for i, j in enumerate(order)):
            continue
        worst = max(
            math.dist([float(v) for v in left[i]["xy"]],
                      [float(v) for v in right[j]["xy"]])
            for i, j in enumerate(order))
        candidates.append(worst)
    if not candidates:
        return False, None
    best = min(candidates)
    return best <= threshold, best


def _xy_margins(region, xy: list[float]) -> dict:
    x, y = [float(value) for value in xy]
    return {"left": x - float(region.xmin), "right": float(region.xmax) - x,
            "lower": y - float(region.ymin), "upper": float(region.ymax) - y,
            "minimum": min(x - float(region.xmin), float(region.xmax) - x,
                           y - float(region.ymin), float(region.ymax) - y)}


def _outside_distance(region, xy: list[float]) -> float:
    x, y = [float(value) for value in xy]
    dx = max(float(region.xmin) - x, 0.0, x - float(region.xmax))
    dy = max(float(region.ymin) - y, 0.0, y - float(region.ymax))
    return math.hypot(dx, dy)


def _ik_evidence(call: dict | None) -> dict | None:
    if call is None:
        return None
    result = call.get("return")
    if not isinstance(result, dict):
        result = call
    return {
        "target_position_m": (call.get("arguments") or {}).get("position"),
        "converged": result.get("converged"),
        "position_error_m": result.get("position_error_m"),
        "rotation_error_rad": result.get("rotation_error_rad"),
        "iterations": result.get("iterations"),
        "joints_rad": result.get("joints_rad"),
        "return_type": call.get("return_type"),
    }


def _ik_record_complete(grasp: dict | None, hover: dict | None,
                        point: list[float], hover_height: float,
                        resting_z: float) -> bool:
    if not grasp or not hover or not grasp.get("converged") or not hover.get("converged"):
        return False
    grasp_target = grasp.get("target_position_m")
    hover_target = hover.get("target_position_m")
    if not isinstance(grasp_target, list) or not isinstance(hover_target, list):
        return False
    if len(grasp_target) != 3 or len(hover_target) != 3:
        return False
    return bool(
        math.isclose(float(grasp_target[0]), point[0], abs_tol=1e-6)
        and math.isclose(float(grasp_target[1]), point[1], abs_tol=1e-6)
        and math.isclose(float(hover_target[0]), point[0], abs_tol=1e-6)
        and math.isclose(float(hover_target[1]), point[1], abs_tol=1e-6)
        and math.isclose(float(grasp_target[2]), resting_z, abs_tol=1e-9)
        and math.isclose(float(hover_target[2]), resting_z + hover_height, abs_tol=1e-9))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--protocol", type=Path,
                        default=common.REPO_ROOT / "configs/so101/clean60_protocol.json")
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--collection", type=Path, action="append", required=True,
                        help="private E1 collection directory containing e1_manifest.json and "
                             "attempts.jsonl; repeat for a merged view")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--near-duplicate-tolerance-m", type=float, default=0.0125,
                        help="half the registered 25 mm cube width")
    parser.add_argument("--expected-episodes", type=int, default=None)
    parser.add_argument("--expected-train", type=int, default=None)
    parser.add_argument("--expected-validation", type=int, default=None)
    parser.add_argument("--expected-attempts", type=int, default=None)
    parser.add_argument("--expected-successes", type=int, default=None)
    parser.add_argument("--expected-failures", type=int, default=None)
    args = parser.parse_args()

    root = args.root or __import__("os").environ.get("SO101_SOURCE_ROOT")
    if not root:
        raise SystemExit("missing --root or SO101_SOURCE_ROOT")
    protocol = common.load_json(args.protocol)
    view = args.view.expanduser().resolve()
    collections = [path.expanduser().resolve() for path in args.collection]
    output = args.output.expanduser().resolve()
    output_file = output if output.suffix == ".json" else output / "scene_audit.json"
    if output_file.exists():
        raise SystemExit("refusing to overwrite an existing scene audit")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    meta = common.load_json(view / "meta.json")
    rows = meta.get("episodes", [])
    total = len(rows)
    train_total = sum(row.get("split") == "train" for row in rows)
    validation_total = sum(row.get("split") == "validation" for row in rows)
    expected_counts = (args.expected_episodes if args.expected_episodes is not None
                       else (60 if total == 60 else total),
                       args.expected_train if args.expected_train is not None
                       else (48 if total == 60 else train_total),
                       args.expected_validation if args.expected_validation is not None
                       else (12 if total == 60 else validation_total))
    if (total, train_total, validation_total) != expected_counts:
        raise SystemExit(f"training view is {total}/{train_total}/{validation_total}, "
                         f"expected {expected_counts}")
    collection_by_name = {collection.name: collection for collection in collections}
    if len(collection_by_name) != len(collections):
        raise SystemExit("duplicate collection names")
    rows_by_collection: dict[str, list[dict]] = {name: [] for name in collection_by_name}
    single = collections[0] if len(collections) == 1 else None
    for row in rows:
        name = row.get("source_collection")
        if name is None:
            if single is None:
                raise SystemExit("merged rows must name their source collection")
            name = single.name
        if name not in rows_by_collection:
            raise SystemExit(f"view row names an unknown collection: {name}")
        rows_by_collection[name].append(row)
    manifest_by_seed: dict[int, dict] = {}
    manifest_rows_total = 0
    for collection in collections:
        manifest_path = collection / "e1_manifest.json"
        if not manifest_path.is_file():
            manifest_path = collection / "manifest.json"
        manifest = common.load_json(manifest_path)
        manifest_rows = manifest.get("episodes", [])
        if len(manifest_rows) != len(rows_by_collection[collection.name]):
            raise SystemExit(f"{collection.name}: manifest has {len(manifest_rows)} episodes, "
                             f"view binds {len(rows_by_collection[collection.name])}")
        manifest_rows_total += len(manifest_rows)
        for row in manifest_rows:
            seed = int(row["seed"])
            if seed in manifest_by_seed:
                raise SystemExit("collection manifests have duplicate successful seeds")
            manifest_by_seed[seed] = {"row": row, "collection": collection}
    if manifest_rows_total != total:
        raise SystemExit("collection manifests do not cover the view")
    if {int(row["seed"]) for row in rows} != set(manifest_by_seed):
        raise SystemExit("collection manifest seed set differs from the exported view")

    config = common.load_external_config(root, protocol)
    from bimanual.tasks.sorting import regions, scene_sampling
    from bimanual.tasks.sorting.certification import check_certificate_current
    from bimanual.sim.environment import BimanualEnv
    from bimanual.control import ik as control_ik
    from clean_layout import install_filter

    # Trace the exact pinned sampler's per-object feasibility predicate and the
    # underlying grasp/hover IK solver calls, preserving all return values.
    feasible_name = "_pose_feasible"
    if not hasattr(scene_sampling, feasible_name):
        raise SystemExit("pinned scene sampler lacks the expected per-object IK predicate")
    original_feasible = getattr(scene_sampling, feasible_name)
    original_signature = inspect.signature(original_feasible)
    original_solver = None
    solver_cls = getattr(control_ik, "PoseIK", None)
    solve_records: list[dict] = []
    if solver_cls is not None and hasattr(solver_cls, "solve"):
        original_solver = solver_cls.solve
        solver_signature = inspect.signature(original_solver)

        def traced_solve(self, *args, **kwargs):
            bound = solver_signature.bind_partial(self, *args, **kwargs)
            result = original_solver(self, *args, **kwargs)
            values = {key: val for key, val in bound.arguments.items() if key != "self"}
            solve_records.append({
                "arguments": _plain(values),
                "return_type": type(result).__name__,
                "return": _plain(result),
                "explicit_converged": _ik_status(result),
            })
            return result

        solver_cls.solve = traced_solve

    feasible_records: list[dict] = []

    def traced_feasible(*args_, **kwargs_):
        bound = original_signature.bind_partial(*args_, **kwargs_)
        before = len(solve_records)
        result = original_feasible(*args_, **kwargs_)
        values = {key: val for key, val in bound.arguments.items()
                  if key.lower() not in {"env", "environment", "config", "certificate"}}
        feasible_records.append({
            "arguments": _plain(values),
            "candidate_xy": _xy_from_call(values),
            "assigned_arm": values.get("arm"),
            "accepted": bool(result),
            "solver_calls": [dict(item) for item in solve_records[before:]],
        })
        return result

    setattr(scene_sampling, feasible_name, traced_feasible)
    # collection.py imports generate_scene by value, so import it after the
    # instrumented sampler is installed, then add the same clean-layout filter.
    collection_module = __import__("bimanual.recording.collection", fromlist=["generate_scene"])
    generate_scene = install_filter(collection_module, protocol)
    certificate = regions.load_certificate(root and Path(root) / config["evaluation"]["certificate"])
    certificate_hashes = certificate.hashes()
    attempts = []
    attempt_source_by_seed: dict[int, Path] = {}
    attempts_by_collection: dict[str, int] = {}
    for collection in collections:
        attempt_path = collection / "attempts.jsonl"
        if not attempt_path.is_file():
            raise SystemExit(f"{collection.name}: collection attempt ledger is missing")
        collection_attempts = [
            json.loads(line) for line in attempt_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
        attempts_by_collection[collection.name] = len(collection_attempts)
        for row in collection_attempts:
            row = dict(row)
            row["source_collection"] = collection.name
            attempts.append(row)
            if row.get("success"):
                seed = int(row["seed"])
                if seed in attempt_source_by_seed:
                    raise SystemExit("duplicate successful seed across collections")
                attempt_source_by_seed[seed] = collection
    attempt_by_seed = {int(row["seed"]): row for row in attempts if row.get("success")}
    if set(attempt_by_seed) != set(manifest_by_seed):
        raise SystemExit("successful attempt ledger does not bind to manifest seeds")
    attempt_seeds = [int(row["seed"]) for row in meta["episodes"]]
    attempt_ordinal_by_seed = {
        int(row["seed"]): ordinal for ordinal, row in enumerate(attempts, start=1)
        if row.get("success")
    }
    failed_attempts = [row for row in attempts if not row.get("success")]
    if any(not row.get("failure_reason") or not row.get("failure_stage")
           for row in failed_attempts):
        raise SystemExit("failed attempt ledger rows lack a stage or reason")
    if any(row.get("profile") != "direct" for row in attempts):
        raise SystemExit("attempt ledger contains a non-direct route")
    expected_attempts = (args.expected_attempts if args.expected_attempts is not None
                         else (175 if total == 60 else len(attempts)))
    expected_successes = (args.expected_successes if args.expected_successes is not None
                          else (60 if total == 60 else len(attempts) - len(failed_attempts)))
    expected_failures = (args.expected_failures if args.expected_failures is not None
                         else (115 if total == 60 else len(failed_attempts)))
    attempt_summary = {
        "attempts": len(attempts),
        "successes": len(attempts) - len(failed_attempts),
        "failures": len(failed_attempts),
        "failure_stages": {
            stage: sum(row.get("failure_stage") == stage for row in failed_attempts)
            for stage in sorted({row.get("failure_stage") for row in failed_attempts})},
        "failure_reasons": {
            reason: sum(row.get("failure_reason") == reason for row in failed_attempts)
            for reason in sorted({row.get("failure_reason") for row in failed_attempts})},
        "profiles": sorted({row.get("profile") for row in attempts}),
        "by_collection": attempts_by_collection,
        "all_rows": attempts,
    }

    # The external environment owns certified workspace, object-size, tray and
    # forbidden-region checks. No geometry or guard is patched by this audit.
    records = []
    with BimanualEnv(config, images=False) as env:
        stale = check_certificate_current(certificate, env)
        if stale:
            raise SystemExit("certified workspace is stale; scene replay is invalid")
        for seed in attempt_seeds:
            feasible_records.clear()
            solve_records.clear()
            scene = generate_scene(
                config, certificate, "direct", seed, env=env,
                sampling=str(config["sorting"].get("sampling_mode", "legacy")))
            expected_row = manifest_by_seed[seed]["row"]
            if scene.sha256 != expected_row["scene_sha256"]:
                raise SystemExit(f"seed {seed}: reconstructed scene hash differs from collection")
            episode_row = next(row for row in meta["episodes"] if int(row["seed"]) == seed)
            if scene.sha256 != episode_row["scene_sha256"]:
                raise SystemExit(f"seed {seed}: reconstructed scene hash differs from training view")
            objects = [obj.as_dict() for obj in scene.objects]
            points = [[float(v) for v in obj.xy] for obj in scene.objects]
            pairs = {f"{a}:{b}": math.dist(points[a], points[b])
                     for a, b in itertools.combinations(range(4), 2)}
            object_audits = []
            pose_resting_z = float(regions.resting_z(config))
            registered_hover_height = float(config["grasp"]["hover_height_m"])
            for object_index, (obj, obj_dict, point) in enumerate(zip(scene.objects, objects, points)):
                matches = [row for row in feasible_records if row["accepted"]
                           and row.get("assigned_arm") == obj_dict.get("supply_arm")
                           and row["candidate_xy"] is not None
                           and math.dist(row["candidate_xy"], point) <= 1e-6]
                feasible = matches[-1] if matches else None
                calls = (feasible or {}).get("solver_calls", [])
                arm = _object_arm(obj_dict, feasible)
                grasp = _ik_evidence(calls[0]) if len(calls) >= 1 else None
                hover = _ik_evidence(calls[1]) if len(calls) >= 2 else None
                source_region = regions.pick_region(config, obj_dict["supply_arm"])
                keepouts = {}
                for box in regions.keepout_boxes(config):
                    keepouts[box.name] = {
                        "contains_with_sampler_margin": bool(box.contains(point, margin=-0.01)),
                        "outside_distance_m": _outside_distance(box, point),
                    }
                table = config["scene"]["table"]
                half = [float(value) for value in table["half_size_xy"]]
                center = [float(value) for value in table["center_xy"]]
                table_safe_margin = min(
                    (half[0] - 0.06) - abs(point[0] - center[0]),
                    (half[1] - 0.06) - abs(point[1] - center[1]))
                object_audits.append({
                    "object_index": object_index,
                    "scene_object": _plain(obj_dict),
                    "assigned_arm": arm,
                    "xy_m": point,
                    "source_region": source_region.as_dict(),
                    "source_region_boundary_margins_m": _xy_margins(source_region, point),
                    "source_region_contains_with_validation_margin": bool(
                        source_region.contains(point, margin=-0.02)),
                    "table_safe_boundary_margin_m": table_safe_margin,
                    "keepout_clearance": keepouts,
                    "grasp_ik": grasp,
                    "hover_ik": hover,
                    "solver_calls_for_accepted_candidate": len(calls),
                    "certified_pose_feasible": bool(feasible and feasible["accepted"]),
                    "ik_record_complete": bool(feasible and len(calls) == 2
                                               and feasible["candidate_xy"] is not None
                                               and _ik_record_complete(
                                                   grasp, hover, point, registered_hover_height,
                                                   pose_resting_z)),
                    "registered_hover_height_m": registered_hover_height,
                    "grasp_resting_z_m": pose_resting_z,
                })
            validation_problems = scene_sampling.validate_scene(
                config, certificate, scene,
                min_separation=float(protocol["scene_layout"]["minimum_pairwise_distance_m"]))
            forbidden_violations = [
                f"{object_row['scene_object']['name']}:{name}"
                for object_row in object_audits
                for name, evidence in object_row["keepout_clearance"].items()
                if evidence["contains_with_sampler_margin"]
            ]
            outside_source = [
                object_row["scene_object"]["name"] for object_row in object_audits
                if not object_row["source_region_contains_with_validation_margin"]
            ]
            summary_path = Path(attempt_by_seed[seed]["diagnostics"]) / "summary.json"
            if not summary_path.is_absolute():
                summary_path = attempt_source_by_seed[seed] / summary_path
            physical_summary = common.load_json(summary_path)
            original_predicates = {}
            for name, status in sorted(
                    (physical_summary.get("status") or {}).get("objects", {}).items()):
                original_predicates[name] = {
                    "color": status.get("color"),
                    "sorted": status.get("sorted"),
                    "contained": status.get("contained"),
                    "supported": status.get("supported"),
                    "supported_geoms": status.get("supported_geoms"),
                    "robot_contacts": status.get("robot_contacts"),
                    "out_of_bounds": status.get("out_of_bounds"),
                    "linear_speed_m_s": status.get("linear_speed_m_s"),
                    "angular_speed_rad_s": status.get("angular_speed_rad_s"),
                }
            predicates_pass = bool(
                physical_summary.get("success") is True
                and physical_summary.get("sorted_count") == 4
                and len(original_predicates) == 4
                and all(row["sorted"] is True and row["contained"] is True
                        and row["supported"] is True and not row["robot_contacts"]
                        and row["out_of_bounds"] is False
                        and f"{row['color']}_tray_0" in row["supported_geoms"]
                        for row in original_predicates.values())
                and physical_summary.get("contact_monitor_error_count") == 0
                and physical_summary.get("soft_contact_count") == 0
                and physical_summary.get("status", {}).get("success_mode") == "strict")
            if len(objects) != 4 or len(pairs) != 6:
                raise SystemExit(f"seed {seed}: scene does not contain four distinct cubes")
            if min(pairs.values()) < float(protocol["scene_layout"]["minimum_pairwise_distance_m"]):
                raise SystemExit(f"seed {seed}: minimum pairwise separation fell below protocol")
            if not scene.extra.get("clean_layout", {}).get("perturbation_present") is False:
                raise SystemExit(f"seed {seed}: scene contains or failed to certify perturbation absence")
            view_replay = {
                "max_state_error_vs_trace": episode_row.get("max_state_error_vs_trace"),
                "max_action_error_vs_trace": episode_row.get("max_action_error_vs_trace"),
            }
            replay_exact = all(value == 0.0 for value in view_replay.values())
            records.append({
                "episode_index": int(episode_row["episode_index"]),
                "split": episode_row["split"],
                "seed": seed,
                "scene_sha256": scene.sha256,
                "scene_spec": _plain(scene.as_dict()),
                "objects": object_audits,
                "positions_xy_m": points,
                "pairwise_distances_m": pairs,
                "minimum_pairwise_distance_m": min(pairs.values()),
                "sampler_accepted_ik_candidates": [row for row in feasible_records
                                                    if row["accepted"]],
                "ik_complete_for_all_objects": all(row["ik_record_complete"]
                                                    and row["certified_pose_feasible"]
                                                    for row in object_audits),
                "region_and_keepout_valid": not validation_problems
                and not forbidden_violations and not outside_source
                and all(row["table_safe_boundary_margin_m"] >= 0.0 for row in object_audits),
                "validation_problems": validation_problems,
                "forbidden_region_violations": forbidden_violations,
                "outside_source_objects": outside_source,
                "original_success_predicate": {
                    "passed": predicates_pass,
                    "success_mode": (physical_summary.get("status") or {}).get("success_mode"),
                    "sorted_count": physical_summary.get("sorted_count"),
                    "contact_monitor_error_count": physical_summary.get("contact_monitor_error_count"),
                    "soft_contact_count": physical_summary.get("soft_contact_count"),
                    "objects": original_predicates,
                },
                "physical_success": bool(attempt_by_seed[seed].get("success")),
                "physical_replay": {**view_replay, "exact_match": replay_exact},
                "episode_frames": int(episode_row["frames"]),
                "attempt_trace": {"attempt_ordinal": attempt_ordinal_by_seed[seed],
                                  "failure_reason": attempt_by_seed[seed].get("failure_reason")},
            })
    if solver_cls is not None and original_solver is not None:
        solver_cls.solve = original_solver
    setattr(scene_sampling, feasible_name, original_feasible)

    tolerance = float(args.near_duplicate_tolerance_m)
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            matched, distance = _same_family(
                [row["scene_object"] for row in records[left]["objects"]],
                [row["scene_object"] for row in records[right]["objects"]],
                tolerance)
            if matched:
                parent[find(right)] = find(left)
    grouped: dict[int, list[int]] = {}
    for index, record in enumerate(records):
        grouped.setdefault(find(index), []).append(record["episode_index"])
    families = list(grouped.values())
    by_episode = {row["episode_index"]: row for row in records}
    split_families = [family for family in families
                      if len({by_episode[index]["split"] for index in family}) > 1]
    source_hash = hashlib.sha256(inspect.getsource(scene_sampling.generate_scene).encode()).hexdigest()
    feasible_source_hash = hashlib.sha256(inspect.getsource(original_feasible).encode()).hexdigest()
    payload = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": common.sha256_file(args.protocol),
        "training_view": view.name,
        "training_view_sha256": meta["view_sha256"],
        "collection_manifest_sha256": {
            collection.name: common.sha256_file(
                (collection / "e1_manifest.json") if (collection / "e1_manifest.json").is_file()
                else (collection / "manifest.json"))
            for collection in collections},
        "attempt_ledger_sha256": {
            collection.name: common.sha256_file(collection / "attempts.jsonl")
            for collection in collections},
        "sampler_generate_scene_source_sha256": source_hash,
        "per_object_ik_source_sha256": feasible_source_hash,
        "certificate_hashes": _plain(certificate_hashes),
        "certificate_summary": certificate.summary(),
        "attempt_ledger_summary": attempt_summary,
        "layout_rule": {"minimum_pairwise_distance_m": protocol["scene_layout"]["minimum_pairwise_distance_m"],
                        "cube_edge_m": protocol["scene_layout"]["cube_edge_m"],
                        "near_duplicate_matching_tolerance_m": tolerance,
                        "near_duplicate_tolerance_basis": (
                            "centrewise half the registered cube edge; matching requires identical "
                            "colour, supply/destination arm, route and source region"),
                        "family_grouping": "transitive closure (union-find) over role-preserving edges",
                        "boundary_and_restricted_regions": (
                            "per-object source-region margins, table safety margin and certified "
                            "tray/shared keepout checks are recorded for every scene")},
        "summary": {
            "episodes": len(records),
            "train": sum(row["split"] == "train" for row in records),
            "validation": sum(row["split"] == "validation" for row in records),
            "unique_scene_hashes": len({row["scene_sha256"] for row in records}),
            "distinct_scene_families": len(families),
            "near_duplicate_families_crossing_split": split_families,
            "all_four_objects_per_episode": all(len(row["objects"]) == 4 for row in records),
            f"all_{total}_scenes_reconstructed_hash_exact": True,
            "all_physical_success_rows": all(row["physical_success"] for row in records),
            "all_physical_replay_exact": all(row["physical_replay"]["exact_match"]
                                              for row in records),
            "all_object_ik_records_complete": all(row["ik_complete_for_all_objects"] for row in records),
            "all_region_keepout_checks_passed": all(
                row["region_and_keepout_valid"] for row in records),
            "all_original_object_predicates_passed": all(
                row["original_success_predicate"]["passed"] for row in records),
            "attempts": attempt_summary["attempts"],
            "successes": attempt_summary["successes"],
            "failures": attempt_summary["failures"],
            "perturbations_present": False,
        },
        "records": records,
    }
    payload["passed"] = bool(
        len(records) == expected_counts[0]
        and payload["summary"]["train"] == expected_counts[1]
        and payload["summary"]["validation"] == expected_counts[2]
        and payload["summary"]["unique_scene_hashes"] == expected_counts[0]
        and not split_families
        and payload["summary"]["all_four_objects_per_episode"]
        and payload["summary"]["all_physical_success_rows"]
        and payload["summary"]["all_physical_replay_exact"]
        and payload["summary"]["all_object_ik_records_complete"]
        and payload["summary"]["all_region_keepout_checks_passed"]
        and payload["summary"]["all_original_object_predicates_passed"]
        and payload["summary"]["attempts"] == expected_attempts
        and payload["summary"]["successes"] == expected_successes
        and payload["summary"]["failures"] == expected_failures)
    common.write_json(output_file, payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
