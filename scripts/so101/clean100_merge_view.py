#!/usr/bin/env python3
"""Merge the preserved clean60 view with 40 new episodes into a versioned clean100 view.

The old 60 episode directories are hardlinked under their original indices, the
new 40 are appended as episodes 60..99, and the split is recomputed so that no
role-preserving near-duplicate family crosses train/validation.  Only the new
32/8 assignment may move; the preserved 48/12 split is immutable.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import clean60_contracts as clean60  # noqa: E402
import clean100_checks  # noqa: E402
import so101_common as common  # noqa: E402
from clean60_scene_audit import _same_family  # noqa: E402


def subset_tree_hash(root: Path, names: list[str]) -> str:
    """so101_common.sha256_tree digest over a named subset of episode directories."""
    import hashlib

    digest = hashlib.sha256()
    for name in names:
        for path in sorted(p for p in (root / "episodes" / name).rglob("*") if p.is_file()):
            relative = path.relative_to(root / "episodes").as_posix().encode()
            digest.update(relative)
            digest.update(b"\0")
            digest.update(common.sha256_file(path).encode())
            digest.update(b"\n")
    return digest.hexdigest()


def stats(arrays: np.ndarray) -> dict:
    arrays = np.asarray(arrays, dtype=np.float64)
    return {
        "mean": np.mean(arrays, axis=0).astype(float).tolist(),
        "std": np.std(arrays, axis=0).astype(float).tolist(),
        "q01": np.quantile(arrays, 0.01, axis=0).astype(float).tolist(),
        "q99": np.quantile(arrays, 0.99, axis=0).astype(float).tolist(),
    }


def scene_index(collection: Path) -> dict[int, dict]:
    """Successful attempt row and scene.json per seed."""
    summary = common.load_json(collection / "summary.json")
    out: dict[int, dict] = {}
    for row in summary["attempts"]:
        if not row.get("success"):
            continue
        scene_path = Path(str(row["diagnostics"])) / "scene.json"
        if not scene_path.is_absolute():
            scene_path = collection / scene_path
        scene = common.load_json(scene_path)
        out[int(row["seed"])] = {"attempt": row, "scene": scene}
    return out


def family_groups(records: list[dict], tolerance: float) -> list[list[int]]:
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            matched, _ = _same_family(records[left]["objects"], records[right]["objects"],
                                      tolerance)
            if matched:
                parent[find(right)] = find(left)
    grouped: dict[int, list[int]] = {}
    for index in range(len(records)):
        grouped.setdefault(find(index), []).append(records[index]["episode_index"])
    return list(grouped.values())


def resolve_new_split(rows: list[dict], records: list[dict],
                      old_split: dict[int, str], tolerance: float) -> tuple[dict[int, str], dict]:
    """Keep every family on one side; the preserved split is never modified."""
    families = family_groups(records, tolerance)
    assignment_default = {int(row["episode_index"]): row["split"] for row in rows}
    crossing_before = [family for family in families
                       if len({assignment_default[index] for index in family}) > 1]
    assignment, report = clean100_checks.resolve_split(rows, old_split, families, 32, 8)
    new_indices = [int(row["episode_index"]) for row in rows if row["collection"] == "new"]
    return assignment, {
        "families": families,
        "crossing_before_adjustment": crossing_before,
        "adjustments": report["adjustments"],
        "crossing_after_adjustment": report["crossing"],
        "new_train": sorted(index for index in new_indices
                            if assignment[index] == "train"),
        "new_validation": sorted(index for index in new_indices
                                 if assignment[index] == "validation"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-view", type=Path, required=True)
    parser.add_argument("--old-collection", type=Path, required=True)
    parser.add_argument("--new-view", type=Path, required=True)
    parser.add_argument("--new-collection", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--near-duplicate-tolerance-m", type=float, default=0.0125)
    args = parser.parse_args()

    old_view = args.old_view.expanduser().resolve()
    new_view = args.new_view.expanduser().resolve()
    old_collection = args.old_collection.expanduser().resolve()
    new_collection = args.new_collection.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit("refusing to reuse the new view directory")
    protocol = common.load_json(args.protocol)
    old_meta = common.load_json(old_view / "meta.json")
    new_meta = common.load_json(new_view / "meta.json")
    old_summary = common.load_json(old_collection / "summary.json")
    new_summary = common.load_json(new_collection / "summary.json")
    new_e1 = common.load_json(new_collection / "e1_manifest.json")
    new_split_file = common.load_json(new_collection / "split.json")
    if len(old_meta["episodes"]) != 60 or len(new_meta["episodes"]) != 40:
        raise SystemExit("expected the preserved 60 and a new 40-episode export")
    old_tree = common.sha256_tree(old_view / "episodes")
    new_tree = common.sha256_tree(new_view / "episodes")
    if old_tree["sha256"] != old_meta["view_sha256"]:
        raise SystemExit("preserved episode tree does not match its metadata")
    if new_tree["sha256"] != new_meta["view_sha256"]:
        raise SystemExit("new episode tree does not match its metadata")
    if old_meta["source_sha256"] != new_meta["source_sha256"]:
        raise SystemExit("the two collections do not share one expert source fingerprint")
    if old_meta.get("protocol_sha256") == new_meta.get("protocol_sha256"):
        raise SystemExit("new collection reused the preserved protocol bytes")
    old_seeds = {int(row["seed"]) for row in old_meta["episodes"]}
    new_seeds = {int(row["seed"]) for row in new_meta["episodes"]}
    if old_seeds & new_seeds:
        raise SystemExit("new seeds overlap the preserved seeds")
    seed_start, seed_stop = (int(value) for value in protocol["collection"]["direct_seed_range"])
    if not all(seed_start <= seed <= seed_stop for seed in new_seeds):
        raise SystemExit("new seeds fall outside the registered block")
    eval_start, eval_stop = (int(value) for value in protocol["evaluation"]["seed_range"])
    if new_seeds & set(range(eval_start, eval_stop + 1)):
        raise SystemExit("new seeds overlap the reserved evaluation block")
    old_scenes = {row["scene_sha256"] for row in old_meta["episodes"]}
    new_scenes = {row["scene_sha256"] for row in new_meta["episodes"]}
    if old_scenes & new_scenes or len(new_scenes) != 40:
        raise SystemExit("scene identities are not 40 new unique hashes")
    if new_meta["episodes"][0]["seed"] not in new_seeds:
        raise SystemExit("new view rows lack seeds")
    if new_e1.get("protocol_sha256") != common.sha256_file(args.protocol):
        raise SystemExit("new collection did not use the clean100 protocol bytes")
    for row in new_e1["episodes"]:
        if int(row["seed"]) not in new_seeds:
            raise SystemExit("collection manifest and exported view seeds differ")

    old_index = scene_index(old_collection)
    new_index = scene_index(new_collection)
    records = []
    for row in old_meta["episodes"]:
        scene = old_index[int(row["seed"])]["scene"]
        records.append({"episode_index": int(row["episode_index"]), "seed": int(row["seed"]),
                        "collection": "old", "scene_sha256": row["scene_sha256"],
                        "objects": scene["objects"]})
    for row in new_meta["episodes"]:
        scene = new_index[int(row["seed"])]["scene"]
        records.append({"episode_index": 60 + int(row["episode_index"]),
                        "seed": int(row["seed"]), "collection": "new",
                        "scene_sha256": row["scene_sha256"], "objects": scene["objects"]})
    if len({record["scene_sha256"] for record in records}) != 100:
        raise SystemExit("100 scene hashes are not unique")
    distances = []
    for record in records:
        points = [[float(v) for v in obj["xy"]] for obj in record["objects"]]
        if len(points) != 4:
            raise SystemExit("a scene does not contain four objects")
        distances.append(min(math.dist(a, b) for a, b in itertools.combinations(points, 2)))
    minimum = float(protocol["scene_layout"]["minimum_pairwise_distance_m"])
    if min(distances) < minimum:
        raise SystemExit(f"a scene fell below the registered {minimum} m separation")
    layout_keys = {
        tuple(sorted(tuple(round(float(v), 6) for v in obj["xy"]) for obj in record["objects"]))
        for record in records}
    if len(layout_keys) != 100:
        raise SystemExit("the 100 scenes do not have 100 fully distinct four-cube layouts")

    rows = []
    for row in old_meta["episodes"]:
        enriched = dict(row)
        enriched.update({"collection": "old",
                         "source_collection": old_collection.name,
                         "source_view": old_view.name})
        rows.append(enriched)
    for row in new_meta["episodes"]:
        enriched = dict(row)
        enriched.update({"collection": "new",
                         "episode_index": 60 + int(row["episode_index"]),
                         "source_collection": new_collection.name,
                         "source_view": new_view.name,
                         "new_collection_episode_index": int(row["episode_index"])})
        rows.append(enriched)
    old_split = {int(row["episode_index"]): row["split"] for row in old_meta["episodes"]}
    assignment, family_report = resolve_new_split(
        rows, records, old_split, args.near_duplicate_tolerance_m)
    for row in rows:
        row["split"] = assignment[int(row["episode_index"])]
    for row in rows:
        if row.get("source_collection") != new_collection.name:
            continue
        count = max(0, int(row["frames"]) - 9)
        n = 16 if row["split"] == "train" else 64
        selected = (sorted(set(np.linspace(0, count - 1, min(n, count), dtype=int).tolist()))
                    if count else [])
        row.update(eligible_windows=count, fixed_windows=selected, event_windows=selected,
                   excluded_cross_event_windows=0)
    for row in rows:
        row.pop("collection", None)
    train_ids = sorted(int(row["episode_index"]) for row in rows if row["split"] == "train")
    val_ids = sorted(int(row["episode_index"]) for row in rows if row["split"] == "validation")
    if len(train_ids) != 80 or len(val_ids) != 20:
        raise SystemExit(f"merged split is {len(train_ids)}/{len(val_ids)}, not 80/20")
    for index, side in old_split.items():
        if assignment[index] != side:
            raise SystemExit("the preserved 48/12 split was modified")
    if len(family_report["crossing_after_adjustment"]):
        raise SystemExit("family isolation failed")

    output.mkdir(parents=True)
    (output / "episodes").mkdir()
    for source in sorted((old_view / "episodes").rglob("*")):
        relative = source.relative_to(old_view / "episodes")
        target = output / "episodes" / relative
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, target)
    for source in sorted((new_view / "episodes").rglob("*")):
        relative = source.relative_to(new_view / "episodes")
        parts = list(relative.parts)
        old_index_str = parts[0]
        new_index = 60 + int(old_index_str.split("_")[-1])
        parts[0] = f"episode_{new_index:03d}"
        target = output / "episodes" / Path(*parts)
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, target)
    for name in ("assets",):
        import shutil
        shutil.copytree(old_view / name, output / name)
    view_tree = common.sha256_tree(output / "episodes")
    preserved_names = [f"episode_{index:03d}" for index in range(60)]
    if subset_tree_hash(output, preserved_names) != old_meta["view_sha256"]:
        raise SystemExit("hardlinked/old episode bytes changed during the merge")

    windows = {}
    old_windows = common.load_json(old_view / "e4t_windows.json")
    for key, value in old_windows.items():
        windows[key] = value
    for row in rows:
        if row.get("source_collection") != new_collection.name:
            continue
        windows[str(int(row["episode_index"]))] = row["fixed_windows"]
    if len(windows) != 100:
        raise SystemExit("window table does not cover 100 episodes")

    states, actions = [], []
    for episode_id in train_ids:
        with np.load(output / "episodes" / f"episode_{episode_id:03d}" / "episode.npz",
                     allow_pickle=False) as data:
            state = np.asarray(data["state"], dtype=np.float64)
            action = np.asarray(data["action"], dtype=np.float64)
        if state.shape != action.shape or state.ndim != 2 or state.shape[1] != 12:
            raise SystemExit(f"episode {episode_id}: invalid aligned arrays")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise SystemExit(f"episode {episode_id}: non-finite arrays")
        states.append(state)
        actions.append(action)
    train_state = np.concatenate(states)
    train_action = np.concatenate(actions)
    new_stats = {"state": stats(train_state), "actions": stats(train_action)}
    near_constant = {
        "state": [index for index, value in enumerate(new_stats["state"]["std"])
                  if float(value) < 1e-6],
        "actions": [index for index, value in enumerate(new_stats["actions"]["std"])
                    if float(value) < 1e-6],
    }
    if near_constant["state"] or near_constant["actions"]:
        raise SystemExit(f"near-constant dimensions in the 80-train statistics: {near_constant}")
    norm_path = output / "assets/so101/norm_stats.json"
    common.write_json(norm_path, {"norm_stats": new_stats})
    norm_sha = common.sha256_file(norm_path)
    canonical = {"state": {field: new_stats["state"][field]
                           for field in ("mean", "std", "q01", "q99")},
                 "actions": {field: new_stats["actions"][field]
                             for field in ("mean", "std", "q01", "q99")}}
    canonical_sha = clean60.sha256_json(canonical)

    binding = {
        "passed": True,
        "problems": [],
        "episodes_bound": 100,
        "checks": {
            "object_counts": [4],
            "routes": ["direct"],
            "perturbation_free": True,
            "minimum_pairwise_distance_m": {
                "registered": minimum,
                "observed_min": round(min(distances), 6),
                "observed_mean": round(sum(distances) / len(distances), 6),
                "observed_max": round(max(distances), 6),
                "episodes_below": sum(1 for value in distances if value < minimum),
            },
            "sources": {
                "preserved": {"collection": old_collection.name, "episodes": 60,
                              "protocol_sha256": old_meta["protocol_sha256"]},
                "new": {"collection": new_collection.name, "episodes": 40,
                        "protocol_sha256": new_meta["protocol_sha256"],
                        "seed_block": [seed_start, seed_stop]},
            },
            "attempts": {
                "preserved": int(old_summary["profiles"]["direct"]["attempts"]),
                "new": int(new_summary["profiles"]["direct"]["attempts"]),
                "total": int(old_summary["profiles"]["direct"]["attempts"])
                         + int(new_summary["profiles"]["direct"]["attempts"]),
            },
            "scene_hashes_unique": 100,
            "unique_layouts": len(layout_keys),
            "families": len(family_report["families"]),
            "families_crossing_split": [],
        },
    }
    if not binding["checks"]["minimum_pairwise_distance_m"]["episodes_below"] == 0:
        raise SystemExit("minimum separation violated")
    binding_sha = clean60.sha256_json(binding)
    split_payload = {
        "rule": protocol["collection"]["split"]["rule"],
        "train": train_ids,
        "validation": val_ids,
        "train_seeds": [int(row["seed"]) for row in rows if row["split"] == "train"],
        "validation_seeds": [int(row["seed"]) for row in rows if row["split"] == "validation"],
        "preserved": {"train": sorted(index for index, side in old_split.items() if side == "train"),
                      "validation": sorted(index for index, side in old_split.items()
                                           if side == "validation")},
        "new": {"train": family_report["new_train"],
                "validation": family_report["new_validation"]},
        "family_isolation": {
            "tolerance_m": args.near_duplicate_tolerance_m,
            "families": family_report["families"],
            "crossing": [],
            "adjustments": family_report["adjustments"],
            "preserved_families_unchanged": True,
        },
    }
    train_stats = {
        "created": __import__("time").strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scope": "train split only",
        "train_episodes": train_ids,
        "validation_episodes": val_ids,
        "train_frames": int(train_state.shape[0]),
        "state_shape": list(train_state.shape),
        "action_shape": list(train_action.shape),
        "constant_dimensions": {
            "state": [index for index, value in enumerate(new_stats["state"]["std"]) if value == 0.0],
            "actions": [index for index, value in enumerate(new_stats["actions"]["std"]) if value == 0.0],
        },
        "near_constant_dimensions_std_lt_1e-6": near_constant,
        "norm_stats_sha256": norm_sha,
        "norm_stats_canonical_sha256": canonical_sha,
        "stats": new_stats,
        "source": "recomputed from the merged 80 clean train episodes only",
    }
    common.write_json(output / "train_stats.json", train_stats)
    meta = dict(old_meta)
    meta.pop("source_collection", None)
    meta.update({
        "schema_version": 2,
        "created": train_stats["created"],
        "phase": "clean60+clean100 merge",
        "episodes": rows,
        "split": split_payload,
        "protocol_sha256": common.sha256_file(args.protocol),
        "preserved_protocol_sha256": old_meta["protocol_sha256"],
        "source_sha256": old_meta["source_sha256"],
        "expert_sha256": old_meta["expert_sha256"],
        "source_collections": {
            "preserved": old_collection.name,
            "new": new_collection.name,
        },
        "clean_profile": {
            "category": "clean",
            "source": clean60.CLEAN_SOURCE,
            "perturbation_free": True,
            "perturbation_frame_field": "present_and_null",
            "episodes_bound": 100,
            "minimum_pairwise_distance_m": minimum,
            "protocol_binding_sha256": binding_sha,
        },
        "materialization": {
            "mode": "hardlink-content-version",
            "preserved_view": old_view.name,
            "preserved_view_sha256": old_meta["view_sha256"],
            "new_view": new_view.name,
            "new_view_sha256": new_meta["view_sha256"],
        },
        "view_sha256": view_tree["sha256"],
        "view_files": view_tree["files"],
        "view_bytes": view_tree["bytes"],
        "norm_stats_sha256": norm_sha,
        "norm_stats_canonical_sha256": canonical_sha,
        "statistics_revision": "clean100_train80_recomputed_v1",
        "claim": "100 clean spread-layout scripted-expert episodes; 60 preserved bytes + 40 new; train-only statistics",
    })
    common.write_json(output / "meta.json", meta)
    common.write_json(output / "splits.json", split_payload)
    common.write_json(output / "protocol_binding.json", binding)
    common.write_json(output / "e4t_windows.json", windows)
    common.write_json(output / "e4t_admission.json", {
        "schema_version": 2, "passed": True,
        "data_gate": {
            "passed": True, "train_only_statistics": True,
            "clean_source": clean60.CLEAN_SOURCE,
            "reason": "preserved clean60 bytes plus a fully replayed clean100 collection; "
                      "train-only statistics from 80 episodes",
        },
        "view_sha256": meta["view_sha256"],
        "dataset_sha256": [old_meta.get("source_dataset_info_sha256"),
                            new_e1.get("dataset_info_sha256")],
        "split_sha256": clean60.sha256_json(split_payload),
        "train_stats_sha256": common.sha256_file(output / "train_stats.json"),
        "totals": {"episodes": 100, "train": 80, "validation": 20,
                   "clean": 100, "recovery": 0},
    })
    common.write_json(output / "merge_lineage.json", {
        "preserved_view": old_view.name,
        "preserved_view_sha256": old_meta["view_sha256"],
        "preserved_protocol_sha256": old_meta["protocol_sha256"],
        "new_view": new_view.name,
        "new_view_sha256": new_meta["view_sha256"],
        "new_collection": new_collection.name,
        "new_collection_summary_sha256": common.sha256_file(new_collection / "summary.json"),
        "new_collection_manifest_sha256": common.sha256_file(new_collection / "e1_manifest.json"),
        "new_collection_split": new_split_file,
        "protocol": args.protocol.name,
        "protocol_sha256": common.sha256_file(args.protocol),
        "seed_block": [seed_start, seed_stop],
        "new_seeds": sorted(new_seeds),
        "scene_hashes_unique": 100,
        "new_episode_index_map": {str(60 + int(row["episode_index"])): int(row["episode_index"])
                                  for row in new_meta["episodes"]},
    })
    problems = clean60.view_contract_problems(
        meta, protocol_binding=binding, expected_counts=(100, 80, 20), require_clean=True)
    problems += clean60.admission_contract_problems(
        common.load_json(output / "e4t_admission.json"), meta=meta,
        split=split_payload,
        train_stats_sha256=common.sha256_file(output / "train_stats.json"))
    report = {
        "output": output.name,
        "passed": not problems,
        "problems": problems,
        "view_sha256": meta["view_sha256"],
        "episodes": 100, "train": 80, "validation": 20,
        "preserved_tree_unchanged": subset_tree_hash(output, preserved_names)
                                    == old_meta["view_sha256"],
        "new_scene_hashes_unique": len(new_scenes) == 40,
        "unique_layouts": len(layout_keys),
        "families": len(family_report["families"]),
        "new_split_adjustments": family_report["adjustments"],
        "norm_stats_sha256": norm_sha,
        "norm_stats_canonical_sha256": canonical_sha,
        "train_frames": int(train_state.shape[0]),
        "minimum_pairwise_distance_m": min(distances),
    }
    common.write_json(output / "merge_report.json", report)
    print(json.dumps(report, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
