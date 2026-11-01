#!/usr/bin/env python3
"""Create a new clean60 metadata version without duplicating episode bytes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import clean60_contracts as clean60  # noqa: E402
import so101_common as common  # noqa: E402
from export_training_view import collection_content_binding, public_episode  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-view", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--acknowledge-protocol-amendment", required=True)
    args = parser.parse_args()

    old = args.old_view.expanduser().resolve()
    collection = args.collection.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit("refusing to reuse the new view directory")
    protocol = common.load_json(args.protocol)
    old_meta = common.load_json(old / "meta.json")
    e1 = common.load_json(collection / "e1_manifest.json")
    summary = common.load_json(collection / "summary.json")
    verification = common.load_json(collection / "verification.json")
    audit_path = collection / "verification.json"
    rows = old_meta.get("episodes", [])
    if len(rows) != 60:
        raise SystemExit("old view does not contain 60 episodes")
    attempt_by_seed = {
        int(row["seed"]): row for row in summary["attempts"] if row.get("success")
    }
    if len(attempt_by_seed) != 60:
        raise SystemExit("collection summary does not contain 60 successful attempts")
    windows = common.load_json(old / "e4t_windows.json")
    old_tree = common.sha256_tree(old / "episodes")
    if old_tree["sha256"] != old_meta.get("view_sha256"):
        raise SystemExit("old episode tree is already inconsistent with its metadata")
    if verification.get("passed") is not True or verification.get("problems"):
        raise SystemExit("collection verification is not cleanly passed")

    binding = collection_content_binding(
        protocol, collection, e1["episodes"], summary)
    if not binding.get("passed"):
        raise SystemExit("collection content binding failed: " + "; ".join(binding["problems"]))
    if e1["protocol_sha256"] != common.sha256_file(args.protocol):
        binding["recorded_protocol_sha256"] = e1["protocol_sha256"]
        binding["live_protocol_sha256"] = common.sha256_file(args.protocol)
        binding["amendment"] = args.acknowledge_protocol_amendment
    else:
        binding["mode"] = "exact_bytes"
        binding["protocol_sha256"] = e1["protocol_sha256"]

    output.mkdir(parents=True)
    (output / "episodes").mkdir()
    for source in sorted((old / "episodes").rglob("*")):
        relative = source.relative_to(old / "episodes")
        target = output / "episodes" / relative
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, target)
    shutil.copytree(old / "assets", output / "assets")

    public_rows = []
    for row in rows:
        seed = int(row["seed"])
        attempt = attempt_by_seed.get(seed)
        if attempt is None or attempt.get("scene_sha256") != row.get("scene_sha256"):
            raise SystemExit(f"seed {seed}: successful attempt does not match the old view")
        diagnostics = Path(attempt["diagnostics"])
        if not diagnostics.is_dir():
            diagnostics = collection / "diagnostics" / diagnostics.name
        scene = common.load_json(diagnostics / "scene.json")
        if ((scene.get("clean_layout") or {}).get("perturbation_present") is not False
                or any(obj.get("route") != "direct" for obj in scene.get("objects", []))
                or len(scene.get("objects", [])) != 4):
            raise SystemExit(f"seed {seed}: scene lacks explicit clean/direct evidence")
        selected = windows.get(str(row["episode_index"]))
        if not isinstance(selected, list):
            raise SystemExit(f"episode {row['episode_index']}: frozen windows are missing")
        enriched = dict(row)
        enriched.update({
            "category": "clean",
            "subgroup": "clean",
            "perturbation_frame": None,
            "eligible_windows": max(0, int(row["frames"]) - 9),
            "excluded_cross_event_windows": 0,
            "fixed_windows": selected,
            "event_windows": selected,
        })
        public_rows.append(public_episode(enriched, require_clean=True))

    new_meta = dict(old_meta)
    protocol_binding_sha = common.sha256_json(binding)
    new_meta.update({
        "schema_version": 2,
        "episodes": public_rows,
        "clean_profile": {
            "category": "clean",
            "source": clean60.CLEAN_SOURCE,
            "perturbation_free": True,
            "perturbation_frame_field": "present_and_null",
            "episodes_bound": len(public_rows),
            "minimum_pairwise_distance_m": float(
                protocol["scene_layout"]["minimum_pairwise_distance_m"]),
            "protocol_binding_sha256": protocol_binding_sha,
        },
        "materialization": {
            "mode": "hardlink-content-version",
            "source_view": old.name,
            "source_view_sha256": old_meta["view_sha256"],
            "reason": "remote free space cannot hold a second 16.8 GiB JPEG/NPZ tree",
            "episode_bytes_changed": False,
        },
        "claim": "versioned clean60 contract metadata over the same replayed episode bytes",
    })
    common.write_json(output / "meta.json", new_meta)
    common.write_json(output / "protocol_binding.json", binding)
    common.write_json(output / "splits.json", old_meta["split"])
    shutil.copy2(old / "train_stats.json", output / "train_stats.json")
    shutil.copy2(old / "e4t_windows.json", output / "e4t_windows.json")
    new_tree = common.sha256_tree(output / "episodes")
    if new_tree["sha256"] != old_meta["view_sha256"]:
        raise SystemExit("hardlinked episode tree hash changed")
    admission = {
        "schema_version": 2, "passed": True,
        "data_gate": {
            "passed": True, "train_only_statistics": True,
            "clean_source": clean60.CLEAN_SOURCE,
            "replay_audit_sha256": common.sha256_file(audit_path),
            "reason": "explicit clean provenance, complete replay and content-bound view",
        },
        "view_sha256": new_meta["view_sha256"],
        "dataset_sha256": e1["dataset_info_sha256"],
        "split_sha256": common.sha256_json(old_meta["split"]),
        "train_stats_sha256": common.sha256_file(output / "train_stats.json"),
        "totals": {
            "episodes": 60, "train": 48, "validation": 12,
            "clean": 60, "recovery": 0,
        },
    }
    common.write_json(output / "e4t_admission.json", admission)
    problems = clean60.view_contract_problems(
        new_meta, protocol_binding=binding, expected_counts=(60, 48, 12),
        require_clean=True)
    problems += clean60.admission_contract_problems(
        admission, meta=new_meta, split=old_meta["split"],
        train_stats_sha256=common.sha256_file(output / "train_stats.json"))
    print(json.dumps({
        "output": output.name, "passed": not problems, "problems": problems,
        "view_sha256": new_meta["view_sha256"],
        "episode_files": new_tree["files"], "episode_bytes": new_tree["bytes"],
        "episodes": 60, "train": 48, "validation": 12,
    }, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
