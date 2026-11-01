#!/usr/bin/env python3
"""Combine Scene1 split, norm and FAST audits into one fail-closed evidence file."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256_file(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--norm-audit", type=Path, required=True)
    parser.add_argument("--fast-audit", type=Path, required=True)
    parser.add_argument("--scene-audit", type=Path, required=True)
    parser.add_argument("--data-audit", type=Path, required=True)
    parser.add_argument("--split-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    meta = json.loads((args.view / "meta.json").read_text(encoding="utf-8"))
    norm = json.loads(args.norm_audit.read_text(encoding="utf-8"))
    fast = json.loads(args.fast_audit.read_text(encoding="utf-8"))
    scene = json.loads(args.scene_audit.read_text(encoding="utf-8"))
    data = json.loads(args.data_audit.read_text(encoding="utf-8"))
    split = json.loads(args.split_evidence.read_text(encoding="utf-8"))
    norm_sha = sha256_file(args.view / "assets/so101/norm_stats.json")
    view_sha = meta["view_sha256"]

    problems = []
    for name, document, expected in (
        ("norm", norm, "pass"),
        ("fast", fast, "pass"),
    ):
        if document.get("status") != expected:
            problems.append(f"{name} audit status is not pass")
        if document.get("view_sha256") != view_sha:
            problems.append(f"{name} audit view hash differs")
    if norm.get("norm_stats_sha256") != norm_sha:
        problems.append("norm audit stats hash differs")
    if fast.get("norm_stats_sha256") != norm_sha:
        problems.append("FAST audit stats hash differs")
    if fast.get("windows") != {"train": 35843, "validation": 8228}:
        problems.append("FAST audit window counts differ")
    if scene.get("view_sha256") != view_sha:
        problems.append("scene audit view hash differs")
    if scene.get("perturbation_free") is not True:
        problems.append("scene audit is not perturbation-free")
    if scene.get("unique_scene_sha256") is not True:
        problems.append("scene hashes are not unique")
    if data.get("split_families_cross") != 0:
        problems.append("data audit reports family leakage")
    if data.get("incomplete_frames") != 0:
        problems.append("data audit reports incomplete frames")
    if data.get("soft_contacts") != 0:
        problems.append("data audit reports soft contacts")
    if split.get("family_crosses") != 0:
        problems.append("split evidence reports family leakage")
    for key, expected in (
        ("episodes", 64),
        ("train", 52),
        ("validation", 12),
        ("train_red", 26),
        ("train_blue", 26),
        ("validation_red", 6),
        ("validation_blue", 6),
    ):
        actual = (
            norm.get("counts", {}).get(key)
            if key != "episodes" else norm.get("counts", {}).get("episodes")
        )
        if actual != expected:
            problems.append(f"norm audit {key} is not {expected}")

    payload = {
        "schema_version": 1,
        "status": "pass" if not problems else "fail",
        "problems": problems,
        "action_horizon": 16,
        "view_sha256": view_sha,
        "norm_stats_sha256": norm_sha,
        "counts": norm.get("counts"),
        "windows": fast.get("windows"),
        "color_windows": fast.get("color_windows"),
        "normalized_action_roundtrip": fast.get(
            "normalized_action_roundtrip"),
        "quantile_forward_inverse": fast.get("quantile_forward_inverse"),
        "evidence": {
            "view_meta": evidence(args.view / "meta.json"),
            "norm_stats": evidence(args.view / "assets/so101/norm_stats.json"),
            "norm_audit": evidence(args.norm_audit),
            "fast_audit": evidence(args.fast_audit),
            "scene_audit": evidence(args.scene_audit),
            "data_audit": evidence(args.data_audit),
            "split_evidence": evidence(args.split_evidence),
        },
        "claim": (
            "All Scene1 train/validation windows and fixed FAST resources were "
            "audited without regenerating or modifying episode data."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": payload["status"],
        "problems": problems,
        "windows": payload["windows"],
    }, sort_keys=True))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
