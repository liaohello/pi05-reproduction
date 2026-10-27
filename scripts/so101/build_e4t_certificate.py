#!/usr/bin/env python3
"""Re-certify the unchanged E4-T geometry under its explicitly longer timeout.

The E1 certificate is not modified.  This command creates a new private
certificate in the patched snapshot and reports whether every prior geometric
point remains certified.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import so101_common as common  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--protocol", type=Path,
                        default=common.REPO_ROOT / "configs/so101/e4t_protocol.json")
    parser.add_argument("--old-certificate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid", type=float, default=0.010)
    args = parser.parse_args()

    root_value = args.root or os.environ.get("SO101_SOURCE_ROOT")
    if not root_value:
        raise SystemExit("missing --root or SO101_SOURCE_ROOT")
    root = common.setup_external(root_value)
    protocol = common.load_json(args.protocol)
    config = common.load_external_config(root, {**protocol, "collection": {
        "direct_seed_range": protocol["seeds"]["recovery_candidates"],
        "target_successes": 1, "maximum_attempts": 1, "recording_fps": protocol["control"]["recording_fps"],
        "control_hz": protocol["control"]["control_hz"], "diagnostics": True,
        "split": {"train": 1, "validation": 0, "rule": "certificate generation only"}}})
    config["sorting"]["timeout_s"] = float(protocol["control"]["timeout_s"])

    from bimanual.sim.environment import BimanualEnv
    from bimanual.tasks.sorting import regions
    from bimanual.tasks.sorting.certification import build_certificate, check_certificate_current

    with BimanualEnv(config, images=False) as env:
        certificate = build_certificate(env, grid=args.grid)
        stale = check_certificate_current(regions.Certificate(certificate), env)
        if stale:
            raise RuntimeError("new certificate is immediately stale: " + "; ".join(stale))
        regions.save_certificate(certificate, args.output)
        old = regions.load_certificate(args.old_certificate)
        old_points = {arm: [tuple(value) for value in old.points(arm)] for arm in ("left", "right")}
        new_points = {arm: [tuple(value) for value in regions.Certificate(certificate).points(arm)]
                      for arm in ("left", "right")}
        result = {
            "schema_version": 1, "output": str(args.output),
            "timeout_s": float(protocol["control"]["timeout_s"]),
            "hashes": regions.Certificate(certificate).hashes(),
            "old_hashes": old.hashes(),
            "geometry_only_hash_match": all(
                old.hashes()[key] == regions.Certificate(certificate).hashes()[key]
                for key in ("model_sha256", "config_sha256", "grasp_sha256")),
            "pick_point_count": {arm: len(new_points[arm]) for arm in ("left", "right")},
            "old_pick_point_count": {arm: len(old_points[arm]) for arm in ("left", "right")},
            "pick_points_identical": {arm: old_points[arm] == new_points[arm] for arm in ("left", "right")},
            "summary": certificate["summary"],
        }
    Path(args.output.with_suffix(".report.json")).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
