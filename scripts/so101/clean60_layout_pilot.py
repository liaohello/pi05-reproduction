#!/usr/bin/env python3
"""Clean60 layout pilot: measure which spread layouts the certified sampler can produce.

Runs the external certified sampler (its own pick-region and IK checks stay in
force) over a fixed seed grid and records, per seed, the four cube centres, all
six pairwise distances, the sampler verdict and the rejection reason.  Nothing is
executed physically here and no episode is saved: the output decides the minimum
separation, sampling region, boundary margin and attempt ceiling that the formal
60-episode collection is then frozen with.

Usage (on the GPU host, external project on PYTHONPATH):
    python3 scripts/so101/clean60_layout_pilot.py --root <external> \
        --protocol configs/so101/clean60_protocol.json --output runs/so101/clean60_layout_pilot
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_common as common  # noqa: E402


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--protocol", type=Path,
                        default=common.REPO_ROOT / "configs/so101/clean60_protocol.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=120,
                        help="certified sampler attempts in the pilot grid")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--thresholds", default="0.08,0.10,0.12,0.14")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    protocol = common.load_json(args.protocol)
    output = args.output.expanduser().resolve()
    if output.exists():
        print(f"refusing to reuse existing pilot directory: {output}", file=sys.stderr)
        return 2
    output.mkdir(parents=True)
    config = common.load_external_config(root, protocol)

    from bimanual.tasks.sorting import regions
    from bimanual.tasks.sorting.scene_sampling import generate_scene, SceneGenerationError
    from bimanual.sim.environment import BimanualEnv

    certificate = regions.load_certificate(root / config["evaluation"]["certificate"])
    start = int(protocol["collection"]["direct_seed_range"][0]) + args.seed_offset
    seeds = list(range(start, start + args.seeds))
    thresholds = sorted(float(value) for value in args.thresholds.split(","))
    records, rejections = [], {}
    with BimanualEnv(config, images=False) as env:
        for seed in seeds:
            try:
                scene = generate_scene(config, certificate, "direct", seed, env=env,
                                       sampling=str(config["sorting"].get("sampling_mode", "legacy")))
            except SceneGenerationError as error:
                reason = str(error).split(":")[0][:80]
                rejections[seed] = reason
                continue
            points = [tuple(float(v) for v in obj.xy) for obj in scene.objects]
            pairs = {f"{a}:{b}": round(math.dist(points[a], points[b]), 4)
                     for a, b in itertools.combinations(range(len(points)), 2)}
            records.append({
                "seed": seed, "scene_sha256": scene.sha256, "objects": len(points),
                "positions": [[round(v, 4) for v in point] for point in points],
                "pairwise_distances_m": pairs,
                "minimum_pairwise_distance_m": round(min(pairs.values()), 4),
                "routes": [obj.route for obj in scene.objects],
                "perturbation": scene.extra.get("perturbation"),
                "certified_reachable": True,
            })
    accepted = {value: sum(1 for row in records if row["minimum_pairwise_distance_m"] >= value)
                for value in thresholds}
    payload = {
        "schema_version": 1, "created": _now(),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": common.sha256_file(args.protocol),
        "external_root_fingerprint": common.sha256_file(Path(__file__)),
        "sampling": {"seed_grid": [seeds[0], seeds[-1]], "attempted": len(seeds),
                     "sampler_accepted": len(records),
                     "sampler_rejected": len(rejections),
                     "rejection_reasons": dict(sorted(
                         (reason, sum(1 for v in rejections.values() if v == reason))
                         for reason in set(rejections.values())), key=lambda kv: -kv[1])
                     if rejections else {}},
        "layout": {
            "objects_per_scene": sorted({row["objects"] for row in records}),
            "perturbation_present": sorted({bool(row["perturbation"]) for row in records}),
            "routes": sorted({route for row in records for route in row["routes"]}),
            "minimum_distance_m": {
                "observed_max": max((row["minimum_pairwise_distance_m"] for row in records), default=None),
                "observed_min": min((row["minimum_pairwise_distance_m"] for row in records), default=None),
            },
            "acceptance_by_threshold": {
                str(value): {"scenes": count,
                             "rate_of_sampler_accepted": round(count / len(records), 4) if records else None,
                             "rate_of_attempts": round(count / len(seeds), 4)}
                for value, count in accepted.items()},
        },
        "registered_candidate_minimum_m": protocol["scene_layout"]["minimum_pairwise_distance_m"],
        "episodes_saved": 0,
        "note": "pilot only: sampler geometry and IK verdicts; no expert execution and no dataset",
        "records": records,
    }
    common.write_json(output / "layout_pilot.json", payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
