#!/usr/bin/env python3
"""Layout/IK pilot for single-block scenes; no episode is executed or saved.

For every seed the sampler draws two tray positions, one block position and a
continuous yaw, then runs the per-scene grasp/hover IK and collision screening.
The resulting acceptance rates decide the frozen tray region, yaw domain and
attempt ceiling used by the physical pilot and formal collection.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_common as common  # noqa: E402
import source_fingerprint  # noqa: E402


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _color(seed: int, start: int) -> str:
    return "red" if (seed - start) % 2 == 0 else "blue"


def _yaw_band(yaw: float) -> str:
    lower = int(yaw // 15) * 15
    return f"{lower:02d}-{lower + 15:02d}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--protocol", type=Path,
                        default=common.REPO_ROOT / "configs/so101/scene1_single_random_protocol.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=15050)
    args = parser.parse_args()

    root = common.setup_external(args.root)
    protocol = common.load_json(args.protocol)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to reuse existing pilot directory: {output}")
    output.mkdir(parents=True)
    from bimanual.config import load_config
    from bimanual.sim.environment import BimanualEnv
    from bimanual.tasks.sorting import regions
    from bimanual.tasks.single_block.scene_sampling import (SceneGenerationError,
                                                            generate_single_scene)

    config = load_config(root=root, profile=protocol["source_binding"]["profile"])
    certificate_path = root / config["evaluation"]["certificate"]
    certificate = regions.load_certificate(certificate_path)
    seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.seeds)))
    records, failures = [], []
    with BimanualEnv(config, images=False) as env:
        env.data.qpos[env.robot.qpos_ids] = env.robot.to_radians(env.robot.home())
        import mujoco
        mujoco.mj_forward(env.model, env.data)
        for seed in seeds:
            color = _color(seed, args.seed_start)
            trace = []
            try:
                scene = generate_single_scene(config, certificate, seed, color, env=env,
                                              max_attempts=400, trace=trace)
            except SceneGenerationError as error:
                failures.append({"seed": seed, "color": color, "reason": str(error)[:400],
                                 "trace": trace})
                continue
            records.append({
                "seed": seed, "color": color, "scene_sha256": scene.sha256,
                "block_xy": scene.extra["block_xy"], "yaw_degrees": scene.extra["yaw_degrees"],
                "trays": scene.extra["trays"], "slot_name": scene.extra["slot_name"],
                "active_arm": scene.extra["active_arm"],
                "sampler_attempts": scene.extra["checks"]["sampler_attempts"],
                "pick": scene.extra["checks"]["pick"],
                "slots": scene.extra["checks"]["slots"],
                "sampler_trace": trace,
            })
    by_color = {}
    for color in ("red", "blue"):
        rows = [row for row in records if row["color"] == color]
        attempts = sum(1 for seed in seeds if _color(seed, args.seed_start) == color)
        by_color[color] = {"attempted": attempts, "accepted": len(rows),
                           "rate": round(len(rows) / attempts, 4) if attempts else None}
    yaws = [row["yaw_degrees"] for row in records]
    yaw_bands = Counter(_yaw_band(value) for value in yaws)
    nominal = {"red": [0.24, 0.26], "blue": [0.24, -0.26]}
    displacements = [max(abs(row["trays"][color][0] - nominal[color][0]),
                         abs(row["trays"][color][1] - nominal[color][1]))
                     for row in records for color in (row["color"],)]
    block_x = [row["block_xy"][0] for row in records]
    block_y = [row["block_xy"][1] for row in records]
    all_draws = [entry for row in records for entry in row["sampler_trace"]]
    all_draws += [entry for row in failures for entry in row.get("trace", [])]
    draw_reasons = Counter(entry["result"] for entry in all_draws)
    first_draw = sum(1 for row in records if len(row["sampler_trace"]) == 1)
    payload = {
        "schema_version": 1, "created": _now(),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": common.sha256_file(args.protocol),
        "base_source_sha256": protocol["source_binding"]["source_sha256"],
        "current_source_fingerprint": source_fingerprint.fingerprint(
            root, os.environ.get("SO101_PYTHON", sys.executable)),
        "certificate_sha256": common.sha256_file(certificate_path),
        "config_sha256": common.sha256_json(config),
        "seed_grid": [seeds[0], seeds[-1]],
        "attempted": len(seeds), "accepted": len(records), "rejected": len(failures),
        "acceptance_rate": round(len(records) / len(seeds), 4),
        "by_color": by_color,
        "yaw": {
            "min": round(min(yaws), 6) if yaws else None,
            "max": round(max(yaws), 6) if yaws else None,
            "bands": dict(sorted(yaw_bands.items())),
            "in_30_60": sum(1 for value in yaws if 30.0 <= value < 60.0),
        },
        "tray_displacement_from_nominal_m": {
            "min": round(min(displacements), 6) if displacements else None,
            "max": round(max(displacements), 6) if displacements else None,
            "above_0.01": sum(1 for value in displacements if value > 0.01),
            "above_0.02": sum(1 for value in displacements if value > 0.02),
            "above_0.04": sum(1 for value in displacements if value > 0.04),
        },
        "block_xy_range": {
            "x": [round(min(block_x), 6), round(max(block_x), 6)] if block_x else None,
            "y": [round(min(block_y), 6), round(max(block_y), 6)] if block_y else None,
        },
        "slot_counts": dict(Counter(row["slot_name"].split("_")[-1] for row in records)),
        "sampler_attempts": {
            "max": max((row["sampler_attempts"] for row in records), default=None),
            "mean": round(float(np.mean([row["sampler_attempts"] for row in records])), 4)
            if records else None,
        },
        "unique_scene_sha256": len({row["scene_sha256"] for row in records}) == len(records),
        "sampler_draws": {
            "total": len(all_draws),
            "first_draw_accepted": first_draw,
            "first_draw_rate": round(first_draw / len(records), 4) if records else None,
            "results": dict(sorted(draw_reasons.items())),
        },
        "episodes_saved": 0,
        "note": "layout/IK pilot only; no expert execution and no dataset",
        "records": records, "failures": failures,
    }
    common.write_json(output / "layout_pilot.json", payload)
    print(json.dumps({key: value for key, value in payload.items()
                      if key not in ("records", "failures", "current_source_fingerprint")}, indent=2))
    return 0 if payload["acceptance_rate"] >= 0.80 else 1


if __name__ == "__main__":
    sys.exit(main())
