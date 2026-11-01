#!/usr/bin/env python3
"""Layout/IK/collision pilot for dual-block counterfactual scenes.

For every seed the sampler draws two tray poses, two block poses and two
continuous yaws, then screens both target instructions (target pick, target
place, initial settling).  No expert episode is executed and no dataset is
written.  The report keeps the draw distribution separate from the accepted
distribution so rejection bias is visible.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene2_common as s2  # noqa: E402


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _band(value: float, width: float = 5.0) -> str:
    lower = int(float(value) // width) * width
    return f"{lower:02.0f}-{lower + width:02.0f}"


def _spread(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
    array = np.asarray(values, dtype=float)
    return {"count": int(array.size), "min": round(float(array.min()), 6),
            "max": round(float(array.max()), 6),
            "mean": round(float(array.mean()), 6), "std": round(float(array.std()), 6)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--protocol", type=Path,
                        default=s2.REPO_ROOT / "configs/so101/scene2_dual_random_protocol.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--max-attempts-per-layout", type=int, default=None)
    parser.add_argument("--wall-clock-limit", type=float, default=None)
    args = parser.parse_args()

    root = s2.configure(args.root)
    protocol = s2.load_protocol(args.protocol)
    s2.set_protocol_id(protocol["protocol_id"])
    certificate_path = (root / protocol["source_binding"]["certificate"]).resolve()
    config = s2.load_config(root, protocol, certificate=certificate_path)
    certificate, certificate_path = s2.load_certificate(config)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to reuse existing pilot directory: {output}")
    output.mkdir(parents=True)

    settings = protocol["collection"]["layout_pilot"]
    seeds = int(args.seeds or settings["seeds"])
    seed_start = int(args.seed_start if args.seed_start is not None else settings["seed_start"])
    max_attempts = int(args.max_attempts_per_layout or settings["maximum_attempts_per_layout"])
    wall_limit = float(args.wall_clock_limit or settings["wall_clock_limit_s"])
    gate = float(settings.get("acceptance_gate_value", 0.5))
    provenance = s2.provenance(root, args.protocol, config, certificate_path)

    BimanualEnv = s2._b("BimanualEnv")
    certification = s2._b("certification")
    started = time.time()
    records, failures = [], []
    with BimanualEnv(config, images=False) as env:
        stale = certification.check_certificate_current(certificate, env)
        if stale:
            raise SystemExit("certificate stale: " + "; ".join(stale))
        s2.install_runtime(config, env)
        for offset in range(seeds):
            if time.time() - started > wall_limit:
                failures.append({"seed": seed_start + offset, "reason": "wall_clock_limit",
                                 "message": f"stopped after {time.time() - started:.1f} s"})
                break
            seed = seed_start + offset
            arrangement = "red_positive_y" if offset % 2 == 0 else "red_negative_y"
            trace = []
            seed_started = time.time()
            try:
                layout = s2.generate_layout(
                    config, certificate, seed, arrangement, env=env,
                    provenance_hashes=provenance, max_attempts=max_attempts, trace=trace)
            except s2._b("SceneGenerationError") as error:
                failures.append({
                    "seed": seed, "arrangement": arrangement, "reason": "scene_generation_failed",
                    "message": getattr(error, "message", str(error))[:500],
                    "attempts": len(trace), "wall_time_s": time.time() - seed_started,
                    "trace": trace,
                })
                continue
            layout["wall_time_s"] = time.time() - seed_started
            layout["trace"] = trace
            records.append(layout)

    accepted = len(records)
    attempted = accepted + len(failures)
    rate = round(accepted / attempted, 4) if attempted else 0.0
    by_arrangement = {}
    for arrangement in ("red_positive_y", "red_negative_y"):
        rows = [row for row in records if row["arrangement"] == arrangement]
        attempts = sum(1 for offset in range(seeds)
                       if (offset % 2 == 0) == (arrangement == "red_positive_y"))
        by_arrangement[arrangement] = {
            "attempted": attempts, "accepted": len(rows),
            "rate": round(len(rows) / attempts, 4) if attempts else None}
    accepted_yaws = {"cube_0": [], "cube_1": []}
    draw_yaws = {"cube_0": [], "cube_1": []}
    for row in records:
        for block in row["blocks"]:
            accepted_yaws[block["name"]].append(float(block["yaw_degrees"]))
    attempt_reasons = Counter()
    for row in records + failures:
        for entry in row.get("trace", []):
            attempt_reasons[str(entry.get("result"))] += 1
            for block in entry.get("blocks", []):
                draw_yaws[block["name"]].append(float(block["yaw_degrees"]))
    accepted_yaw_all = [value for values in accepted_yaws.values() for value in values]
    draw_yaw_all = [value for values in draw_yaws.values() for value in values]
    accepted_bands = Counter(_band(value) for value in accepted_yaw_all)
    draw_bands = Counter(_band(value) for value in draw_yaw_all)
    tray_x = [row["trays"][color][0] for row in records for color in ("red", "blue")]
    tray_y = [row["trays"][color][1] for row in records for color in ("red", "blue")]
    block_x = [block["xy"][0] for row in records for block in row["blocks"]]
    block_y = [block["xy"][1] for row in records for block in row["blocks"]]
    rejection_reasons = Counter()
    for row in failures:
        rejection_reasons[row["reason"]] += 1
    by_color_arm = {}
    for color in ("red", "blue"):
        rows = [row for row in records if any(block["color"] == color for block in row["blocks"])]
        arms = Counter(next(block["arm"] for block in row["blocks"] if block["color"] == color)
                       for row in rows)
        by_color_arm[color] = {"accepted_layouts": len(rows), "by_arm": dict(arms),
                               "yaw": _spread([float(next(block["yaw_degrees"] for block in row["blocks"]
                                                         if block["color"] == color))
                                               for row in rows])}
    payload = {
        "schema_version": 1,
        "created": _now(),
        "phase": "layout_ik_collision_pilot",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": s2.common.sha256_file(args.protocol),
        "profile_sha256": provenance["profile_sha256"],
        "adapter_sha256": provenance["adapter_sha256"],
        "certificate_sha256": provenance["certificate_sha256"],
        "source_fingerprint": {key: provenance[key] for key in
                               ("source_sha256", "source_files", "expert_sha256",
                                "model_sha256", "config_sha256", "packages")},
        "seed_grid": [seed_start, seed_start + max(0, seeds - 1)],
        "attempted": attempted, "accepted": accepted, "rejected": len(failures),
        "acceptance_rate": rate, "acceptance_gate": gate,
        "passed": bool(rate >= gate and accepted > 0),
        "by_arrangement": by_arrangement,
        "by_color_arm": by_color_arm,
        "yaw": {
            "draw": {"values": _spread(draw_yaw_all), "bands_5deg": dict(sorted(draw_bands.items()))},
            "accepted": {"values": _spread(accepted_yaw_all),
                         "bands_5deg": dict(sorted(accepted_bands.items()))},
            "by_object": {name: {"draw": _spread(values),
                                 "accepted": _spread(accepted_yaws[name])}
                          for name, values in draw_yaws.items()},
            "note": "draw and accepted distributions are reported separately; the accepted set is rejection-biased and is never described as uniform",
        },
        "tray": {"x_m": _spread(tray_x), "y_m": _spread(tray_y)},
        "block": {"x_m": _spread(block_x), "y_m": _spread(block_y)},
        "sampler_attempts": {
            "max": max((row["checks"]["sampler_attempts"] for row in records), default=None),
            "mean": round(float(np.mean([row["checks"]["sampler_attempts"] for row in records])), 4)
            if records else None,
        },
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "attempt_reasons": dict(sorted(attempt_reasons.items())),
        "unique_physical_hashes": len({row["physical_sha256"] for row in records}),
        "all_accepted_have_both_conditions": all(
            set(row["checks"]["conditions"]) == {"red", "blue"} for row in records),
        "all_accepted_initial_checks_passed": all(
            row["checks"]["initial"]["passed"] for row in records),
        "episodes_saved": 0,
        "expert_executions": 0,
        "wall_time_s": time.time() - started,
        "note": "layout/IK/collision pilot only; no expert execution and no dataset",
        "records": records,
        "failures": failures,
    }
    s2.common.write_json(output / "layout_pilot.json", payload)
    summary = {key: value for key, value in payload.items()
               if key not in ("records", "failures")}
    s2.common.write_json(output / "layout_pilot_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
