#!/usr/bin/env python3
"""Group an open-loop evaluation by colour, yaw, block position and tray layout.

The evaluator records one row per window with the episode id; this script joins
those rows to the frozen view metadata so the required red/blue and
position/tray failure distributions are measured, not inferred.
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path


def _mean(values):
    values = [value for value in values if value is not None]
    return round(sum(values) / len(values), 6) if values else None


def _band(value, edges):
    for edge in edges:
        if value < edge:
            return f"lt_{edge}"
    return f"ge_{edges[-1]}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True,
                        help="evaluation directory containing records.jsonl.gz")
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    meta = json.loads((args.view / "meta.json").read_text())
    episodes = {int(row["episode_index"]): row for row in meta["episodes"]}
    rows = []
    with gzip.open(args.report / "records.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    groups = defaultdict(list)
    for row in rows:
        episode = episodes[int(row["episode"])]
        color = episode["color"]
        yaw = float(episode["yaw_degrees"])
        block = episode["block_xy"]
        tray = episode["trays"][color]
        nominal = (0.24, 0.26 if color == "red" else -0.26)
        displacement = max(abs(tray[0] - nominal[0]), abs(tray[1] - nominal[1]))
        keys = {
            "all": "all",
            "color": color,
            "yaw": _band(yaw, [15, 30, 45, 60]),
            "block_x": _band(float(block[0]), [0.20, 0.24, 0.28]),
            "block_y_abs": _band(abs(float(block[1])), [0.08, 0.13, 0.18]),
            "tray_displacement": _band(displacement, [0.01, 0.02, 0.03]),
            "tray_side": color,
        }
        for family, key in keys.items():
            groups[f"{family}:{key}"].append(row)
    summary = {}
    for key, values in sorted(groups.items()):
        summary[key] = {
            "windows": len(values),
            "episodes": len({int(row["episode"]) for row in values}),
            "joint_mae_deg": _mean([row["joint_mae_deg"] for row in values]),
            "gripper_mae_pct": _mean([row["gripper_mae_pct"] for row in values]),
            "p95_joint_deg": _mean([row["p95_joint_deg"] for row in values]),
            "hold_joint_mae_deg": _mean([row["hold_joint_mae_deg"] for row in values]),
            "train_mean_joint_mae_deg": _mean([row["mean_joint_mae_deg"] for row in values]),
            "hold_gripper_mae_pct": _mean([row["hold_gripper_mae_pct"] for row in values]),
            "train_mean_gripper_mae_pct": _mean([row["mean_gripper_mae_pct"] for row in values]),
        }
    payload = {"schema_version": 1, "report": args.report.name,
               "view_sha256": meta.get("view_sha256"), "windows": len(rows),
               "groups": summary,
               "claim": "open-loop action prediction grouped by frozen scene attributes; not closed-loop success"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
