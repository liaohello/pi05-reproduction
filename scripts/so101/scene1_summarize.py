#!/usr/bin/env python3
"""Summarise a scene1 collection for the reports and the experiment ledger.

Reads the collection manifest, attempts ledger and optional view audit.  All
counts are derived from the recorded rows; no success is inferred from a log.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _bands(value, edges):
    for index, edge in enumerate(edges):
        if value < edge:
            return f"lt_{edge}"
    return f"ge_{edges[-1]}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--view", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collection = args.collection.expanduser().resolve()
    manifest = json.loads((collection / "collection_manifest.json").read_text())
    attempts = [json.loads(line) for line in (collection / "attempts.jsonl").read_text().splitlines()
                if line.strip()]
    rows = manifest["successes_rows"]
    physical = [row for row in attempts if row.get("kind") == "episode"]
    layout = [row for row in attempts if row.get("kind") == "layout_reject"]
    failures = [row for row in physical if not row.get("success")]
    by_color = {}
    for color in ("red", "blue"):
        all_color = [row for row in physical if row.get("color") == color]
        ok_color = [row for row in all_color if row.get("success")]
        by_color[color] = {"attempts": len(all_color), "successes": len(ok_color),
                           "failures": len(all_color) - len(ok_color),
                           "failure_reasons": dict(Counter(row.get("failure_reason")
                                                           for row in all_color if not row.get("success")))}
    bands = defaultdict(lambda: defaultdict(int))
    failure_bands = defaultdict(lambda: defaultdict(int))
    for row in rows:
        scene = row["scene"]
        color = scene["color"]
        tray = scene["trays"][color]
        nominal = (0.24, 0.26 if color == "red" else -0.26)
        displacement = max(abs(tray[0] - nominal[0]), abs(tray[1] - nominal[1]))
        bands["yaw"][_bands(float(scene["yaw_degrees"]), [15, 30, 45, 60, 75])] += 1
        bands["block_x"][_bands(float(scene["block_xy"][0]), [0.20, 0.24, 0.28])] += 1
        bands["block_y"][_bands(abs(float(scene["block_xy"][1])), [0.08, 0.13, 0.18])] += 1
        bands["tray_displacement"][_bands(displacement, [0.01, 0.02, 0.03, 0.04])] += 1
        bands["tray_side"][color] += 1
    for row in failures:
        scene = row.get("scene") or {}
        color = scene.get("color", row.get("color"))
        if not scene:
            continue
        tray = scene["trays"][color]
        nominal = (0.24, 0.26 if color == "red" else -0.26)
        displacement = max(abs(tray[0] - nominal[0]), abs(tray[1] - nominal[1]))
        failure_bands["yaw"][_bands(float(scene["yaw_degrees"]), [15, 30, 45, 60, 75])] += 1
        failure_bands["block_x"][_bands(float(scene["block_xy"][0]), [0.20, 0.24, 0.28])] += 1
        failure_bands["block_y"][_bands(abs(float(scene["block_xy"][1])), [0.08, 0.13, 0.18])] += 1
        failure_bands["tray_displacement"][_bands(displacement, [0.01, 0.02, 0.03, 0.04])] += 1
        failure_bands["tray_side"][color] += 1
    summary = {
        "schema_version": 1,
        "protocol_id": manifest["protocol_id"],
        "collection_run": manifest.get("seed_range"),
        "attempts_physical": len(physical), "successes": len(rows),
        "failures": len(failures), "layout_rejects": len(layout),
        "failure_reasons": dict(Counter(row.get("failure_reason") for row in failures)),
        "by_color": by_color,
        "success_bands": {key: dict(value) for key, value in bands.items()},
        "failure_bands": {key: dict(value) for key, value in failure_bands.items()},
        "split": {key: manifest["split"][key] for key in
                  ("train", "validation", "train_red", "train_blue",
                   "validation_red", "validation_blue", "family_crosses")},
        "attempts_file": str(collection / "attempts.jsonl"),
    }
    if args.view is not None:
        view = args.view.expanduser().resolve()
        audit = json.loads((view / "data_audit.json").read_text())
        stats = json.loads((view / "train_stats.json").read_text())
        summary["view"] = {
            "view_sha256": json.loads((view / "meta.json").read_text())["view_sha256"],
            "train_frames": stats["train_frames"],
            "constant_dimensions": stats["constant_dimensions"],
            "norm_stats_sha256": stats["norm_stats_sha256"],
            "frames_total": audit["frames_total"], "frames_min": audit["frames_min"],
            "frames_max": audit["frames_max"],
            "control_tick_s": audit["control_tick_s"],
            "substeps_all_full": audit["substeps_all_full"],
            "incomplete_frames": audit["incomplete_frames"],
            "max_state_error_vs_trace": audit["max_state_error_vs_trace"],
            "max_action_error_vs_trace": audit["max_action_error_vs_trace"],
            "camera_sync_passed": audit["camera_sync"].get("passed"),
            "video_timestamps_aligned": audit["video_timestamps_aligned_across_cameras"],
            "contact_monitor_errors": audit["contact_monitor_errors"],
            "soft_contacts": audit["soft_contacts"],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
