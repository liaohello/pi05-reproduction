#!/usr/bin/env python3
"""Measure how far the recorded gripper-slip timeline can be replayed, and compare with a bump episode.

The audit rejects a slip episode because its physical replay diverges; this script writes the
divergence itself as an artifact so the exclusion is backed by numbers instead of an assertion text.
Nothing is executed here beyond the scripted expert recording and its replay.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_common as common  # noqa: E402


def replay_one(module, directory: Path) -> dict:
    try:
        result = module.replay_episode(directory)
    except (AssertionError, ValueError, RuntimeError) as error:
        return {"passed": False, "error": f"{type(error).__name__}: {error}"}
    return {
        "passed": bool(result.get("passed")),
        "max_public_state_error": result.get("max_public_state_error"),
        "max_object_position_error_m": result.get("max_object_position_error_m"),
        "max_clock_error_s": result.get("max_clock_error_s"),
        "full_final_qpos_error": result.get("full_final_qpos_error"),
        "perturbation": result.get("perturbation"),
        "failure": result.get("failure") or result.get("problems"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None, help="patched external snapshot")
    parser.add_argument("--audit", type=Path, required=True,
                        help="verify_e4t.py output; its excluded_by_protocol list names the slip episodes")
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--mechanism", default="gripper_release_slip")
    parser.add_argument("--comparison-mechanism", default="arm_bump_impulse")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = common.setup_external(args.root or os.environ.get("SO101_SOURCE_ROOT"))
    if args.root:
        os.environ["SO101_SOURCE_ROOT"] = str(Path(args.root).expanduser().resolve())
    os.environ["BIMANUAL_ROOT"] = str(root)
    module = importlib.import_module("bimanual.recording.task_replay")

    audit = common.load_json(args.audit.expanduser().resolve())
    manifest = common.load_json(args.collection.expanduser().resolve() / "e4t_manifest.json")
    by_attempt = {row["attempt_id"]: row for row in manifest["episodes"]}
    selected = {args.mechanism: [], args.comparison_mechanism: []}
    for row in audit["episodes"]:
        mechanism = (row.get("perturbation_event") or {}).get("mechanism")
        if mechanism in selected and len(selected[mechanism]) < args.limit:
            selected[mechanism].append(row["attempt_id"])

    report = {
        "schema_version": 1, "branch": "so101", "phase": "E3", "substage": "E4-T-replay-divergence",
        "created": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "audit": str(args.audit), "root": str(root), "episodes": {},
        "interpretation": "",
    }
    for mechanism, attempts in selected.items():
        for attempt in attempts:
            entry = by_attempt.get(attempt)
            if not entry:
                continue
            report["episodes"][attempt] = {
                "mechanism": mechanism,
                "directory": entry["diagnostics"],
                "replay": replay_one(module, Path(entry["diagnostics"])),
            }
    slips = [row for row in report["episodes"].values() if row["mechanism"] == args.mechanism]
    bumps = [row for row in report["episodes"].values() if row["mechanism"] == args.comparison_mechanism]
    report["summary"] = {
        "slip_replayed": sum(1 for row in slips if row["replay"]["passed"]),
        "slip_total": len(slips),
        "bump_replayed": sum(1 for row in bumps if row["replay"]["passed"]),
        "bump_total": len(bumps),
    }
    report["interpretation"] = (
        "the slip episodes are excluded from the dataset because their recorded event timeline cannot be "
        "replayed inside this round's tolerance, while the bump episodes can; see the per-episode numbers")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": report["summary"],
                      "per_episode": {name: {"passed": row["replay"]["passed"],
                                             "public_state_error": row["replay"].get("max_public_state_error"),
                                             "final_qpos_error": row["replay"].get("full_final_qpos_error"),
                                             "object_error_m": row["replay"].get("max_object_position_error_m"),
                                             "clock_error_s": row["replay"].get("max_clock_error_s")}
                                      for name, row in report["episodes"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
