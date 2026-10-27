#!/usr/bin/env python3
"""Summarise an E4-T main-collection run: attempts, per-profile quota and saved episodes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_common as common  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--protocol", type=Path,
                        default=common.REPO_ROOT / "configs/so101/e4t_protocol.json")
    args = parser.parse_args()
    path = args.run / "attempts.jsonl"
    if not path.is_file():
        print(json.dumps({"run": str(args.run), "state": "no_attempts_recorded"}))
        return 1
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    targets = common.load_json(args.protocol)["collection"]["target_successes_per_profile"]
    budgets = common.load_json(args.protocol)["collection"]["maximum_attempts_per_profile"]
    profiles = {}
    for row in rows:
        bucket = profiles.setdefault(row["profile"], {"attempts": 0, "episode_success": 0,
                                                     "saved": 0, "last_reason": None})
        bucket["attempts"] += 1
        bucket["episode_success"] += 1 if row.get("success") else 0
        bucket["saved"] += 1 if row.get("source_episode_index") is not None else 0
        if not row.get("success"):
            bucket["last_reason"] = row.get("raw_failure_reason") or row.get("failure_reason")
    print(json.dumps({
        "run": str(args.run), "total_attempts": len(rows),
        "global_attempt_budget": common.load_json(args.protocol)["collection"]["main_maximum_attempts"],
        "profiles": {name: {**values, "target": targets.get(name),
                            "attempt_budget": budgets.get(name)}
                     for name, values in sorted(profiles.items())},
        "quota_met": all(profiles.get(name, {}).get("saved", 0) >= target
                         for name, target in targets.items()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
