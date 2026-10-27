#!/usr/bin/env python3
"""Freeze the E4-T sampling plan into its own file so the run config can bind its hash.

The weights themselves are pre-registered in configs/so101/e4t_training.json before training;
this script only copies them out, refuses any group that the exported view does not contain,
and records which groups the view actually provides.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_common as common  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-config", type=Path,
                        default=common.REPO_ROOT / "configs/so101/e4t_training.json")
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    training = common.load_json(args.training_config)
    plan = dict(training["sampling_plan"])
    meta = common.load_json(args.view / "meta.json")
    train_groups = sorted({row.get("subgroup") for row in meta["episodes"] if row["split"] == "train"})
    configured = sorted(plan["group_weights"])
    missing = [group for group in train_groups if group not in configured]
    absent = [group for group in configured if group not in train_groups]
    if missing:
        raise SystemExit(f"training view has groups without a configured weight: {missing}")
    if absent:
        raise SystemExit(f"sampling plan weights groups that the train split does not contain: {absent}")
    document = {
        "schema_version": 1, "branch": "so101", "phase": "E3", "substage": "E4-T-sampling-plan",
        "mode": plan["mode"], "group_weights": plan["group_weights"],
        "episode_equal_within_group": True,
        "train_groups_present": train_groups,
        "train_episodes": sum(1 for row in meta["episodes"] if row["split"] == "train"),
        "validation_episodes": sum(1 for row in meta["episodes"] if row["split"] == "validation"),
        "view_sha256": meta["view_sha256"],
        "fixed_before_training": True,
        "claim": "group-balanced window sampling; validation episodes are never part of it",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    common.write_json(args.output, document)
    print(json.dumps({"output": str(args.output), "groups": train_groups,
                      "weights": plan["group_weights"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
