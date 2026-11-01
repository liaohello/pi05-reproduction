#!/usr/bin/env python3
"""Record the pinned Optax learning-rate schedule with update-number semantics."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import optax

from openpi.training import optimizer


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    schedule = optimizer.CosineDecaySchedule(
        warmup_steps=1000,
        peak_lr=2.5e-5,
        decay_steps=40000,
        decay_lr=2.5e-6,
    ).create()
    indexes = (0, 1, 998, 999, 1000, 1001, 39998, 39999, 40000, 40001)
    values = {str(index): float(schedule(index)) for index in indexes}
    update_points = {
        "1": values["0"],
        "999": values["998"],
        "1000": values["999"],
        "1001": values["1000"],
        "40000": values["39999"],
    }
    payload = {
        "schema_version": 1,
        "status": "pass",
        "optax_version": optax.__version__,
        "optimizer_source": str(args.optimizer_source),
        "optimizer_source_sha256": sha256_file(args.optimizer_source),
        "schedule": {
            "type": "warmup_cosine_decay_schedule",
            "init_value": 2.5e-5 / (1000 + 1),
            "peak_value": 2.5e-5,
            "warmup_steps": 1000,
            "decay_steps_includes_warmup": True,
            "decay_steps": 40000,
            "end_value": 2.5e-6,
        },
        "schedule_index_values": values,
        "optimizer_update_to_schedule_index": update_points,
        "semantics": (
            "The optimizer uses the pre-increment TrainState.step. The 40,000th "
            "completed update therefore reads schedule index 39,999 and leaves "
            "TrainState.step=40,000."),
        "configured_final_lr": 2.5e-6,
        "actual_last_update_lr": update_points["40000"],
        "actual_minus_configured": update_points["40000"] - 2.5e-6,
        "within_numerical_tolerance": abs(
            update_points["40000"] - 2.5e-6) <= 1e-12,
        "source_definition": inspect.getsource(
            optax.warmup_cosine_decay_schedule),
    }
    if not payload["within_numerical_tolerance"]:
        payload["status"] = "fail"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": payload["status"],
        "first_update_lr": update_points["1"],
        "update_1000_lr": update_points["1000"],
        "update_1001_lr": update_points["1001"],
        "update_40000_lr": update_points["40000"],
    }, sort_keys=True))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
