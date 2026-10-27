#!/usr/bin/env python3
"""Emit the realised learning-rate trace of an E4-T run.

The OpenPI trainer does not print the learning rate, so the value has to come from the schedule the
trainer actually uses. This script imports ``CosineDecaySchedule`` from the same image and the same
config the run was built from and evaluates it at the steps of interest, instead of re-implementing
the cosine formula here and hoping it matches.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_common as common  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-config", type=Path, required=True,
                        help="config_initial.json written by the training entry point")
    parser.add_argument("--training-config", type=Path, required=True,
                        help="configs/so101/e4t_training.json, to check the registered values")
    parser.add_argument("--steps", default="",
                        help="comma separated steps to evaluate (the checkpoint steps)")
    parser.add_argument("--dense-grid", type=int, default=0,
                        help="also emit this many evenly spaced points between 1 and the last step")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from openpi.training import optimizer as _optimizer  # noqa: E402

    run_config = common.load_json(args.run_config.expanduser().resolve())
    registered = common.load_json(args.training_config.expanduser().resolve())["optimisation"]
    schedule_config = run_config["lr_schedule"]
    for field, key in (("warmup_steps", "warmup_steps"), ("peak_lr", "peak_lr"),
                       ("decay_steps", "decay_steps"), ("decay_lr", "decay_lr")):
        if float(schedule_config[field]) != float(registered[key]):
            raise SystemExit(f"the run's {field}={schedule_config[field]} differs from the registered "
                             f"{registered[key]}; the trace would not describe this round's run")

    schedule = _optimizer.CosineDecaySchedule(
        warmup_steps=int(schedule_config["warmup_steps"]), peak_lr=float(schedule_config["peak_lr"]),
        decay_steps=int(schedule_config["decay_steps"]), decay_lr=float(schedule_config["decay_lr"]))

    # openpi schedules are optax-style factories: the object itself is not callable
    learning_rate = schedule.create()

    def lr(step: int) -> float:
        import jax.numpy as jnp

        return float(learning_rate(jnp.asarray(step, dtype=jnp.float32)))

    steps = sorted({int(value) for value in args.steps.split(",") if value.strip()})
    last = max([steps[-1]] if steps else [int(run_config["steps"])])
    if args.dense_grid:
        step_count = max(1, args.dense_grid)
        steps = sorted(set(steps) | {int(round(1 + index * (last - 1) / max(1, step_count - 1)))
                                     for index in range(step_count)})
    trace = [{"step": step, "lr": lr(step)} for step in steps]
    document = {
        "schema_version": 1, "branch": "so101", "phase": "E3", "substage": "E4-T-T1-lr-trace",
        "created": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "source": "openpi.training.optimizer.CosineDecaySchedule evaluated in the training image, "
                  "with the schedule object the trainer itself builds",
        "run_config": str(args.run_config), "run_config_sha256": common.sha256_file(
            args.run_config.expanduser().resolve()),
        "schedule": schedule_config, "points": trace,
        "peak_step": max(trace, key=lambda row: row["lr"])["step"],
        "final_lr": trace[-1]["lr"] if trace else None,
        "claim": "configuration trace; the trainer does not log the learning rate, so this is the "
                 "value the registered schedule produces at each step",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: document[key] for key in ("schedule", "peak_step", "final_lr")}, indent=2))
    print(json.dumps(document["points"][:6], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
