#!/usr/bin/env python3
"""Turn the timestamped E2 training log into machine-readable evidence."""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

STEP = re.compile(r"^(?:(\d+(?:\.\d+)?)\s+)?Step\s+(\d+):\s+(.*)$")
VALUE = re.compile(r"(loss|grad_norm|param_norm)=([-+0-9.eE]+)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("initial", "resume", "smoke"), required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    run = json.loads((run_dir / f"run_{args.mode}.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / f"config_{args.mode}.json").read_text(encoding="utf-8"))
    log_path = run_dir / f"train_{args.mode}.log"
    steps = []
    last_epoch = None
    for raw in log_path.read_text(encoding="utf-8", errors="replace").split("\n"):
        line = raw.replace("\r", "").strip()
        if not line:
            continue
        match = STEP.match(line)
        if match:
            epoch_text, step, rest = match.groups()
            epoch = float(epoch_text) if epoch_text else last_epoch
            if epoch is None:
                continue
            values = {key: float(value) for key, value in VALUE.findall(rest)}
            steps.append({"epoch": epoch, "step": int(step), **values})
            continue
        # tqdm writes the Step text after its own CR; the timestamp may be on
        # its own line. Keep the most recent leading timestamp.
        epoch_match = re.match(r"^(\d+(?:\.\d+)?)\b", line)
        if epoch_match:
            last_epoch = float(epoch_match.group(1))
    steps.sort(key=lambda item: item["step"])
    losses = [item["loss"] for item in steps if "loss" in item]
    grads = [item["grad_norm"] for item in steps if "grad_norm" in item]
    params_norm = [item["param_norm"] for item in steps if "param_norm" in item]
    report = {
        "mode": args.mode,
        "run": run,
        "config": {key: config.get(key) for key in
                   ("mode", "run_root", "train_view", "base_checkpoint", "exp_name",
                    "norm_stats_sha256", "train_view_meta_sha256")},
        "updates_completed": len(steps),
        "first_step": steps[0] if steps else None,
        "last_step": steps[-1] if steps else None,
        "time_to_first_step_s": (steps[0]["epoch"] - run["started_unix"]) if steps else None,
        "step_interval_s": ((steps[-1]["epoch"] - steps[0]["epoch"]) / (len(steps) - 1)
                            if len(steps) > 1 else None),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_min": min(losses) if losses else None,
        "loss_max": max(losses) if losses else None,
        "loss_last10_mean": statistics.fmean(losses[-10:]) if losses else None,
        "grad_norm_first": grads[0] if grads else None,
        "grad_norm_last": grads[-1] if grads else None,
        "param_norm_first": params_norm[0] if params_norm else None,
        "param_norm_last": params_norm[-1] if params_norm else None,
        "gpu_peak_mib_sampled": run.get("gpu_peak_mib_sampled"),
        "wall_time_s": run.get("wall_time_s"),
        "exit_code": run.get("exit_code"),
        "stop_method": "configured num_train_steps reached; process exited",
        "note": "time_to_first_step_s includes data-loader start, base-weight load and JAX compile",
    }
    text = json.dumps(report, indent=2) + "\n"
    output = args.output or (run_dir / f"summary_{args.mode}.json")
    output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
