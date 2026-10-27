#!/usr/bin/env python3
"""Launch the frozen E4-T run from configs/so101/e4t_training.json.

Every hyper-parameter comes from the protocol file so the training command cannot drift from what
was registered, and the private base-checkpoint and tokenizer paths are read from the deploy config
without ever being printed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "deploy"))
sys.path.insert(0, str(REPO / "scripts" / "so101"))
import common as deploy  # noqa: E402
import so101_common as common  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True, help="external python for the launcher")
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--required-memory-mib", type=int, default=15500)
    parser.add_argument("--memory-margin-mib", type=int, default=1500)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--steps", type=int, default=None, help="override the registered update budget")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol = common.load_json(REPO / "configs/so101/e4t_training.json")
    optimisation = protocol["optimisation"]
    assets = deploy.load_assets()
    view = args.view.expanduser().resolve()
    admission = view / "e4t_admission.json"
    plan = view / "e4t_sampling_plan.json"
    for required in (admission, plan, view / "meta.json"):
        if not required.is_file():
            raise SystemExit(f"missing frozen artifact: {required}")
    argv = [str(args.python), str(REPO / "scripts/so101/e4t_run_training.py"),
            "--mode", "initial", "--run-dir", str(REPO / "runs/so101/e4t"),
            "--train-view", str(view), "--base-checkpoint", assets["checkpoint"],
            "--tokenizer", assets["tokenizer"], "--exp-name", "e4t_30k_state_on_b8",
            "--state-mode", protocol["initialisation"]["state_mode"],
            "--gpu-index", str(args.gpu_index),
            "--required-memory-mib", str(args.required_memory_mib),
            "--memory-margin-mib", str(args.memory_margin_mib),
            "--batch-size", str(optimisation["primary_batch_size"]),
            "--steps", str(args.steps or optimisation["planned_updates"]),
            "--save-interval", str(protocol["checkpointing"]["save_interval"]),
            "--data-loader-workers", str(optimisation["data_loader_workers"]),
            "--lr-warmup-steps", str(optimisation["warmup_steps"]),
            "--peak-lr", str(optimisation["peak_lr"]),
            "--decay-steps", str(optimisation["decay_steps"]),
            "--decay-lr", str(optimisation["decay_lr"]),
            "--sampling-plan", str(plan), "--data-admission", str(admission),
            "--dataset-label", str(view.name),
            "--timeout-seconds", str(optimisation["training_wall_limit_s"])]
    if args.run_id:
        argv += ["--run-id", args.run_id]
    shown = [value if value not in (assets["checkpoint"], assets["tokenizer"]) else "<asset>"
             for value in argv]
    print(json.dumps({"argv": shown, "view": str(view),
                      "admission_sha256": common.sha256_file(admission),
                      "sampling_plan_sha256": common.sha256_file(plan)}, indent=2))
    if args.dry_run:
        return 0
    return subprocess.call(argv)


if __name__ == "__main__":
    raise SystemExit(main())
