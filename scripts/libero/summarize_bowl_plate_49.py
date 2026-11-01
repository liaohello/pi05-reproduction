#!/usr/bin/env python3
"""Combine the training, audit, checkpoint, and closed-loop evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-run", type=Path, required=True)
    parser.add_argument("--data-audit", type=Path, required=True)
    parser.add_argument("--model-audit", type=Path, required=True)
    parser.add_argument("--checkpoint-audit", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config_sha = sha256(args.config)
    data_audit = json.loads(args.data_audit.read_text(encoding="utf-8"))
    model_audit = json.loads(args.model_audit.read_text(encoding="utf-8"))
    checkpoint_audit = json.loads(
        args.checkpoint_audit.read_text(encoding="utf-8"))
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    metrics = [
        json.loads(line)
        for line in (args.train_run / "metrics.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not metrics:
        raise ValueError("training metrics are empty")
    losses = [float(row["metrics"]["loss"]) for row in metrics]
    final_step = int(metrics[-1]["train_state_step"])
    completed = int(evaluation.get("completed_episodes", 0))
    successes = int(evaluation.get("successes", 0))
    if completed != int(evaluation.get("episodes_per_task", -1)):
        raise ValueError("evaluation did not complete every requested episode")
    if successes != sum(bool(row["success"]) for row in evaluation["episodes"]):
        raise ValueError("evaluation success count is inconsistent")
    passed = (
        data_audit.get("passed") is True
        and model_audit.get("passed") is True
        and checkpoint_audit.get("passed") is True
        and evaluation.get("status") == "complete"
        and final_step == 6000
    )
    payload = {
        "schema_version": 1,
        "experiment": "pi05_libero_bowl_plate_49_lora",
        "config": {
            "path": str(args.config),
            "sha256": config_sha,
        },
        "protocol_sha256": data_audit.get("protocol_sha256"),
        "view_sha256": data_audit.get("view_sha256"),
        "norm_stats_sha256": data_audit["norm_stats"]["sha256"],
        "data_audit": {
            "path": str(args.data_audit),
            "sha256": sha256(args.data_audit),
            "passed": data_audit["passed"],
            "episodes": data_audit["dataset"]["selected_episodes"],
            "frames": data_audit["dataset"]["selected_frames"],
            "action_horizon": 10,
        },
        "model_audit": {
            "path": str(args.model_audit),
            "sha256": sha256(args.model_audit),
            "passed": model_audit["passed"],
            "base_checkpoint_tree_sha256": (
                model_audit["base_checkpoint_tree_sha256"]),
            "total_parameters": model_audit["total_parameters"],
            "trainable_parameters": model_audit["trainable_parameters"],
            "ema_decay": model_audit["ema_decay"],
        },
        "training": {
            "batch_size": 8,
            "fsdp_devices": 2,
            "final_train_state_step": final_step,
            "effective_sample_budget": final_step * 8,
            "metrics_file": "metrics.jsonl",
            "metrics_sha256": sha256(args.train_run / "metrics.jsonl"),
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "loss_min": min(losses),
            "loss_max": max(losses),
            "loss_is_not_closed_loop_evidence": True,
        },
        "checkpoint": {
            "audit_path": str(args.checkpoint_audit),
            "audit_sha256": sha256(args.checkpoint_audit),
            "passed": checkpoint_audit["passed"],
            "final_step": checkpoint_audit["restored_train_state_step"],
            "retained_steps": checkpoint_audit["retained_steps"],
            "step_files": checkpoint_audit["step_files"],
            "step_bytes": checkpoint_audit["step_bytes"],
            "parameter_tree_norm": checkpoint_audit["parameter_tree_norm"],
        },
        "closed_loop": {
            "path": str(args.evaluation),
            "sha256": sha256(args.evaluation),
            "status": evaluation["status"],
            "suite": evaluation["suite"],
            "task_ids": evaluation["task_ids"],
            "seed": evaluation["seed"],
            "replan_steps": evaluation["replan_steps"],
            "completed_episodes": completed,
            "successes": successes,
            "success_rate": successes / completed,
        },
        "evidence_level": (
            "R3-local-complete" if passed else "R3-local-incomplete"
        ),
        "passed": passed,
    }
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(args.out),
        "sha256": sha256(args.out),
        **payload,
    }, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
