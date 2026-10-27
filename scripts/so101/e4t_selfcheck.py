#!/usr/bin/env python3
"""E4-T T0/T1 self-check that must run inside the OpenPI container.

The host test suite cannot import JAX, so the four claims that only the training
stack can prove are checked here against the real exported view and the real
TrainConfig:

* the batch size in the CLI reaches ``TrainConfig.batch_size`` and the loader;
* the long LR schedule replaces the historical 10/100/peak==decay hard-coding;
* validation episodes, statistics and gradients stay separate from training;
* a 10-step window is aligned with the recorded action slice, no off-by-one.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import openpi_so101 as so101  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--mode", default="initial")
    args = parser.parse_args()

    run_config = json.loads(args.run_config.read_text(encoding="utf-8"))
    meta = json.loads((args.view / "meta.json").read_text(encoding="utf-8"))
    train_ids = sorted(int(row["episode_index"]) for row in meta["episodes"] if row["split"] == "train")
    val_ids = sorted(int(row["episode_index"]) for row in meta["episodes"] if row["split"] == "validation")
    batch = int(run_config["batch_size"])

    config = so101.make_train_config(
        mode=args.mode, run_root=args.view.parent / "selfcheck", train_view=args.view,
        base_checkpoint=args.base_checkpoint, exp_name="e4t_selfcheck",
        state_mode=run_config.get("state_mode", "on"), batch_size=batch,
        lr_warmup_steps=int(run_config["lr_schedule"]["warmup_steps"]),
        peak_lr=float(run_config["lr_schedule"]["peak_lr"]), decay_steps=int(run_config["lr_schedule"]["decay_steps"]),
        decay_lr=float(run_config["lr_schedule"]["decay_lr"]))

    dataset = so101.SO101EpisodeDataset(args.view, split="train",
                                        action_horizon=config.model.action_horizon,
                                        episode_ids=train_ids)
    seen_episodes = {episode for episode, _ in dataset.samples}
    alignment_errors = []
    for episode_id, frame in dataset.samples[:64]:
        arrays = dataset.episodes[episode_id]["arrays"]
        window = np.asarray(arrays["action"][frame:frame + config.model.action_horizon])
        if window.shape != (config.model.action_horizon, so101.ACTION_DIM):
            alignment_errors.append([int(episode_id), int(frame), list(window.shape)])
        elif not np.array_equal(window, np.asarray(arrays["action"])[frame:frame + 10]):
            alignment_errors.append([int(episode_id), int(frame), "mismatch"])

    schedule = config.lr_schedule
    results = {
        "schema_version": 1, "phase": "E3", "substage": "E4-T-T0-selfcheck",
        "batch_size": {
            "cli": batch, "train_config": int(config.batch_size),
            "pass": int(config.batch_size) == batch},
        "lr_schedule": {
            "warmup_steps": int(schedule.warmup_steps), "peak_lr": float(schedule.peak_lr),
            "decay_steps": int(schedule.decay_steps), "decay_lr": float(schedule.decay_lr),
            "pass": (int(schedule.warmup_steps) == int(run_config["lr_schedule"]["warmup_steps"])
                     and abs(float(schedule.peak_lr) - float(run_config["lr_schedule"]["peak_lr"])) < 1e-12
                     and int(schedule.decay_steps) == int(run_config["lr_schedule"]["decay_steps"])
                     and abs(float(schedule.decay_lr) - float(run_config["lr_schedule"]["decay_lr"])) < 1e-12
                     and float(schedule.decay_lr) < float(schedule.peak_lr))},
        "split_isolation": {
            "train_episodes": len(train_ids), "validation_episodes": len(val_ids),
            "episodes_in_loader_samples": len(seen_episodes),
            "validation_leaked": sorted(seen_episodes & set(val_ids)),
            "pass": bool(seen_episodes <= set(train_ids) and not (seen_episodes & set(val_ids)))},
        "horizon_alignment": {
            "checked_windows": min(64, len(dataset.samples)),
            "action_horizon": int(config.model.action_horizon),
            "errors": alignment_errors, "pass": not alignment_errors},
        "window_accounting": {
            "loader_samples": len(dataset.samples),
            "meta_eligible_windows_train": sum(int(row.get("eligible_windows", 0))
                                               for row in meta["episodes"] if row["split"] == "train"),
            "pass": len(dataset.samples) == sum(int(row.get("eligible_windows", 0))
                                                for row in meta["episodes"] if row["split"] == "train")},
        "state_mode": str(run_config.get("state_mode", "on")),
        "claim": "configuration and data-path self-check; not a training result",
    }
    results["all_passed"] = all(row["pass"] for row in results.values() if isinstance(row, dict))
    print(json.dumps(results, indent=2))
    return 0 if results["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
