#!/usr/bin/env python3
"""Prove non-finite training metrics and corrupt view data stop before continuation."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

SCRIPTS = "/work/scripts/so101"
sys.path.insert(0, SCRIPTS)

import openpi_so101  # noqa: E402
import train_so101  # noqa: E402


def guard_test(loss, grad_norm, require_initial):
    guard = train_so101._FiniteOutputGuard(
        io.StringIO(), require_initial_update=require_initial)
    guard.write(
        f"Step 0: loss={loss}, grad_norm={grad_norm}, param_norm=1.0\n")


def corrupt_view_test():
    meta = {
        "feature_names": ["x"] * 12,
        "episodes": [{
            "episode_index": 0, "split": "train", "frames": 10,
            "seed": 1, "scene_sha256": "0" * 64,
        }],
    }
    state = np.zeros((10, 12), dtype=np.float32)
    state[0, 0] = np.nan
    with tempfile.TemporaryDirectory() as directory:
        view = Path(directory)
        episode = view / "episodes" / "episode_000"
        episode.mkdir(parents=True)
        (view / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        np.savez_compressed(
            episode / "episode.npz",
            state=state, action=np.zeros_like(state),
            timestamp=np.arange(10) / 50.0, frame_index=np.arange(10),
            task=np.asarray("test"),
        )
        try:
            openpi_so101.SO101EpisodeDataset(
                view, split="train", action_horizon=10)
        except ValueError as error:
            return "non-finite" in str(error), error
        return False, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checks = {}
    checks["finite_metrics_continue"] = True
    guard_test(0.5, 0.1, False)
    try:
        guard_test("nan", 0.1, False)
        checks["nonfinite_loss_stopped"] = False
        loss_error = None
    except FloatingPointError as error:
        checks["nonfinite_loss_stopped"] = True
        loss_error = str(error)
    try:
        guard_test(0.5, "inf", False)
        checks["nonfinite_grad_stopped"] = False
        grad_error = None
    except FloatingPointError as error:
        checks["nonfinite_grad_stopped"] = True
        grad_error = str(error)
    try:
        guard_test(0.5, 0.0, True)
        checks["zero_first_update_stopped"] = False
        zero_error = None
    except RuntimeError as error:
        checks["zero_first_update_stopped"] = True
        zero_error = str(error)
    corrupt, corrupt_error = corrupt_view_test()
    checks["corrupt_data_stopped"] = bool(corrupt)
    payload = {
        "schema_version": 1,
        "checks": checks,
        "errors": {
            "nonfinite_loss": loss_error,
            "nonfinite_grad": grad_error,
            "zero_first_update": zero_error,
            "corrupt_data": str(corrupt_error) if corrupt_error else None,
        },
        "passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
