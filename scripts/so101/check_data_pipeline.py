#!/usr/bin/env python3
"""CPU-only check of the SO101 training-view -> OpenPI transform pipeline.

Used before spending GPU time on E2: verifies image parsing, normalisation,
padding to 32 dimensions, action-chunk shape and prompt tokenisation.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import openpi_so101  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-view", type=Path, required=True)
    parser.add_argument("--temp-root", type=Path, required=True)
    parser.add_argument("--state-mode", choices=("off", "on"), default="off")
    args = parser.parse_args()
    view = args.train_view.expanduser().resolve()
    temp = args.temp_root.expanduser().resolve()
    os.environ["SO101_TRAIN_VIEW"] = str(view)
    config = openpi_so101.make_train_config(
        mode="serve", run_root=temp, train_view=view,
        base_checkpoint=temp, exp_name="pipeline_check", state_mode=args.state_mode,
    )
    loader = openpi_so101.create_so101_data_loader(config, sharding=None, shuffle=False)
    observation, actions = next(iter(loader))
    actions = np.asarray(actions)
    state = np.asarray(observation.state)
    images = {key: np.asarray(value) for key, value in observation.images.items()}
    masks = {key: np.asarray(value) for key, value in observation.image_masks.items()}
    report = {
        "state_shape": list(state.shape),
        "action_shape": list(actions.shape),
        "images": {key: [list(value.shape), str(value.dtype), float(value.min()), float(value.max())]
                   for key, value in images.items()},
        "masks": {key: bool(value.all()) for key, value in masks.items()},
        "tokenized_prompt_shape": list(np.asarray(observation.tokenized_prompt).shape),
        "state_mode": args.state_mode,
        "finite_state": bool(np.isfinite(state).all()),
        "finite_actions": bool(np.isfinite(actions).all()),
        "action_padding_zero": bool(np.allclose(actions[..., 12:], 0.0)),
        "state_padding_zero": bool(np.allclose(state[..., 12:], 0.0)),
        "state_first_12_range": [float(state[..., :12].min()), float(state[..., :12].max())],
        "action_first_12_range": [float(actions[..., :12].min()), float(actions[..., :12].max())],
    }
    assert state.shape == (1, 32), state.shape
    assert actions.shape == (1, 10, 32), actions.shape
    assert set(images) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert all(value.shape == (1, 224, 224, 3) for value in images.values())
    assert all(value.all() for value in masks.values())
    assert report["finite_state"] and report["finite_actions"]
    assert report["action_padding_zero"] and report["state_padding_zero"]
    print(__import__("json").dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
