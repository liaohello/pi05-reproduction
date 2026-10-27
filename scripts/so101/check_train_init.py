#!/usr/bin/env python3
"""E2 preflight: load pi05_libero into the LoRA train state and count params."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import flax.traverse_util as traverse_util
import jax
import numpy as np
import os

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import openpi_so101  # noqa: E402
import openpi.training.sharding as _sharding  # noqa: E402


def _load_train_module():
    for path in (Path("/opt/openpi/scripts/train.py"),):
        if path.is_file():
            spec = importlib.util.spec_from_file_location("openpi_train_entry", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError("pinned OpenPI scripts/train.py not found")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-view", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--temp-root", type=Path, required=True)
    parser.add_argument("--state-mode", choices=("off", "on"), default="off")
    parser.add_argument("--train-episodes", default="")
    parser.add_argument("--save-checkpoint", type=Path, default=None,
                        help="save the initialized TrainState at step 0 to this checkpoint root")
    args = parser.parse_args()
    train = _load_train_module()
    if args.train_episodes.strip():
        os.environ["SO101_TRAIN_EPISODES"] = args.train_episodes.strip()
    config = openpi_so101.make_train_config(
        mode="serve", run_root=args.temp_root, train_view=args.train_view,
        base_checkpoint=args.base_checkpoint, exp_name="init_check",
        state_mode=args.state_mode,
    )
    mesh = _sharding.make_mesh(1)
    state, _ = train.init_train_state(config, jax.random.key(0), mesh, resume=False)
    jax.block_until_ready(state)
    flat = traverse_util.flatten_dict(state.params.to_pure_dict(), sep="/")
    import hashlib
    digest = hashlib.sha256()
    for key in sorted(flat):
        value = flat[key]
        digest.update(key.encode()); digest.update(b"\0")
        digest.update(str(value.dtype).encode()); digest.update(b"\0")
        digest.update(str(value.shape).encode()); digest.update(b"\0")
        digest.update(np.asarray(value).tobytes())
    trainable = state.params.filter(config.trainable_filter)
    trainable_flat = traverse_util.flatten_dict(trainable.to_pure_dict(), sep="/")
    report = {
        "total_tensors": len(flat),
        "total_parameters": int(sum(value.size for value in flat.values())),
        "trainable_tensors": len(trainable_flat),
        "trainable_parameters": int(sum(value.size for value in trainable_flat.values())),
        "trainable_top_level": sorted({key.split("/")[0] for key in trainable_flat}),
        "trainable_lora_tensors": sum("lora" in key for key in trainable_flat),
        "trainable_examples": sorted(trainable_flat)[:12],
        "train_state_step": int(state.step),
        "parameters_sha256": digest.hexdigest(),
        "state_mode": args.state_mode,
        "train_episodes": args.train_episodes,
        "freeze_filter": repr(config.freeze_filter),
        "dtype_examples": {key: str(value.dtype) for key, value in list(flat.items())[:5]},
    }
    if args.save_checkpoint is not None:
        if args.train_episodes.strip():
            os.environ["SO101_TRAIN_EPISODES"] = args.train_episodes.strip()
        os.environ["SO101_TRAIN_VIEW"] = str(args.train_view.expanduser().resolve())
        loader = openpi_so101.create_so101_data_loader(config, sharding=None, shuffle=False)
        root = args.save_checkpoint.expanduser().resolve()
        manager, _ = train._checkpoints.initialize_checkpoint_dir(
            root, keep_period=1, overwrite=True, resume=False)
        train._checkpoints.save_state(manager, state, loader, 0)
        manager.wait_until_finished()
        manager.close()
        report["saved_initial_checkpoint"] = str(root / "0")
    print(json.dumps(report, indent=2))
    assert report["train_state_step"] == 0
    assert report["trainable_parameters"] > 0
    assert report["trainable_parameters"] < report["total_parameters"]
    return 0


if __name__ == "__main__":
    sys.exit(main())
