#!/usr/bin/env python3
"""Read the authoritative TrainState.step and optimizer state from an E2 checkpoint."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import jax
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import openpi_so101  # noqa: E402
import openpi.training.checkpoints as _checkpoints  # noqa: E402
import openpi.training.sharding as _sharding  # noqa: E402


def _load_train_module():
    path = Path("/opt/openpi/scripts/train.py")
    spec = importlib.util.spec_from_file_location("openpi_train_entry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--step", type=int, default=None, help="Orbax directory index; default latest")
    parser.add_argument("--train-view", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--state-mode", choices=("off", "on"), default="off")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.checkpoint_root.expanduser().resolve()
    train = _load_train_module()
    config = openpi_so101.make_train_config(
        mode="serve", run_root=root.parent, train_view=args.train_view,
        base_checkpoint=args.base_checkpoint, exp_name="state_probe",
        state_mode=args.state_mode,
    )
    mesh = _sharding.make_mesh(1)
    template, _ = train.init_train_state(config, jax.random.key(0), mesh, resume=True)
    manager, _ = _checkpoints.initialize_checkpoint_dir(
        root, keep_period=1, overwrite=False, resume=True)
    steps = tuple(int(value) for value in manager.all_steps())
    step = int(args.step) if args.step is not None else max(steps)
    state = _checkpoints.restore_state(manager, template, None, step=step)
    step_value = int(jax.device_get(state.step))
    params = state.params.to_pure_dict()
    flat = __import__("flax.traverse_util", fromlist=["flatten_dict"]).flatten_dict(params, sep="/")
    import hashlib
    import re
    def _trainable(name: str) -> bool:
        # mirrors openpi_so101.make_freeze_filter: the LLM and SigLIP image tower are frozen
        # unless the tensor is a LoRA adapter; every other parameter (the projections) trains.
        return not ((re.search(r"llm", name) or re.search(r"PaliGemma/img", name))
                    and not re.search(r"lora", name))
    trainable_names = [key for key in sorted(flat) if _trainable(key)]
    digest = hashlib.sha256()
    for key in sorted(flat):
        value = flat[key]
        digest.update(key.encode()); digest.update(b"\0")
        digest.update(str(value.dtype).encode()); digest.update(b"\0")
        digest.update(str(value.shape).encode()); digest.update(b"\0")
        digest.update(np.asarray(value).tobytes())
    opt_leaves = jax.tree_util.tree_leaves(state.opt_state)
    report = {
        "checkpoint_root": str(root),
        "orbax_step_directory": step,
        "train_state_step": step_value,
        "directory_to_train_state_offset": step_value - step,
        "orbax_steps": list(steps),
        "ema_params_present": state.ema_params is not None,
        "optimizer_leaves": len(opt_leaves),
        "optimizer_elements": int(sum(np.asarray(leaf).size for leaf in opt_leaves)),
        "optimizer_dtypes": sorted({str(np.asarray(leaf).dtype) for leaf in opt_leaves}),
        "param_tensors": int(sum(1 for _ in jax.tree_util.tree_leaves(params))),
        "trainable_parameter_tensors": len(trainable_names),
        "trainable_parameter_count": int(sum(np.asarray(flat[key]).size for key in trainable_names)),
        "trainable_parameter_names": trainable_names,
        "trainable_rule": "not ((path contains 'llm' or 'PaliGemma/img') and path does not contain 'lora')",
        "parameters_sha256": digest.hexdigest(),
        "state_mode": args.state_mode,
        "note": "TrainState.step is authoritative; the Orbax directory name is a loop index",
    }
    manager.close()
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    # openpi saves after applying the update, so Orbax directory index d carries TrainState.step d+1;
    # measured on the batch-8 probe (dir 0 -> step 1) and on the E2 run (dir 99 -> step 100).
    expected = step + 1
    if step_value != expected:
        print(f"unexpected directory/step relation: dir {step}, TrainState.step {step_value}, "
              f"expected {expected}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
