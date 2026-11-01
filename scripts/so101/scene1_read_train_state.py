#!/usr/bin/env python3
"""Read a Scene1 checkpoint's authoritative TrainState.step and full-state digest."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import scene1_openpi_so101 as sx  # noqa: E402


def digest_tree(values: dict) -> str:
    def visit(tree: dict, prefix: str = ""):
        for key in sorted(tree):
            value = tree[key]
            path = f"{prefix}/{key}" if prefix else str(key)
            if isinstance(value, dict):
                yield from visit(value, path)
            else:
                yield path, value

    digest = hashlib.sha256()
    for path, value in visit(values):
        import jax
        import numpy as np

        array = np.asarray(jax.device_get(value))
        digest.update(path.encode())
        digest.update(b"\0")
        digest.update(str(array.dtype).encode())
        digest.update(b"\0")
        digest.update(str(array.shape).encode())
        digest.update(b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--config-initial", type=Path, required=True)
    parser.add_argument("--train-view", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--fsdp-devices", type=int, choices=(2, 4), required=True)
    parser.add_argument("--expected-step", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    from flax.traverse_util import flatten_dict
    from openpi.training import checkpoints
    from openpi.training import sharding

    saved = json.loads(args.config_initial.read_text(encoding="utf-8"))
    model_scope = saved["model_scope"]
    config = sx.make_train_config(
        mode="probe",
        run_root=args.checkpoint_root.parent,
        train_view=args.train_view,
        base_checkpoint=args.base_checkpoint,
        exp_name=saved["exp_name"],
        state_mode=saved["state_mode"],
        steps_override=saved["steps"],
        save_interval=saved["save_interval"],
        batch_size=saved["batch_size"],
        lr_warmup_steps=saved["lr_schedule"]["warmup_steps"],
        peak_lr=saved["lr_schedule"]["peak_lr"],
        decay_steps=saved["lr_schedule"]["decay_steps"],
        decay_lr=saved["lr_schedule"]["decay_lr"],
        fsdp_devices=args.fsdp_devices,
        **model_scope,
        action_horizon=saved["action_horizon"],
        seed=saved["seed"],
        gradient_clip_norm=saved["gradient_clip_norm"],
        optimizer_name=saved["optimizer"],
        ema_decay=saved["ema_decay"],
        dtype=saved["dtype"],
    )
    spec = importlib.util.spec_from_file_location(
        "openpi_train_entry", "/opt/openpi/scripts/train.py")
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    mesh = sharding.make_mesh(args.fsdp_devices)
    template, _ = train.init_train_state(
        config, jax.random.key(0), mesh, resume=True)
    manager, _ = checkpoints.initialize_checkpoint_dir(
        args.checkpoint_root, keep_period=1, overwrite=False, resume=True)
    state = checkpoints.restore_state(manager, template, None)
    state = jax.block_until_ready(state)
    step = int(jax.device_get(state.step))
    if args.expected_step is not None and step != args.expected_step:
        raise ValueError(f"TrainState.step {step} differs from {args.expected_step}")
    params = state.params.to_pure_dict()
    ema = state.ema_params.to_pure_dict() if state.ema_params is not None else None
    optimizer_leaves = jax.tree_util.tree_leaves(state.opt_state)
    optimizer_paths = jax.tree_util.tree_leaves_with_path(state.opt_state)
    payload = {
        "schema_version": 1,
        "checkpoint_root": str(args.checkpoint_root),
        "orbax_steps": [int(value) for value in manager.all_steps()],
        "train_state_step": step,
        "ema_present": ema is not None,
        "ema_decay": float(state.ema_decay) if state.ema_decay is not None else None,
        "parameters_sha256": digest_tree(
            flatten_dict(params, sep="/")),
        "ema_sha256": digest_tree(
            flatten_dict(ema, sep="/")) if ema is not None else None,
        "optimizer_sha256": digest_tree({
            jax.tree_util.keystr(path): value
            for path, value in optimizer_paths
        }),
        "optimizer_tensors": len(optimizer_leaves),
        "optimizer_parameter_count": int(sum(
            jnp.asarray(value).size for value in optimizer_leaves)),
        "optimizer_dtypes": sorted({
            str(jnp.asarray(value).dtype) for value in optimizer_leaves}),
        "state_mode": config.model.discrete_state_input,
        "action_horizon": config.model.action_horizon,
        "freeze_filter": repr(config.freeze_filter),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    manager.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
