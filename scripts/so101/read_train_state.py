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
from model_scope import resolve_model_scope  # noqa: E402
import openpi.training.checkpoints as _checkpoints  # noqa: E402
import openpi.training.sharding as _sharding  # noqa: E402


def _load_train_module():
    path = Path("/opt/openpi/scripts/train.py")
    spec = importlib.util.spec_from_file_location("openpi_train_entry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trainable_summary(params, trainable_filter) -> dict:
    """Use the trainer's selected parameter tree, including its exact freeze scope."""
    from flax.traverse_util import flatten_dict

    flat = flatten_dict(params.filter(trainable_filter).to_pure_dict(), sep="/")
    return {
        "trainable_parameter_tensors": len(flat),
        "trainable_parameter_count": int(sum(value.size for value in flat.values())),
        "trainable_parameter_names": sorted(flat),
        "trainable_rule": "config.trainable_filter",
    }


def _checkpoint_settings(root: Path, run_config: Path | None, state_mode: str | None) -> dict:
    # Checkpoint roots are <run>/checkpoints/<config>/<experiment>.
    config_path = run_config or root.parent.parent.parent / "config_initial.json"
    document = (json.loads(config_path.read_text(encoding="utf-8"))
                if run_config is not None or config_path.is_file() else {})
    saved_mode = document.get("state_mode")
    if state_mode is not None and saved_mode is not None and state_mode != saved_mode:
        raise ValueError("explicit state_mode differs from the saved training config")
    return {"model_scope": resolve_model_scope(document),
            "state_mode": saved_mode or state_mode or "off"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--step", type=int, default=None, help="Orbax directory index; default latest")
    parser.add_argument("--train-view", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--state-mode", choices=("off", "on"), default=None)
    parser.add_argument("--fsdp-devices", type=int, default=1)
    parser.add_argument("--run-config", type=Path, default=None,
                        help="saved training config; defaults to the run's config_initial.json")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.checkpoint_root.expanduser().resolve()
    settings = _checkpoint_settings(root, args.run_config, args.state_mode)
    train = _load_train_module()
    config = openpi_so101.make_train_config(
        mode="serve", run_root=root.parent, train_view=args.train_view,
        base_checkpoint=args.base_checkpoint, exp_name="state_probe",
        state_mode=settings["state_mode"], **settings["model_scope"],
    )
    mesh = _sharding.make_mesh(args.fsdp_devices)
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
    trainable_summary = _trainable_summary(state.params, config.trainable_filter)
    digest = hashlib.sha256()
    for key in sorted(flat):
        value = flat[key]
        digest.update(key.encode()); digest.update(b"\0")
        digest.update(str(value.dtype).encode()); digest.update(b"\0")
        digest.update(str(value.shape).encode()); digest.update(b"\0")
        digest.update(np.asarray(value).tobytes())
    opt_leaves = jax.tree_util.tree_leaves(state.opt_state)
    optimizer_digest = hashlib.sha256()
    for path, value in sorted(jax.tree_util.tree_leaves_with_path(state.opt_state),
                              key=lambda item: jax.tree_util.keystr(item[0])):
        array = np.asarray(value)
        optimizer_digest.update(jax.tree_util.keystr(path).encode())
        optimizer_digest.update(b"\0")
        optimizer_digest.update(str(array.dtype).encode())
        optimizer_digest.update(b"\0")
        optimizer_digest.update(str(array.shape).encode())
        optimizer_digest.update(b"\0")
        optimizer_digest.update(array.tobytes())
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
        **trainable_summary,
        "parameters_sha256": digest.hexdigest(),
        "optimizer_state_sha256": optimizer_digest.hexdigest(),
        "optimizer_state_hash_scope": (
            "sorted JAX paths plus dtype, shape and bytes over the complete optimizer state"),
        "fsdp_devices": args.fsdp_devices,
        "state_mode": settings["state_mode"],
        "model_scope": settings["model_scope"],
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
