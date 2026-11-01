#!/usr/bin/env python3
"""Strictly compare the real initialized TrainState with the pi05 base tree."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

import jax
import numpy as np
from flax.traverse_util import flatten_dict

sys.path.insert(0, "/work/scripts/so101")
import openpi_so101  # noqa: E402
import openpi.models.model as model_lib  # noqa: E402
import openpi.training.sharding as sharding  # noqa: E402

LORA = re.compile(r".*lora.*")


def load_train_module():
    spec = importlib.util.spec_from_file_location(
        "openpi_train_entry", Path("/opt/openpi/scripts/train.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-view", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base-checkpoint", default="/base")
    parser.add_argument("--fsdp-devices", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train = load_train_module()
    config = openpi_so101.make_train_config(
        mode="serve", run_root=out, train_view=Path(args.train_view),
        base_checkpoint=Path(args.base_checkpoint), exp_name="base_load_audit",
        state_mode="on", batch_size=args.batch_size, fsdp_devices=args.fsdp_devices,
        seed=args.seed, paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m", freeze_vision_encoder=False,
        action_horizon=10, dtype="bfloat16")
    mesh = sharding.make_mesh(args.fsdp_devices)
    init_rng = jax.random.split(jax.random.key(args.seed))[1]
    state, _ = train.init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(state)
    target_tree = state.params.to_pure_dict()
    target = flatten_dict(target_tree, sep="/")
    loaded_tree = config.weight_loader.load(target_tree)
    loaded = flatten_dict(loaded_tree, sep="/")
    source = flatten_dict(model_lib.restore_params(
        str(Path(args.base_checkpoint) / "params"), restore_type=np.ndarray), sep="/")
    trainable = flatten_dict(
        state.params.filter(config.trainable_filter).to_pure_dict(), sep="/")
    problems = []
    if set(target) != set(loaded):
        problems.append("CheckpointWeightLoader changed the parameter path set")
    if sorted(set(source) - set(target)):
        problems.append("base checkpoint has paths absent from the TrainState")
    target_without_source = [name for name in set(target) - set(source)
                             if not LORA.fullmatch(name)]
    if target_without_source:
        problems.append("non-LoRA TrainState parameters are absent from the base checkpoint")
    records = []
    for name in sorted(target):
        train_array = np.asarray(jax.device_get(target[name]))
        loaded_array = np.asarray(loaded[name])
        base_array = source.get(name)
        if train_array.shape != loaded_array.shape:
            problems.append(f"{name}: loader shape mismatch")
            continue
        if not np.isfinite(train_array).all() or not np.isfinite(loaded_array).all():
            problems.append(f"{name}: non-finite parameter")
            continue
        if base_array is None:
            if not LORA.fullmatch(name):
                continue
            exact = bool(np.array_equal(train_array, loaded_array))
            source_dtype = None
            cast_exact = None
            expected_loaded = train_array
        else:
            if base_array.shape != train_array.shape:
                problems.append(f"{name}: base shape mismatch")
                continue
            expected_loaded = base_array.astype(train_array.dtype)
            exact = bool(np.array_equal(expected_loaded, loaded_array))
            cast_exact = bool(np.array_equal(expected_loaded, train_array))
            source_dtype = str(base_array.dtype)
            if not exact:
                problems.append(f"{name}: loader did not preserve the cast base value")
            if not cast_exact:
                problems.append(f"{name}: base cast differs from TrainState")
        records.append({
            "path": name, "shape": list(train_array.shape),
            "source_dtype": source_dtype, "target_dtype": str(train_array.dtype),
            "loaded_dtype": str(loaded_array.dtype),
            "trainable": name in trainable, "lora": bool(LORA.fullmatch(name)),
            "loader_exact": exact, "source_cast_exact": cast_exact,
        })
    payload = {
        "schema_version": 1,
        "fsdp_devices": args.fsdp_devices,
        "global_batch_size": args.batch_size,
        "seed": args.seed,
        "target_tensors": len(target),
        "source_tensors": len(source),
        "loaded_tensors": len(loaded),
        "trainable_tensors": len(trainable),
        "records": records,
        "problems": problems,
        "passed": not problems,
    }
    output = out / "base_load_audit.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"},
                     indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
