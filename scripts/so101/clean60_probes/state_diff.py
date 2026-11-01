#!/usr/bin/env python3
"""Restore two complete TrainStates and quantify parameter/optimizer differences."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys

import jax
import numpy as np
from flax.traverse_util import flatten_dict

sys.path.insert(0, "/work/scripts/so101")
import openpi_so101  # noqa: E402
import openpi.training.checkpoints as checkpoints  # noqa: E402
import openpi.training.sharding as sharding  # noqa: E402
from model_scope import resolve_model_scope  # noqa: E402


def train_module():
    spec = importlib.util.spec_from_file_location(
        "openpi_train_entry", Path("/opt/openpi/scripts/train.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def group_of(path: str) -> str:
    if re.fullmatch(r".*llm.*_1.*", path):
        return "action_expert"
    if "/img/" in path:
        return "vision_encoder"
    if "lora" in path:
        return "language_lora"
    if "llm" in path:
        return "language_backbone"
    for name in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"):
        if path.startswith(name + "/"):
            return f"projection:{name}"
    return "other"


def restore(root: Path, config, template, train):
    manager, _ = checkpoints.initialize_checkpoint_dir(
        root, keep_period=1, overwrite=False, resume=True)
    steps = tuple(int(value) for value in manager.all_steps())
    if not steps:
        raise RuntimeError(f"no checkpoints in {root.name}")
    state = checkpoints.restore_state(manager, template, None, step=max(steps))
    jax.block_until_ready(state)
    params = flatten_dict(state.params.to_pure_dict(), sep="/")
    params = {name: np.asarray(jax.device_get(value)) for name, value in params.items()}
    opt = {}
    for path, value in jax.tree_util.tree_leaves_with_path(state.opt_state):
        key = jax.tree_util.keystr(path)
        opt[key] = np.asarray(jax.device_get(value))
    result = {
        "steps": list(steps), "step": int(jax.device_get(state.step)),
        "params": params, "opt": opt, "ema_present": state.ema_params is not None,
    }
    manager.close()
    return result


def compare(first: dict, second: dict) -> tuple[dict, dict, list[str]]:
    problems = []
    params = {}
    for group in sorted({group_of(name) for name in first["params"]}):
        names = sorted(name for name in first["params"] if group_of(name) == group)
        fields = []
        for name in names:
            if name not in second["params"]:
                problems.append(f"missing parameter {name}")
                continue
            a, b = first["params"][name], second["params"][name]
            if a.shape != b.shape or a.dtype != b.dtype:
                problems.append(f"parameter structure differs: {name}")
                continue
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                problems.append(f"non-finite parameter: {name}")
                continue
            delta = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
            fields.append((name, delta))
        params[group] = {
            "tensors": len(fields),
            "changed_tensors": sum(delta > 0 for _, delta in fields),
            "max_abs_delta": max((delta for _, delta in fields), default=0.0),
            "sample": max(fields, key=lambda item: item[1])[0] if fields else None,
        }
    opt_fields = []
    for name in sorted(first["opt"]):
        if name not in second["opt"]:
            problems.append(f"missing optimizer state {name}")
            continue
        a, b = first["opt"][name], second["opt"][name]
        if a.shape != b.shape or a.dtype != b.dtype:
            problems.append(f"optimizer structure differs: {name}")
            continue
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            problems.append(f"non-finite optimizer state: {name}")
            continue
        opt_fields.append(float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))))
    optimizer = {
        "leaves": len(opt_fields),
        "changed_leaves": sum(delta > 0 for delta in opt_fields),
        "max_abs_delta": max(opt_fields, default=0.0),
    }
    return params, optimizer, problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-root", type=Path, required=True)
    parser.add_argument("--second-root", type=Path, required=True)
    parser.add_argument("--first-config", type=Path, required=True)
    parser.add_argument("--second-config", type=Path, required=True)
    parser.add_argument("--train-view", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--fsdp-devices", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train = train_module()
    saved = json.loads(args.first_config.read_text(encoding="utf-8"))
    scope = resolve_model_scope(saved)
    config = openpi_so101.make_train_config(
        mode="serve", run_root=args.output.parent, train_view=args.train_view,
        base_checkpoint=args.base_checkpoint, exp_name="state_diff",
        state_mode=saved.get("state_mode", "on"), fsdp_devices=args.fsdp_devices,
        action_horizon=int(saved.get("action_horizon", 10)), seed=int(saved.get("seed", 42)),
        **scope)
    mesh = sharding.make_mesh(args.fsdp_devices)
    template, _ = train.init_train_state(
        config, jax.random.key(int(saved.get("seed", 42))), mesh, resume=True)
    first = restore(args.first_root, config, template, train)
    second = restore(args.second_root, config, template, train)
    params, optimizer, problems = compare(first, second)
    payload = {
        "schema_version": 1,
        "first": {"root": args.first_root.name, "step": first["step"],
                  "checkpoint_steps": first["steps"], "ema_present": first["ema_present"]},
        "second": {"root": args.second_root.name, "step": second["step"],
                   "checkpoint_steps": second["steps"], "ema_present": second["ema_present"]},
        "parameter_groups": params,
        "optimizer": optimizer,
        "problems": problems,
        "bitwise_equal": not problems and all(
            row["changed_tensors"] == 0 for row in params.values())
        and optimizer["changed_leaves"] == 0,
        "passed_structure_and_finite": not problems,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed_structure_and_finite"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
