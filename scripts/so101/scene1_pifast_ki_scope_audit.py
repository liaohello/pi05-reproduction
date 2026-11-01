#!/usr/bin/env python3
"""CPU audit of the actual Scene1 joint TrainState and optimizer scope."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import flax.traverse_util as traverse
import jax
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import scene1_pifast_ki_runtime as runtime  # noqa: E402
import scene1_train_so101 as launcher  # noqa: E402
from openpi.models import model as model_lib  # noqa: E402


def dtype_summary(values) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for value in values:
        dtype = str(value.dtype)
        entry = result.setdefault(dtype, {"arrays": 0, "elements": 0})
        entry["arrays"] += 1
        entry["elements"] += int(value.size)
    return result


def group_for(path: str) -> str:
    if "fast_token_residual" in path:
        return "fast_adapter"
    if "lora" in path:
        return "action_expert_lora" if "_1/" in path else "language_lora"
    if "PaliGemma/img" in path:
        return "vision"
    if any(key in path for key in (
            "action_in_proj", "action_out_proj",
            "time_mlp_in", "time_mlp_out")):
        return "action_time_projections"
    if "_1/" in path:
        return "action_expert_base"
    if "PaliGemma/llm" in path:
        return "language_base"
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-base-tree-sha256", required=True)
    parser.add_argument("--train-view", type=Path, required=True)
    parser.add_argument("--fast-tokenizer-dir", type=Path, required=True)
    parser.add_argument("--paligemma-tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "cpu":
        raise RuntimeError(f"expected CPU backend, got {jax.default_backend()}")
    base_tree_sha256 = launcher.clean60.sha256_tree(args.base_checkpoint)
    if base_tree_sha256 != args.expected_base_tree_sha256:
        raise ValueError(
            f"pi05_base tree hash {base_tree_sha256} differs from the plan")

    config = runtime.make_train_config(
        mode="smoke",
        run_root=args.output.parent,
        train_view=args.train_view,
        base_checkpoint=args.base_checkpoint,
        exp_name="scope-audit",
        steps=1,
        batch_size=1,
        save_interval=1,
        warmup_steps=1000,
        peak_lr=2.5e-5,
        decay_steps=40000,
        decay_lr=2.5e-6,
        seed=42,
        fsdp_devices=1,
        num_workers=0,
        fast_ce_weight=1.0,
        fast_tokenizer_dir=args.fast_tokenizer_dir,
        paligemma_tokenizer_path=args.paligemma_tokenizer,
        dataset_label="scene1_single_random_v1",
    )
    train_module, train_path = launcher._load_train_module()
    from openpi.training import sharding

    mesh = sharding.make_mesh(1)
    state, _ = train_module.init_train_state(
        config,
        jax.random.key(config.seed),
        mesh,
        resume=False,
    )
    jax.block_until_ready(state.step)

    params_state = traverse.flatten_dict(
        state.params.to_pure_dict(), sep="/")
    trainable_state = traverse.flatten_dict(
        state.params.filter(config.trainable_filter).to_pure_dict(),
        sep="/",
    )
    trainable_paths = set(trainable_state)
    total = sum(int(value.size) for value in params_state.values())
    trainable = sum(int(value.size) for value in trainable_state.values())

    groups: dict[str, dict] = {}
    for path, value in params_state.items():
        group = groups.setdefault(
            group_for(path),
            {"arrays": 0, "elements": 0, "trainable": 0, "paths": []},
        )
        group["arrays"] += 1
        group["elements"] += int(value.size)
        if path in trainable_paths:
            group["trainable"] += int(value.size)
        group["paths"].append(path)

    expected_trainable_groups = {
        "vision", "language_lora", "action_expert_lora",
        "action_time_projections", "fast_adapter",
    }
    unexpected = {
        name for name, group in groups.items()
        if group["trainable"] and name not in expected_trainable_groups
    }
    if unexpected:
        raise RuntimeError(f"unexpected trainable groups: {sorted(unexpected)}")
    for name in ("language_base", "action_expert_base"):
        if groups.get(name, {}).get("trainable", 0) != 0:
            raise RuntimeError(f"{name} is not frozen")

    fast_paths = [
        path for path in params_state if path.startswith("fast_token_residual/")
    ]
    if fast_paths != ["fast_token_residual/table"]:
        raise RuntimeError(f"unexpected FAST adapter paths: {fast_paths}")
    fast_value = params_state[fast_paths[0]]
    if fast_value.shape != (2048, 2048) or np.any(np.asarray(fast_value) != 0):
        raise RuntimeError("FAST residual is not zero-initialized [2048,2048]")

    expert_lora = {
        path: list(value.shape)
        for path, value in params_state.items()
        if "_1/" in path and "lora" in path
    }
    language_lora = {
        path: list(value.shape)
        for path, value in params_state.items()
        if "_1/" not in path and "lora" in path
    }
    if len(expert_lora) != 10 or any(64 not in shape
                                     for shape in expert_lora.values()):
        raise RuntimeError("Action Expert LoRA did not resolve to rank64")
    if len(language_lora) != 10 or any(16 not in shape
                                       for shape in language_lora.values()):
        raise RuntimeError("language LoRA did not resolve to rank16")

    projection_paths = {
        path: path in trainable_paths
        for path in params_state
        if any(key in path for key in (
            "action_in_proj", "action_out_proj",
            "time_mlp_in", "time_mlp_out"))
    }
    if not projection_paths or not all(projection_paths.values()):
        raise RuntimeError("a projection kernel or bias is frozen")
    if not any(path.endswith("/kernel") for path in projection_paths):
        raise RuntimeError("projection kernels are absent")
    if not any(path.endswith("/bias") for path in projection_paths):
        raise RuntimeError("projection biases are absent")

    optimizer_leaves = jax.tree_util.tree_leaves(state.opt_state)
    loaded = traverse.flatten_dict(
        model_lib.restore_params(
            str(args.base_checkpoint / "params"),
            restore_type=np.ndarray,
        ),
        sep="/",
    )
    loaded_mismatches = []
    for path, source_value in loaded.items():
        if path not in params_state:
            loaded_mismatches.append(f"unexpected:{path}")
            continue
        target = np.asarray(params_state[path])
        if tuple(source_value.shape) != tuple(target.shape):
            loaded_mismatches.append(f"shape:{path}")
            continue
        cast = source_value.astype(target.dtype)
        if not np.array_equal(cast, target):
            loaded_mismatches.append(f"value:{path}")
    missing_inherited = sorted(
        set(params_state) - set(loaded)
        - {path for path in params_state if "lora" in path}
        - {"fast_token_residual/table"}
    )
    if loaded_mismatches or missing_inherited:
        raise RuntimeError(
            "base checkpoint load failed: "
            f"mismatches={loaded_mismatches[:8]}, missing={missing_inherited[:8]}")
    payload = {
        "schema_version": 1,
        "status": "pass",
        "backend": jax.default_backend(),
        "claim": (
            "Actual CPU TrainState initialization from pi05_base with the v2 "
            "rank64 joint model; no update or checkpoint."),
        "source": {
            "train_entry": str(train_path),
            "train_entry_sha256": launcher._sha_file(train_path),
            "runtime_module_sha256": launcher._sha_file(
                HERE / "scene1_pifast_ki_runtime.py"),
            "model_module_sha256": launcher._sha_file(
                HERE / "scene1_pifast_ki_model.py"),
        },
        "base_checkpoint_tree_sha256": base_tree_sha256,
        "base_checkpoint_load": {
            "loaded_paths": len(loaded),
            "loaded_elements": sum(
                int(value.size) for value in loaded.values()),
            "source_parameter_dtypes": dtype_summary(loaded.values()),
            "target_parameter_dtypes": {
                dtype: {
                    "arrays": sum(
                        str(np.asarray(params_state[path]).dtype) == dtype
                        for path in loaded),
                    "elements": sum(
                        int(np.asarray(params_state[path]).size)
                        for path in loaded
                        if str(np.asarray(params_state[path]).dtype) == dtype),
                }
                for dtype in sorted({
                    str(np.asarray(params_state[path]).dtype)
                    for path in loaded})
            },
            "all_paths_shapes_values_match_after_required_cast": True,
            "missing_inherited_parameters": 0,
            "unexpected_source_parameters": 0,
        },
        "action_horizon": config.model.action_horizon,
        "internal_action_dim": config.model.action_dim,
        "max_token_len": config.model.max_token_len,
        "parameter_paths": len(params_state),
        "trainable_arrays": len(trainable_state),
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "parameter_dtypes": dtype_summary(params_state.values()),
        "trainable_parameter_dtypes": dtype_summary(trainable_state.values()),
        "groups": groups,
        "language_lora_paths": language_lora,
        "action_expert_lora_paths": expert_lora,
        "projection_paths": projection_paths,
        "fast_adapter": {
            "path": fast_paths[0],
            "shape": list(fast_value.shape),
            "dtype": str(fast_value.dtype),
            "trainable": fast_paths[0] in trainable_paths,
            "zero_initialized": True,
        },
        "optimizer_state": {
            "tensors": len(optimizer_leaves),
            "elements": sum(int(value.size) for value in optimizer_leaves),
            "dtype_summary": dtype_summary(optimizer_leaves),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "pass",
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "expert_lora_paths": len(expert_lora),
        "optimizer_tensors": len(optimizer_leaves),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
