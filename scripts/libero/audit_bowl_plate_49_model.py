#!/usr/bin/env python3
"""Audit pi05_base loading, LoRA scope, dtypes, and optimizer state."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bowl_plate_49  # noqa: E402

LORA_RE = re.compile(r".*lora.*")
LLM_RE = re.compile(r".*llm.*")
EXPERT_RE = re.compile(r".*llm.*_1.*")
PROJECTION_NAMES = (
    "action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out",
    "state_proj",
)


def group(path: str) -> str:
    is_lora = bool(LORA_RE.fullmatch(path))
    is_llm = bool(LLM_RE.fullmatch(path))
    is_expert = bool(EXPERT_RE.fullmatch(path))
    if is_lora and is_expert:
        return "action_expert_lora"
    if is_lora and is_llm:
        return "language_lora"
    if is_expert:
        return "action_expert_backbone"
    if is_llm:
        return "language_backbone"
    if "/img/" in path or "/img." in path:
        return "vision_encoder"
    if any(name in path for name in PROJECTION_NAMES):
        return "projection"
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import jax
    import numpy as np
    from flax.traverse_util import flatten_dict
    from openpi.models import gemma
    from openpi.models import model as model_lib
    from openpi.training import sharding

    protocol = bowl_plate_49.load_protocol(
        args.protocol, args.expected_protocol_sha256)
    view = bowl_plate_49.load_view(args.view, protocol)
    base_tree = bowl_plate_49.sha256_tree(args.base_checkpoint)
    if base_tree["sha256"] != protocol["base"]["tree_sha256"]:
        raise ValueError("base tree differs from registered pi05_base")
    config_args = SimpleNamespace(
        exp_name="libero-bowl-plate-model-audit",
        mode="probe",
        view=str(args.view),
        base_checkpoint=str(args.base_checkpoint),
        batch_size=32,
        fsdp_devices=1,
    )
    config = bowl_plate_49.make_config(
        config_args, protocol, view, args.out.parent / "audit_checkpoints", 1)
    train, _ = bowl_plate_49.load_train_module()
    mesh = sharding.make_mesh(1)
    rng = jax.random.split(jax.random.key(config.seed))[1]
    state, state_sharding = train.init_train_state(
        config, rng, mesh, resume=False)
    state = jax.block_until_ready(state)

    def unwrap(value):
        return getattr(value, "value", value)

    target_tree = state.params.to_pure_dict()
    target = flatten_dict(target_tree, sep="/")
    loaded_tree = config.weight_loader.load(target_tree)
    loaded = flatten_dict(loaded_tree, sep="/")
    source = flatten_dict(
        model_lib.restore_params(
            str(args.base_checkpoint / "params"), restore_type=np.ndarray),
        sep="/",
    )
    trainable = flatten_dict(
        state.params.filter(config.trainable_filter).to_pure_dict(), sep="/")
    problems = []
    records = []
    if set(loaded) != set(target):
        problems.append("weight loader changed the target parameter path set")
    if set(source) - set(target):
        problems.append(
            "base checkpoint contains parameters absent from the target model")
    missing_base = [
        path for path in sorted(set(target) - set(source))
        if not LORA_RE.fullmatch(path)
    ]
    if missing_base:
        problems.append(
            "non-LoRA target parameters absent from pi05_base: "
            + ",".join(missing_base)
        )
    for path in sorted(target):
        target_value = np.asarray(jax.device_get(unwrap(target[path])))
        loaded_value = unwrap(loaded[path])
        loaded_is_abstract = not isinstance(loaded_value, np.ndarray)
        record = {
            "path": path,
            "group": group(path),
            "target_shape": list(target_value.shape),
            "target_dtype": str(target_value.dtype),
            "trainable": path in trainable,
            "lora": bool(LORA_RE.fullmatch(path)),
            "finite": bool(np.isfinite(target_value).all()),
            "loader_is_abstract_missing_value": loaded_is_abstract,
        }
        if record["loader_is_abstract_missing_value"]:
            record["loader_shape_exact"] = (
                tuple(loaded_value.shape) == tuple(target_value.shape)
            )
            record["loader_exact"] = None
        else:
            loaded_array = np.asarray(loaded_value)
            record["loader_shape_exact"] = loaded_array.shape == target_value.shape
            record["loader_exact"] = bool(
                np.array_equal(loaded_array.astype(target_value.dtype), target_value)
            )
        if not record["loader_shape_exact"]:
            problems.append(f"{path}: loader shape mismatch")
        if not record["finite"]:
            problems.append(f"{path}: non-finite initialized value")
        base_value = source.get(path)
        if base_value is None:
            record["source_shape"] = None
            record["source_exact_after_cast"] = None
        else:
            expected = np.asarray(base_value).astype(target_value.dtype)
            record["source_shape"] = list(base_value.shape)
            record["source_exact_after_cast"] = bool(
                np.array_equal(expected, target_value))
            if base_value.shape != target_value.shape:
                problems.append(f"{path}: base shape mismatch")
            if not record["source_exact_after_cast"]:
                problems.append(f"{path}: initialized value differs from pi05_base")
        records.append(record)

    groups = {}
    for record in records:
        entry = groups.setdefault(record["group"], {
            "tensors": 0,
            "parameters": 0,
            "trainable_tensors": 0,
            "trainable_parameters": 0,
            "dtypes": set(),
        })
        parameters = int(np.prod(record["target_shape"]))
        entry["tensors"] += 1
        entry["parameters"] += parameters
        entry["dtypes"].add(record["target_dtype"])
        if record["trainable"]:
            entry["trainable_tensors"] += 1
            entry["trainable_parameters"] += parameters
    for entry in groups.values():
        entry["dtypes"] = sorted(entry["dtypes"])

    scope_checks = {
        "language_lora_trainable": (
            groups.get("language_lora", {}).get("trainable_parameters", 0) > 0
        ),
        "action_expert_lora_trainable": (
            groups.get("action_expert_lora", {}).get("trainable_parameters", 0) > 0
        ),
        "language_non_lora_frozen": (
            groups.get("language_backbone", {}).get("trainable_parameters", -1) == 0
        ),
        "action_expert_non_lora_frozen": (
            groups.get("action_expert_backbone", {}).get(
                "trainable_parameters", -1) == 0
        ),
        "vision_encoder_trainable_per_reference_filter": (
            groups.get("vision_encoder", {}).get("trainable_parameters", 0)
            == groups.get("vision_encoder", {}).get("parameters", -1)
        ),
        "projection_trainable": (
            groups.get("projection", {}).get("trainable_parameters", 0)
            == groups.get("projection", {}).get("parameters", -1)
        ),
        "ema_disabled": state.ema_decay is None and state.ema_params is None,
    }
    for name, passed in scope_checks.items():
        if not passed:
            problems.append(f"scope check failed: {name}")

    optimizer_dtype_counts = {}
    optimizer_state_kinds = {}
    optimizer_parameters = 0
    for path, value in jax.tree_util.tree_leaves_with_path(state.opt_state):
        array = np.asarray(jax.device_get(value))
        dtype = str(array.dtype)
        optimizer_dtype_counts[dtype] = (
            optimizer_dtype_counts.get(dtype, 0) + int(array.size))
        text = jax.tree_util.keystr(path)
        kind = next(
            (name for name in ("mu", "nu", "count", "scale", "trace") if name in text),
            "other",
        )
        optimizer_state_kinds[kind] = (
            optimizer_state_kinds.get(kind, 0) + int(array.size))
        if kind in {"mu", "nu"}:
            optimizer_parameters += int(array.size)
    trainable_parameters = sum(
        int(np.prod(record["target_shape"]))
        for record in records if record["trainable"]
    )
    if optimizer_parameters != 2 * trainable_parameters:
        problems.append(
            "optimizer mu/nu arrays do not cover the trainable parameter tree")

    lora_runtime = {}
    for variant in (
        config.model.paligemma_variant,
        config.model.action_expert_variant,
    ):
        lora_configs = gemma.get_config(variant).lora_configs or {}
        lora_runtime[variant] = {
            name: {
                "rank": int(value.rank),
                "alpha": float(value.alpha),
                "axes": list(value.axes),
                "rslora": bool(value.rslora),
            }
            for name, value in lora_configs.items()
        }
    param_dtype_counts = {}
    for record in records:
        dtype = record["target_dtype"]
        param_dtype_counts[dtype] = (
            param_dtype_counts.get(dtype, 0)
            + int(np.prod(record["target_shape"]))
        )
    payload = {
        "schema_version": 1,
        "protocol_sha256": protocol["_sha256"],
        "view_sha256": view["view_sha256"],
        "base_checkpoint_tree_sha256": base_tree["sha256"],
        "model_config": {
            "pi05": bool(config.model.pi05),
            "paligemma_variant": config.model.paligemma_variant,
            "action_expert_variant": config.model.action_expert_variant,
            "action_horizon": int(config.model.action_horizon),
            "discrete_state_input": bool(config.model.discrete_state_input),
            "dtype": config.model.dtype,
        },
        "freeze_filter": repr(config.freeze_filter),
        "trainable_filter": repr(config.trainable_filter),
        "source_tensors": len(source),
        "target_tensors": len(target),
        "loaded_reference_tensors": len(loaded),
        "trainable_tensors": len(trainable),
        "total_parameters": sum(
            int(np.prod(record["target_shape"])) for record in records),
        "trainable_parameters": trainable_parameters,
        "groups": groups,
        "scope_checks": scope_checks,
        "lora_runtime": lora_runtime,
        "parameter_dtype_parameter_counts": param_dtype_counts,
        "optimizer_dtype_parameter_counts": optimizer_dtype_counts,
        "optimizer_state_kind_parameter_counts": optimizer_state_kinds,
        "optimizer_mu_nu_arrays_cover_parameters": (
            optimizer_parameters == 2 * trainable_parameters),
        "ema_decay": state.ema_decay,
        "ema_parameters": None if state.ema_params is None else "present",
        "records": records,
        "problems": problems,
        "passed": not problems,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.out),
        "sha256": bowl_plate_49.sha256_file(args.out),
        "passed": payload["passed"],
        "problems": problems,
        "source_tensors": len(source),
        "target_tensors": len(target),
        "trainable_tensors": len(trainable),
        "total_parameters": payload["total_parameters"],
        "trainable_parameters": trainable_parameters,
        "groups": groups,
        "scope_checks": scope_checks,
        "lora_runtime": lora_runtime,
        "optimizer_dtype_parameter_counts": optimizer_dtype_counts,
        "optimizer_state_kind_parameter_counts": optimizer_state_kinds,
    }, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
