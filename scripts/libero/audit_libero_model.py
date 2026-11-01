#!/usr/bin/env python3
"""Strict pi05_base load, scope, dtype, and optimizer-state audit."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import libero_data  # noqa: E402
import train_libero  # noqa: E402

LORA_RE = re.compile(r".*lora.*")
LLM_RE = re.compile(r".*llm.*")
EXPERT_RE = re.compile(r".*llm.*_1.*")
PROJECTION_NAMES = (
    "action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out", "state_proj",
)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def group(path: str) -> str:
    if LORA_RE.fullmatch(path):
        return "language_lora"
    if LLM_RE.fullmatch(path) and EXPERT_RE.fullmatch(path):
        return "action_expert"
    if "/img/" in path or "/img." in path:
        return "vision_encoder"
    if "llm" in path:
        return "language_backbone"
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
    import jax.numpy as jnp
    import numpy as np
    from flax.traverse_util import flatten_dict
    from openpi.models import gemma
    from openpi.models import model as model_lib
    from openpi.training import sharding

    protocol = train_libero.load_protocol(
        args.protocol, args.expected_protocol_sha256)
    view = libero_data.load_view(args.view)
    if view["view_sha256"] != protocol["data"]["view_sha256"]:
        raise ValueError("model audit view differs from protocol")
    base_tree = train_libero.sha256_tree(args.base_checkpoint)
    if base_tree["sha256"] != protocol["base"]["tree_sha256"]:
        raise ValueError("model audit base differs from the registered pi05_base")
    config = train_libero.make_config(
        SimpleNamespace(
            exp_name="libero-pi05-model-audit",
            mode="probe",
            view=str(args.view),
            base_checkpoint=str(args.base_checkpoint),
            fsdp_devices=4,
        ),
        protocol,
        view,
        args.out.parent / "checkpoint_config",
        int(protocol["training"]["optimizer_updates"]),
    )
    train, _ = train_libero._load_train_module()
    mesh = sharding.make_mesh(1)
    state, _ = train.init_train_state(
        config, jax.random.split(jax.random.key(config.seed))[1], mesh, resume=False)
    jax.block_until_ready(state)

    target = flatten_dict(state.params.to_pure_dict(), sep="/")
    loaded_tree = config.weight_loader.load(state.params.to_pure_dict())
    loaded = flatten_dict(loaded_tree, sep="/")
    source = flatten_dict(
        model_lib.restore_params(
            str(args.base_checkpoint / "params"), restore_type=np.ndarray),
        sep="/",
    )
    trainable = flatten_dict(
        state.params.filter(config.trainable_filter).to_pure_dict(), sep="/")
    problems = []
    if set(target) != set(loaded):
        problems.append("weight loader changed the target path set")
    if set(source) - set(target):
        problems.append("base checkpoint contains paths absent from the model")
    missing_non_lora = [
        name for name in sorted(set(target) - set(source))
        if not LORA_RE.fullmatch(name)
    ]
    if missing_non_lora:
        problems.append(
            "non-LoRA model paths absent from base: " + ",".join(missing_non_lora))
    records = []
    kind = lambda value: getattr(value, "value", value)
    for name in sorted(target):
        target_value = np.asarray(jax.device_get(kind(target[name])))
        loaded_value = np.asarray(kind(loaded[name]))
        base_value = source.get(name)
        record = {
            "path": name,
            "group": group(name),
            "target_shape": list(target_value.shape),
            "target_dtype": str(target_value.dtype),
            "trainable": name in trainable,
            "lora": bool(LORA_RE.fullmatch(name)),
            "finite": bool(np.isfinite(target_value).all()),
            "loader_shape_exact": loaded_value.shape == target_value.shape,
        }
        if not record["finite"]:
            problems.append(f"{name}: non-finite target value")
        if not record["loader_shape_exact"]:
            problems.append(f"{name}: loader shape mismatch")
        if base_value is None:
            record.update({
                "source_shape": None,
                "source_dtype": None,
                "loader_exact": bool(np.array_equal(target_value, loaded_value)),
                "source_cast_exact": None,
            })
            if not LORA_RE.fullmatch(name):
                problems.append(f"{name}: missing base source but is not LoRA")
        else:
            expected = base_value.astype(target_value.dtype)
            record.update({
                "source_shape": list(base_value.shape),
                "source_dtype": str(base_value.dtype),
                "loader_exact": bool(np.array_equal(expected, loaded_value)),
                "source_cast_exact": bool(np.array_equal(expected, target_value)),
            })
            if base_value.shape != target_value.shape:
                problems.append(f"{name}: source shape mismatch")
            if not record["loader_exact"]:
                problems.append(f"{name}: loader changed the cast source value")
            if not record["source_cast_exact"]:
                problems.append(f"{name}: initialized value differs from cast source")
        records.append(record)

    groups = {}
    for record in records:
        row = groups.setdefault(record["group"], {
            "tensors": 0, "parameters": 0, "trainable_tensors": 0,
            "trainable_parameters": 0, "dtypes": set(),
        })
        parameters = int(np.prod(record["target_shape"]))
        row["tensors"] += 1
        row["parameters"] += parameters
        row["dtypes"].add(record["target_dtype"])
        if record["trainable"]:
            row["trainable_tensors"] += 1
            row["trainable_parameters"] += parameters
    for row in groups.values():
        row["dtypes"] = sorted(row["dtypes"])

    required_projection_parts = {
        name: {part for part in ("kernel", "bias") if any(
            path.startswith(name + "/" + part) for path in target)}
        for name in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out")
    }
    projection_scope_complete = all(
        value == {"kernel", "bias"} for value in required_projection_parts.values())
    if not projection_scope_complete:
        problems.append(f"projection kernel/bias scope incomplete: {required_projection_parts}")
    scope_checks = {
        "action_expert_all_trainable": (
            groups.get("action_expert", {}).get("trainable_parameters", 0)
            == groups.get("action_expert", {}).get("parameters", -1)
        ),
        "vision_all_frozen": (
            groups.get("vision_encoder", {}).get("trainable_parameters", -1)
            == 0
        ),
        "projection_all_trainable": (
            groups.get("projection", {}).get("trainable_parameters", 0)
            == groups.get("projection", {}).get("parameters", -1)
        ),
        "projection_kernel_and_bias_all_included": projection_scope_complete,
        "language_lora_trainable": (
            groups.get("language_lora", {}).get("trainable_parameters", 0) > 0),
        "language_non_lora_frozen": (
            groups.get("language_backbone", {}).get("trainable_parameters", -1) == 0),
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
        optimizer_dtype_counts[dtype] = optimizer_dtype_counts.get(dtype, 0) + int(array.size)
        text = jax.tree_util.keystr(path)
        state_kind = next(
            (name for name in ("mu", "nu", "count", "scale", "trace") if name in text),
            "other",
        )
        optimizer_state_kinds[state_kind] = (
            optimizer_state_kinds.get(state_kind, 0) + int(array.size))
        if state_kind in {"mu", "nu"}:
            optimizer_parameters += int(array.size)

    lora_configs = gemma.get_config(config.model.paligemma_variant).lora_configs or {}
    lora_runtime = {
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
            param_dtype_counts.get(dtype, 0) + int(np.prod(record["target_shape"])))
    ema_pure = state.ema_params.to_pure_dict() if state.ema_params is not None else None
    ema_flat = flatten_dict(ema_pure, sep="/") if ema_pure is not None else {}
    ema_dtype_counts = {}
    ema_matches_initial = True
    for name, value in ema_flat.items():
        array = np.asarray(jax.device_get(kind(value)))
        dtype = str(array.dtype)
        ema_dtype_counts[dtype] = ema_dtype_counts.get(dtype, 0) + int(array.size)
        initial = np.asarray(jax.device_get(kind(target[name])))
        ema_matches_initial &= bool(np.array_equal(array, initial))
    if float(state.ema_decay) != float(protocol["training"]["ema_decay"]):
        problems.append("actual TrainState EMA decay differs from protocol")
    if set(ema_flat) != set(target):
        problems.append("EMA parameter tree does not cover the full parameter tree")
    if not ema_matches_initial:
        problems.append("EMA parameters do not start equal to initial parameters")
    payload = {
        "schema_version": 1,
        "protocol_sha256": protocol["_sha256"],
        "view_sha256": view["view_sha256"],
        "base_tree_sha256": base_tree["sha256"],
        "model_config": train_libero._jsonable(config.model),
        "freeze_filter": repr(config.freeze_filter),
        "trainable_filter": repr(config.trainable_filter),
        "lora_runtime": lora_runtime,
        "source_tensors": len(source),
        "target_tensors": len(target),
        "loaded_tensors": len(loaded),
        "trainable_tensors": len(trainable),
        "total_parameters": sum(
            int(np.prod(record["target_shape"])) for record in records),
        "trainable_parameters": sum(
            int(np.prod(record["target_shape"]))
            for record in records if record["trainable"]),
        "parameter_dtype_parameter_counts": param_dtype_counts,
        "optimizer_dtype_parameter_counts": optimizer_dtype_counts,
        "optimizer_state_kind_parameter_counts": optimizer_state_kinds,
        "optimizer_mu_nu_arrays_cover_parameters": (
            optimizer_parameters == 2 * sum(
                int(np.prod(record["target_shape"]))
                for record in records if record["trainable"])),
        "ema_decay": float(state.ema_decay),
        "ema_parameters": int(sum(value.size for value in ema_flat.values())),
        "ema_tensors": len(ema_flat),
        "ema_dtype_parameter_counts": ema_dtype_counts,
        "ema_covers_full_parameter_tree": set(ema_flat) == set(target),
        "ema_initial_equals_parameters": ema_matches_initial,
        "groups": groups,
        "scope_checks": scope_checks,
        "projection_parts": {
            name: sorted(value) for name, value in required_projection_parts.items()
        },
        "records": records,
        "problems": problems,
        "passed": not problems,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.out),
        "sha256": sha256_file(args.out),
        "passed": payload["passed"],
        "problems": problems,
        "source_tensors": len(source),
        "target_tensors": len(target),
        "trainable_tensors": len(trainable),
        "total_parameters": payload["total_parameters"],
        "trainable_parameters": payload["trainable_parameters"],
        "groups": groups,
        "parameter_dtype_parameter_counts": param_dtype_counts,
        "optimizer_dtype_parameter_counts": optimizer_dtype_counts,
        "optimizer_state_kind_parameter_counts": optimizer_state_kinds,
        "optimizer_mu_nu_arrays_cover_parameters": payload[
            "optimizer_mu_nu_arrays_cover_parameters"],
        "ema_decay": payload["ema_decay"],
        "ema_parameters": payload["ema_parameters"],
        "ema_tensors": payload["ema_tensors"],
        "ema_dtype_parameter_counts": ema_dtype_counts,
        "ema_covers_full_parameter_tree": payload[
            "ema_covers_full_parameter_tree"],
        "ema_initial_equals_parameters": payload[
            "ema_initial_equals_parameters"],
        "scope_checks": scope_checks,
        "lora_runtime": lora_runtime,
    }, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
