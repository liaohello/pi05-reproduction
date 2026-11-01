#!/usr/bin/env python3
"""Strict clean60 base-loading, freeze-scope and parameter-delta audit.

The reference is a real ``init_train_state`` result, not a prematurely flattened
checkpoint dictionary.  Run this on the same multi-GPU topology and registered
scope used by the short update probe.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re

import flax.nnx as nnx
import jax
import numpy as np

import openpi.models.gemma as gemma
import openpi.models.model as model_lib
import openpi.training.sharding as sharding

import clean60_contracts as clean60
import openpi_so101

BASE = Path(os.environ.get("AUDIT_BASE", "/base"))
CKPT = Path(os.environ.get("AUDIT_CKPT", "/ckpt"))
OUT = Path(os.environ.get("AUDIT_OUT", "/work/runs/so101/clean60_probes"))
FSDP_DEVICES = int(os.environ.get("AUDIT_FSDP_DEVICES", "4"))
BATCH_SIZE = int(os.environ.get("AUDIT_BATCH_SIZE", "8"))
SEED = int(os.environ.get("AUDIT_SEED", "42"))
VARIANTS = {"paligemma_variant": "gemma_2b_lora", "action_expert_variant": "gemma_300m"}

EXPERT_RE = re.compile(r".*llm.*_1.*")
LORA_RE = re.compile(r".*lora.*")
LLM_RE = re.compile(r".*llm.*")
PROJECTIONS = ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out")


def group_of(path: str) -> str:
    if LLM_RE.fullmatch(path) and EXPERT_RE.fullmatch(path) and not LORA_RE.fullmatch(path):
        return "action_expert"
    if "/img/" in path or "/img." in path:
        return "vision_encoder"
    if LORA_RE.fullmatch(path):
        return "language_lora"
    if LLM_RE.fullmatch(path):
        return "language_backbone"
    for name in PROJECTIONS:
        if path.startswith(name + "/") or path == name:
            return f"projection:{name}"
    return "other"


def flatten_tree(node, prefix: str = "") -> dict[str, np.ndarray]:
    """Flatten a nested checkpoint state without changing its key structure."""
    if hasattr(node, "to_pure_dict"):
        node = node.to_pure_dict()
    if isinstance(node, dict):
        result = {}
        for key, child in node.items():
            name = f"{prefix}/{key}" if prefix else str(key)
            result.update(flatten_tree(child, name))
        return result
    if hasattr(node, "value") and not hasattr(node, "shape"):
        return flatten_tree(node.value, prefix)
    array = np.asarray(node)
    if not hasattr(array, "dtype"):
        raise TypeError(f"unsupported parameter node at {prefix or '<root>'}")
    if prefix == "":
        raise TypeError("parameter tree cannot be a bare array")
    return {prefix: array}


def tree_sha(flat: dict[str, np.ndarray]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for name in sorted(flat):
        value = flat[name]
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(value.dtype).encode())
        digest.update(b"\0")
        digest.update(str(value.shape).encode())
        digest.update(b"\0")
        digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


def strict_problems(reference: dict[str, np.ndarray],
                    other: dict[str, np.ndarray],
                    *, require_bit_equal: bool = True) -> list[str]:
    problems = []
    missing = sorted(set(reference) - set(other))
    extra = sorted(set(other) - set(reference))
    if missing:
        problems.append(f"{len(missing)} reference paths are absent")
    if extra:
        problems.append(f"{len(extra)} unexpected paths are present")
    for name in sorted(set(reference) & set(other)):
        before, after = reference[name], other[name]
        if before.shape != after.shape:
            problems.append(f"{name}: shape {before.shape} != {after.shape}")
            continue
        if before.dtype != after.dtype:
            problems.append(f"{name}: dtype {before.dtype} != {after.dtype}")
            continue
        if not np.isfinite(before).all() or not np.isfinite(after).all():
            problems.append(f"{name}: non-finite value")
            continue
        if require_bit_equal and not np.array_equal(before, after):
            problems.append(f"{name}: values differ")
    return problems


def loader_problems(target: dict[str, np.ndarray],
                    loaded: dict[str, np.ndarray],
                    base_raw: dict[str, np.ndarray]) -> list[str]:
    """Check the loader's real nested-tree output, including its licensed cast."""
    problems = []
    if set(target) != set(loaded):
        problems.append("CheckpointWeightLoader path set differs from the TrainState tree")
    for name in sorted(set(target) & set(loaded)):
        expected, actual = target[name], loaded[name]
        if expected.shape != actual.shape:
            problems.append(f"{name}: loader shape mismatch")
            continue
        if not np.isfinite(actual).all():
            problems.append(f"{name}: loader produced a non-finite value")
            continue
        if name in base_raw:
            source = base_raw[name]
            if source.shape != expected.shape:
                problems.append(f"{name}: base source shape mismatch")
            elif not np.array_equal(source, actual):
                problems.append(f"{name}: loader did not preserve its source value")
            elif not np.array_equal(source.astype(expected.dtype), expected):
                problems.append(f"{name}: source cast differs from initial TrainState")
        elif not LORA_RE.fullmatch(name):
            problems.append(f"{name}: non-LoRA path is absent from the base checkpoint")
        elif not np.array_equal(actual, expected):
            problems.append(f"{name}: random LoRA init was not retained")
    return problems


def summarise_delta(reference: dict[str, np.ndarray],
                    trained: dict[str, np.ndarray],
                    trainable: set[str]) -> tuple[dict, list[str]]:
    groups = {}
    problems = []
    for name in sorted(set(reference) & set(trained)):
        before, after = reference[name], trained[name]
        if before.shape != after.shape:
            problems.append(f"{name}: trained shape mismatch")
            continue
        if before.dtype != after.dtype:
            problems.append(f"{name}: trained dtype mismatch")
            continue
        if not np.isfinite(before).all() or not np.isfinite(after).all():
            problems.append(f"{name}: non-finite trained/reference value")
            continue
        delta = float(np.max(np.abs(after.astype(np.float64) - before.astype(np.float64))))
        group = group_of(name)
        entry = groups.setdefault(group, {
            "tensors": 0, "changed_tensors": 0, "max_abs_delta": 0.0,
            "trainable_tensors": 0, "frozen_tensors": 0, "sample": name,
        })
        entry["tensors"] += 1
        entry["changed_tensors"] += int(delta > 0.0)
        entry["max_abs_delta"] = max(entry["max_abs_delta"], delta)
        if name in trainable:
            entry["trainable_tensors"] += 1
        else:
            entry["frozen_tensors"] += 1
    return groups, problems


def _load_train_module():
    path = Path("/opt/openpi/scripts/train.py")
    spec = importlib.util.spec_from_file_location("openpi_train_entry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    train = _load_train_module()
    run_root = OUT / "param_delta_runtime"
    config = openpi_so101.make_train_config(
        mode="serve", run_root=run_root, train_view=run_root / "view",
        base_checkpoint=BASE, exp_name="param_delta", state_mode="on",
        batch_size=BATCH_SIZE, fsdp_devices=FSDP_DEVICES, seed=SEED,
        paligemma_variant=VARIANTS["paligemma_variant"],
        action_expert_variant=VARIANTS["action_expert_variant"],
        action_horizon=10, dtype="bfloat16")
    mesh = sharding.make_mesh(FSDP_DEVICES)
    init_rng = jax.random.split(jax.random.key(SEED))[1]
    state, _ = train.init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(state)
    reference = flatten_tree(state.params)
    trainable = {name for name, value in
                 flatten_tree(state.params.filter(config.trainable_filter)).items()}

    # The loader receives the real nested TrainState tree, never a flattened dict.
    loaded_tree = config.weight_loader.load(state.params.to_pure_dict())
    loaded = flatten_tree(loaded_tree)
    base_raw = flatten_tree(model_lib.restore_params(
        str(BASE / "params"), restore_type=np.ndarray))
    load_problems = loader_problems(reference, loaded, base_raw)
    non_lora_reference = {name: value for name, value in reference.items()
                          if not LORA_RE.fullmatch(name)}
    base_problems = []
    base_paths_absent = sorted(set(non_lora_reference) - set(base_raw))
    base_extra = sorted(set(base_raw) - set(non_lora_reference))
    if base_paths_absent:
        base_problems.append(f"{len(base_paths_absent)} non-LoRA target paths absent from base")
    if base_extra:
        base_problems.append(f"{len(base_extra)} base paths absent from target")
    loaded_records = []
    for name, target in sorted(non_lora_reference.items()):
        source = base_raw.get(name)
        if source is None:
            continue
        if source.shape != target.shape:
            base_problems.append(f"{name}: source/target shape mismatch")
            continue
        cast = source.astype(target.dtype)
        if not np.isfinite(source).all() or not np.array_equal(cast, target):
            base_problems.append(f"{name}: source-cast/target values differ")
        loaded_records.append({
            "path": name, "source_shape": list(source.shape),
            "target_shape": list(target.shape), "source_dtype": str(source.dtype),
            "target_dtype": str(target.dtype), "cast_exact": bool(np.array_equal(cast, target)),
            "group": group_of(name), "trainable": name in trainable,
        })
    non_lora_extras = sorted(name for name in set(reference) - set(base_raw)
                             if not LORA_RE.fullmatch(name))
    if non_lora_extras:
        base_problems.append("non-LoRA parameters were randomly initialised")

    trained = flatten_tree(model_lib.restore_params(
        str(CKPT / "params"), restore_type=np.ndarray))
    structure_problems = clean60.tree_structure_problems(
        {name: {"shape": value.shape, "dtype": str(value.dtype)}
         for name, value in reference.items()},
        {name: {"shape": value.shape, "dtype": str(value.dtype)}
         for name, value in trained.items()},
    )
    trained_problems = strict_problems(
        {name: value for name, value in reference.items()},
        trained, require_bit_equal=False)
    delta_groups, delta_problems = summarise_delta(reference, trained, trainable)
    frozen_language = {name for name in reference
                       if group_of(name) == "language_backbone"}
    frozen_changed = sorted(
        name for name in frozen_language
        if name not in trained or not np.array_equal(reference[name], trained[name]))
    frozen_dtype_mismatch = sorted(
        name for name in frozen_language
        if name not in trained or reference[name].dtype != trained[name].dtype)

    must_move = ["action_expert", "vision_encoder", "language_lora",
                 *[f"projection:{name}" for name in PROJECTIONS]]
    missing_updates = [
        name for name in must_move
        if delta_groups.get(name, {}).get("max_abs_delta", 0.0) <= 0.0]
    projection_missing_components = {}
    for projection in PROJECTIONS:
        paths = [name for name in trained if group_of(name) == f"projection:{projection}"]
        missing = [component for component in ("kernel", "bias")
                   if not any(name.endswith("/" + component) for name in paths)]
        unchanged = [name for name in paths
                     if name in reference and np.array_equal(reference[name], trained[name])]
        projection_missing_components[projection] = {
            "missing_kernel_or_bias": missing, "unchanged_tensors": unchanged}

    gemma_config = gemma.get_config(VARIANTS["paligemma_variant"])
    lora_runtime = {
        name: {"rank": int(value.rank), "alpha": float(value.alpha),
               "axes": list(value.axes), "rslora": bool(value.rslora)}
        for name, value in (gemma_config.lora_configs or {}).items()
    }
    passed = bool(
        not load_problems and not base_problems and not structure_problems
        and not trained_problems
        and not delta_problems and not missing_updates
        and not any(value["missing_kernel_or_bias"] or value["unchanged_tensors"]
                    for value in projection_missing_components.values())
        and not frozen_changed and not frozen_dtype_mismatch
        and set(lora_runtime) == {"attn", "ffn"})
    payload = {
        "schema_version": 2,
        "reference": "actual init_train_state output from CheckpointWeightLoader(base)",
        "base_checkpoint": BASE.name,
        "trained_checkpoint": CKPT.name,
        "fsdp_devices": FSDP_DEVICES,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "fresh_tensors": len(reference),
        "loader_loaded_tensors": len(loaded),
        "base_source_tensors": len(base_raw),
        "trained_tensors": len(trained),
        "loader_path_shape_dtype_value_problems": load_problems,
        "base_load_problems": base_problems,
        "base_paths_absent_from_target": base_paths_absent,
        "base_paths_extra_in_target": base_extra,
        "non_lora_random_initialisation": non_lora_extras,
        "base_load_records": loaded_records,
        "trained_structure_problems": trained_problems,
        "trained_path_shape_dtype_problems": structure_problems,
        "trained_delta_problems": delta_problems,
        "groups": delta_groups,
        "must_move_missing": missing_updates,
        "projection_components": projection_missing_components,
        "language_backbone_changed_tensors": frozen_changed[:20],
        "language_backbone_dtype_mismatches": frozen_dtype_mismatch[:20],
        "lora_runtime": lora_runtime,
        "reference_params_sha256": tree_sha(reference),
        "trained_params_sha256": tree_sha(trained),
        "initial_state_dtypes": sorted({str(value.dtype) for value in reference.values()}),
        "trained_state_dtypes": sorted({str(value.dtype) for value in trained.values()}),
        "passed": passed,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "param_delta_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items()
                      if key != "base_load_records"}, indent=2, sort_keys=True)[:12000])
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
