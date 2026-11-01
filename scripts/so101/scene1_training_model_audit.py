#!/usr/bin/env python3
"""Strict Scene1 pi05_base load, scope, dtype, optimizer and EMA audit."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scene1_openpi_so101 as sx  # noqa: E402

LORA = re.compile(r".*lora.*")
EXPERT = re.compile(r".*llm.*_1.*")
LLM = re.compile(r".*llm.*")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(root: Path) -> str:
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def group(path: str) -> str:
    if LORA.fullmatch(path):
        return "language_lora"
    if LLM.fullmatch(path) and EXPERT.fullmatch(path):
        return "action_expert"
    if "/img/" in path or "/img." in path:
        return "vision_encoder"
    if LLM.fullmatch(path):
        return "language_backbone"
    if any(name in path for name in (
            "action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out")):
        return "projection"
    return "other"


def load_train_module():
    spec = importlib.util.spec_from_file_location(
        "openpi_train_entry", "/opt/openpi/scripts/train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-view", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--expected-training-config-sha256", required=True)
    parser.add_argument("--expected-base-tree-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    from flax.traverse_util import flatten_dict
    from openpi.models import gemma
    from openpi.models import model as model_lib
    from openpi.training import sharding

    if sha256_file(args.training_config) != args.expected_training_config_sha256:
        raise ValueError("training config SHA256 mismatch")
    if sha256_tree(args.base_checkpoint) != args.expected_base_tree_sha256:
        raise ValueError("base checkpoint tree SHA256 mismatch")
    registered = json.loads(args.training_config.read_text())
    config = sx.make_train_config(
        mode="serve",
        run_root=args.out.parent / "model_audit_run",
        train_view=args.train_view,
        base_checkpoint=args.base_checkpoint,
        exp_name="scene1_pi05base_model_audit",
        state_mode="on",
        steps_override=25000,
        save_interval=3000,
        batch_size=8,
        lr_warmup_steps=1000,
        peak_lr=5e-5,
        decay_steps=25000,
        decay_lr=5e-6,
        fsdp_devices=1,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m",
        freeze_vision_encoder=True,
        action_horizon=10,
        seed=42,
        gradient_clip_norm=1.0,
        optimizer_name="AdamW",
        ema_decay=0.999,
        dtype="bfloat16",
    )
    train = load_train_module()
    mesh = sharding.make_mesh(1)
    state, _ = train.init_train_state(
        config, jax.random.split(jax.random.key(config.seed))[1], mesh, resume=False)
    jax.block_until_ready(state)
    target = flatten_dict(state.params.to_pure_dict(), sep="/")
    target_set = flatten_dict(
        state.params.filter(config.trainable_filter).to_pure_dict(), sep="/")
    loaded = flatten_dict(
        config.weight_loader.load(state.params.to_pure_dict()), sep="/")
    source = flatten_dict(model_lib.restore_params(
        str(args.base_checkpoint / "params"), restore_type=np.ndarray), sep="/")
    problems = []
    if set(target) != set(loaded):
        problems.append("CheckpointWeightLoader changed target paths")
    if set(source) - set(target):
        problems.append("base contains paths absent from the model")
    missing = [
        path for path in set(target) - set(source) if not LORA.fullmatch(path)]
    if missing:
        problems.append("non-LoRA parameters missing from base: " + ",".join(sorted(missing)))

    value_of = lambda value: getattr(value, "value", value)
    records = []
    for path in sorted(target):
        train_value = np.asarray(jax.device_get(value_of(target[path])))
        loaded_value = value_of(loaded[path])
        source_value = source.get(path)
        if not np.isfinite(train_value).all():
            problems.append(f"{path}: non-finite TrainState value")
        record = {
            "path": path,
            "group": group(path),
            "shape": list(train_value.shape),
            "target_dtype": str(train_value.dtype),
            "trainable": path in target_set,
            "lora": bool(LORA.fullmatch(path)),
        }
        if hasattr(loaded_value, "shape") and tuple(loaded_value.shape) != train_value.shape:
            problems.append(f"{path}: loader shape mismatch")
        if source_value is None:
            if not LORA.fullmatch(path):
                problems.append(f"{path}: random initialization outside LoRA")
            record.update({
                "source_dtype": None,
                "source_shape": None,
                "source_cast_exact": None,
            })
        else:
            source_value = np.asarray(source_value)
            expected = source_value.astype(train_value.dtype)
            loaded_array = np.asarray(jax.device_get(loaded_value))
            record.update({
                "source_dtype": str(source_value.dtype),
                "source_shape": list(source_value.shape),
                "source_cast_exact": bool(np.array_equal(expected, train_value)),
                "loader_exact": bool(np.array_equal(expected, loaded_array)),
            })
            if source_value.shape != train_value.shape:
                problems.append(f"{path}: base shape mismatch")
            if not record["source_cast_exact"]:
                problems.append(f"{path}: TrainState differs from cast base value")
            if not record["loader_exact"]:
                problems.append(f"{path}: loader differs from cast base value")
        records.append(record)

    groups = {}
    for record in records:
        entry = groups.setdefault(record["group"], {
            "tensors": 0, "parameters": 0, "trainable_parameters": 0,
            "trainable_tensors": 0, "dtypes": set(),
        })
        count = int(np.prod(record["shape"]))
        entry["tensors"] += 1
        entry["parameters"] += count
        entry["dtypes"].add(record["target_dtype"])
        if record["trainable"]:
            entry["trainable_parameters"] += count
            entry["trainable_tensors"] += 1
    for entry in groups.values():
        entry["dtypes"] = sorted(entry["dtypes"])

    expected_groups = {
        "action_expert": (427932672, 427932672),
        "language_backbone": (2508531712, 0),
        "language_lora": (27869184, 27869184),
        "projection": (2165792, 2165792),
        "vision_encoder": (414803696, 0),
    }
    for name, (parameters, trainable) in expected_groups.items():
        actual = groups.get(name, {})
        if (actual.get("parameters"), actual.get("trainable_parameters")) != (
                parameters, trainable):
            problems.append(f"{name}: scope count mismatch")
    if set(groups) != set(expected_groups):
        problems.append("unexpected parameter groups: " + ",".join(sorted(set(groups) - set(expected_groups))))

    parameter_dtypes = {}
    for record in records:
        parameter_dtypes[record["target_dtype"]] = (
            parameter_dtypes.get(record["target_dtype"], 0)
            + int(np.prod(record["shape"])))
    if parameter_dtypes != {"bfloat16": 2923335408, "float32": 457967648}:
        problems.append(f"parameter dtype coverage mismatch: {parameter_dtypes}")

    optimizer_shapes = {}
    optimizer_mu_nu = 0
    for path, value in jax.tree_util.tree_leaves_with_path(state.opt_state):
        array = jnp.asarray(value)
        key = str(array.dtype)
        optimizer_shapes[key] = optimizer_shapes.get(key, 0) + int(array.size)
        text = jax.tree_util.keystr(path)
        if ".mu" in text or ".nu" in text:
            optimizer_mu_nu += int(array.size)
    if optimizer_mu_nu != 2 * 457967648:
        problems.append("AdamW mu/nu does not cover every trainable parameter")
    if optimizer_shapes.get("float32", 0) != 2 * 457967648:
        problems.append(f"AdamW trainable-state dtype coverage mismatch: {optimizer_shapes}")

    ema = flatten_dict(state.ema_params.to_pure_dict(), sep="/")
    ema_dtypes = {}
    ema_matches = set(ema) == set(target)
    for path, value in ema.items():
        array = np.asarray(jax.device_get(value_of(value)))
        ema_dtypes[str(array.dtype)] = ema_dtypes.get(str(array.dtype), 0) + int(array.size)
        if path not in target or not np.array_equal(
                array, np.asarray(jax.device_get(value_of(target[path])))):
            ema_matches = False
    if float(state.ema_decay) != 0.999:
        problems.append("actual EMA decay is not 0.999")
    if not ema_matches:
        problems.append("EMA does not initialize over the full parameter tree")

    lora = gemma.get_config(config.model.paligemma_variant).lora_configs or {}
    lora_runtime = {
        name: {"rank": int(value.rank), "alpha": float(value.alpha),
               "axes": list(value.axes), "rslora": bool(value.rslora)}
        for name, value in lora.items()
    }
    checks = {
        "config_sha256": sha256_file(args.training_config),
        "base_tree_sha256": sha256_tree(args.base_checkpoint),
        "target_episodes": len(json.loads(
            (args.train_view / "meta.json").read_text())["episodes"]),
        "view_sha256": json.loads(
            (args.train_view / "meta.json").read_text())["view_sha256"],
        "source_tensors": len(source),
        "target_tensors": len(target),
        "trainable_tensors": len(target_set),
        "total_parameters": sum(int(np.prod(record["shape"])) for record in records),
        "trainable_parameters": sum(
            int(np.prod(record["shape"])) for record in records if record["trainable"]),
        "vision_encoder_trainable": groups["vision_encoder"]["trainable_parameters"],
        "language_backbone_trainable": groups["language_backbone"]["trainable_parameters"],
        "parameter_dtypes": parameter_dtypes,
        "optimizer_dtypes": optimizer_shapes,
        "ema_decay": float(state.ema_decay),
        "ema_parameters": sum(
            int(np.prod(list(np.asarray(jax.device_get(value_of(value))).shape)))
            for value in ema.values()),
        "ema_dtypes": ema_dtypes,
        "ema_initial_equals_parameters": ema_matches,
    }
    payload = {
        "schema_version": 1,
        "training_config_sha256": checks["config_sha256"],
        "base_tree_sha256": checks["base_tree_sha256"],
        "view_sha256": checks["view_sha256"],
        "model_config": {
            "pi05": True,
            "state_mode": "on",
            "action_horizon": config.model.action_horizon,
            "paligemma_variant": config.model.paligemma_variant,
            "action_expert_variant": config.model.action_expert_variant,
            "freeze_vision_encoder": True,
            "ema_decay": config.ema_decay,
        },
        "lora_runtime": lora_runtime,
        "groups": groups,
        "records": records,
        "checks": checks,
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
        "checks": checks,
        "groups": groups,
        "lora_runtime": lora_runtime,
    }, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
