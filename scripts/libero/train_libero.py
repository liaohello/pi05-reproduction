#!/usr/bin/env python3
"""Pinned one-task S2 pi05_base trainer for the LIBERO work line."""
from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import libero_data  # noqa: E402

PROTOCOL_FIELDS = (
    "schema_version", "protocol_id", "suite", "task_id", "dataset_task_index",
    "task", "base", "data", "runtime", "model", "training", "validation",
    "checkpoint", "resource",
)
PROTOCOL_KEYS = PROTOCOL_FIELDS + ("experiment_prefix", "policy_config", "evaluation")
MODE_KINDS = {
    "initial": "formal", "probe": "probe", "probe_resume": "probe", "resume": "formal",
}
VALIDATION_LOSS_DEFINITION = (
    "model.compute_loss(rng, observation, actions, train=False); "
    "mean over action horizon and action dimensions, then equal-weight mean "
    "over all valid validation samples"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> dict:
    """Hash file bytes and relative paths, excluding transient Python bytecode."""
    path = Path(path)
    if path.is_file():
        data = path.read_bytes()
        return {"files": 1, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    files = sorted(
        item for item in path.rglob("*")
        if item.is_file()
        and "__pycache__" not in item.parts
        and item.suffix not in {".pyc", ".pyo"}
    )
    if not files:
        raise ValueError(f"empty tree: {path}")
    digest = hashlib.sha256()
    total = 0
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        file_digest = hashlib.sha256()
        size = 0
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                size += len(block)
                file_digest.update(block)
        total += size
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(file_digest.digest())
    return {"files": len(files), "bytes": total, "sha256": digest.hexdigest()}


def canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def action_semantics_token(value: str) -> str:
    return value.replace("-", "_").replace(" ", "_")


def load_protocol(path: Path, expected_sha256: str) -> dict:
    path = Path(path).resolve()
    actual = sha256_file(path)
    if not expected_sha256 or actual != expected_sha256:
        raise ValueError(f"training protocol SHA256 mismatch: {actual}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if tuple(sorted(document)) != tuple(sorted(PROTOCOL_KEYS)):
        raise ValueError("training protocol has an unexpected field set")
    required = {
        "schema_version": 1,
        "protocol_id": "libero-pi05-goal8-s2-v5-validation",
        "experiment_prefix": "libero-pi05-",
        "policy_config": "pi05_libero",
        "suite": "libero_goal",
        "task_id": 8,
        "dataset_task_index": 10,
        "task": "put the bowl on the plate",
    }
    for key, value in required.items():
        if document.get(key) != value:
            raise ValueError(f"training protocol {key} differs from S2")
    model = document["model"]
    expected_model = {
        "pi05": True,
        "paligemma_variant": "gemma_2b_lora",
        "action_expert_variant": "gemma_300m",
        "discrete_state_input": False,
        "dtype": "bfloat16",
        "scope": "s2_language_lora_action_full_vision_frozen_projection_full",
        "freeze_language_non_lora": True,
        "freeze_vision_encoder": True,
        "train_language_attention_lora": True,
        "train_language_mlp_lora": True,
        "train_action_expert_full": True,
        "train_projection_full": True,
    }
    if model != expected_model:
        raise ValueError("training protocol model scope differs from S2")
    training = document["training"]
    exact_training = {
        "seed": 42,
        "global_batch_size": 8,
        "optimizer_updates": 25000,
        "optimizer": "AdamW",
        "adamw_b1": 0.9,
        "adamw_b2": 0.95,
        "adamw_eps": 1e-8,
        "adamw_weight_decay": 1e-10,
        "gradient_clip_norm": 1.0,
        "warmup_steps": 500,
        "peak_lr": 5e-5,
        "cosine_decay_steps": 25000,
        "final_lr": 5e-6,
        "ema_decay": 0.999,
        "data_loader_workers": 8,
    }
    for key, value in exact_training.items():
        if training.get(key) != value:
            raise ValueError(f"training protocol training.{key} differs from S2")
    if list(training.get("fsdp_devices", [])) != [2, 4]:
        raise ValueError("training protocol must admit only two- or four-device FSDP")
    if document["validation"] != {
        "split": "validation",
        "expected_samples": 895,
        "norm_stats_source": "train_only",
        "train_argument": False,
        "loss": "flow_matching_mse",
        "seed": 42,
        "sample_order": "episode_index_then_frame_index",
        "sample_weighting": "uniform_per_valid_sample",
        "gradient_updates": 0,
        "evaluate_after_checkpoint": True,
    }:
        raise ValueError("training protocol validation settings differ from S2")
    expected_checkpoints = [
        500, 1000, 1500, 2000, 3000, 6000, 9000,
        12000, 15000, 18000, 21000, 24000, 25000,
    ]
    if document["checkpoint"] != {
        "save_steps": expected_checkpoints,
        "full_train_state": True,
        "retain_all_saved_steps": True,
        "orbax_directory_is_not_train_state_step": True,
    }:
        raise ValueError("training protocol checkpoint settings differ from S2")
    if int(document["resource"].get("minimum_memory_margin_mib", -1)) != 10:
        raise ValueError("training protocol must require a 10 MiB GPU margin")
    return {**document, "_path": str(path), "_sha256": actual}


def checkpoint_steps(protocol: dict) -> tuple[int, ...]:
    steps = tuple(int(value) for value in protocol["checkpoint"]["save_steps"])
    if not steps or tuple(sorted(set(steps))) != steps:
        raise ValueError("checkpoint save_steps must be positive, sorted, and unique")
    if steps[-1] != int(protocol["training"]["optimizer_updates"]):
        raise ValueError("the final checkpoint step must equal optimizer_updates")
    if any(value <= 0 for value in steps):
        raise ValueError("checkpoint steps must be positive optimizer-update counts")
    return steps


def weighted_validation_loss(
    per_sample_losses: Sequence[Sequence[float]],
    valid_samples_per_batch: Sequence[int],
) -> dict:
    """Equal-weight all valid samples while excluding deterministic final-batch padding."""
    if len(per_sample_losses) != len(valid_samples_per_batch):
        raise ValueError("validation batch count differs from the fixed plan")
    weighted_sum = 0.0
    weight_sum = 0.0
    for batch_index, (values, valid) in enumerate(
        zip(per_sample_losses, valid_samples_per_batch)
    ):
        valid = int(valid)
        if valid < 0 or valid > len(values):
            raise ValueError(f"invalid validation sample count at batch {batch_index}")
        for raw in values[:valid]:
            value = float(raw)
            if not math.isfinite(value):
                raise FloatingPointError(
                    f"non-finite validation loss at batch {batch_index}")
            weighted_sum += value
            weight_sum += 1.0
    if weight_sum <= 0.0:
        raise ValueError("validation produced no valid samples")
    return {
        "validation_loss": weighted_sum / weight_sum,
        "weighted_loss_sum": weighted_sum,
        "weight_sum": weight_sum,
        "samples_evaluated": int(weight_sum),
        "sample_weighting": "uniform_per_valid_sample",
    }


def public_validation_plan(plan: dict) -> dict:
    return {key: value for key, value in plan.items() if not key.startswith("_")}


def finite_metrics(info: dict) -> tuple[bool, dict]:
    import jax
    import numpy as np

    values = {
        key: float(np.asarray(jax.device_get(value)).reshape(-1)[0])
        for key, value in info.items()
    }
    return all(np.isfinite(value) for value in values.values()), values


def validation_sample_losses(state, observation, actions, sample_keys):
    """Gradient-free validation using the model's training loss definition."""
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp

    model = nnx.merge(state.model_def, state.params)
    model.eval()

    def one_sample(sample_key, sample_observation, sample_actions):
        chunked = model.compute_loss(
            sample_key, sample_observation, sample_actions, train=False)
        return jnp.mean(chunked)

    return jax.vmap(one_sample)(sample_keys, observation, actions)


def validation_rngs(seed: int, sample_indices: Sequence[int]):
    """Create one stable RNG key per validation row and final-batch padding position."""
    import jax
    import jax.numpy as jnp

    base = jax.random.key(int(seed))
    indices = jnp.asarray(sample_indices, dtype=jnp.uint32)
    return jax.vmap(lambda index: jax.random.fold_in(base, index))(indices)


def validation_rng_sha256(sample_keys) -> str:
    import jax
    import numpy as np

    data = np.asarray(jax.device_get(jax.random.key_data(sample_keys)), dtype="<u4")
    return hashlib.sha256(data.tobytes()).hexdigest()


def validation_randomness_digests(
    sample_keys, action_horizon: int, action_dim: int
) -> dict:
    """Hash the exact Flow time and Gaussian noise produced by compute_loss keys."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    def one_sample(sample_key):
        _, noise_rng, time_rng = jax.random.split(sample_key, 3)
        noise = jax.random.normal(noise_rng, (int(action_horizon), int(action_dim)))
        flow_time = jax.random.beta(time_rng, 1.5, 1, ()) * 0.999 + 0.001
        return noise, flow_time

    noises, times = jax.vmap(one_sample)(sample_keys)
    noise_array = np.asarray(jax.device_get(noises), dtype="<f4")
    time_array = np.asarray(jax.device_get(times), dtype="<f4")
    return {
        "noise_sha256": hashlib.sha256(noise_array.tobytes()).hexdigest(),
        "flow_time_sha256": hashlib.sha256(time_array.tobytes()).hexdigest(),
        "noise_shape": list(noise_array.shape),
        "flow_time_shape": list(time_array.shape),
    }


def collect_validation_loss(
    batches,
    valid_samples_per_batch: Sequence[int],
    evaluate_batch,
    progress=None,
) -> dict:
    """Evaluate every fixed batch once and aggregate only unpadded samples."""
    rows: list[list[float]] = []
    for batch_index, batch in enumerate(batches):
        if batch_index >= len(valid_samples_per_batch):
            raise ValueError("validation loader returned more batches than planned")
        values = evaluate_batch(batch, batch_index)
        rows.append([float(value) for value in values])
        if progress is not None:
            progress(batch_index + 1, len(valid_samples_per_batch))
    if len(rows) != len(valid_samples_per_batch):
        raise ValueError(
            f"validation loader returned {len(rows)} batches, "
            f"expected {len(valid_samples_per_batch)}")
    result = weighted_validation_loss(rows, valid_samples_per_batch)
    result["batches_evaluated"] = len(rows)
    return result


def _jsonable(value):
    if dataclasses.is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return repr(value)


def make_config(args, protocol: dict, view: dict, run_root: Path, target_steps: int):
    import openpi.training.config as C
    import openpi.training.optimizer as opt
    import flax.nnx as nnx
    from openpi.shared import nnx_utils
    from openpi.training import weight_loaders

    base = C.get_config(protocol["policy_config"])
    model_settings = protocol["model"]
    model = dataclasses.replace(
        base.model,
        pi05=bool(model_settings["pi05"]),
        paligemma_variant=model_settings["paligemma_variant"],
        action_expert_variant=model_settings["action_expert_variant"],
        action_horizon=int(protocol["data"]["action_horizon"]),
        discrete_state_input=bool(model_settings["discrete_state_input"]),
        dtype=model_settings["dtype"],
    )
    data_base = dataclasses.replace(
        base.data.base_config or C.DataConfig(), prompt_from_task=True)
    data = dataclasses.replace(
        base.data,
        repo_id="local/libero-goal8",
        assets=C.AssetsConfig(
            assets_dir=str(Path(args.view) / "assets"), asset_id="libero_goal8"),
        base_config=data_base,
        extra_delta_transform=False,
    )
    training = protocol["training"]
    save_steps = checkpoint_steps(protocol)
    base_freeze_filter = model.get_freeze_filter()
    freeze_filter = base_freeze_filter
    if model_settings["freeze_vision_encoder"]:
        freeze_filter = nnx.Any(
            base_freeze_filter,
            nnx_utils.PathRegex(r".*PaliGemma/img.*"),
        )
    schedule = opt.CosineDecaySchedule(
        warmup_steps=int(training["warmup_steps"]),
        peak_lr=float(training["peak_lr"]),
        decay_steps=int(training["cosine_decay_steps"]),
        decay_lr=float(training["final_lr"]),
    )
    return dataclasses.replace(
        base,
        name="pi05_libero",
        project_name="libero-pi05",
        exp_name=args.exp_name,
        model=model,
        data=data,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            str(Path(args.base_checkpoint) / "params")),
        freeze_filter=freeze_filter,
        batch_size=int(training["global_batch_size"]),
        num_train_steps=int(target_steps),
        num_workers=int(training["data_loader_workers"]),
        save_interval=min(save_steps),
        keep_period=1,
        seed=int(training["seed"]),
        ema_decay=float(training["ema_decay"]),
        fsdp_devices=int(args.fsdp_devices),
        assets_base_dir=str(Path(args.view) / "assets"),
        checkpoint_base_dir=str(run_root / "checkpoints"),
        lr_schedule=schedule,
        optimizer=opt.AdamW(
            b1=float(training["adamw_b1"]),
            b2=float(training["adamw_b2"]),
            eps=float(training["adamw_eps"]),
            weight_decay=float(training["adamw_weight_decay"]),
            clip_gradient_norm=float(training["gradient_clip_norm"]),
        ),
        overwrite=args.mode in {"initial", "probe"},
        resume=args.mode in {"resume", "probe_resume"},
        wandb_enabled=False,
        log_interval=1,
        policy_metadata={
            "suite": protocol["suite"],
            "task_id": protocol["task_id"],
            "dataset_task_index": protocol["dataset_task_index"],
            "task": protocol["task"],
            "view_sha256": view["view_sha256"],
            "base": protocol["base"]["name"],
            "scope": protocol["model"]["scope"],
            "protocol_sha256": protocol["_sha256"],
            "checkpoint_steps": list(save_steps),
            "retain_all_saved_steps": True,
        },
    )


def _identity(args, protocol: dict, view: dict, runtime: dict, base_tree: dict) -> dict:
    return {
        "protocol_sha256": protocol["_sha256"],
        "image_digest": args.image_digest,
        "runtime": runtime,
        "view_sha256": view["view_sha256"],
        "norm_stats_sha256": view["norm_stats"]["sha256"],
        "base_checkpoint_tree": base_tree,
        "suite": protocol["suite"],
        "task_id": protocol["task_id"],
        "dataset_task_index": protocol["dataset_task_index"],
        "task": protocol["task"],
        "model": protocol["model"],
        "data": protocol["data"],
        "training": protocol["training"],
        "checkpoint": protocol["checkpoint"],
        "fsdp_devices": int(args.fsdp_devices),
        "data_loader_workers": int(protocol["training"]["data_loader_workers"]),
        "exp_name": args.exp_name,
    }


def validate_saved_identity(saved: dict, current: dict) -> None:
    old = saved.get("identity")
    if not isinstance(old, dict):
        raise ValueError("saved run has no bound identity")
    old_hash = saved.get("identity_sha256")
    if not old_hash or canonical_sha256(old) != old_hash:
        raise ValueError("saved run identity hash is invalid")
    current_hash = canonical_sha256(current)
    if old_hash != current_hash:
        differences = sorted(
            key for key in set(old) | set(current) if old.get(key) != current.get(key)
        )
        raise ValueError(
            "resume identity differs from config snapshot: " + ", ".join(differences))


def _snapshot(mode: str, run_id: str, identity: dict, run_root: Path,
              protocol_path: Path, target_steps: int) -> dict:
    payload = {
        "schema_version": 1,
        "mode": mode,
        "run_id": run_id,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_root": str(run_root),
        "protocol_path": str(protocol_path),
        "target_steps_for_this_process": int(target_steps),
        "identity": identity,
        "checkpoint_steps": list(identity["checkpoint"]["save_steps"]),
        "validation": identity["validation"],
    }
    payload["identity_sha256"] = canonical_sha256(identity)
    return payload


def _load_train_module():
    path = Path("/opt/openpi/scripts/train.py")
    if not path.is_file():
        raise FileNotFoundError("pinned OpenPI scripts/train.py is unavailable")
    spec = importlib.util.spec_from_file_location("openpi_train_entry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def _consume_batches(data_iter, count: int) -> None:
    for _ in range(count):
        next(data_iter)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def build_validation_plan(protocol: dict, view: dict, sample_keys) -> dict:
    validation = protocol["validation"]
    plan = libero_data.fixed_validation_sample_plan(
        int(validation["expected_samples"]),
        int(protocol["training"]["global_batch_size"]),
        int(validation["seed"]),
    )
    plan.update({
        "protocol_sha256": protocol["_sha256"],
        "view_sha256": view["view_sha256"],
        "norm_stats_sha256": view["norm_stats"]["sha256"],
        "norm_stats_source": validation["norm_stats_source"],
        "train_argument": bool(validation["train_argument"]),
        "gradient_updates": int(validation["gradient_updates"]),
        "loss": validation["loss"],
        "loss_definition": VALIDATION_LOSS_DEFINITION,
        "evaluate_after_checkpoint": bool(validation["evaluate_after_checkpoint"]),
        "sample_key_sha256": validation_rng_sha256(sample_keys),
    })
    plan.update(validation_randomness_digests(
        sample_keys[: int(validation["expected_samples"])],
        int(protocol["data"]["action_horizon"]),
        int(protocol["data"]["internal_action_dim"]),
    ))
    return plan


def register_validation_plan(
    run_root: Path, plan: dict, *, fresh: bool
) -> tuple[dict, str]:
    path = run_root / "validation_plan.json"
    public = public_validation_plan(plan)
    public["validation_plan_sha256"] = canonical_sha256(public)
    if fresh:
        if path.exists():
            raise ValueError(f"refusing to overwrite validation plan: {path}")
        _write_json(path, public)
    else:
        if not path.is_file():
            raise ValueError("resume requires validation_plan.json")
        saved = json.loads(path.read_text(encoding="utf-8"))
        claimed = saved.get("validation_plan_sha256")
        payload = {key: value for key, value in saved.items()
                   if key != "validation_plan_sha256"}
        if not claimed or canonical_sha256(payload) != claimed:
            raise ValueError("saved validation plan hash is invalid")
        if saved != public:
            raise ValueError("resume validation plan differs from the registered plan")
    return public, public["validation_plan_sha256"]


def _prepare_run_dir(args, run_root: Path, snapshot_name: str) -> None:
    snapshot = run_root / snapshot_name
    if args.mode in {"initial", "probe"}:
        if run_root.exists() and any(run_root.iterdir()):
            raise ValueError(f"refusing non-empty fresh run directory: {run_root}")
        if snapshot.exists():
            raise ValueError(f"refusing to overwrite {snapshot_name}")
    elif not snapshot.is_file():
        raise ValueError(f"{args.mode} requires {snapshot_name}")
    run_root.mkdir(parents=True, exist_ok=True)


def _target_steps(args, protocol: dict, saved: dict | None) -> int:
    if args.mode in {"initial", "resume"}:
        target = int(protocol["training"]["optimizer_updates"])
    else:
        target = int(args.probe_steps)
        if args.mode == "probe_resume":
            restored = int(saved.get("target_steps_for_this_process", -1))
            if target <= restored:
                raise ValueError("probe_resume target must exceed the saved probe step")
    if args.mode in {"probe", "probe_resume"} and not 1 <= target <= 5:
        raise ValueError("probe targets must be 1..5 optimizer updates")
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("initial", "probe", "probe_resume", "resume"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--probe-steps", type=int, default=2)
    parser.add_argument("--fsdp-devices", type=int, choices=(2, 4), required=True)
    parser.add_argument("--image-digest", required=True)
    args = parser.parse_args()

    protocol = load_protocol(args.protocol, args.expected_protocol_sha256)
    if not args.exp_name.startswith(protocol["experiment_prefix"]):
        raise ValueError(
            f"experiment name must start with {protocol['experiment_prefix']}")
    if not args.run_id.startswith(protocol["experiment_prefix"]):
        raise ValueError(f"run id must start with {protocol['experiment_prefix']}")
    if args.image_digest != protocol["runtime"]["image_digest"]:
        raise ValueError("runtime image digest differs from the selected protocol")

    view = libero_data.load_view(args.view)
    expected_view = protocol["data"]
    checks = {
        "view_sha256": view["view_sha256"],
        "norm_stats_sha256": view["norm_stats"]["sha256"],
        "suite": view["suite"],
        "task_id": view["task_id"],
        "dataset_task_index": view["dataset_task_index"],
        "task": view["task"],
        "train_episodes": len(view["train_episode_ids"]),
        "validation_episodes": len(view["validation_episode_ids"]),
        "train_frames": sum(
            row["frames"] for row in view["episodes"] if row["split"] == "train"),
        "validation_frames": sum(
            row["frames"] for row in view["episodes"] if row["split"] == "validation"),
        "fps": view["fps"],
        "state_dim": view["state_dim"],
        "action_dim": view["action_dim"],
        "action_horizon": view["action_horizon"],
        "action_semantics": action_semantics_token(
            view["constraints"]["action_semantics"]),
        "validation_in_norm_stats": view["constraints"]["validation_in_norm_stats"],
    }
    expected = {
        "view_sha256": expected_view["view_sha256"],
        "norm_stats_sha256": expected_view["norm_stats_sha256"],
        "suite": protocol["suite"],
        "task_id": protocol["task_id"],
        "dataset_task_index": protocol["dataset_task_index"],
        "task": protocol["task"],
        "train_episodes": expected_view["train_episodes"],
        "validation_episodes": expected_view["validation_episodes"],
        "train_frames": expected_view["train_frames"],
        "validation_frames": expected_view["validation_frames"],
        "fps": expected_view["fps"],
        "state_dim": expected_view["state_dim"],
        "action_dim": expected_view["action_dim"],
        "action_horizon": expected_view["action_horizon"],
        "action_semantics": expected_view["action_semantics"],
        "validation_in_norm_stats": expected_view["validation_in_norm_stats"],
    }
    bad = [key for key in expected if checks[key] != expected[key]]
    if bad:
        raise ValueError("data view differs from protocol: " + ", ".join(bad))

    base_tree = sha256_tree(args.base_checkpoint)
    if base_tree["sha256"] != protocol["base"]["tree_sha256"]:
        raise ValueError("pi05_base tree hash differs from the registered base")
    runtime = {
        "openpi_src": sha256_tree("/opt/openpi/src/openpi"),
        "openpi_scripts": sha256_tree("/opt/openpi/scripts"),
        "libero_source": sha256_tree("/opt/openpi/third_party/libero"),
    }
    for key, value in runtime.items():
        if value != protocol["runtime"][key]:
            raise ValueError(f"runtime source {key} differs from the registered image")

    run_root = args.run_root.resolve()
    snapshot_name = (
        "config_probe.json"
        if MODE_KINDS[args.mode] == "probe"
        else "config_initial.json"
    )
    _prepare_run_dir(args, run_root, snapshot_name)
    saved = (
        json.loads((run_root / snapshot_name).read_text(encoding="utf-8"))
        if args.mode in {"resume", "probe_resume"}
        else None
    )
    target_steps = _target_steps(args, protocol, saved)
    current_identity = _identity(args, protocol, view, runtime, base_tree)
    if saved is not None:
        validate_saved_identity(saved, current_identity)
        if saved.get("run_id") != args.run_id:
            raise ValueError("resume run_id differs from the saved run")

    config = make_config(args, protocol, view, run_root, target_steps)
    train, train_path = _load_train_module()

    import jax
    from openpi.training import sharding

    jax.config.update(
        "jax_compilation_cache_dir",
        os.environ.get(
            "JAX_COMPILATION_CACHE_DIR", str(Path("~/.cache/jax").expanduser())),
    )
    device_count = jax.device_count()
    if device_count != args.fsdp_devices:
        raise ValueError(
            f"expected exactly {args.fsdp_devices} visible devices, got {device_count}")
    if config.batch_size % device_count:
        raise ValueError("global batch must divide evenly across visible devices")
    mesh = sharding.make_mesh(args.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec())
    manager, resuming = train._checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    loader = libero_data.create_filtered_data_loader(
        config,
        view_path=args.view,
        data_root=args.data_root,
        sharding=data_sharding,
        shuffle=True,
        split="train",
    )
    data_iter = iter(loader)
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    validation_settings = protocol["validation"]
    validation_sample_count = int(validation_settings["expected_samples"])
    validation_probe_plan = libero_data.fixed_validation_sample_plan(
        validation_sample_count,
        int(protocol["training"]["global_batch_size"]),
        int(validation_settings["seed"]),
    )
    validation_keys = validation_rngs(
        int(validation_settings["seed"]),
        validation_probe_plan["_sampler_indices"],
    )
    validation_plan = build_validation_plan(protocol, view, validation_keys)
    registered_validation_plan, validation_plan_sha256 = register_validation_plan(
        run_root,
        validation_plan,
        fresh=args.mode in {"initial", "probe"},
    )
    state, state_sharding = train.init_train_state(
        config, init_rng, mesh, resume=resuming)
    state = jax.block_until_ready(state)
    if resuming:
        state = train._checkpoints.restore_state(manager, state, loader)
        state = jax.block_until_ready(state)
    start_step = int(jax.device_get(state.step))
    if args.mode in {"initial", "probe"} and start_step != 0:
        raise ValueError("fresh run restored a non-zero TrainState.step")
    if start_step > target_steps:
        raise ValueError("restored TrainState.step exceeds the requested target")
    registered_checkpoint_steps = checkpoint_steps(protocol)

    snapshot = _snapshot(
        str(args.mode), args.run_id, current_identity, run_root,
        Path(protocol["_path"]), target_steps)
    snapshot["start_train_state_step"] = start_step
    snapshot["save_interval"] = min(registered_checkpoint_steps)
    snapshot["train_episode_ids"] = view["train_episode_ids"]
    snapshot["validation_episode_ids"] = view["validation_episode_ids"]
    snapshot["validation_plan_sha256"] = validation_plan_sha256
    if args.mode in {"initial", "probe"}:
        _write_json(run_root / snapshot_name, snapshot)
    else:
        _write_json(run_root / f"config_{args.mode}.json", snapshot)

    ptrain_step = jax.jit(
        functools.partial(train.train_step, config),
        in_shardings=(replicated_sharding, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    pvalidation = jax.jit(
        validation_sample_losses,
        in_shardings=(state_sharding, data_sharding, data_sharding, data_sharding),
        out_shardings=data_sharding,
    )
    validation_loader = None

    def ensure_validation_loader():
        nonlocal validation_loader
        if validation_loader is None:
            validation_loader, loader_plan = libero_data.create_fixed_validation_loader(
                config,
                view_path=args.view,
                data_root=args.data_root,
                sharding=data_sharding,
                seed=int(validation_settings["seed"]),
                expected_samples=validation_sample_count,
            )
            if (
                loader_plan["_sampler_indices"]
                != validation_plan["_sampler_indices"]
                or loader_plan["valid_samples_per_batch"]
                != validation_plan["valid_samples_per_batch"]
            ):
                raise ValueError("validation loader differs from the frozen plan")
        return validation_loader

    def validate_saved_checkpoint(new_step: int, checkpoint_info: dict) -> dict:
        import numpy as np

        if int(jax.device_get(state.step)) != int(new_step):
            raise RuntimeError("validation checkpoint step differs from TrainState.step")
        path = run_root / f"validation_loss_{new_step}.json"
        if path.is_file():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if (
                int(saved.get("train_state_step", -1)) != int(new_step)
                or saved.get("validation_plan_sha256") != validation_plan_sha256
            ):
                raise ValueError(f"validation record is inconsistent: {path.name}")
            return saved

        loader_to_validate = ensure_validation_loader()

        def evaluate_batch(batch, batch_index):
            start = int(batch_index) * int(config.batch_size)
            stop = start + int(config.batch_size)
            with sharding.set_mesh(mesh):
                values = pvalidation(
                    state,
                    batch[0],
                    batch[1],
                    validation_keys[start:stop],
                )
            return np.asarray(jax.device_get(values), dtype=np.float32).reshape(-1)

        result = collect_validation_loss(
            loader_to_validate,
            validation_plan["valid_samples_per_batch"],
            evaluate_batch,
            progress=lambda completed, total: print(json.dumps({
                "event": "validation_progress",
                "train_state_step": int(new_step),
                "batches_completed": int(completed),
                "batches_total": int(total),
            }), flush=True),
        )
        if result["samples_evaluated"] != validation_sample_count:
            raise RuntimeError(
                f"validation evaluated {result['samples_evaluated']} of "
                f"{validation_sample_count} samples")
        if int(jax.device_get(state.step)) != int(new_step):
            raise RuntimeError("validation changed TrainState.step")
        payload = {
            "schema_version": 1,
            "event": "validation_loss",
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "split": "validation",
            "train_state_step": int(new_step),
            "checkpoint_saved": True,
            "checkpoint_info": checkpoint_info,
            "samples_expected": validation_sample_count,
            **result,
            "validation_plan_sha256": validation_plan_sha256,
            "view_sha256": view["view_sha256"],
            "norm_stats_sha256": view["norm_stats"]["sha256"],
            "norm_stats_source": "train_only",
            "validation_seed": int(validation_settings["seed"]),
            "sample_key_sha256": validation_plan["sample_key_sha256"],
            "noise_sha256": validation_plan["noise_sha256"],
            "flow_time_sha256": validation_plan["flow_time_sha256"],
            "train_false": True,
            "loss": "flow_matching_mse",
            "loss_definition": VALIDATION_LOSS_DEFINITION,
            "gradient_updates": 0,
        }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
        print(json.dumps(payload, sort_keys=True), flush=True)
        return payload

    if (
        MODE_KINDS[args.mode] == "formal"
        and start_step in registered_checkpoint_steps
    ):
        manager.wait_until_finished()
        checkpoint_metadata_step = int(manager.latest_step())
        if checkpoint_metadata_step != start_step:
            raise RuntimeError(
                "restored checkpoint metadata step differs from TrainState.step: "
                f"{checkpoint_metadata_step} != {start_step}")
        validate_saved_checkpoint(
            start_step,
            {
                "train_state_step": start_step,
                "checkpoint_metadata_step": checkpoint_metadata_step,
                "retained_steps": [int(value) for value in manager.all_steps()],
                "retain_all_saved_steps": True,
                "full_train_state": True,
            },
        )

    _consume_batches(data_iter, start_step)
    batch = next(data_iter)
    devices = list(jax.devices())
    print(json.dumps({
        "mode": args.mode,
        "run_id": args.run_id,
        "start_train_state_step": start_step,
        "target_train_state_step": target_steps,
        "global_batch_size": config.batch_size,
        "device_count": len(devices),
        "device_ids": [device.id for device in devices],
        "fsdp_devices": config.fsdp_devices,
        "mesh_axis_names": list(mesh.axis_names),
        "mesh_shape": {
            str(key): int(value) for key, value in mesh.shape.items()},
        "data_parallel_replicas": len(devices) // config.fsdp_devices,
        "batch_per_device": config.batch_size // len(devices),
        "checkpoint_dir": str(config.checkpoint_dir),
        "checkpoint_steps": list(registered_checkpoint_steps),
        "validation": validation_settings,
        "validation_plan_sha256": validation_plan_sha256,
        "train_episode_ids": view["train_episode_ids"],
        "validation_episode_ids": view["validation_episode_ids"],
        "train_entry": str(train_path),
        "config": _jsonable(config),
    }), flush=True)

    for _ in range(start_step, target_steps):
        old_step = int(jax.device_get(state.step))
        with sharding.set_mesh(mesh):
            state, info = ptrain_step(train_rng, state, batch)
        finite, metrics = finite_metrics(info)
        new_step = int(jax.device_get(state.step))
        if not finite:
            raise FloatingPointError(
                f"non-finite training metrics at update {old_step}: {metrics}")
        if new_step != old_step + 1:
            raise RuntimeError(
                f"TrainState did not advance exactly once: {old_step} -> {new_step}")
        if not metrics.get("grad_norm", 0.0) > 0.0:
            raise RuntimeError(
                f"zero gradient / no effective update at update {old_step}")
        print(
            f"Step {new_step}: loss={metrics['loss']:.8f}, "
            f"grad_norm={metrics['grad_norm']:.8f}, "
            f"param_norm={metrics['param_norm']:.8f}",
            flush=True,
        )
        print(json.dumps({
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "scheduled_step": old_step,
            "train_state_step": new_step,
            "metrics": metrics,
        }), flush=True)
        should_save = (
            new_step == target_steps
            or (
                MODE_KINDS[args.mode] == "formal"
                and new_step in registered_checkpoint_steps
            )
        )
        if should_save:
            if int(jax.device_get(state.step)) != new_step:
                raise RuntimeError("checkpoint step differs from TrainState.step")
            train._checkpoints.save_state(manager, state, loader, new_step)
            manager.wait_until_finished()
            checkpoint_metadata_step = int(manager.latest_step())
            if checkpoint_metadata_step != new_step:
                raise RuntimeError(
                    "checkpoint manager latest step differs from TrainState.step: "
                    f"{checkpoint_metadata_step} != {new_step}")
            retained_steps = [int(value) for value in manager.all_steps()]
            checkpoint_info = {
                "train_state_step": new_step,
                "checkpoint_metadata_step": checkpoint_metadata_step,
                "retained_steps": retained_steps,
                "retain_all_saved_steps": True,
                "full_train_state": True,
            }
            _write_json(
                run_root / f"checkpoint_steps_{new_step}.json", checkpoint_info)
            if MODE_KINDS[args.mode] == "formal":
                validate_saved_checkpoint(new_step, checkpoint_info)
        batch = next(data_iter)

    manager.wait_until_finished()
    final_step = int(jax.device_get(state.step))
    if final_step != target_steps:
        raise RuntimeError(
            f"final TrainState.step {final_step} differs from target {target_steps}")
    payload = {
        "mode": args.mode,
        "run_id": args.run_id,
        "start_train_state_step": start_step,
        "final_train_state_step": final_step,
        "target_train_state_step": target_steps,
        "checkpoint_dir": str(config.checkpoint_dir),
        "view_sha256": view["view_sha256"],
        "protocol_sha256": protocol["_sha256"],
        "base_checkpoint_tree_sha256": base_tree["sha256"],
        "image_digest": args.image_digest,
    }
    _write_json(run_root / f"done_{args.mode}.json", payload)
    print(json.dumps(payload), flush=True)
    manager.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
