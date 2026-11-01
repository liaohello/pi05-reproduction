#!/usr/bin/env python3
"""Deterministic validation loss support for the Scene1 SO101 trainer."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Sequence

LOSS_DEFINITION = (
    "model.compute_loss(rng, observation, actions, train=False); "
    "mean over the 10-step horizon and action dimensions, then equal-weight "
    "mean over every valid window in the fixed validation plan"
)
EXPECTED_CHECKPOINT_STEPS = (
    500, 1000, 1500, 2000, 3000, 6000, 9000,
    12000, 15000, 18000, 21000, 24000, 25000,
)


def fixed_sample_plan(
    sample_indices: Sequence[int], global_batch_size: int, seed: int
) -> dict:
    unique = [int(value) for value in sample_indices]
    sample_count = len(unique)
    if sample_count < 1 or global_batch_size < 1 or seed < 0:
        raise ValueError("validation sample count, batch size, and seed are invalid")
    if len(set(unique)) != len(unique) or any(value < 0 for value in unique):
        raise ValueError("validation sample indices must be unique and non-negative")
    batch_count = (sample_count + global_batch_size - 1) // global_batch_size
    padded_count = batch_count * global_batch_size
    padding = padded_count - sample_count
    sampler_indices = unique + unique[:padding]
    valid_per_batch = [
        min(global_batch_size, sample_count - batch * global_batch_size)
        for batch in range(batch_count)
    ]
    return {
        "schema_version": 1,
        "seed": int(seed),
        "sample_count": int(sample_count),
        "global_batch_size": int(global_batch_size),
        "batch_count": int(batch_count),
        "padded_count": int(padded_count),
        "padding": int(padding),
        "valid_samples_per_batch": valid_per_batch,
        "sample_order": "episode_index_then_frame_index",
        "sample_weighting": "uniform_per_valid_sample",
        "sample_order_sha256": hashlib.sha256(
            json.dumps(unique, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "sampler_indices_sha256": hashlib.sha256(
            json.dumps(sampler_indices, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "_sampler_indices": sampler_indices,
    }


def fixed_validation_sample_plan(
    sample_count: int, global_batch_size: int, seed: int
) -> dict:
    return fixed_sample_plan(range(int(sample_count)), global_batch_size, seed)


def balanced_probe_indices(
    sample_metadata: Sequence[dict], count: int, seed: int
) -> tuple[list[int], dict]:
    """Select a deterministic episode-balanced subset of validation windows."""
    if count < 1:
        raise ValueError("probe sample count must be positive")
    by_episode: dict[int, list[int]] = {}
    for row in sample_metadata:
        by_episode.setdefault(int(row["episode_id"]), []).append(
            int(row["sample_index"]))
    if not by_episode or count < len(by_episode):
        raise ValueError("probe sample count cannot cover every validation episode")

    def episode_priority(episode_id: int) -> str:
        return hashlib.sha256(
            f"scene1-probe:{int(seed)}:{episode_id}".encode("ascii")
        ).hexdigest()

    base, remainder = divmod(int(count), len(by_episode))
    extra = set(sorted(by_episode, key=episode_priority)[:remainder])
    selected: list[int] = []
    per_episode: dict[str, int] = {}
    for episode_id, indices in sorted(by_episode.items()):
        take = base + int(episode_id in extra)
        ranked = sorted(
            indices,
            key=lambda index: (
                hashlib.sha256(
                    f"scene1-probe:{int(seed)}:{int(index)}".encode("ascii")
                ).hexdigest(),
                index,
            ),
        )
        if take > len(ranked):
            raise ValueError(f"episode {episode_id} has too few validation windows")
        selected.extend(ranked[:take])
        per_episode[str(episode_id)] = take
    selected.sort()
    if len(selected) != count or len(set(selected)) != count:
        raise ValueError("probe selection did not produce unique samples")
    return selected, {
        "per_episode": per_episode,
        "covered_episodes": len(by_episode),
        "balanced": True,
    }


def public_plan(plan: dict) -> dict:
    return {key: value for key, value in plan.items() if not key.startswith("_")}


def validation_batch_keys(seed: int, batch_count: int):
    import jax
    import jax.numpy as jnp

    base = jax.random.key(int(seed))
    indices = jnp.arange(int(batch_count), dtype=jnp.uint32)
    return jax.vmap(lambda index: jax.random.fold_in(base, index))(indices)


def validation_key_sha256(batch_keys) -> str:
    import jax
    import numpy as np

    values = np.asarray(jax.device_get(jax.random.key_data(batch_keys)), dtype="<u4")
    return hashlib.sha256(values.tobytes()).hexdigest()


def randomness_digests(
    batch_keys, batch_size: int, action_horizon: int, action_dim: int
) -> dict:
    import jax
    import jax.numpy as jnp
    import numpy as np

    def one_batch(batch_key):
        _, noise_rng, time_rng = jax.random.split(batch_key, 3)
        noise = jax.random.normal(
            noise_rng, (int(batch_size), int(action_horizon), int(action_dim)))
        flow_time = (
            jax.random.beta(time_rng, 1.5, 1, (int(batch_size),)) * 0.999 + 0.001)
        return noise, flow_time

    noises, times = jax.vmap(one_batch)(batch_keys)
    noise = np.asarray(jax.device_get(noises), dtype="<f4")
    flow_time = np.asarray(jax.device_get(times), dtype="<f4")
    return {
        "noise_sha256": hashlib.sha256(noise.tobytes()).hexdigest(),
        "flow_time_sha256": hashlib.sha256(flow_time.tobytes()).hexdigest(),
        "noise_shape": list(noise.shape),
        "flow_time_shape": list(flow_time.shape),
    }


def validation_batch_losses(state, observation, actions, batch_key):
    """Call the model's Flow-only validation loss once per fixed batch."""
    import flax.nnx as nnx
    import jax.numpy as jnp

    model = nnx.merge(state.model_def, state.params)
    model.eval()
    loss_fn = getattr(model, "compute_flow_loss", model.compute_loss)
    losses = loss_fn(batch_key, observation, actions, train=False)
    return jnp.mean(losses, axis=-1) if losses.ndim > 1 else losses


def weighted_validation_loss(
    per_sample_losses: Sequence[Sequence[float]],
    valid_samples_per_batch: Sequence[int],
) -> dict:
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
    if weight_sum <= 0:
        raise ValueError("validation produced no valid samples")
    return {
        "validation_loss": weighted_sum / weight_sum,
        "weighted_loss_sum": weighted_sum,
        "weight_sum": weight_sum,
        "samples_evaluated": int(weight_sum),
    }


def collect_validation_loss(
    batches,
    valid_samples_per_batch: Sequence[int],
    evaluate_batch,
    progress=None,
) -> dict:
    rows: list[list[float]] = []
    for batch_index, batch in enumerate(batches):
        if batch_index >= len(valid_samples_per_batch):
            raise ValueError("validation loader returned more batches than planned")
        rows.append([float(value) for value in evaluate_batch(batch, batch_index)])
        if progress is not None:
            progress(batch_index + 1, len(valid_samples_per_batch))
    if len(rows) != len(valid_samples_per_batch):
        raise ValueError(
            f"validation loader returned {len(rows)} batches, "
            f"expected {len(valid_samples_per_batch)}")
    result = weighted_validation_loss(rows, valid_samples_per_batch)
    result["batches_evaluated"] = len(rows)
    return result


def canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def build_plan(
    *,
    sample_indices: Sequence[int],
    global_batch_size: int,
    seed: int,
    metric: str,
    validation_every: int | None,
    action_horizon: int,
    internal_action_dim: int,
    view_sha256: str,
    norm_stats_sha256: str,
    batch_keys,
    selection: dict | None = None,
    episode_ids: Sequence[int] | None = None,
) -> dict:
    plan = fixed_sample_plan(sample_indices, global_batch_size, seed)
    plan.update({
        "split": "validation",
        "metric": metric,
        "validation_every_updates": (
            int(validation_every) if validation_every is not None else None),
        "norm_stats_source": "train_only",
        "weights_source": "TrainState.params",
        "train_false": True,
        "gradient_updates": 0,
        "loss": "flow_matching_mse",
        "loss_definition": LOSS_DEFINITION,
        "view_sha256": view_sha256,
        "norm_stats_sha256": norm_stats_sha256,
        "validation_episode_ids": (
            [int(value) for value in episode_ids] if episode_ids is not None else None),
        "batch_key_sha256": validation_key_sha256(batch_keys),
    })
    if selection is not None:
        plan["selection"] = selection
        plan["selected_sample_indices"] = [int(value) for value in sample_indices]
    plan.update(randomness_digests(
        batch_keys, global_batch_size, action_horizon, internal_action_dim))
    return plan


def register_plan(
    run_root: Path, plan: dict, *, name: str, fresh: bool
) -> tuple[dict, str]:
    path = Path(run_root) / name
    public = public_plan(plan)
    public["validation_plan_sha256"] = canonical_sha256(public)
    if fresh:
        if path.exists():
            raise ValueError(f"refusing to overwrite validation plan: {path}")
        path.write_text(
            json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        if not path.is_file():
            raise ValueError(f"resume requires {name}")
        saved = json.loads(path.read_text(encoding="utf-8"))
        claimed = saved.get("validation_plan_sha256")
        payload = {key: value for key, value in saved.items()
                   if key != "validation_plan_sha256"}
        if not claimed or canonical_sha256(payload) != claimed:
            raise ValueError("saved validation plan hash is invalid")
        if saved != public:
            raise ValueError("resume validation plan differs from the registered plan")
    return public, public["validation_plan_sha256"]


def archive_future_validation_records(
    run_root: Path, restore_step: int, *, archive_name: str
) -> list[str]:
    """Preserve speculative post-checkpoint probe files when resuming an older state."""
    archive = Path(run_root) / "resume_validation_archive" / archive_name
    moved = []
    for metric in ("val_loss_probe", "val_loss_full"):
        for path in sorted(Path(run_root).glob(f"{metric}_*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                step = int(payload["train_state_step"])
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                raise ValueError(f"malformed validation record: {path.name}") from None
            if step <= int(restore_step):
                continue
            archive.mkdir(parents=True, exist_ok=True)
            destination = archive / path.name
            if destination.exists():
                raise ValueError(f"validation archive would overwrite {destination.name}")
            shutil.move(str(path), str(destination))
            moved.append(str(destination))
    if moved:
        manifest = {
            "schema_version": 1,
            "restore_step": int(restore_step),
            "reason": "records were ahead of the restored Checkpoint",
            "moved": [Path(value).name for value in moved],
        }
        (archive / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return moved


def checkpoint_steps(config: dict, updates: int) -> tuple[int, ...]:
    values = tuple(int(value) for value in
                   config["checkpointing"]["checkpoint_steps"])
    if not values or tuple(sorted(set(values))) != values:
        raise ValueError("checkpoint_steps must be positive, sorted, and unique")
    if values[-1] != int(updates) or any(value <= 0 for value in values):
        raise ValueError("checkpoint_steps must end at the optimizer update target")
    if values != EXPECTED_CHECKPOINT_STEPS:
        raise ValueError("checkpoint_steps differ from the registered Scene1 schedule")
    return values


def validation_steps(updates: int, every: int) -> tuple[int, ...]:
    if every < 1 or updates < every:
        raise ValueError("invalid validation interval")
    return tuple(range(int(every), int(updates) + 1, int(every)))
