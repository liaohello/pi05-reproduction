#!/usr/bin/env python3
"""Run real global-batch-8 clean60 updates, optimizer steps and a no-save probe."""
from __future__ import annotations

import argparse
import functools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.traverse_util import flatten_dict

SCRIPTS = "/work/scripts/so101"
sys.path.insert(0, SCRIPTS)

import openpi_so101  # noqa: E402
import openpi.training.sharding as sharding  # noqa: E402


def nvidia_rows() -> list[dict]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,memory.total,memory.used,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "nvidia-smi failed")
    rows = []
    for line in result.stdout.splitlines():
        uuid, total, used, free = [part.strip() for part in line.split(",", 3)]
        rows.append({"uuid": uuid, "memory_total_mib": int(total),
                     "memory_used_mib": int(used), "memory_free_mib": int(free)})
    return rows


def memory_stats(devices) -> dict:
    rows = {}
    for device in devices:
        stats = dict(device.memory_stats() or {})
        rows[str(device.id)] = {
            key: int(value) for key, value in stats.items()
            if key in {"bytes_limit", "bytes_in_use", "persistent_bytes",
                       "allocated_size", "reserved_size", "free_size"}}
    return rows


def shard_summary(tree) -> dict:
    rows = {}
    for leaf in jax.tree_util.tree_leaves(
            tree, is_leaf=lambda value: isinstance(value, jax.Array)):
        if not isinstance(leaf, jax.Array):
            continue
        for shard in leaf.addressable_shards:
            rows.setdefault(str(shard.device.id), []).append({
                "global_shape": list(leaf.shape),
                "local_shape": list(shard.data.shape),
                "dtype": str(leaf.dtype),
                "index": str(shard.index),
            })
    return rows


def replicated_evidence(value) -> dict:
    array = value
    if hasattr(array, "addressable_shards"):
        values = [np.asarray(shard.data).tolist() for shard in array.addressable_shards]
        return {"shards": values, "all_equal": all(item == values[0] for item in values[1:])}
    return {"value": np.asarray(jax.device_get(array)).tolist(), "all_equal": True}


def group_of(path: str) -> str:
    if re_full(path, r".*llm.*_1.*"):
        return "action_expert"
    if "/img/" in path or "/img." in path:
        return "vision_encoder"
    if "lora" in path:
        return "language_lora"
    if "llm" in path:
        return "language_backbone"
    for name in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"):
        if path.startswith(name + "/") or path == name:
            return f"projection:{name}"
    return "other"


def re_full(value: str, pattern: str) -> bool:
    import re

    return bool(re.fullmatch(pattern, value))


def load_train_module():
    import importlib.util

    path = Path("/opt/openpi/scripts/train.py")
    spec = importlib.util.spec_from_file_location("openpi_train_entry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-view", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--base-checkpoint", default="/base")
    parser.add_argument("--out", required=True)
    parser.add_argument("--fsdp-devices", type=int, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    if args.batch_size != 8:
        raise ValueError("this probe preserves the registered global batch of 8")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": 1, "label": args.label,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "formal_training": False, "checkpoint_written": False,
        "requested": {"global_batch": args.batch_size, "steps": args.steps,
                      "fsdp_devices": args.fsdp_devices, "seed": args.seed},
    }
    baseline = nvidia_rows()
    report["gpu_baseline"] = baseline
    usage_free = shutil.disk_usage(out)
    report["disk_preflight"] = {
        "free_mib": usage_free.free // (1024 * 1024),
        "one_checkpoint_reference_mib": 12482,
        "formal_checkpoints": 5,
        "formal_estimated_mib": 12482 * 5,
        "checkpoint_probe_possible_without_cleanup": usage_free.free // (1024 * 1024) >= 12482,
    }

    stop = threading.Event()
    peak = {row["uuid"]: {"used_mib": row["memory_used_mib"],
                          "free_mib": row["memory_free_mib"]} for row in baseline}

    def sample():
        while not stop.is_set():
            try:
                for row in nvidia_rows():
                    entry = peak.setdefault(row["uuid"], {"used_mib": 0, "free_mib": 999999})
                    entry["used_mib"] = max(entry["used_mib"], row["memory_used_mib"])
                    entry["free_mib"] = min(entry["free_mib"], row["memory_free_mib"])
            except Exception as error:
                report.setdefault("monitor_errors", []).append(
                    f"{type(error).__name__}: {error}")
            stop.wait(0.5)

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    stage = "imports"
    failure = None
    try:
        registered = json.loads(Path(args.train_config).read_text(encoding="utf-8"))
        train = load_train_module()
        devices = list(jax.local_devices())
        if len(devices) != args.fsdp_devices:
            raise RuntimeError(f"expected {args.fsdp_devices} visible devices, got {len(devices)}")
        if args.batch_size % len(devices):
            raise RuntimeError("global batch is not divisible by the visible devices")
        config = openpi_so101.make_train_config(
            mode="probe", run_root=out, train_view=Path(args.train_view),
            base_checkpoint=Path(args.base_checkpoint), exp_name=args.label,
            state_mode="on", steps_override=args.steps, save_interval=5000,
            batch_size=args.batch_size, lr_warmup_steps=500, peak_lr=5e-5,
            decay_steps=25000, decay_lr=5e-6, dataset_label="clean60_full_update_probe",
            phase="E4-T", num_workers=0, fsdp_devices=args.fsdp_devices,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m",
            freeze_vision_encoder=False, action_horizon=10, seed=args.seed,
            gradient_clip_norm=1.0, optimizer_name="AdamW", ema_decay=None,
            dtype="bfloat16")
        mesh = sharding.make_mesh(args.fsdp_devices)
        replicas = len(devices) // args.fsdp_devices
        data_sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        report["mesh"] = {
            "axis_names": list(mesh.axis_names),
            "shape": {str(key): int(value) for key, value in mesh.shape.items()},
            "fsdp_devices": args.fsdp_devices, "data_parallel_replicas": replicas,
            "global_batch_size": args.batch_size,
            "batch_per_data_replica": args.batch_size // replicas,
            "batch_per_device_shard": args.batch_size // len(devices),
            "batch_partitioned_across_all_devices": True,
            "fsdp_axis_replicates_batch": False,
            "device_uuids": [row["uuid"] for row in nvidia_rows()],
        }
        stage = "init_train_state"
        init_rng = jax.random.split(jax.random.key(config.seed))[1]
        t0 = time.time()
        state, state_sharding = train.init_train_state(
            config, init_rng, mesh, resume=False)
        jax.block_until_ready(state)
        report["init_wall_s"] = time.time() - t0
        report["init_memory_stats"] = memory_stats(devices)
        initial_tree = state.params.to_pure_dict()
        initial_flat = flatten_dict(initial_tree, sep="/")
        initial = {}
        for name, value in initial_flat.items():
            initial[name] = np.asarray(jax.device_get(value))
        initial_dtypes = {name: str(value.dtype) for name, value in initial.items()}
        trainable_initial = flatten_dict(
            state.params.filter(config.trainable_filter).to_pure_dict(), sep="/")
        initial_opt = jax.tree_util.tree_leaves(state.opt_state)
        report["parameter_tree"] = {
            "tensors": len(initial), "elements": int(sum(value.size for value in initial.values())),
            "dtypes": sorted(set(initial_dtypes.values())),
            "paths": [{"path": name, "shape": list(value.shape),
                       "dtype": str(value.dtype),
                       "trainable": name in trainable_initial}
                      for name, value in sorted(initial.items())],
        }
        report["optimizer_state"] = {
            "leaves": len(initial_opt),
            "elements": int(sum(value.size for value in initial_opt)),
            "dtypes": sorted({str(value.dtype) for value in initial_opt}),
        }
        report["sharding"] = {
            "params_by_device": shard_summary(state.params),
            "state_sharding_tree": str(state_sharding)[:20000],
        }
        stage = "batch"
        os.environ["SO101_TRAIN_VIEW"] = args.train_view
        raw_dataset = openpi_so101.SO101EpisodeDataset(
            Path(args.train_view), split="train", action_horizon=10)
        weights, sampling_audit = raw_dataset.weighted_sample_probabilities(
            registered["sampling_plan"])
        sample_indices = raw_dataset.precomputed_sample_indices(
            weights, args.steps * args.batch_size, seed=args.seed)
        sample_path = out / "probe_sample_ids.jsonl"
        sample_path.write_text("".join(f"{value}\n" for value in sample_indices),
                               encoding="utf-8")
        os.environ["SO101_TRAINING_SAMPLE_IDS"] = str(sample_path)
        loader = openpi_so101.create_so101_data_loader(
            config, sharding=data_sharding, shuffle=True, num_batches=args.steps)
        data_iter = iter(loader)
        first_batch = next(data_iter)
        jax.block_until_ready(first_batch)
        observation, actions = first_batch
        report["batch"] = {
            "opportunistic_batch_bytes_by_device": shard_summary(first_batch),
            "action_shape": list(actions.shape),
            "state_shape": list(observation.state.shape),
            "sampling_audit": sampling_audit,
            "first_8_sample_ids": sample_indices[:8],
        }
        stage = "gradient_audit"

        def gradient_metrics(rng, state, batch):
            model = nnx.merge(state.model_def, state.params)
            model.train()
            observation, actions = batch

            def loss_fn(model, rng, observation, actions):
                return jnp.mean(model.compute_loss(rng, observation, actions, train=True))

            diff_state = nnx.DiffState(0, config.trainable_filter)
            loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
                model, rng, observation, actions)
            flat = flatten_dict(grads.to_pure_dict(), sep="/")
            metrics = {
                name: jnp.stack([
                    jnp.all(jnp.isfinite(value)), jnp.max(jnp.abs(value)),
                    jnp.sum(jnp.abs(value)), jnp.asarray(value.size, dtype=jnp.float32),
                ])
                for name, value in flat.items()
            }
            return loss, optax.global_norm(grads), metrics

        pgrad = jax.jit(
            gradient_metrics,
            in_shardings=(replicated, state_sharding, data_sharding),
            out_shardings=(replicated, replicated,
                           {name: replicated for name in flatten_dict(
                               state.params.filter(config.trainable_filter).to_pure_dict(),
                               sep="/")}))
        with sharding.set_mesh(mesh):
            grad_info = pgrad(jax.random.fold_in(jax.random.key(config.seed), 0),
                              state, first_batch)
        loss_replicated = replicated_evidence(grad_info[0])
        norm_replicated = replicated_evidence(grad_info[1])
        grad_info = jax.device_get(grad_info)
        report["gradient_audit"] = {
            "loss": float(grad_info[0]), "global_norm": float(grad_info[1]),
            "loss_replicated": loss_replicated,
            "norm_replicated": norm_replicated,
            "groups": {
                name: {"finite": bool(value[0]), "max_abs": float(value[1]),
                       "sum_abs": float(value[2]), "elements": int(value[3])}
                for name, value in sorted(grad_info[2].items())},
        }

        stage = "update"
        ptrain_step = jax.jit(
            functools.partial(train.train_step, config),
            in_shardings=(replicated, state_sharding, data_sharding),
            out_shardings=(state_sharding, replicated),
            donate_argnums=(1,))
        step_rows = []
        batch = first_batch
        for step in range(args.steps):
            t0 = time.time()
            with sharding.set_mesh(mesh):
                state, info = ptrain_step(
                    jax.random.fold_in(jax.random.key(config.seed), step), state, batch)
            jax.block_until_ready(state)
            elapsed = time.time() - t0
            values = {key: float(value) for key, value in info.items()}
            if not all(math.isfinite(value) for value in values.values()):
                raise FloatingPointError(f"non-finite update metrics at step {step + 1}")
            if int(jax.device_get(state.step)) != step + 1:
                raise RuntimeError("TrainState.step did not advance exactly once")
            step_rows.append({"step": step + 1, **values, "wall_s": elapsed,
                              "memory_stats": memory_stats(devices)})
            if step + 1 < args.steps:
                batch = next(data_iter)
        report["updates"] = step_rows
        steady = step_rows[1:] if len(step_rows) > 1 else step_rows
        report["throughput"] = {
            "steady_s_per_update": float(np.median([row["wall_s"] for row in steady])),
            "last_update_s": float(step_rows[-1]["wall_s"]),
            "projected_25k_wall_s": float(np.median(
                [row["wall_s"] for row in steady]) * 25000),
        }
        stage = "parameter_delta"
        final_flat = flatten_dict(state.params.to_pure_dict(), sep="/")
        trainable_flat = flatten_dict(
            state.params.filter(config.trainable_filter).to_pure_dict(), sep="/")
        delta_rows = []
        for name in sorted(final_flat):
            before = initial.get(name)
            after = np.asarray(jax.device_get(final_flat[name]))
            if before is None or before.shape != after.shape or before.dtype != after.dtype:
                raise RuntimeError(f"{name}: parameter structure changed during the probe")
            if not np.isfinite(before).all() or not np.isfinite(after).all():
                raise FloatingPointError(f"{name}: non-finite parameter")
            delta = float(np.max(np.abs(after.astype(np.float64)
                                        - before.astype(np.float64))))
            delta_rows.append({"path": name, "shape": list(after.shape),
                               "dtype": str(after.dtype),
                               "trainable": name in trainable_flat,
                               "max_abs_delta": delta})
        report["parameter_delta"] = delta_rows
        frozen = [row for row in delta_rows if "/llm/" in "/" + row["path"]
                  and "lora" not in row["path"] and "_1/" not in "/" + row["path"]
                  and row["path"].startswith("PaliGemma/llm/")]
        report["frozen_language"] = {
            "tensors": len(frozen),
            "changed_tensors": sum(row["max_abs_delta"] > 0 for row in frozen),
            "nonfinite": 0,
        }
        delta_groups = {}
        for row in delta_rows:
            entry = delta_groups.setdefault(group_of(row["path"]), {
                "tensors": 0, "changed_tensors": 0, "max_abs_delta": 0.0,
                "trainable_tensors": 0, "frozen_tensors": 0})
            entry["tensors"] += 1
            entry["changed_tensors"] += int(row["max_abs_delta"] > 0)
            entry["max_abs_delta"] = max(entry["max_abs_delta"], row["max_abs_delta"])
            entry["trainable_tensors" if row["trainable"] else "frozen_tensors"] += 1
        report["delta_groups"] = delta_groups
        projection_changes = {
            name: {
                "kernel_changed": any(
                    row["path"].startswith(name + "/kernel") and row["max_abs_delta"] > 0
                    for row in delta_rows),
                "bias_changed": any(
                    row["path"].startswith(name + "/bias") and row["max_abs_delta"] > 0
                    for row in delta_rows),
            }
            for name in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out")
        }
        report["projection_changes"] = projection_changes
        report["lora_changed_tensors"] = sum(
            "lora" in row["path"] and row["max_abs_delta"] > 0 for row in delta_rows)
        report["final_train_state_step"] = int(jax.device_get(state.step))
        report["final_optimizer_state"] = {
            "leaves": len(jax.tree_util.tree_leaves(state.opt_state)),
            "dtypes": sorted({str(value.dtype) for value in
                              jax.tree_util.tree_leaves(state.opt_state)}),
        }
        report["passed_compute_without_checkpoint"] = bool(
            report["gradient_audit"]["loss_replicated"]["all_equal"]
            and report["gradient_audit"]["norm_replicated"]["all_equal"]
            and report["gradient_audit"]["global_norm"] > 0
            and all(row["finite"] for row in report["gradient_audit"]["groups"].values())
            and report["frozen_language"]["changed_tensors"] == 0
            and report["lora_changed_tensors"] > 0
            and delta_groups.get("action_expert", {}).get("max_abs_delta", 0.0) > 0
            and delta_groups.get("vision_encoder", {}).get("max_abs_delta", 0.0) > 0
            and all(value["kernel_changed"] and value["bias_changed"]
                    for value in projection_changes.values())
            and report["final_train_state_step"] == args.steps)
    except Exception as error:
        import traceback

        report["error"] = {"stage": stage, "type": type(error).__name__,
                           "message": str(error)[-5000:]}
        traceback.print_exc()
        failure = error
    finally:
        stop.set()
        monitor.join(timeout=5)
        report["gpu_peak_sampled_total"] = peak
        report["gpu_process_delta_at_peak_mib"] = {
            row["uuid"]: peak[row["uuid"]]["used_mib"] - row["memory_used_mib"]
            for row in baseline if row["uuid"] in peak}
        effective = config.fsdp_devices if "config" in locals() else args.fsdp_devices
        report["resource_gate"] = {
            "whole_process_demand_includes_cuda_nonpool": True,
            "margin_requirement_mib": 512,
            "all_observed_margins_mib": {
                uuid: peak[uuid]["free_mib"] for uuid in peak},
            "all_observed_margins_at_least_512": all(
                value >= 512 for value in (peak[uuid]["free_mib"] for uuid in peak)),
        }
    output = out / f"full_update_{args.label}.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"parameter_tree", "parameter_delta",
                                     "sharding", "batch"}},
                     indent=2, sort_keys=True)[:20000])
    if failure is not None:
        return 1
    return 0 if report.get("passed_compute_without_checkpoint") else 1


if __name__ == "__main__":
    raise SystemExit(main())
