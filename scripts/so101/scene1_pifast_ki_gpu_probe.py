#!/usr/bin/env python3
"""Real-data gradient, update and checkpoint-continuity probe for Scene1."""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import flax.nnx as nnx
import flax.traverse_util as traverse
import jax
import jax.numpy as jnp
import numpy as np
import optax

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import openpi.models.model as model_lib  # noqa: E402
import openpi.training.checkpoints as checkpoints  # noqa: E402
from openpi.training import sharding  # noqa: E402
import scene1_openpi_so101 as openpi_so101  # noqa: E402
import scene1_pifast_ki_runtime as runtime  # noqa: E402
import scene1_train_so101 as launcher  # noqa: E402
import scene1_validation as validation  # noqa: E402


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


def group_norms(tree) -> dict:
    state = tree if isinstance(tree, nnx.State) else tree
    flat = traverse.flatten_dict(state.to_pure_dict(), sep="/")
    groups: dict[str, dict] = {}
    for path, value in flat.items():
        group = groups.setdefault(
            group_for(path),
            {"arrays": 0, "elements": 0, "squared_norm": 0.0,
             "finite": True, "nonzero_arrays": 0},
        )
        norm_sq = float(jnp.linalg.norm(value.astype(jnp.float32)) ** 2)
        finite = bool(jnp.isfinite(value).all())
        group["arrays"] += 1
        group["elements"] += int(value.size)
        group["squared_norm"] += norm_sq
        group["finite"] = group["finite"] and finite
        group["nonzero_arrays"] += int(norm_sq > 0.0)
    for group in groups.values():
        group["norm"] = math.sqrt(group["squared_norm"])
    return groups


def parameter_deltas(before, after, trainable_filter) -> dict:
    before_all = traverse.flatten_dict(before.to_pure_dict(), sep="/")
    after_all = traverse.flatten_dict(after.to_pure_dict(), sep="/")
    trainable = set(traverse.flatten_dict(
        before.filter(trainable_filter).to_pure_dict(), sep="/"))
    if set(before_all) != set(after_all):
        raise RuntimeError("parameter tree changed during update")
    result: dict[str, dict] = {}
    for path in sorted(before_all):
        group = result.setdefault(
            group_for(path),
            {"arrays": 0, "elements": 0, "delta_squared_norm": 0.0,
             "parameter_squared_norm": 0.0, "changed_elements": 0,
             "trainable": path in trainable, "nontrainable_changed": 0},
        )
        delta = after_all[path] - before_all[path]
        changed = int(jnp.count_nonzero(delta))
        delta_sq = float(jnp.linalg.norm(delta.astype(jnp.float32)) ** 2)
        param_sq = float(
            jnp.linalg.norm(before_all[path].astype(jnp.float32)) ** 2)
        group["arrays"] += 1
        group["elements"] += int(delta.size)
        group["delta_squared_norm"] += delta_sq
        group["parameter_squared_norm"] += param_sq
        group["changed_elements"] += changed
        if path not in trainable and changed:
            group["nontrainable_changed"] += changed
    for group in result.values():
        group["delta_norm"] = math.sqrt(group["delta_squared_norm"])
        group["parameter_norm"] = math.sqrt(group["parameter_squared_norm"])
        group["relative_delta_norm"] = (
            group["delta_norm"] / group["parameter_norm"]
            if group["parameter_norm"] else 0.0)
    return result


def hash_leaf(leaf) -> dict:
    logical = np.asarray(jax.device_get(leaf))
    digest = hashlib.sha256(logical.tobytes())
    shard_metadata = []
    if hasattr(leaf, "addressable_shards"):
        for shard in leaf.addressable_shards:
            shard_metadata.append({
                "index": [str(value) for value in shard.index],
                "device_id": int(shard.device.id),
            })
    return {
        "shape": [int(value) for value in leaf.shape],
        "dtype": str(leaf.dtype),
        "sha256": digest.hexdigest(),
        "sharding": shard_metadata,
    }


def state_digest(state) -> dict:
    params = traverse.flatten_dict(state.params.to_pure_dict(), sep="/")
    optimizer_leaves, _ = jax.tree_util.tree_flatten_with_path(
        state.opt_state)
    optimizer = {
        jax.tree_util.keystr(path): value
        for path, value in optimizer_leaves
    }
    return {
        "step": int(jax.device_get(state.step)),
        "params": {
            path: hash_leaf(value) for path, value in sorted(params.items())
        },
        "optimizer": {
            path: hash_leaf(value) for path, value in sorted(optimizer.items())
        },
    }


def compare_digests(left: dict, right: dict) -> list[str]:
    problems = []
    if left["step"] != right["step"]:
        problems.append(f"step {left['step']} != {right['step']}")
    for section in ("params", "optimizer"):
        if set(left[section]) != set(right[section]):
            problems.append(f"{section} paths differ")
            continue
        for path in sorted(left[section]):
            if left[section][path] != right[section][path]:
                problems.append(f"{section}/{path} differs")
    return problems


def make_config(args):
    return runtime.make_train_config(
        mode=args.mode,
        run_root=args.run_dir,
        train_view=args.train_view,
        base_checkpoint=args.base_checkpoint,
        exp_name=args.exp_name,
        steps=args.target_updates,
        batch_size=args.batch_size,
        save_interval=1,
        warmup_steps=1000,
        peak_lr=2.5e-5,
        decay_steps=40000,
        decay_lr=2.5e-6,
        seed=args.seed,
        fsdp_devices=args.fsdp_devices,
        num_workers=args.data_loader_workers,
        fast_ce_weight=args.fast_ce_weight,
        fast_tokenizer_dir=args.fast_tokenizer_dir,
        paligemma_tokenizer_path=args.paligemma_tokenizer,
        dataset_label="scene1_single_random_v1",
    )


def build_batches(config, args, *, start_update: int, count: int):
    raw = openpi_so101.SO101EpisodeDataset(
        args.train_view,
        split="train",
        action_horizon=config.model.action_horizon,
    )
    total = (start_update + count) * config.batch_size
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(
        len(raw), size=total, replace=True).astype(np.int64).tolist()
    active = indices[start_update * config.batch_size:]
    data_sharding = jax.sharding.NamedSharding(
        sharding.make_mesh(config.fsdp_devices),
        jax.sharding.PartitionSpec(sharding.DATA_AXIS),
    )
    loader = openpi_so101.create_so101_data_loader(
        config,
        sharding=data_sharding,
        shuffle=False,
        num_batches=count,
        split="train",
        sampler_indices=active,
        expected_samples=len(raw),
    )
    return list(loader), indices


def gradient_step(config, mode, ki, state, batch):
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(module, rng, observation, actions):
        if mode == "flow":
            values = module.compute_flow_loss(
                rng, observation, actions, train=True, ki=ki)
        elif mode == "fast":
            values = module.compute_fast_loss(
                rng, observation, actions, train=True)
        elif mode == "joint":
            values = module.compute_loss(
                rng, observation, actions, train=True)
        else:
            raise ValueError(f"unknown gradient mode {mode!r}")
        return jnp.mean(values)

    train_rng, _ = jax.random.split(jax.random.key(config.seed))
    rng = jax.random.fold_in(train_rng, state.step)
    observation, actions = batch
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(
        loss_fn, argnums=diff_state)(
            model, rng, observation, actions)
    return loss, grads


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        choices=("gradients", "update", "restore", "validation", "compare"),
        required=True)
    parser.add_argument("--mode", choices=("initial", "resume"), default="initial")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-base-tree-sha256", required=True)
    parser.add_argument("--train-view", type=Path, required=True)
    parser.add_argument("--fast-tokenizer-dir", type=Path, required=True)
    parser.add_argument("--paligemma-tokenizer", type=Path, required=True)
    parser.add_argument("--fast-ce-weight", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--fsdp-devices", type=int, required=True)
    parser.add_argument("--target-updates", type=int, default=40000)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-loader-workers", type=int, default=8)
    parser.add_argument("--restore-step", type=int, default=None)
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--save-at-step", type=int, default=None)
    parser.add_argument("--digest-at-step", action="append", type=int, default=[])
    parser.add_argument("--skip-state-digest", action="store_true")
    parser.add_argument("--skip-parameter-deltas", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--left-digest", type=Path)
    parser.add_argument("--right-digest", type=Path)
    parser.add_argument("--compare-output", type=Path)
    parser.add_argument(
        "--validation-samples", type=int, default=256,
        help="-1 evaluates every validation window")
    parser.add_argument("--validation-seed", type=int, default=42)
    args = parser.parse_args()

    if args.action == "compare":
        if not args.left_digest or not args.right_digest or not args.compare_output:
            raise ValueError("compare requires left, right and compare outputs")
        left = json.loads(args.left_digest.read_text(encoding="utf-8"))
        right = json.loads(args.right_digest.read_text(encoding="utf-8"))
        problems = compare_digests(left, right)
        payload = {
            "status": "pass" if not problems else "fail",
            "problems": problems,
            "left_step": left["step"],
            "right_step": right["step"],
            "bitwise_equal": not problems,
        }
        args.compare_output.parent.mkdir(parents=True, exist_ok=True)
        args.compare_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, sort_keys=True))
        return 0 if not problems else 1

    if jax.default_backend() != "gpu":
        raise RuntimeError(f"expected GPU backend, got {jax.default_backend()}")
    if args.batch_size % jax.device_count():
        raise ValueError("global batch must be divisible by visible GPU count")
    if args.fsdp_devices < 1 or jax.device_count() % args.fsdp_devices:
        raise ValueError("fsdp_devices must divide the visible GPU count")
    base_tree_sha256 = launcher.clean60.sha256_tree(args.base_checkpoint)
    if base_tree_sha256 != args.expected_base_tree_sha256:
        raise ValueError(
            f"pi05_base tree hash {base_tree_sha256} differs from the plan")

    os.environ["SO101_TRAIN_VIEW"] = str(args.train_view)
    config = make_config(args)
    train_module, train_path = launcher._load_train_module()
    mesh = sharding.make_mesh(args.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec())
    start_update = int(args.restore_step or 0)
    if args.action == "validation":
        batches = []
        sample_indices = []
    else:
        batches, sample_indices = build_batches(
            config, args, start_update=start_update, count=args.steps)

    if args.mode == "resume":
        template, state_sharding = train_module.init_train_state(
            config, jax.random.key(config.seed), mesh, resume=True)
        manager, _ = checkpoints.initialize_checkpoint_dir(
            config.checkpoint_dir,
            keep_period=1,
            overwrite=False,
            resume=True,
        )
        state = checkpoints.restore_state(
            manager, template, None, step=args.restore_step)
        restored_step = int(jax.device_get(state.step))
        if restored_step != args.restore_step:
            raise RuntimeError(
                f"restored TrainState.step {restored_step} != {args.restore_step}")
        restored_digest = state_digest(state)
    else:
        state, state_sharding = train_module.init_train_state(
            config, jax.random.key(config.seed), mesh, resume=False)
        manager = None
        restored_step = None
        restored_digest = None

    compile_start = time.perf_counter()
    gradient_rows = {}
    if args.action == "validation":
        validation_dataset = openpi_so101.SO101EpisodeDataset(
            args.train_view,
            split="validation",
            action_horizon=config.model.action_horizon,
        )
        if args.validation_samples == -1:
            validation_indices = list(range(len(validation_dataset)))
            selection = None
        else:
            validation_indices, selection = validation.balanced_probe_indices(
                validation_dataset.sample_metadata,
                args.validation_samples,
                args.validation_seed,
            )
        fixed_plan = validation.fixed_sample_plan(
            validation_indices,
            config.batch_size,
            args.validation_seed,
        )
        keys = validation.validation_batch_keys(
            args.validation_seed, fixed_plan["batch_count"])
        validation_loader = openpi_so101.create_so101_data_loader(
            config,
            sharding=data_sharding,
            shuffle=False,
            num_batches=fixed_plan["batch_count"],
            split="validation",
            sampler_indices=fixed_plan["_sampler_indices"],
        )
        compiled_validation = jax.jit(
            validation.validation_batch_losses,
            in_shardings=(
                state_sharding, data_sharding, data_sharding, replicated),
        )
        started = time.perf_counter()
        validation_iter = iter(validation_loader)
        losses = []
        for key in keys:
            batch = next(validation_iter)
            losses.append(
                compiled_validation(state, batch[0], batch[1], key))
        summary = validation.weighted_validation_loss(
            [np.asarray(item).tolist() for item in losses],
            fixed_plan["valid_samples_per_batch"],
        )
        jax.block_until_ready(state.step)
        elapsed = time.perf_counter() - compile_start
        result = {
            "schema_version": 1,
            "status": "pass",
            "backend": jax.default_backend(),
            "device_count": jax.device_count(),
            "device_ids": [int(device.id) for device in jax.devices()],
            "fsdp_devices": args.fsdp_devices,
            "mesh_shape": {str(key): int(value)
                           for key, value in mesh.shape.items()},
            "global_batch_size": args.batch_size,
            "batch_per_device_shard": args.batch_size // jax.device_count(),
            "validation_samples": fixed_plan["sample_count"],
            "validation_batches": fixed_plan["batch_count"],
            "valid_samples_per_batch": fixed_plan[
                "valid_samples_per_batch"],
            "seed": args.validation_seed,
            "selection": selection,
            "train_false": True,
            "gradient_updates": 0,
            "fast_ce_weight": args.fast_ce_weight,
            "base_checkpoint_tree_sha256": base_tree_sha256,
            "source_sha256": {
                "train_entry": launcher._sha_file(train_path),
                "trainer": launcher._sha_file(
                    HERE / "scene1_train_so101.py"),
                "joint_model": launcher._sha_file(
                    HERE / "scene1_pifast_ki_model.py"),
                "runtime": launcher._sha_file(
                    HERE / "scene1_pifast_ki_runtime.py"),
                "fast_data": launcher._sha_file(
                    HERE / "scene1_pifast_ki_fast_data.py"),
                "validation": launcher._sha_file(
                    HERE / "scene1_validation.py"),
            },
            "sample_indices_sha256": fixed_plan["sample_order_sha256"],
            "batch_key_sha256": validation.validation_key_sha256(keys),
            "randomness": validation.randomness_digests(
                keys,
                config.batch_size,
                config.model.action_horizon,
                config.model.action_dim,
            ),
            "loss": summary,
            "elapsed_s": elapsed,
            "claim": (
                "Real validation windows, train=False, no optimizer update; "
                "validation loss is Flow-MSE only"),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "status": "pass",
            "samples": fixed_plan["sample_count"],
            "batches": fixed_plan["batch_count"],
            "validation_loss": summary["validation_loss"],
            "elapsed_s": elapsed,
        }, sort_keys=True))
        return 0

    if args.action == "gradients":
        batch = batches[0]
        compiled = {}
        for mode, ki in (
            ("flow", True),
            ("flow", False),
            ("fast", True),
            ("joint", True),
        ):
            name = f"{mode}_ki_{str(ki).lower()}"
            fn = jax.jit(
                functools.partial(gradient_step, config, mode, ki),
                in_shardings=(state_sharding, data_sharding),
            )
            started = time.perf_counter()
            loss, grads = fn(state, batch)
            jax.block_until_ready((loss, grads))
            gradient_rows[name] = {
                "loss": float(loss),
                "elapsed_s": time.perf_counter() - started,
                "groups": group_norms(grads),
                "global_norm": float(optax.global_norm(grads)),
            }
            compiled[name] = fn
        elapsed = time.perf_counter() - compile_start
    else:
        if args.mode == "initial" and (args.save_checkpoint or args.save_at_step):
            manager, _ = checkpoints.initialize_checkpoint_dir(
                config.checkpoint_dir,
                keep_period=1,
                overwrite=True,
                resume=False,
            )
            checkpoint_loader = openpi_so101.create_so101_data_loader(
                config,
                sharding=data_sharding,
                shuffle=False,
                num_batches=1,
                split="train",
                sampler_indices=sample_indices[:config.batch_size],
            )
        else:
            checkpoint_loader = None
        ptrain_step = jax.jit(
            functools.partial(train_module.train_step, config),
            in_shardings=(replicated, state_sharding, data_sharding),
            out_shardings=(state_sharding, replicated),
            donate_argnums=(1,) if args.skip_parameter_deltas else (),
        )
        before = None if args.skip_parameter_deltas else state
        started = time.perf_counter()
        infos = []
        update_times = []
        saved_steps = []
        intermediate_digests = {}
        train_rng, _ = jax.random.split(jax.random.key(config.seed))
        for batch in batches:
            step_started = time.perf_counter()
            state, info = ptrain_step(
                train_rng, state, batch)
            jax.block_until_ready((state.step, info))
            update_times.append(time.perf_counter() - step_started)
            infos.append(info)
            current_step = int(jax.device_get(state.step))
            if args.save_at_step == current_step:
                if manager is None or checkpoint_loader is None:
                    raise RuntimeError("intermediate checkpoint manager is absent")
                checkpoints.save_state(
                    manager, state, checkpoint_loader, current_step)
                manager.wait_until_finished()
                metadata_step = int(manager.latest_step())
                if metadata_step != current_step:
                    raise RuntimeError(
                        f"checkpoint metadata {metadata_step} != {current_step}")
                saved_steps.append(current_step)
            if current_step in set(args.digest_at_step):
                intermediate_digests[current_step] = state_digest(state)
        update_elapsed = time.perf_counter() - started
        elapsed = time.perf_counter() - compile_start

    result = {
        "schema_version": 1,
        "status": "pass",
        "backend": jax.default_backend(),
        "device_count": jax.device_count(),
        "device_ids": [int(device.id) for device in jax.devices()],
        "fsdp_devices": args.fsdp_devices,
        "mesh_shape": {str(key): int(value)
                       for key, value in mesh.shape.items()},
        "global_batch_size": args.batch_size,
        "batch_per_device_shard": args.batch_size // jax.device_count(),
        "data_parallel_replicas": jax.device_count() // args.fsdp_devices,
        "sample_indices_sha256": hashlib.sha256(
            json.dumps(sample_indices, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "fast_ce_weight": args.fast_ce_weight,
        "seed": args.seed,
        "base_checkpoint_tree_sha256": base_tree_sha256,
        "source_sha256": {
            "train_entry": launcher._sha_file(train_path),
            "trainer": launcher._sha_file(HERE / "scene1_train_so101.py"),
            "joint_model": launcher._sha_file(
                HERE / "scene1_pifast_ki_model.py"),
            "runtime": launcher._sha_file(
                HERE / "scene1_pifast_ki_runtime.py"),
            "fast_data": launcher._sha_file(
                HERE / "scene1_pifast_ki_fast_data.py"),
            "validation": launcher._sha_file(HERE / "scene1_validation.py"),
            "gemma_overlay": launcher._sha_file(
                Path(os.environ["PYTHONPATH"].split(":", 1)[0])
                / "openpi/models/gemma.py"),
        },
        "tokenizer_sha256": {
            "fast_tokenizer_json": launcher._sha_file(
                args.fast_tokenizer_dir / "tokenizer.json"),
            "paligemma_model": launcher._sha_file(
                args.paligemma_tokenizer),
        },
        "train_entry": str(train_path),
        "train_entry_sha256": launcher._sha_file(train_path),
        "elapsed_s": elapsed,
        "restored_train_state_step": restored_step,
    }
    if restored_digest is not None:
        result["restored_state_digest"] = restored_digest
    if args.action == "gradients":
        flow_true = gradient_rows["flow_ki_true"]["groups"]
        flow_false = gradient_rows["flow_ki_false"]["groups"]
        fast_groups = gradient_rows["fast_ki_true"]["groups"]
        joint_groups = gradient_rows["joint_ki_true"]["groups"]
        shared = ("vision", "language_lora", "fast_adapter")
        flow_groups = ("action_expert_lora", "action_time_projections")
        checks = {
            "flow_ki_blocks_shared_vlm": all(
                flow_true[name]["norm"] == 0.0 for name in shared),
            "flow_ki_off_activates_shared_vlm": all(
                flow_false[name]["norm"] > 0.0
                for name in ("vision", "language_lora")),
            "flow_ki_off_keeps_fast_adapter_zero": (
                flow_false["fast_adapter"]["norm"] == 0.0),
            "fast_blocks_flow_action_expert": all(
                fast_groups[name]["norm"] == 0.0 for name in flow_groups),
            "joint_updates_every_trainable_group": all(
                joint_groups[name]["norm"] > 0.0
                for name in shared + flow_groups),
        }
        result["gradient_contract_checks"] = checks
        if not all(checks.values()):
            result["status"] = "fail"
        result["gradient_scenarios"] = gradient_rows
        result["complete_updates_executed"] = 0
    else:
        after_step = int(jax.device_get(state.step))
        if after_step != start_update + args.steps:
            raise RuntimeError(
                f"TrainState.step {after_step} != "
                f"{start_update + args.steps}")
        if args.skip_parameter_deltas:
            deltas = {}
            delta_checks = {
                "parameter_deltas_checked": False,
                "finite_metrics": all(
                    math.isfinite(float(value))
                    for value in jax.device_get(info).values()),
            }
        else:
            if before is None:
                raise RuntimeError("parameter delta baseline is absent")
            deltas = parameter_deltas(
                before.params, state.params, config.trainable_filter)
            expected_delta_groups = (
                "vision", "language_lora", "action_expert_lora",
                "action_time_projections", "fast_adapter")
            delta_checks = {
                "parameter_deltas_checked": True,
                "every_trainable_group_changed": all(
                    deltas[name]["delta_norm"] > 0.0
                    for name in expected_delta_groups),
                "frozen_parameters_bitwise_unchanged": all(
                    group["nontrainable_changed"] == 0
                    for group in deltas.values()),
                "finite_metrics": all(
                    math.isfinite(float(value))
                    for value in jax.device_get(info).values()),
            }
            if not all(
                    value for key, value in delta_checks.items()
                    if key != "parameter_deltas_checked"):
                result["status"] = "fail"
        pre_clip = float(jax.device_get(info["grad_norm"]))
        result.update({
            "complete_updates_executed": args.steps,
            "train_state_step": after_step,
            "update_elapsed_s": update_elapsed,
            "update_times_s": update_times,
            "steady_update_median_s": (
                float(np.median(update_times[1:]))
                if len(update_times) > 1 else None),
            "metrics": {
                key: float(np.mean([
                    float(jax.device_get(row[key])) for row in infos
                ]))
                for key in infos[0]
            },
            "gradient_clip": {
                "pre_clip_global_norm": pre_clip,
                "clip_norm": 1.0,
                "clip_coefficient": min(1.0, 1.0 / pre_clip),
                "post_clip_global_norm": min(pre_clip, 1.0),
            },
            "update_checks": delta_checks,
            "parameter_deltas": deltas,
            "intermediate_state_digests": intermediate_digests,
        })
        checkpoint_started = time.perf_counter()
        if args.save_checkpoint or saved_steps:
            if manager is None or checkpoint_loader is None:
                raise RuntimeError("checkpoint manager is absent")
            if args.save_checkpoint:
                checkpoints.save_state(
                    manager, state, checkpoint_loader, after_step)
                saved_steps.append(after_step)
            manager.wait_until_finished()
            latest = int(manager.latest_step())
            expected_latest = saved_steps[-1]
            if latest != expected_latest:
                raise RuntimeError(
                    f"checkpoint step {latest} != expected {expected_latest}")
            result["checkpoint"] = {
                "saved": True,
                "orbax_latest_step": latest,
                "saved_train_state_steps": saved_steps,
                "elapsed_s": time.perf_counter() - checkpoint_started,
            }
        else:
            result["checkpoint"] = {"saved": False}
        if not args.skip_state_digest:
            result["state_digest"] = state_digest(state)
    result["claim"] = (
        "Real Scene1 batch, real pi05_base, full FSDP TrainState; diagnostic only")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if "state_digest" in result:
        digest_path = args.output.with_suffix(".digest.json")
        digest_path.write_text(
            json.dumps(result["state_digest"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if "restored_state_digest" in result:
        restored_path = args.output.with_suffix(".restored.digest.json")
        restored_path.write_text(
            json.dumps(
                result["restored_state_digest"], indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    for step, digest in result.get("intermediate_state_digests", {}).items():
        digest_path = args.output.with_suffix(f".step{step}.digest.json")
        digest_path.write_text(
            json.dumps(digest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({
        "status": result["status"],
        "step": result.get("train_state_step"),
        "updates": result["complete_updates_executed"],
        "batch": result["global_batch_size"],
        "devices": result["device_count"],
        "elapsed_s": result["elapsed_s"],
    }, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
