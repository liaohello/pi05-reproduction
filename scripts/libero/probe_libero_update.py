#!/usr/bin/env python3
"""Two-update S2 probe with real gradients, parameter deltas, and sharding."""
from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import libero_data  # noqa: E402
import train_libero  # noqa: E402


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fsdp-devices", type=int, choices=(2, 4), required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--phase", choices=("both", "gradient", "update"), default="both")
    args = parser.parse_args()

    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from flax.traverse_util import flatten_dict
    from openpi.models import model as model_lib
    from openpi.shared import nnx_utils
    from openpi.training import sharding

    protocol = train_libero.load_protocol(
        args.protocol, args.expected_protocol_sha256)
    view = libero_data.load_view(args.view)
    if view["view_sha256"] != protocol["data"]["view_sha256"]:
        raise ValueError("probe view differs from the protocol")
    if args.image_digest != protocol["runtime"]["image_digest"]:
        raise ValueError("probe image digest differs from protocol")
    runtime = {
        "openpi_src": train_libero.sha256_tree("/opt/openpi/src/openpi"),
        "openpi_scripts": train_libero.sha256_tree("/opt/openpi/scripts"),
        "libero_source": train_libero.sha256_tree("/opt/openpi/third_party/libero"),
    }
    for key, value in runtime.items():
        if value != protocol["runtime"][key]:
            raise ValueError(f"runtime source {key} differs from protocol")
    base_tree = train_libero.sha256_tree(args.base_checkpoint)
    if base_tree["sha256"] != protocol["base"]["tree_sha256"]:
        raise ValueError("probe base differs from protocol")
    config = train_libero.make_config(
        SimpleNamespace(
            exp_name="libero-pi05-group-probe",
            mode="probe",
            view=str(args.view),
            base_checkpoint=str(args.base_checkpoint),
            fsdp_devices=args.fsdp_devices,
        ),
        protocol,
        view,
        args.out.parent / "probe_checkpoints",
        2,
    )
    train, _ = train_libero._load_train_module()
    devices = list(jax.devices())
    if len(devices) != args.fsdp_devices:
        raise ValueError(f"expected {args.fsdp_devices} devices, got {len(devices)}")
    mesh = sharding.make_mesh(args.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    loader = libero_data.create_filtered_data_loader(
        config,
        view_path=args.view,
        data_root=args.data_root,
        sharding=data_sharding,
        shuffle=True,
        split="train",
    )
    data_iter = iter(loader)
    batch = next(data_iter)
    state, state_sharding = train.init_train_state(
        config, jax.random.split(jax.random.key(config.seed))[1], mesh, resume=False)
    state = jax.block_until_ready(state)

    language_backbone = nnx.All(
        nnx.Param,
        nnx_utils.PathRegex(r".*llm.*"),
        nnx.Not(nnx_utils.PathRegex(r".*llm.*_1.*")),
        nnx.Not(nnx_utils.PathRegex(r".*lora.*")),
    )
    trainable_filters = {
        "language_lora": nnx.All(
            nnx.Param, nnx_utils.PathRegex(r".*lora.*")),
        "action_expert": nnx.All(
            nnx.Param, nnx_utils.PathRegex(r".*llm.*_1.*")),
        "projection": nnx.All(
            nnx.Param,
            nnx_utils.PathRegex(
                r".*(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out).*"),
        ),
    }
    frozen_filters = {
        "language_backbone": language_backbone,
        "vision_encoder": nnx.All(
            nnx.Param, nnx_utils.PathRegex(r".*PaliGemma/img.*")),
    }
    filters = {**trainable_filters, **frozen_filters}

    def state_stats(state):
        leaves = jax.tree.leaves(state)
        finite = True
        max_abs = jnp.asarray(0.0, dtype=jnp.float32)
        for value in leaves:
            finite = finite & jnp.all(jnp.isfinite(value))
            max_abs = jnp.maximum(max_abs, jnp.max(jnp.abs(value)))
        return {
            "tensor_count": len(leaves),
            "global_norm": optax.global_norm(leaves),
            "finite": finite,
            "max_abs": max_abs,
        }

    def comparison_stats(left, right):
        left_leaves = jax.tree.leaves(left)
        right_leaves = jax.tree.leaves(right)
        if len(left_leaves) != len(right_leaves):
            raise ValueError("comparison trees have different leaf counts")
        finite = True
        sum_squares = jnp.asarray(0.0, dtype=jnp.float32)
        max_abs = jnp.asarray(0.0, dtype=jnp.float32)
        for old, new in zip(left_leaves, right_leaves):
            delta = new - old
            finite = finite & jnp.all(jnp.isfinite(delta))
            sum_squares = sum_squares + jnp.sum(delta.astype(jnp.float32) ** 2)
            max_abs = jnp.maximum(max_abs, jnp.max(jnp.abs(delta)))
        return {
            "tensor_count": len(left_leaves),
            "global_norm": jnp.sqrt(sum_squares),
            "finite": finite,
            "max_abs": max_abs,
        }

    def gradient_audit(rng, state_value, batch_value):
        model = nnx.merge(state_value.model_def, state_value.params)
        model.train()

        def loss_fn(loss_model, loss_rng, observation, actions):
            return jnp.mean(loss_model.compute_loss(
                loss_rng, observation, actions, train=True))

        train_rng = jax.random.fold_in(rng, state_value.step)
        loss, grads = nnx.value_and_grad(
            loss_fn, argnums=nnx.DiffState(0, config.trainable_filter))(
                model, train_rng, batch_value[0], batch_value[1])
        stats = {}
        for name, filter_value in trainable_filters.items():
            filtered = grads.filter(filter_value).to_pure_dict()
            stats[name] = state_stats(filtered)
        return loss, stats

    def update_audit(rng, state_value, batch_value):
        new_state, info = train.train_step(config, rng, state_value, batch_value)
        delta_stats = {}
        for name, filter_value in filters.items():
            old_group = state_value.params.filter(filter_value).to_pure_dict()
            new_group = new_state.params.filter(filter_value).to_pure_dict()
            delta_stats[name] = comparison_stats(old_group, new_group)
        ema_delta = comparison_stats(
            state_value.ema_params.to_pure_dict(),
            new_state.ema_params.to_pure_dict(),
        )
        return new_state, info, delta_stats, ema_delta

    p_gradient = (
        jax.jit(
            gradient_audit,
            in_shardings=(replicated, state_sharding, data_sharding),
            out_shardings=None,
        )
        if args.phase != "update"
        else None
    )
    p_update = jax.jit(
        update_audit,
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated, None, None),
        donate_argnums=(1,),
    )

    phase_results = []
    for update_index in range(2):
        start = time.monotonic()
        if args.phase != "update":
            with sharding.set_mesh(mesh):
                loss, grad_stats = p_gradient(jax.random.key(config.seed), state, batch)
            gradient_time = time.monotonic() - start
        else:
            loss = None
            grad_stats = {}
            gradient_time = None
        if args.phase != "gradient":
            before = int(jax.device_get(state.step))
            update_start = time.monotonic()
            with sharding.set_mesh(mesh):
                state, info, delta_stats, ema_delta = p_update(
                    jax.random.key(config.seed), state, batch)
            state = jax.block_until_ready(state)
            update_time = time.monotonic() - update_start
            after = int(jax.device_get(state.step))
        else:
            before = int(jax.device_get(state.step))
            after = before
            info = {"loss": loss}
            delta_stats = {}
            ema_delta = None
            update_time = None
        metrics = {
            name: float(np.asarray(jax.device_get(value)).reshape(-1)[0])
            for name, value in info.items()
        }
        payload_stats = {}
        if args.phase != "update":
            for name in trainable_filters:
                payload_stats[name] = {
                    key: (
                        bool(np.asarray(jax.device_get(grad_stats[name][key])))
                        if key == "finite"
                        else float(np.asarray(jax.device_get(grad_stats[name][key])))
                        if key in {"global_norm", "max_abs"}
                        else int(jax.device_get(grad_stats[name][key]))
                    )
                    for key in grad_stats[name]
                }
        delta_payload = {}
        for name in filters:
            if name not in delta_stats:
                continue
            delta_payload[name] = {
                key: (
                    bool(np.asarray(jax.device_get(delta_stats[name][key])))
                    if key == "finite"
                    else float(np.asarray(jax.device_get(delta_stats[name][key])))
                    if key in {"global_norm", "max_abs"}
                    else int(jax.device_get(delta_stats[name][key]))
                )
                for key in delta_stats[name]
            }
        ema_delta_payload = None
        if ema_delta is not None:
            ema_delta_payload = {
                key: (
                    bool(np.asarray(jax.device_get(ema_delta[key])))
                    if key == "finite"
                    else float(np.asarray(jax.device_get(ema_delta[key])))
                    if key in {"global_norm", "max_abs"}
                    else int(jax.device_get(ema_delta[key]))
                )
                for key in ema_delta
            }
        phase_results.append({
            "update_index": update_index,
            "train_state_before": before,
            "train_state_after": after,
            "loss": (
                float(np.asarray(jax.device_get(loss)))
                if loss is not None else None
            ),
            "metrics": metrics,
            "gradient_stats": payload_stats,
            "parameter_delta_stats": delta_payload,
            "ema_parameter_delta_stats": ema_delta_payload,
            "gradient_and_compile_s": gradient_time,
            "actual_update_compile_or_run_s": update_time,
        })
        if not all(np.isfinite(value) for value in metrics.values()):
            raise FloatingPointError(f"non-finite training metrics: {metrics}")
        if loss is not None and not np.isfinite(float(np.asarray(jax.device_get(loss)))):
            raise FloatingPointError("non-finite training loss")
        if args.phase != "gradient" and not metrics.get("grad_norm", 0.0) > 0.0:
            raise RuntimeError("zero global gradient")
        if args.phase != "gradient" and after != before + 1:
            raise RuntimeError(f"TrainState did not advance: {before} -> {after}")
        if args.phase != "gradient":
            for name in ("language_backbone", "vision_encoder"):
                if not delta_payload[name]["max_abs"] == 0.0:
                    raise RuntimeError(f"frozen {name} changed")
            for name in ("language_lora", "action_expert", "projection"):
                if delta_payload[name]["global_norm"] == 0.0:
                    raise RuntimeError(f"{name} parameter delta is zero")
            if not ema_delta_payload["finite"]:
                raise FloatingPointError("non-finite EMA update")
            if ema_delta_payload["global_norm"] == 0.0:
                raise RuntimeError("EMA parameters did not update")
        if args.phase != "update":
            for name in ("language_lora", "action_expert", "projection"):
                if not payload_stats[name]["finite"]:
                    raise FloatingPointError(f"non-finite {name} gradient")
                if payload_stats[name]["max_abs"] == 0.0:
                    raise RuntimeError(f"zero {name} gradient")
        batch = next(data_iter)
        if args.phase == "gradient":
            break

    params = flatten_dict(state.params.to_pure_dict(), sep="/")
    sharding_pure = state_sharding.params.to_pure_dict()
    param_shardings = flatten_dict(
        jax.tree.map(
            lambda value: getattr(value, "sharding", value),
            sharding_pure,
        ),
        sep="/",
    )
    sharding_examples = {}
    for name in sorted(params)[:]:
        group = (
            "language_lora" if "lora" in name
            else "vision_encoder" if "/img/" in name
            else "action_expert" if "llm" in name and "_1" in name
            else "projection" if any(token in name for token in (
                "action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"))
            else "language_backbone" if "llm" in name
            else "other"
        )
        if group not in sharding_examples:
            value = param_shardings[name]
            sharding_examples[group] = {
                "parameter": name,
                "spec": str(value.spec),
                "device_count": len(value.device_set),
            }
    observation, actions = batch
    batch_sharding = {
        "state": {
            "shape": list(observation.state.shape),
            "sharding": str(observation.state.sharding),
            "per_device_shapes": [
                list(value.data.shape)
                for value in observation.state.addressable_shards],
        },
        "actions": {
            "shape": list(actions.shape),
            "sharding": str(actions.sharding),
            "per_device_shapes": [
                list(value.data.shape) for value in actions.addressable_shards],
        },
        "image_shapes": {
            name: list(value.shape) for name, value in observation.images.items()
        },
    }
    output = {
        "schema_version": 1,
        "phase": args.phase,
        "passed": True,
        "protocol_sha256": protocol["_sha256"],
        "view_sha256": view["view_sha256"],
        "base_tree_sha256": base_tree["sha256"],
        "image_digest": args.image_digest,
        "runtime_sources": runtime,
        "global_batch_size": config.batch_size,
        "num_train_steps_for_probe": config.num_train_steps,
        "device_count": len(devices),
        "device_ids": [value.id for value in devices],
        "device_kinds": sorted({value.device_kind for value in devices}),
        "fsdp_devices": config.fsdp_devices,
        "mesh_axis_names": list(mesh.axis_names),
        "mesh_shape": {str(key): int(value) for key, value in mesh.shape.items()},
        "data_parallel_replicas": len(devices) // config.fsdp_devices,
        "sharding_examples": sharding_examples,
        "batch_sharding": batch_sharding,
        "phases": phase_results,
        "final_train_state_step": int(jax.device_get(state.step)),
        "checkpoint_written": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.out),
        "sha256": sha256_file(args.out),
        "passed": True,
        "mesh_shape": output["mesh_shape"],
        "sharding_examples": sharding_examples,
        "batch_sharding": batch_sharding,
        "phases": phase_results,
        "final_train_state_step": output["final_train_state_step"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
