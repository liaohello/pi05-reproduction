#!/usr/bin/env python3
"""Clean60 multi-card FSDP capacity probe.

Runs inside the pinned ``openpi-server`` image with N visible GPUs and measures
how much memory the registered clean60 training scope actually occupies per
card: first the materialised train state (params + AdamW moments), then one
forward/backward with the real global batch.

It stops before ``tx.update``/``apply_updates``: no weight is modified, no
checkpoint is written and the dataset split is never read as evidence.  The
batch comes from a real exported training view so the shapes are the ones the
25k run will see, not invented ones.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_SCRIPTS = "/work/scripts/so101"


def _bytes_by_device(tree) -> dict[str, int]:
    """Sum the resident bytes of every addressable shard, keyed by device id.

    Walks with ``is_leaf`` so nnx ``Variable`` wrappers and ``DiffState``/``State``
    containers are flattened by JAX instead of being skipped as unknown objects.
    """
    import jax

    totals: dict[str, int] = {}

    def is_edge(node):
        return isinstance(node, jax.Array) or hasattr(node, "value")

    for leaf in jax.tree_util.tree_leaves(tree, is_leaf=is_edge):
        array = getattr(leaf, "value", leaf)
        if not hasattr(array, "addressable_shards"):
            continue
        for shard in array.addressable_shards:
            key = str(shard.device.id)
            totals[key] = totals.get(key, 0) + int(shard.data.nbytes)
    return totals


def _shape_tree(tree):
    def leaf(value):
        shape = getattr(value, "shape", None)
        dtype = str(getattr(value, "dtype", ""))
        if shape is None:
            return repr(value)[:80]
        return {"shape": list(shape), "dtype": dtype}

    import jax

    flat = jax.tree_util.tree_leaves(tree)
    sizes = {}
    for value in flat:
        entry = leaf(value)
        if isinstance(entry, dict):
            key = f"{entry['dtype']}:{'x'.join(str(s) for s in entry['shape'])}"
            sizes[key] = sizes.get(key, 0) + 1
    return {"leaves": len(flat), "distinct_by_dtype_shape": dict(sorted(sizes.items()))}


def _shard_shapes_by_device(tree):
    """Record each local shard's real shape and dtype, including FSDP replication."""
    import jax

    rows: dict[str, list[dict]] = {}
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
                "replica_id": int(shard.replica_id),
            })
    return {device: {"leaf_count": len(values), "leaves": values}
            for device, values in sorted(rows.items())}


def _nvidia_rows() -> list[dict]:
    """Container-visible device rows; empty on a GPU-less container (no nvidia-smi)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.used",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    rows = []
    for line in result.stdout.splitlines():
        index, uuid, total, used = [part.strip() for part in line.split(",", 3)]
        rows.append({"index": int(index), "uuid": uuid, "memory_total_mib": int(total),
                     "memory_used_mib": int(used)})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsdp-devices", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True,
                        help="global batch; OpenPI requires batch_size % device_count == 0")
    parser.add_argument("--train-view", required=True, help="container path of an exported view")
    parser.add_argument("--base-checkpoint", default="/base")
    parser.add_argument("--out", required=True, help="container run dir for the report")
    parser.add_argument("--state-mode", default="on", choices=("on", "off"))
    parser.add_argument("--label", default="clean60-capacity")
    parser.add_argument("--shape-only", action="store_true",
                        help="trace the forward/backward with jax.eval_shape instead of "
                             "executing it: validates the wiring on a GPU-less container")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ["SO101_TRAIN_VIEW"] = args.train_view
    os.environ.setdefault("PYTHONPATH", REPO_SCRIPTS)
    if REPO_SCRIPTS not in sys.path:
        sys.path.insert(0, REPO_SCRIPTS)

    # Imported after the environment is fixed: the pinned runtime resolves the
    # tokenizer location from OPENPI_DATA_HOME at import time.
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import optax

    import openpi_so101
    import openpi.training.sharding as jsh

    train = _load_train_module()

    devices = list(jax.local_devices())
    report: dict = {
        "schema_version": 1,
        "label": args.label,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backend": jax.default_backend(),
        "device_count": len(devices),
        "devices": [{"jax_id": device.id, "kind": getattr(device, "device_kind", "unknown"),
                     "process_index": device.process_index}
                    for device in devices],
        "nvidia_visible_before": _nvidia_rows(),
        "requested": {"fsdp_devices": args.fsdp_devices, "global_batch_size": args.batch_size,
                      "state_mode": args.state_mode, "trainable_scope": "clean60 registered scope"},
        "stages": [],
        "jax_memory_fraction_env": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
        "jax_preallocate_env": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "tokenizer_from_env": str(os.environ.get("OPENPI_DATA_HOME", "")),
    }

    def snapshot(name: str, started: float, extra: dict | None = None) -> dict:
        per_device = {}
        for device in devices:
            stats = dict(device.memory_stats() or {})
            stats.pop("peak_bytes", None)
            per_device[str(device.id)] = {
                key: int(value) for key, value in stats.items()
                if key in {"bytes_limit", "bytes_in_use", "persistent_bytes",
                           "allocated_size", "reserved_size", "free_size"}
            }
        row = {"stage": name, "wall_s": round(time.time() - started, 2),
               "jax_memory_stats": per_device,
               "nvidia_visible": _nvidia_rows()}
        if extra:
            row.update(extra)
        report["stages"].append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        return row

    stage = "config"
    try:
        config = openpi_so101.make_train_config(
            mode="serve", run_root=out, train_view=Path(args.train_view),
            base_checkpoint=Path(args.base_checkpoint), exp_name="clean60_capacity_probe",
            state_mode=args.state_mode, batch_size=args.batch_size,
            fsdp_devices=args.fsdp_devices, paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m", dataset_label="clean60_capacity_probe",
            phase="clean60-capacity-probe", num_workers=0)
        if config.batch_size % len(devices):
            raise RuntimeError(f"batch {config.batch_size} not divisible by {len(devices)} devices")
        mesh = jsh.make_mesh(config.fsdp_devices)
        data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(jsh.DATA_AXIS))
        replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        report["mesh"] = {"axis_names": list(mesh.axis_names),
                          "shape": {str(k): int(v) for k, v in mesh.shape.items()},
                          "fsdp_devices": config.fsdp_devices,
                          "data_parallel_replicas": len(devices) // config.fsdp_devices,
                          "global_batch_size": config.batch_size,
                          "batch_per_data_replica": config.batch_size // (
                              len(devices) // config.fsdp_devices),
                          "batch_per_device_shard": config.batch_size // (
                              len(devices) // config.fsdp_devices),
                          "batch_per_data_shard": config.batch_size // (
                              len(devices) // config.fsdp_devices),
                          "fsdp_axis_replicates_batch": True}

        stage = "init_train_state"
        started = time.time()
        state, state_sharding = train.init_train_state(config, jax.random.key(0), mesh, resume=False)
        jax.block_until_ready(state)
        params_bytes = _bytes_by_device(state.params)
        opt_bytes = _bytes_by_device(state.opt_state)
        snapshot(stage, started, {
            "sharded_bytes_per_device": {
                "params": params_bytes, "opt_state": opt_bytes,
                "total": {key: params_bytes.get(key, 0) + opt_bytes.get(key, 0)
                          for key in set(params_bytes) | set(opt_bytes)}},
            "param_tensors": len(jax.tree_util.tree_leaves(state.params.to_pure_dict())),
            "params_shape_dtype": _shape_tree(state.params.to_pure_dict()),
            "optimizer_state_shape_dtype": _shape_tree(state.opt_state),
        })

        stage = "batch"
        started = time.time()
        loader = openpi_so101.create_so101_data_loader(
            config, sharding=data_sharding, shuffle=True, num_batches=1)
        batch = next(iter(loader))
        jax.block_until_ready(batch)
        observation, actions = batch
        snapshot(stage, started, {
            "observation": _shape_tree(observation), "actions": _shape_tree(actions),
            "batch_bytes_per_device": _bytes_by_device(batch),
            "batch_shard_shapes_by_device": _shard_shapes_by_device(batch),
        })

        stage = "forward_backward"
        trainable_filter = config.trainable_filter

        def grad_step(rng, state, batch):
            model = nnx.merge(state.model_def, state.params)
            model.train()
            observation, actions = batch

            def loss_fn(model, rng, observation, actions):
                return jnp.mean(model.compute_loss(rng, observation, actions, train=True))

            diff_state = nnx.DiffState(0, trainable_filter)
            loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, rng, observation, actions)
            return {"loss": loss, "grad_norm": optax.global_norm(grads)}, grads

        pgrad = jax.jit(grad_step,
                        in_shardings=(replicated_sharding, state_sharding, data_sharding),
                        out_shardings=({"loss": replicated_sharding,
                                        "grad_norm": replicated_sharding}, None))
        started = time.time()
        if args.shape_only:
            with jsh.set_mesh(mesh):
                traced = jax.eval_shape(grad_step, jax.random.key(config.seed), state, batch)
            snapshot(stage, started, {
                "executed": False,
                "grad_structure": _shape_tree(traced[1]),
                "info_shapes": _shape_tree(traced[0]),
            })
        else:
            with jsh.set_mesh(mesh):
                info, grads = pgrad(jax.random.fold_in(jax.random.key(config.seed), 0), state, batch)
            jax.block_until_ready(grads)
            grads_bytes = _bytes_by_device(grads)
            snapshot(stage, started, {
                "executed": True,
                "info": {key: float(value) for key, value in info.items()},
                "grads_bytes_per_device": grads_bytes,
                "grad_leaves": len(jax.tree_util.tree_leaves(grads)),
                "not_covered_by_this_probe": {
                    "updates_and_new_opt_state": (
                        "optax.apply_updates is deliberately not run; its transient is one more "
                        "trainable-sized pytree, so add the largest grads_bytes_per_device entry "
                        "to the measured peak."),
                    "checkpoint_write_buffers": "orbax save buffers are not allocated here",
                },
            })
        report["result"] = {
            "shape_only": args.shape_only,
            "max_nvidia_used_mib": max((row["memory_used_mib"] for entry in report["stages"]
                                        for row in entry["nvidia_visible"]), default=0),
            "max_sharded_bytes_per_device": max(
                (value for entry in report["stages"] for tree in _stage_trees(entry)
                 for value in tree.values()), default=0),
            "weights_modified": False,
            "checkpoint_written": False,
        }
    except Exception as error:  # an OOM during the probe is the answer, not a lost log
        report["error"] = {"stage": stage, "type": type(error).__name__,
                           "message": str(error)[-3000:]}
        path = out / f"capacity_{args.label}.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"CAPACITY PROBE FAILED in stage {stage}: {type(error).__name__}: "
              f"{str(error)[:400]}", file=sys.stderr)
        raise
    path = out / f"capacity_{args.label}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"capacity_report={path}")
    return 0


def _load_train_module():
    import importlib.util
    from pathlib import Path as _Path

    path = _Path("/opt/openpi/scripts/train.py")
    if not path.is_file():
        raise FileNotFoundError("pinned OpenPI scripts/train.py not found")
    spec = importlib.util.spec_from_file_location("openpi_train_entry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stage_trees(stage: dict) -> list[dict]:
    trees = []
    for key in ("sharded_bytes_per_device", "batch_bytes_per_device", "grads_bytes_per_device"):
        value = stage.get(key, {})
        trees.extend(sub for sub in value.values() if isinstance(sub, dict))
    return trees


if __name__ == "__main__":
    sys.exit(main())
