#!/usr/bin/env python3
"""Train the requested all-49 LIBERO bowl-to-plate LoRA adapter with FSDP."""
from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bowl_plate_49  # noqa: E402


def jsonable(value):
    if dataclasses.is_dataclass(value):
        return {
            field.name: jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return repr(value)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_train_module():
    import importlib.util

    path = Path("/opt/openpi/scripts/train.py")
    if not path.is_file():
        raise FileNotFoundError("pinned OpenPI scripts/train.py is unavailable")
    spec = importlib.util.spec_from_file_location("openpi_train_entry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def finite_metrics(info: dict) -> tuple[bool, dict]:
    import jax
    import numpy as np

    values = {
        key: float(np.asarray(jax.device_get(value)).reshape(-1)[0])
        for key, value in info.items()
    }
    return all(np.isfinite(value) for value in values.values()), values


def snapshot_sources(
    run_root: Path,
    protocol_path: Path,
    view_path: Path,
    trainer_path: Path,
) -> dict:
    source = run_root / "source_snapshot"
    source.mkdir(parents=True, exist_ok=False)
    copied = {}
    targets = {
        "protocol": (protocol_path, source / "configs/libero/bowl_plate_49_lora.json"),
        "loader": (HERE / "bowl_plate_49.py", source / "scripts/libero/bowl_plate_49.py"),
        "trainer": (trainer_path, source / "scripts/libero/train_bowl_plate_49.py"),
        "view": (view_path / "view.json", source / "view/view.json"),
        "norm_stats": (
            view_path / "assets/libero_goal8/norm_stats.json",
            source / "view/assets/libero_goal8/norm_stats.json",
        ),
    }
    for name, (origin, destination) in targets.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, destination)
        copied[name] = {
            "path": str(destination.relative_to(run_root)),
            "sha256": bowl_plate_49.sha256_file(destination),
            "bytes": destination.stat().st_size,
        }
    write_json(
        source / "source_manifest.json",
        {"schema_version": 1, "files": copied},
    )
    return copied


def identity(
    args,
    protocol: dict,
    view: dict,
    runtime: dict,
    base_tree: dict,
    schedule: dict,
) -> dict:
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
        "batch_size": int(args.batch_size),
        "fsdp_devices": int(args.fsdp_devices),
        "data_loader_workers": int(protocol["training"]["data_loader_workers"]),
        "exp_name": args.exp_name,
        "effective_schedule": schedule,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("probe", "full", "resume"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--probe-steps", type=int, default=1)
    parser.add_argument("--fsdp-devices", type=int, choices=(2, 4), required=True)
    parser.add_argument("--image-digest", required=True)
    args = parser.parse_args()

    protocol = bowl_plate_49.load_protocol(
        args.protocol, args.expected_protocol_sha256)
    view = bowl_plate_49.load_view(args.view, protocol)
    if not args.run_id.startswith(protocol["experiment_prefix"]):
        raise ValueError(
            f"run id must start with {protocol['experiment_prefix']}"
        )
    if args.image_digest != protocol["runtime"]["image_digest"]:
        raise ValueError("runtime image digest differs from the protocol")
    allowed = [int(value) for value in protocol["training"]["allowed_batch_sizes"]]
    if args.batch_size not in allowed:
        raise ValueError(f"batch size must be one of {allowed}")
    if args.mode == "probe" and not 1 <= args.probe_steps <= 5:
        raise ValueError("probe steps must be 1 through 5")

    run_root = args.run_root.resolve()
    fresh = args.mode in {"probe", "full"}
    if fresh and run_root.exists() and any(run_root.iterdir()):
        raise ValueError(f"refusing non-empty run directory: {run_root}")
    if not fresh and not (run_root / "config_initial.json").is_file():
        raise ValueError("resume requires config_initial.json")
    run_root.mkdir(parents=True, exist_ok=True)

    base_tree = bowl_plate_49.sha256_tree(args.base_checkpoint)
    if base_tree["sha256"] != protocol["base"]["tree_sha256"]:
        raise ValueError("base checkpoint tree differs from registered pi05_base")
    runtime = bowl_plate_49.runtime_hashes()
    for key, expected in protocol["runtime"].items():
        if key in runtime and runtime[key] != expected:
            raise ValueError(f"runtime source {key} differs from the protocol")
    schedule = bowl_plate_49.effective_schedule(protocol, args.batch_size)
    run_identity = identity(args, protocol, view, runtime, base_tree, schedule)
    if fresh:
        copied = snapshot_sources(
            run_root, args.protocol, args.view, Path(__file__))
    else:
        saved = json.loads(
            (run_root / "config_initial.json").read_text(encoding="utf-8"))
        saved_identity = saved.get("identity")
        saved_hash = saved.get("identity_sha256")
        if (
            not isinstance(saved_identity, dict)
            or not saved_hash
            or bowl_plate_49.canonical_sha256(saved_identity) != saved_hash
        ):
            raise ValueError("saved run identity is invalid")
        if saved_identity != run_identity:
            raise ValueError("resume identity differs from config_initial.json")
        if saved.get("run_id") != args.run_id:
            raise ValueError("resume run ID differs from config_initial.json")
        copied = saved.get("source_snapshot", {})

    import jax
    from openpi.training import sharding

    jax.config.update(
        "jax_compilation_cache_dir",
        os.environ.get(
            "JAX_COMPILATION_CACHE_DIR",
            str(Path("~/.cache/jax").expanduser()),
        ),
    )
    device_count = jax.device_count()
    if device_count != args.fsdp_devices:
        raise ValueError(
            f"expected {args.fsdp_devices} visible devices, got {device_count}"
        )
    if args.batch_size % device_count:
        raise ValueError("batch size must divide across visible devices")
    mesh = sharding.make_mesh(args.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec())

    target_steps = (
        int(schedule["optimizer_updates"])
        if args.mode in {"full", "resume"}
        else int(args.probe_steps)
    )
    config = bowl_plate_49.make_config(args, protocol, view, run_root, target_steps)
    train, train_path = load_train_module()

    manager = None
    resuming = False
    if args.mode in {"full", "resume"}:
        manager, resuming = train._checkpoints.initialize_checkpoint_dir(
            config.checkpoint_dir,
            keep_period=config.keep_period,
            overwrite=args.mode == "full",
            resume=args.mode == "resume",
        )
        if args.mode == "full" and resuming:
            raise RuntimeError("fresh run unexpectedly requested checkpoint resume")
        if args.mode == "resume" and not resuming:
            raise RuntimeError("resume found no checkpoint to restore")

    loader = bowl_plate_49.create_train_loader(
        config,
        view_path=args.view,
        data_root=args.data_root,
        sharding=data_sharding,
    )
    data_iter = iter(loader)

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    state, state_sharding = train.init_train_state(
        config, init_rng, mesh, resume=resuming)
    state = jax.block_until_ready(state)
    if resuming:
        state = train._checkpoints.restore_state(manager, state, loader)
        state = jax.block_until_ready(state)
    start_step = int(jax.device_get(state.step))
    if not resuming and start_step != 0:
        raise ValueError("fresh pi05_base load did not start at TrainState.step 0")
    if start_step > target_steps:
        raise ValueError("restored step exceeds the target schedule")

    snapshot = {
        "schema_version": 1,
        "mode": args.mode,
        "run_id": args.run_id,
        "run_root": str(run_root),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol_path": str(args.protocol),
        "protocol_sha256": protocol["_sha256"],
        "target_steps": target_steps,
        "effective_schedule": schedule,
        "batch_size": args.batch_size,
        "fsdp_devices": args.fsdp_devices,
        "mesh_shape": {str(key): int(value) for key, value in mesh.shape.items()},
        "batch_per_device": args.batch_size // device_count,
        "identity": run_identity,
        "identity_sha256": bowl_plate_49.canonical_sha256(run_identity),
        "source_snapshot": copied,
        "train_episode_ids": list(bowl_plate_49.EXPECTED_EPISODES),
        "train_entry": str(train_path),
        "config": jsonable(config),
    }
    write_json(
        run_root / (
            "config_initial.json" if fresh else "config_resume.json"
        ),
        snapshot,
    )
    for _ in range(start_step):
        next(data_iter)
    batch = next(data_iter)

    ptrain_step = jax.jit(
        functools.partial(train.train_step, config),
        in_shardings=(replicated_sharding, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    print(json.dumps({
        "event": "training_started",
        "run_id": args.run_id,
        "mode": args.mode,
        "device_count": device_count,
        "device_ids": [device.id for device in jax.devices()],
        "fsdp_devices": args.fsdp_devices,
        "mesh_shape": snapshot["mesh_shape"],
        "global_batch_size": args.batch_size,
        "batch_per_device": snapshot["batch_per_device"],
        "target_steps": target_steps,
        "start_train_state_step": start_step,
        "train_episode_count": len(bowl_plate_49.EXPECTED_EPISODES),
        "train_frames": 4563,
        "action_horizon": config.model.action_horizon,
        "ema_decay": config.ema_decay,
    }), flush=True)

    metrics_path = run_root / "metrics.jsonl"
    started = time.monotonic()
    last_metrics = None
    with metrics_path.open("a", encoding="utf-8") as metrics_log:
        for update in range(start_step + 1, target_steps + 1):
            with sharding.set_mesh(mesh):
                state, info = ptrain_step(train_rng, state, batch)
            finite, metrics = finite_metrics(info)
            actual_step = int(jax.device_get(state.step))
            if not finite:
                raise FloatingPointError(
                    f"non-finite training metrics at update {update}: {metrics}"
                )
            if actual_step != update:
                raise RuntimeError(
                    f"TrainState step {actual_step} differs from update {update}"
                )
            if not metrics.get("grad_norm", 0.0) > 0.0:
                raise RuntimeError(f"zero gradient at update {update}")
            record = {
                "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "train_state_step": actual_step,
                "metrics": metrics,
            }
            metrics_log.write(json.dumps(record, sort_keys=True) + "\n")
            metrics_log.flush()
            if update % config.log_interval == 0 or update == target_steps:
                print(
                    f"Step {update}: loss={metrics['loss']:.8f}, "
                    f"grad_norm={metrics['grad_norm']:.8f}, "
                    f"param_norm={metrics['param_norm']:.8f}",
                    flush=True,
                )
                print(json.dumps(record, sort_keys=True), flush=True)
            last_metrics = metrics
            if manager is not None and update in schedule["checkpoint_steps"]:
                train._checkpoints.save_state(
                    manager, state, loader, actual_step)
                manager.wait_until_finished()
                metadata_step = int(manager.latest_step())
                if metadata_step != actual_step:
                    raise RuntimeError(
                        "checkpoint step differs from TrainState.step: "
                        f"{metadata_step} != {actual_step}"
                    )
                retained = [int(value) for value in manager.all_steps()]
                checkpoint = {
                    "train_state_step": actual_step,
                    "checkpoint_metadata_step": metadata_step,
                    "retained_steps": retained,
                }
                write_json(
                    run_root / f"checkpoint_steps_{actual_step}.json", checkpoint)
                print(json.dumps({
                    "event": "checkpoint_saved",
                    **checkpoint,
                }, sort_keys=True), flush=True)
            batch = next(data_iter)

    elapsed = time.monotonic() - started
    final_step = int(jax.device_get(state.step))
    updates_this_process = target_steps - start_step
    result = {
        "schema_version": 1,
        "status": "complete",
        "run_id": args.run_id,
        "mode": args.mode,
        "final_train_state_step": final_step,
        "start_train_state_step": start_step,
        "optimizer_updates": target_steps,
        "updates_this_process": updates_this_process,
        "effective_schedule": schedule,
        "batch_size": args.batch_size,
        "fsdp_devices": args.fsdp_devices,
        "elapsed_s": elapsed,
        "seconds_per_update": (
            elapsed / updates_this_process if updates_this_process else 0.0
        ),
        "last_metrics": last_metrics,
        "checkpoint_directory": str(config.checkpoint_dir),
        "checkpoint_steps": (
            [int(value) for value in manager.all_steps()]
            if manager is not None else []
        ),
        "closed_loop_success_evaluated": False,
    }
    write_json(run_root / "training_result.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
