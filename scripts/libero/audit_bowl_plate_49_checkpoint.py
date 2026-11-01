#!/usr/bin/env python3
"""Read back one trained checkpoint and emit a compact checkpoint audit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bowl_plate_49  # noqa: E402


def _restore_state(manager, template, step, checkpoints, at, jax, ocp):
    """Restore into a shape template; Orbax rejects sharding=None off-device."""
    with at.disable_typechecking():
        train_state_item, params_item = checkpoints._split_params(template)
    device_sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])

    def restore_args(target):
        shardings = jax.tree.map(
            lambda value: device_sharding
            if isinstance(value, jax.ShapeDtypeStruct) else None,
            target,
            is_leaf=lambda value: value is None,
        )
        return ocp.checkpoint_utils.construct_restore_args(target, shardings)

    items = {
        "train_state": train_state_item,
        "params": {"params": params_item},
    }
    restored = manager.restore(
        step,
        items=items,
        restore_kwargs={
            name: {"restore_args": restore_args(item)}
            for name, item in items.items()
        },
    )
    return checkpoints._merge_params(restored["train_state"], restored["params"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import jax
    import numpy as np
    import orbax.checkpoint as ocp
    from openpi.shared import array_typing as at
    from openpi.training import checkpoints, sharding

    protocol = bowl_plate_49.load_protocol(
        args.protocol, args.expected_protocol_sha256)
    view = bowl_plate_49.load_view(args.view, protocol)
    root = args.checkpoint_root.resolve()
    step_dir = root / str(args.step)
    if not step_dir.is_dir():
        raise FileNotFoundError(f"checkpoint step directory is missing: {step_dir}")
    config = bowl_plate_49.make_config(
        SimpleNamespace(
            exp_name="bowl_plate_49_lora",
            mode="probe",
            view=str(args.view),
            base_checkpoint="/not-used-for-inspection",
            batch_size=8,
            fsdp_devices=1,
        ),
        protocol,
        view,
        root.parents[2],
        int(protocol["training"]["optimizer_updates"]),
    )
    train, _ = bowl_plate_49.load_train_module()
    mesh = sharding.make_mesh(1)
    template, _ = train.init_train_state(
        config,
        jax.random.split(jax.random.key(config.seed))[1],
        mesh,
        resume=True,
    )
    manager = ocp.CheckpointManager(
        root,
        item_handlers={
            "assets": checkpoints.CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(create=False),
    )
    state = _restore_state(manager, template, args.step, checkpoints, at, jax, ocp)
    restored_step = int(jax.device_get(state.step))
    if restored_step != args.step:
        raise ValueError(
            f"restored TrainState.step {restored_step} != {args.step}"
        )
    params = jax.tree.leaves(state.params.to_pure_dict())
    param_norm_squared = sum(
        float(np.sum(np.asarray(jax.device_get(value), dtype=np.float64) ** 2))
        for value in params
    )
    norm_path = step_dir / "assets/libero_goal8/norm_stats.json"
    norm_sha = bowl_plate_49.sha256_file(norm_path)
    if norm_sha != protocol["data"]["norm_stats_sha256"]:
        raise ValueError("checkpoint Norm Stats differ from the protocol")
    step_files = [
        path for path in step_dir.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    payload = {
        "schema_version": 1,
        "checkpoint_root": str(root),
        "requested_step": args.step,
        "restored_train_state_step": restored_step,
        "retained_steps": [int(value) for value in manager.all_steps()],
        "step_files": len(step_files),
        "step_bytes": int(sum(path.stat().st_size for path in step_files)),
        "has_params": (step_dir / "params").is_dir(),
        "has_train_state": (step_dir / "train_state").is_dir(),
        "ema_params_present": state.ema_params is not None,
        "optimizer_state_restored": state.opt_state is not None,
        "parameter_tree_norm": param_norm_squared ** 0.5,
        "norm_stats_sha256": norm_sha,
        "protocol_sha256": protocol["_sha256"],
        "view_sha256": view["view_sha256"],
        "passed": (
            restored_step == args.step
            and (step_dir / "params").is_dir()
            and (step_dir / "train_state").is_dir()
            and state.ema_params is None
            and state.opt_state is not None
            and norm_sha == protocol["data"]["norm_stats_sha256"]
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.out),
        "sha256": bowl_plate_49.sha256_file(args.out),
        **payload,
    }, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
