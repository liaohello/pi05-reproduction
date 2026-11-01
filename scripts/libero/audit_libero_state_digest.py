#!/usr/bin/env python3
"""Restore one complete LIBERO TrainState and emit tensor-level digests."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import libero_data  # noqa: E402
import train_libero  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def flatten_nnx_state(value):
    from flax.traverse_util import flatten_dict

    return flatten_dict(value.to_pure_dict(), sep="/")


def sharding_summary(sharding):
    import jax
    from flax.traverse_util import flatten_dict

    def group(name: str) -> str:
        if "lora" in name:
            return "language_lora"
        if "/img/" in name:
            return "vision_encoder"
        if "llm" in name and "_1" in name:
            return "action_expert"
        if any(token in name for token in (
            "action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out")):
            return "projection"
        if "llm" in name:
            return "language_backbone"
        return "other"

    def group_examples(tree):
        values = flatten_dict(tree.to_pure_dict(), sep="/")
        examples = {}
        spec_counts = {}
        for name, value in sorted(values.items()):
            spec = str(value.spec)
            spec_counts[spec] = spec_counts.get(spec, 0) + 1
            name_group = group(name)
            if name_group not in examples:
                examples[name_group] = {
                    "parameter": name,
                    "spec": spec,
                    "device_count": len(value.device_set),
                }
        return {"group_examples": examples, "spec_tensor_counts": spec_counts}

    optimizer = {}
    for path, value in jax.tree_util.tree_leaves_with_path(sharding.opt_state):
        text = jax.tree_util.keystr(path)
        kind = next(
            (name for name in ("mu", "nu", "count") if name in text), "other")
        row = optimizer.setdefault(kind, {
            "tensor_count": 0, "spec_counts": {}, "example_path": text,
            "example_spec": str(value.spec),
        })
        row["tensor_count"] += 1
        spec = str(value.spec)
        row["spec_counts"][spec] = row["spec_counts"].get(spec, 0) + 1
    return {
        "params": group_examples(sharding.params),
        "ema_params": group_examples(sharding.ema_params),
        "opt_state": optimizer,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--fsdp-devices", type=int, choices=(2, 4), required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    import numpy as np
    from openpi.training import sharding

    protocol = train_libero.load_protocol(
        args.protocol, args.expected_protocol_sha256)
    view = libero_data.load_view(args.view)
    if view["view_sha256"] != protocol["data"]["view_sha256"]:
        raise ValueError("state digest view differs from protocol")
    saved_path = args.run_root / args.config_file
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    train_libero.validate_saved_identity(saved, saved["identity"])
    if saved["identity"]["protocol_sha256"] != protocol["_sha256"]:
        raise ValueError("saved state protocol hash differs from the current protocol")
    if saved["identity"]["view_sha256"] != view["view_sha256"]:
        raise ValueError("saved state view hash differs from the current view")
    exp_name = saved["identity"]["exp_name"]
    config = train_libero.make_config(
        SimpleNamespace(
            exp_name=exp_name,
            mode="probe_resume",
            view=str(args.view),
            base_checkpoint=str(args.base_checkpoint),
            fsdp_devices=args.fsdp_devices,
        ),
        protocol,
        view,
        args.run_root,
        int(saved["target_steps_for_this_process"]),
    )
    train, _ = train_libero._load_train_module()
    mesh = sharding.make_mesh(args.fsdp_devices)
    manager, resuming = train._checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=False,
        resume=True,
    )
    if not resuming:
        raise ValueError("state digest did not find a checkpoint")
    template, template_sharding = train.init_train_state(
        config, jax.random.split(jax.random.key(config.seed))[1], mesh, resume=True)
    state = train._checkpoints.restore_state(manager, template, None)
    state = jax.block_until_ready(state)
    step = int(jax.device_get(state.step))
    if step != args.expected_step:
        raise ValueError(
            f"restored TrainState.step {step} differs from expected {args.expected_step}")

    sources = {
        "params": flatten_nnx_state(state.params),
        "ema_params": flatten_nnx_state(state.ema_params),
    }
    optimizer = {}
    for path, value in jax.tree_util.tree_leaves_with_path(state.opt_state):
        optimizer[jax.tree_util.keystr(path)] = value
    sources["opt_state"] = optimizer

    records = {}
    for section, values in sources.items():
        section_records = {}
        for name, value in sorted(values.items(), key=lambda item: str(item[0])):
            array = np.asarray(jax.device_get(value))
            dtype_text = str(value.dtype)
            finite = (
                bool(jax.device_get(jnp.all(jnp.isfinite(value))))
                if dtype_text.startswith(("float", "bfloat")) else True
            )
            section_records[str(name)] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "bytes": int(array.nbytes),
                "finite": finite,
                "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            }
        records[section] = section_records
    payload = {
        "schema_version": 1,
        "run_root": str(args.run_root),
        "config_file": args.config_file,
        "config_sha256": sha256_file(saved_path),
        "identity_sha256": saved["identity_sha256"],
        "train_state_step": step,
        "ema_decay": float(state.ema_decay),
        "sections": records,
        "section_counts": {
            section: len(values) for section, values in records.items()
        },
        "section_bytes": {
            section: sum(row["bytes"] for row in values.values())
            for section, values in records.items()
        },
        "sharding": sharding_summary(template_sharding),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.out),
        "sha256": sha256_file(args.out),
        "train_state_step": step,
        "ema_decay": payload["ema_decay"],
        "section_counts": payload["section_counts"],
        "section_bytes": payload["section_bytes"],
        "sharding": payload["sharding"],
    }, indent=2, sort_keys=True))
    manager.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
