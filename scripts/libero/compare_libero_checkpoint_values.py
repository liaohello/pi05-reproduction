#!/usr/bin/env python3
"""Dump mismatched reference tensors and quantify resumed-state differences."""
from __future__ import annotations

import argparse
import gc
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


def optimizer_values(state):
    import jax

    return {
        jax.tree_util.keystr(path): value
        for path, value in jax.tree_util.tree_leaves_with_path(state.opt_state)
    }


def load_state(args, *, run_root: str, config_file: str):
    import jax
    from openpi.training import sharding

    protocol = train_libero.load_protocol(
        args.protocol, args.expected_protocol_sha256)
    view = libero_data.load_view(args.view)
    if view["view_sha256"] != protocol["data"]["view_sha256"]:
        raise ValueError("state comparison view differs from protocol")
    saved_path = args.run_root / run_root / config_file
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    train_libero.validate_saved_identity(saved, saved["identity"])
    config = train_libero.make_config(
        SimpleNamespace(
            exp_name=saved["identity"]["exp_name"],
            mode="probe_resume",
            view=str(args.view),
            base_checkpoint=str(args.base_checkpoint),
            fsdp_devices=args.fsdp_devices,
        ),
        protocol,
        view,
        Path(run_root),
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
        raise ValueError("state comparison found no checkpoint")
    template, _ = train.init_train_state(
        config, jax.random.split(jax.random.key(config.seed))[1], mesh, resume=True)
    state = train._checkpoints.restore_state(manager, template, None)
    state = jax.block_until_ready(state)
    step = int(jax.device_get(state.step))
    if step != args.expected_step:
        raise ValueError(f"restored step {step} differs from {args.expected_step}")
    return state, manager, {
        "params": flatten_nnx_state(state.params),
        "ema_params": flatten_nnx_state(state.ema_params),
        "opt_state": optimizer_values(state),
    }


def dump(args, mismatches):
    import jax
    import numpy as np

    state, manager, values = load_state(
        args,
        run_root=args.reference_run_root,
        config_file=args.reference_config,
    )
    records = []
    values_dir = args.out / "values"
    values_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(mismatches):
        section, path = item["section"], item["path"]
        array = np.asarray(jax.device_get(values[section][path]))
        relative = Path(section) / f"{index:04d}.npy"
        destination = args.out / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, array)
        records.append({
            "section": section,
            "path": path,
            "file": str(relative),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "bytes": int(array.nbytes),
            "sha256": sha256_file(destination),
        })
    payload = {
        "schema_version": 1,
        "mode": "dump",
        "train_state_step": args.expected_step,
        "tensor_count": len(records),
        "total_bytes": sum(row["bytes"] for row in records),
        "records": records,
    }
    manifest = args.out / "manifest.json"
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "manifest": str(manifest),
        "sha256": sha256_file(manifest),
        "tensor_count": payload["tensor_count"],
        "total_bytes": payload["total_bytes"],
    }, indent=2, sort_keys=True))
    manager.close()
    del state, values
    gc.collect()
    jax.clear_caches()
    return 0


def quantify(args):
    import jax
    import numpy as np

    manifest = json.loads((args.out / "manifest.json").read_text(encoding="utf-8"))
    state, manager, values = load_state(
        args,
        run_root=args.resumed_run_root,
        config_file=args.resumed_config,
    )
    records = []
    sections = {}
    for item in manifest["records"]:
        section, path = item["section"], item["path"]
        reference = np.load(args.out / item["file"], mmap_mode="r")
        resumed = np.asarray(jax.device_get(values[section][path]))
        if list(reference.shape) != list(resumed.shape):
            raise ValueError(f"{section}/{path}: shape mismatch")
        if str(reference.dtype) != str(resumed.dtype):
            raise ValueError(f"{section}/{path}: dtype mismatch")
        if reference.dtype.kind in "fc" or str(reference.dtype).startswith("bfloat"):
            left = np.asarray(reference, dtype=np.float32)
            right = np.asarray(resumed, dtype=np.float32)
            difference = np.abs(left - right)
            max_abs = float(np.max(difference)) if difference.size else 0.0
            mean_abs = float(np.mean(difference)) if difference.size else 0.0
            rms = float(np.sqrt(np.mean(difference * difference))) if difference.size else 0.0
            reference_max_abs = float(np.max(np.abs(left))) if left.size else 0.0
            relative_max_abs = (
                max_abs / reference_max_abs if reference_max_abs > 0 else max_abs)
            unequal = int(np.count_nonzero(reference != resumed))
            finite = bool(np.isfinite(reference).all() and np.isfinite(resumed).all())
        else:
            unequal = int(np.count_nonzero(reference != resumed))
            max_abs = mean_abs = rms = 0.0
            reference_max_abs = relative_max_abs = 0.0
            finite = True
        row = {
            "section": section,
            "path": path,
            "elements": int(reference.size),
            "unequal_elements": unequal,
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "rms": rms,
            "reference_max_abs": reference_max_abs,
            "relative_max_abs": relative_max_abs,
            "finite": finite,
        }
        summary = sections.setdefault(section, {
            "tensors": 0,
            "elements": 0,
            "unequal_elements": 0,
            "max_abs": 0.0,
            "max_relative_abs": 0.0,
            "finite": True,
        })
        summary["tensors"] += 1
        summary["elements"] += int(reference.size)
        summary["unequal_elements"] += unequal
        summary["max_abs"] = max(summary["max_abs"], max_abs)
        summary["max_relative_abs"] = max(
            summary["max_relative_abs"], relative_max_abs)
        summary["finite"] &= finite
        records.append(row)
    parameter_within_tolerance = sections["params"]["max_abs"] <= 1e-5
    ema_within_tolerance = sections["ema_params"]["max_abs"] <= 1e-5
    optimizer_difference_quantified = sections["opt_state"]["finite"]
    passed = (
        parameter_within_tolerance
        and ema_within_tolerance
        and optimizer_difference_quantified
    )
    payload = {
        "schema_version": 1,
        "mode": "compare",
        "passed": passed,
        "parameter_tolerance_max_abs": 1e-5,
        "ema_tolerance_max_abs": 1e-5,
        "optimizer_tolerance": "quantified_only",
        "parameter_within_tolerance": parameter_within_tolerance,
        "ema_within_tolerance": ema_within_tolerance,
        "optimizer_difference_quantified": optimizer_difference_quantified,
        "bitwise_identical": False,
        "train_state_step": args.expected_step,
        "sections": sections,
        "overall_max_abs": max((row["max_abs"] for row in records), default=0.0),
        "records": records,
    }
    destination = args.out / "comparison.json"
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(destination),
        "sha256": sha256_file(destination),
        "passed": passed,
        "bitwise_identical": payload["bitwise_identical"],
        "parameter_within_tolerance": parameter_within_tolerance,
        "ema_within_tolerance": ema_within_tolerance,
        "optimizer_difference_quantified": optimizer_difference_quantified,
        "overall_max_abs": payload["overall_max_abs"],
        "sections": sections,
    }, indent=2, sort_keys=True))
    manager.close()
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dump", "compare"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--reference-run-root", required=True)
    parser.add_argument("--reference-config", required=True)
    parser.add_argument("--resumed-run-root", required=True)
    parser.add_argument("--resumed-config", required=True)
    parser.add_argument("--fsdp-devices", type=int, choices=(2, 4), required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    mismatches = [
        {"section": item["section"], "path": item["path"]}
        for item in comparison["mismatches"] if "path" in item
    ]
    if not mismatches:
        raise ValueError("digest comparison has no mismatched tensor paths")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "dump":
        return dump(args, mismatches)
    return quantify(args)


if __name__ == "__main__":
    sys.exit(main())
