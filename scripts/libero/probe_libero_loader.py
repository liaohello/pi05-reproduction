#!/usr/bin/env python3
"""CPU check of the exact filtered LIBERO OpenPI data loader."""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import libero_data  # noqa: E402
import train_libero  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import jax
    import numpy as np

    protocol = train_libero.load_protocol(
        args.protocol, args.expected_protocol_sha256)
    view = libero_data.load_view(args.view)
    if view["view_sha256"] != protocol["data"]["view_sha256"]:
        raise ValueError("loader probe view differs from protocol")
    config = train_libero.make_config(
        SimpleNamespace(
            exp_name="libero-pi05-loader-probe",
            mode="probe",
            view=str(args.view),
            base_checkpoint="/not-used",
            fsdp_devices=4,
        ),
        protocol,
        view,
        args.out.parent / "loader_checkpoints",
        2,
    )
    config = dataclasses.replace(config, num_workers=0)
    loader = libero_data.create_filtered_data_loader(
        config,
        view_path=args.view,
        data_root=args.data_root,
        sharding=None,
        shuffle=False,
        num_batches=1,
        split="train",
    )
    batch = next(iter(loader))
    observation, actions = batch
    state = np.asarray(jax.device_get(observation.state))
    actions = np.asarray(jax.device_get(actions))
    images = {
        name: np.asarray(jax.device_get(value))
        for name, value in observation.images.items()
    }
    masks = {
        name: np.asarray(jax.device_get(value))
        for name, value in observation.image_masks.items()
    }
    mask_summary = {
        name: {
            "shape": list(value.shape),
            "all_true": bool(value.all()),
            "all_false": bool((~value).all()),
        }
        for name, value in masks.items()
    }
    right = images["right_wrist_0_rgb"]
    right_stats = {
        "min": float(np.min(right)),
        "max": float(np.max(right)),
        "mean": float(np.mean(right)),
        "nan_count": int(np.isnan(right).sum()),
        "nonzero_count": int(np.count_nonzero(right)),
        "numel": int(right.size),
    }
    checks = {
        "global_batch_8": state.shape[0] == 8 and actions.shape[0] == 8,
        "state_32": state.shape[1:] == (32,),
        "actions_25x32": (
            actions.shape[1:]
            == (protocol["data"]["action_horizon"], 32)
        ),
        "three_camera_slots": set(images) == {
            "base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"},
        "images_224_rgb": all(
            value.shape[1:] == (224, 224, 3) for value in images.values()),
        "right_wrist_model_fill_minus_one": (
            right_stats["nonzero_count"] == right_stats["numel"]
            and right_stats["min"] == -1.0
            and right_stats["max"] == -1.0
        ),
        "right_wrist_finite": right_stats["nan_count"] == 0,
        "right_wrist_masked": bool(
            (~masks["right_wrist_0_rgb"]).all()),
        "state_finite": bool(np.isfinite(state).all()),
        "actions_finite": bool(np.isfinite(actions).all()),
    }
    output = {
        "schema_version": 1,
        "protocol_sha256": protocol["_sha256"],
        "view_sha256": view["view_sha256"],
        "train_episode_ids": view["train_episode_ids"],
        "validation_episode_ids": view["validation_episode_ids"],
        "batch_shapes": {
            "state": list(state.shape),
            "actions": list(actions.shape),
            "images": {name: list(value.shape) for name, value in images.items()},
        },
        "image_masks": mask_summary,
        "right_wrist_pixel_stats": right_stats,
        "right_wrist_pre_scale_zero_evidence": "norm_audit_v3c sample_metadata.right_wrist_zero",
        "right_wrist_note": (
            "LiberoInputs creates a pre-scale zero image. "
            "Observation.from_dict maps uint8 zero to -1.0 for the model; "
            "the registered image mask remains false."
        ),
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
