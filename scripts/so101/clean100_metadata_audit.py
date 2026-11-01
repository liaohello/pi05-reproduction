#!/usr/bin/env python3
"""Compare the view statistics with the training loader and the service metadata path.

Runs inside the pinned OpenPI image.  It executes the same
``make_train_config`` -> ``DataConfig`` -> ``load_norm_stats`` path that
``serve_so101.py`` uses to publish its metadata, without starting a server or a
model.  Field order, camera order, action dimension, transform order and every
per-dimension statistic are compared and reported.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, "/work/scripts/so101")

import openpi_so101  # noqa: E402
import openpi.training.checkpoints as _checkpoints  # noqa: E402
from openpi.policies import policy_config as _policy_config  # noqa: E402
import clean60_contracts as clean60  # noqa: E402

STAT_FIELDS = ("mean", "std", "q01", "q99")


def canonical_norm_sha(norm_stats: dict) -> str:
    payload = {
        split: {field: np.asarray(getattr(stats, field)).tolist()
                for field in STAT_FIELDS}
        for split, stats in norm_stats.items()
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def group_items(group, name: str) -> list:
    return list(getattr(group, name, ()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fsdp-devices", type=int, default=1)
    args = parser.parse_args()
    view = (args.view or Path(os.environ["SO101_TRAIN_VIEW"])).resolve()
    output_path = (args.output or (view / "metadata_audit.json")).resolve()
    meta = json.loads((view / "meta.json").read_text(encoding="utf-8"))
    norm_path = view / "assets/so101/norm_stats.json"
    norm_file = json.loads(norm_path.read_text(encoding="utf-8"))["norm_stats"]
    norm_file_sha = hashlib.sha256(norm_path.read_bytes()).hexdigest()

    run_root = Path(tempfile.mkdtemp(prefix="clean100_metadata_audit_"))
    config = openpi_so101.make_train_config(
        mode="serve", run_root=run_root, train_view=view,
        base_checkpoint=view, exp_name="metadata_audit", state_mode="on",
        fsdp_devices=args.fsdp_devices, action_horizon=10, seed=42,
        dtype="bfloat16", num_workers=0, batch_size=8, lr_warmup_steps=500,
        peak_lr=5e-5, decay_steps=25000, decay_lr=5e-6)
    data_config = config.data.create(config.assets_dirs, config.model)
    loader_norm = data_config.norm_stats
    loader_canonical_sha = canonical_norm_sha(loader_norm)

    with tempfile.TemporaryDirectory() as temporary:
        assets = Path(temporary) / "assets"
        asset_dir = assets / str(data_config.asset_id)
        asset_dir.mkdir(parents=True)
        (asset_dir / "norm_stats.json").write_bytes(norm_path.read_bytes())
        served_norm = _checkpoints.load_norm_stats(assets, data_config.asset_id)
        served_canonical_sha = canonical_norm_sha(served_norm)
        served_file_sha = hashlib.sha256(
            (asset_dir / "norm_stats.json").read_bytes()).hexdigest()

    per_dim = {}
    for split in ("state", "actions"):
        per_dim[split] = {}
        for field in STAT_FIELDS:
            view_values = np.asarray(norm_file[split][field], dtype=np.float64)
            loader_values = np.asarray(getattr(loader_norm[split], field), dtype=np.float64)
            served_values = np.asarray(getattr(served_norm[split], field), dtype=np.float64)
            per_dim[split][field] = {
                "view_vs_loader_max_abs": float(np.max(np.abs(view_values - loader_values))),
                "view_vs_served_max_abs": float(np.max(np.abs(view_values - served_values))),
                "loader_vs_served_max_abs": float(np.max(np.abs(loader_values - served_values))),
                "length": int(view_values.size),
                "finite": bool(np.isfinite(view_values).all()
                               and np.isfinite(loader_values).all()
                               and np.isfinite(served_values).all()),
            }

    dataset = openpi_so101.SO101EpisodeDataset(view, split="train", action_horizon=10)
    camera_order = list(openpi_so101.CAMERAS.values())
    data_inputs = group_items(data_config.data_transforms, "inputs")
    data_outputs = group_items(data_config.data_transforms, "outputs")
    model_inputs = group_items(data_config.model_transforms, "inputs")
    model_outputs = group_items(data_config.model_transforms, "outputs")
    repack_inputs = group_items(data_config.repack_transforms, "inputs")
    repack_outputs = group_items(data_config.repack_transforms, "outputs")
    from openpi import transforms as _transforms
    normalise = _transforms.Normalize(loader_norm,
                                      use_quantiles=bool(data_config.use_quantile_norm))
    unnormalise = _transforms.Unnormalize(loader_norm,
                                          use_quantiles=bool(data_config.use_quantile_norm))
    inference_source = inspect.getsource(_policy_config.create_trained_policy)
    source_unnorm = inference_source.find("Unnormalize")
    source_data_output = inference_source.find("data_transforms.outputs")
    source_repack_output = inference_source.find("repack_transforms.outputs")
    inference_order_verified = (
        source_unnorm >= 0 and source_data_output > source_unnorm
        and source_repack_output > source_data_output)
    worst = max(
        entry[key]
        for split in per_dim.values()
        for entry in split.values()
        for key in ("view_vs_loader_max_abs", "view_vs_served_max_abs",
                    "loader_vs_served_max_abs"))
    feature_order_ok = (meta.get("feature_names") == clean60.FEATURE_NAMES
                        == dataset.feature_names)
    cameras_ok = (meta.get("cameras") == clean60.CAMERAS == camera_order)
    passed = bool(
        worst <= 1e-6
        and norm_file_sha == meta.get("norm_stats_sha256")
        and served_file_sha == norm_file_sha
        and feature_order_ok and cameras_ok
        and meta.get("state_action_dim") == openpi_so101.ACTION_DIM
        and meta.get("feature_units") == clean60.FEATURE_UNITS
        and inference_order_verified)
    payload = {
        "schema_version": 1,
        "view": view.name,
        "view_sha256": meta["view_sha256"],
        "norm_stats_sha256": norm_file_sha,
        "norm_stats_canonical_sha256": meta.get("norm_stats_canonical_sha256"),
        "loader_norm_stats_canonical_sha256": loader_canonical_sha,
        "served_norm_stats_canonical_sha256": served_canonical_sha,
        "meta_norm_stats_canonical_sha256": meta.get("norm_stats_canonical_sha256"),
        "canonical_hashes_equal": bool(
            loader_canonical_sha == served_canonical_sha
            == meta.get("norm_stats_canonical_sha256")),
        "served_norm_stats_file_sha256": served_file_sha,
        "asset_id": str(data_config.asset_id),
        "per_dimension": per_dim,
        "worst_max_abs_error": worst,
        "field_order": {
            "meta_feature_names": meta.get("feature_names"),
            "contract_feature_names": clean60.FEATURE_NAMES,
            "loader_feature_names": dataset.feature_names,
            "matches": feature_order_ok,
        },
        "camera_order": {
            "meta": meta.get("cameras"),
            "loader": camera_order,
            "matches": cameras_ok,
        },
        "action_dim": {"meta": meta.get("state_action_dim"),
                       "contract": openpi_so101.ACTION_DIM,
                       "loader": openpi_so101.ACTION_DIM},
        "transform_order": {
            "train_input": [type(item).__name__ for item in
                            [*repack_inputs, *data_inputs, normalise, *model_inputs]],
            "inference_output": [type(item).__name__ for item in
                                 [*model_outputs, unnormalise, *data_outputs,
                                  *repack_outputs]],
            "inference_source_sha256": hashlib.sha256(
                inference_source.encode()).hexdigest(),
            "inference_order_verified": inference_order_verified,
        },
        "service_metadata_fields": {
            "action_horizon": 10,
            "control_hz": 50,
            "state_key": "observation.state",
            "cameras": camera_order,
            "norm_stats_sha256": served_canonical_sha,
            "norm_stats_file_sha256": served_file_sha,
        },
        "note": "the metadata path is executed with the pinned image's own "
                "make_train_config/load_norm_stats code; no server or model is started",
        "passed": passed,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items()
                      if key != "per_dimension"}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
