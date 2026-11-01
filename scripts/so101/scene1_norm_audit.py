#!/usr/bin/env python3
"""Audit clean60 action/state normalization using the pinned OpenPI transforms.

The report is written into the ignored, versioned training view. It verifies the
52/12 split, train-only statistics, the exact training data-loader transform
order, inverse policy output transforms, and the 12 -> 32 -> 12 action path.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import scene1_openpi_so101 as openpi_so101  # noqa: E402
import openpi.transforms as transforms  # noqa: E402
from openpi.policies import policy_config  # noqa: E402
import clean60_contracts as clean60  # noqa: E402

UNITS = ["degree"] * 5 + ["percent"] + ["degree"] * 5 + ["percent"]
STAT_FIELDS = ("mean", "std", "q01", "q99")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    """Hash every relative file path and byte in a checkpoint directory."""
    if path.is_file():
        return sha256_file(path)
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"checkpoint contains no files: {path.name}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def per_dimension_stats(values: np.ndarray) -> dict:
    return {
        "mean": [float(v) for v in values.mean(axis=0)],
        "std": [float(v) for v in values.std(axis=0)],
        "q01": [float(v) for v in np.quantile(values, 0.01, axis=0)],
        "q99": [float(v) for v in np.quantile(values, 0.99, axis=0)],
        "min": [float(v) for v in values.min(axis=0)],
        "max": [float(v) for v in values.max(axis=0)],
    }


def group_items(group, name: str) -> list:
    return list(getattr(group, name, ()))


def apply_transforms(items: list, values: dict) -> dict:
    result = values
    for item in items:
        result = item(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", type=Path, default=None)
    parser.add_argument("--run-config", type=Path, default=None)
    parser.add_argument("--base-checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None,
                        help="audit destination; defaults to view/action_norm_audit.json")
    parser.add_argument("--fsdp-devices", type=int, default=1,
                        help="only resolves OpenPI config; this audit itself is CPU-only")
    parser.add_argument("--paligemma-variant", default="gemma_2b_lora")
    parser.add_argument("--action-expert-variant", default="gemma_300m")
    parser.add_argument("--limit-transform-samples", type=int, default=0,
                        help="deprecated; 0 audits every real action")
    parser.add_argument("--expected-episodes", type=int, default=64)
    parser.add_argument("--expected-train", type=int, default=52)
    parser.add_argument("--expected-validation", type=int, default=12)
    args = parser.parse_args()

    view = (args.view or Path(os.environ["SO101_TRAIN_VIEW"])).resolve()
    registered_config = (
        json.loads(args.run_config.read_text(encoding="utf-8"))
        if args.run_config and args.run_config.is_file() else {})
    optimisation = registered_config.get("optimisation", {})
    warmup_steps = int(optimisation.get("warmup_updates", 1000))
    decay_steps = int(optimisation.get("cosine_decay_updates", 25000))
    peak_lr = float(optimisation.get("peak_lr", 5e-5))
    final_lr = float(optimisation.get("final_lr", 5e-6))
    output_path = (args.output or (view / "action_norm_audit.json")).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta = json.loads((view / "meta.json").read_text(encoding="utf-8"))
    episodes = sorted(meta["episodes"], key=lambda row: row["episode_index"])
    train_ids = [int(row["episode_index"]) for row in episodes if row["split"] == "train"]
    val_ids = [int(row["episode_index"]) for row in episodes if row["split"] == "validation"]
    if len(episodes) != args.expected_episodes:
        raise ValueError(f"view has {len(episodes)} episodes, expected {args.expected_episodes}")
    if (len(train_ids), len(val_ids)) != (args.expected_train, args.expected_validation):
        raise ValueError(f"view split is {len(train_ids)}/{len(val_ids)}, expected "
                         f"{args.expected_train}/{args.expected_validation}")

    def load_split(ids: list[int]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        states, actions, per_episode = [], [], []
        for episode_id in ids:
            with np.load(view / "episodes" / f"episode_{episode_id:03d}" / "episode.npz",
                         allow_pickle=False) as data:
                state = np.asarray(data["state"], dtype=np.float64)
                action = np.asarray(data["action"], dtype=np.float64)
            if state.shape != action.shape or state.ndim != 2 or state.shape[1] != 12:
                raise ValueError(f"episode {episode_id}: expected aligned [frames,12] state/action")
            if not np.isfinite(state).all() or not np.isfinite(action).all():
                raise ValueError(f"episode {episode_id}: non-finite state/action")
            states.append(state)
            actions.append(action)
            per_episode.append({"episode_index": episode_id, "state": state, "action": action})
        return np.concatenate(states), np.concatenate(actions), per_episode

    train_state, train_action, train_episodes = load_split(train_ids)
    val_state, val_action, _ = load_split(val_ids)
    stats_file = view / "assets/so101/norm_stats.json"
    recorded = json.loads(stats_file.read_text(encoding="utf-8"))["norm_stats"]
    recomputed = {"state": per_dimension_stats(train_state),
                  "actions": per_dimension_stats(train_action)}
    combined = {
        "state": per_dimension_stats(np.concatenate([train_state, val_state])),
        "actions": per_dimension_stats(np.concatenate([train_action, val_action])),
    }
    stats_match = all(
        key in recorded and all(
            field in recorded[key]
            and np.allclose(recorded[key][field], recomputed[key][field], rtol=1e-6, atol=1e-6)
            for field in STAT_FIELDS)
        for key in ("state", "actions"))
    finite = bool(np.isfinite(train_action).all() and np.isfinite(train_state).all()
                  and np.isfinite(val_action).all() and np.isfinite(val_state).all()
                  and all(np.isfinite(np.asarray(recorded[key][field], dtype=np.float64)).all()
                          for key in ("state", "actions") for field in STAT_FIELDS))
    near_constant = {
        key: [index for index, std in enumerate(recorded[key]["std"]) if float(std) < 1e-6]
        for key in ("state", "actions")
    }
    leakage = {
        key: {field: float(np.max(np.abs(np.asarray(combined[key][field])
                                      - np.asarray(recorded[key][field]))))
              for field in STAT_FIELDS}
        for key in ("state", "actions")
    }

    same_frame_errors, next_frame_errors = [], []
    for episode in train_episodes:
        state, action = episode["state"], episode["action"]
        same_frame_errors.append(np.abs(action - state))
        if len(state) > 1:
            next_frame_errors.append(np.abs(action[:-1] - state[1:]))
    same = np.concatenate(same_frame_errors)
    next_frame = np.concatenate(next_frame_errors)

    config = openpi_so101.make_train_config(
        mode="serve", run_root=view / "_audit", train_view=view,
        base_checkpoint=(args.base_checkpoint or view), exp_name="norm_audit",
        state_mode="on", fsdp_devices=args.fsdp_devices,
        paligemma_variant=args.paligemma_variant,
        action_expert_variant=args.action_expert_variant,
        action_horizon=10, seed=42, dtype="bfloat16", num_workers=0,
        batch_size=4, lr_warmup_steps=warmup_steps, peak_lr=peak_lr,
        decay_steps=decay_steps, decay_lr=final_lr)
    data_config = config.data.create(config.assets_dirs, config.model)
    use_quantiles = bool(data_config.use_quantile_norm)
    norm_stats = data_config.norm_stats
    normalise = transforms.Normalize(norm_stats, use_quantiles=use_quantiles)
    unnormalise = transforms.Unnormalize(norm_stats, use_quantiles=use_quantiles)
    model_inputs = group_items(data_config.model_transforms, "inputs")
    model_outputs = group_items(data_config.model_transforms, "outputs")
    data_inputs = group_items(data_config.data_transforms, "inputs")
    data_outputs = group_items(data_config.data_transforms, "outputs")
    repack_outputs = group_items(data_config.repack_transforms, "outputs")
    pad = next((item for item in model_inputs
                if type(item).__name__ == "PadStatesAndActions"), None)
    if pad is None:
        raise SystemExit("pinned OpenPI model input pipeline has no PadStatesAndActions transform")

    all_actions = np.concatenate([train_action, val_action], axis=0)
    raw = {"actions": all_actions.astype(np.float32),
           "state": np.concatenate([train_state, val_state], axis=0).astype(np.float32)}
    normalized = normalise(dict(raw))
    padded = pad(dict(normalized))
    padded_actions = np.asarray(padded["actions"])
    internal_dim = int(padded_actions.shape[-1])
    if internal_dim != 32:
        raise ValueError(f"runtime padded actions to {internal_dim} dimensions, expected 32")
    pad_error = float(np.max(np.abs(padded_actions[..., :12] - normalized["actions"])))
    pad_tail_max = float(np.max(np.abs(padded_actions[..., 12:])))

    policy_output = apply_transforms(
        model_outputs, {**padded, "actions": padded_actions})
    policy_output = unnormalise(policy_output)
    restored_state = np.asarray(policy_output["state"], dtype=np.float64)
    policy_output = apply_transforms(data_outputs, policy_output)
    policy_output = apply_transforms(repack_outputs, policy_output)
    restored = np.asarray(policy_output["actions"], dtype=np.float64)
    if restored.shape != raw["actions"].shape:
        raise ValueError(f"runtime inverse output shape {restored.shape} != {raw['actions'].shape}")
    roundtrip = restored - raw["actions"].astype(np.float64)
    internal_state_dim = int(restored_state.shape[-1])
    if internal_state_dim < 12:
        raise ValueError(f"runtime returned only {internal_state_dim} state dimensions")
    state_roundtrip = restored_state[..., :12] - raw["state"].astype(np.float64)
    state_tail_max = float(np.max(np.abs(restored_state[..., 12:]))) if internal_state_dim > 12 else 0.0
    per_dim_error = [float(np.max(np.abs(roundtrip[:, index]))) for index in range(12)]
    tails_outside_unit = [index for index in range(12)
                          if normalized["actions"][:, index].min() < -1.0
                          or normalized["actions"][:, index].max() > 1.0]

    os.environ["SO101_TRAIN_VIEW"] = str(view)
    os.environ.pop("SO101_TRAINING_SAMPLE_IDS", None)
    os.environ.pop("SO101_TRAIN_EPISODES", None)
    raw_dataset = openpi_so101.SO101EpisodeDataset(
        view, split="train", action_horizon=config.model.action_horizon)
    selected_indices = []
    selected_by_episode = {}
    for episode_id in train_ids:
        hits = [index for index, sample in enumerate(raw_dataset.samples)
                if sample[0] == episode_id]
        if not hits:
            raise ValueError(f"episode {episode_id} has no eligible action window")
        selected = hits[len(hits) // 2]
        selected_indices.append(selected)
        selected_by_episode[episode_id] = selected
    sample_ids_path = output_path.parent / "_norm_audit_loader_ids.jsonl"
    sample_ids_path.write_text("".join(f"{value}\n" for value in selected_indices),
                               encoding="utf-8")
    os.environ["SO101_TRAINING_SAMPLE_IDS"] = str(sample_ids_path)
    loader = openpi_so101.create_so101_data_loader(
        config, shuffle=False, num_batches=len(selected_indices) // config.batch_size)
    loader_rows = []
    observed_samples = 0
    for observation, loader_actions in loader:
        loader_state = np.asarray(observation.state)
        loader_actions = np.asarray(loader_actions)
        count = int(loader_actions.shape[0])
        for local in range(count):
            sample_index = selected_indices[observed_samples + local]
            episode_id, frame = raw_dataset.samples[sample_index]
            expected = raw_dataset[sample_index]
            expected = apply_transforms(
                group_items(data_config.repack_transforms, "inputs"), expected)
            expected = apply_transforms(data_inputs, expected)
            expected = normalise(expected)
            expected = apply_transforms(model_inputs, expected)
            action_error = float(np.max(np.abs(
                loader_actions[local] - np.asarray(expected["actions"]))))
            state_error = float(np.max(np.abs(
                loader_state[local] - np.asarray(expected["state"]))))
            loader_rows.append({
                "episode_id": episode_id,
                "frame": frame,
                "action_max_abs_error": action_error,
                "state_max_abs_error": state_error,
            })
        observed_samples += count
    os.environ.pop("SO101_TRAINING_SAMPLE_IDS", None)
    if observed_samples != len(selected_indices):
        raise ValueError(f"real loader covered {observed_samples} of {len(selected_indices)} episodes")
    covered_episodes = {row["episode_id"] for row in loader_rows}
    if covered_episodes != set(train_ids):
        raise ValueError("real loader comparison did not cover every train episode")
    loader_action_error = max(row["action_max_abs_error"] for row in loader_rows)
    loader_state_error = max(row["state_max_abs_error"] for row in loader_rows)

    inference_source = inspect.getsource(policy_config.create_trained_policy)
    source_unnorm = inference_source.find("Unnormalize")
    source_data_output = inference_source.find("data_transforms.outputs")
    source_repack_output = inference_source.find("repack_transforms.outputs")
    inference_order_verified = (
        source_unnorm >= 0 and source_data_output > source_unnorm
        and source_repack_output > source_data_output)
    executed_output_types = [
        *[type(item).__name__ for item in model_outputs],
        type(unnormalise).__name__,
        *[type(item).__name__ for item in data_outputs],
        *[type(item).__name__ for item in repack_outputs],
    ]
    run_config_hash = (sha256_file(args.run_config) if args.run_config and args.run_config.is_file()
                       else None)
    base_checkpoint_sha = sha256_tree(args.base_checkpoint) if args.base_checkpoint else None
    # Same real frame through the training reader and through the inference
    # input transform.  The JPG is loaded independently from the view; both
    # dictionaries then pass the identical OpenPI transform stack.
    image_check = {}
    try:
        first_index = selected_indices[0]
        episode_id, frame = raw_dataset.samples[first_index]
        sample = raw_dataset[first_index]
        raw_images = {key: np.asarray(sample[key]) for key in openpi_so101.CAMERAS.values()}
        direct = {}
        for key, name in openpi_so101.CAMERA_FILES.items():
            from PIL import Image
            path = view / "episodes" / f"episode_{episode_id:03d}" / name / f"{frame:06d}.jpg"
            with Image.open(path) as handle:
                direct[key] = np.asarray(handle.convert("RGB"))
        raw_frame_exact = all(np.array_equal(raw_images[key], direct[key])
                              for key in raw_images)
        base_dict = {"observation.state": np.asarray(sample["observation.state"], dtype=np.float32),
                     "action": np.asarray(sample["action"], dtype=np.float32),
                     "task": sample["task"], **raw_images}
        direct_dict = {"observation.state": np.asarray(sample["observation.state"], dtype=np.float32),
                       "action": np.asarray(sample["action"], dtype=np.float32),
                       "task": sample["task"], **direct}
        def pipeline(values):
            result = apply_transforms(data_inputs, dict(values))
            result = normalise(result)
            return apply_transforms(model_inputs, result)
        loader_inputs = pipeline(base_dict)
        direct_inputs = pipeline(direct_dict)
        image_errors = {}
        for camera in loader_inputs.get("image", {}):
            image_errors[camera] = float(np.max(np.abs(
                np.asarray(loader_inputs["image"][camera])
                - np.asarray(direct_inputs["image"][camera]))))
        image_check = {
            "episode_id": int(episode_id), "frame": int(frame),
            "raw_jpg_equals_loader_frame": bool(raw_frame_exact),
            "image_max_abs_error": max(image_errors.values()) if image_errors else None,
            "cameras": sorted(image_errors),
            "passed": bool(raw_frame_exact and image_errors
                           and max(image_errors.values()) < 1e-6),
        }
    except Exception as error:
        image_check = {"passed": False, "error": f"{type(error).__name__}: {error}"}

    payload = {
        "schema_version": 3,
        "view": view.name,
        "view_sha256": meta["view_sha256"],
        "norm_stats_sha256": sha256_file(stats_file),
        "run_config": args.run_config.name if args.run_config else None,
        "run_config_sha256": run_config_hash,
        "training_config_sha256": run_config_hash,
        "base_checkpoint_sha256": base_checkpoint_sha,
        "base_checkpoint_hash_scope": "SHA256 of sorted relative file paths and each file SHA256",
        "binding": {"train_episodes": len(train_ids), "validation_episodes": len(val_ids),
                    "train_frames": int(train_action.shape[0]),
                    "validation_frames_excluded": int(val_action.shape[0])},
        "semantics": {
            "dim_order": "left six then right six", "units": UNITS,
            "feature_units": UNITS,
            "action_horizon": config.model.action_horizon,
            "action_shape": [config.model.action_horizon, openpi_so101.ACTION_DIM],
            "action_is_control_target_from_current_frame": True,
            "action_minus_same_frame_state_max_abs": float(same.max()),
            "action_equals_same_frame_state_fraction": float(np.mean(same == 0.0)),
            "action_minus_next_frame_state_max_abs_within_episode": float(next_frame.max()),
            "episode_boundaries_excluded_from_alignment": True,
            "note": "each action row is aligned to the control frame whose target was applied; "
                    "the measured state is the pre-control observation",
        },
        "statistics": {
            "recomputed_from_train_only_matches_view": bool(stats_match),
            "recomputed_train_only": recomputed,
            "recorded_train_only": recorded,
            "combined_train_validation": combined,
            "validation_inclusion_max_abs_by_split_and_field": leakage,
            "all_values_finite": finite,
            "near_constant_dims_std_lt_1e-6": near_constant,
            "normalisation_uses_quantiles": use_quantiles,
        },
        "transform": {
            "train_input_order": [type(item).__name__ for item in
                                  [*group_items(data_config.repack_transforms, "inputs"),
                                   *data_inputs, normalise, *model_inputs]],
            "inference_output_order": [type(item).__name__ for item in
                                       [*model_outputs, unnormalise, *data_outputs,
                                        *repack_outputs]],
            "inference_source_sha256": hashlib.sha256(inference_source.encode()).hexdigest(),
            "inference_output_order_verified_against_pinned_source": inference_order_verified,
            "inference_output_execution_order": executed_output_types,
            "inference_output_execution_order_verified": bool(
                inference_order_verified and max(per_dim_error) < 1e-4),
            "normalise_class": type(normalise).__name__,
            "unnormalise_class": type(unnormalise).__name__,
            "samples_round_tripped": int(len(raw["actions"])),
            "per_dim_max_recovery_error": per_dim_error,
            "worst_recovery_error": float(max(per_dim_error)),
            "state_per_dim_max_recovery_error": [
                float(np.max(np.abs(state_roundtrip[:, index]))) for index in range(12)],
            "state_worst_recovery_error": float(np.max(np.abs(state_roundtrip))),
            "internal_state_dim": internal_state_dim,
            "state_padded_tail_max_abs": state_tail_max,
            "dims_exceeding_unit_interval_after_normalisation": tails_outside_unit,
            "clipping_applied": False,
            "internal_action_dim": internal_dim,
            "pad_first_12_max_abs_error": pad_error,
            "padded_tail_max_abs": pad_tail_max,
        },
        "image_pipeline": image_check,
        "loader_consistency": {
            "episodes_covered": len(covered_episodes),
            "samples_checked": observed_samples,
            "per_episode": loader_rows,
            "state_max_abs_error_vs_runtime_pipeline": loader_state_error,
            "action_max_abs_error_vs_runtime_pipeline": loader_action_error,
            "raw_action_range": [float(train_action.min()), float(train_action.max())],
        },
    }
    payload["expected_counts"] = {"episodes": args.expected_episodes,
                                  "train": len(train_ids), "validation": len(val_ids)}
    payload["passed"] = bool(
        len(train_ids) + len(val_ids) == args.expected_episodes and finite and stats_match
        and inference_order_verified and restored.shape[-1] == 12
        and covered_episodes == set(train_ids)
        and not near_constant["state"] and not near_constant["actions"]
        and max(per_dim_error) < 1e-4 and pad_error == 0.0 and pad_tail_max == 0.0
        and loader_action_error < 1e-6 and loader_state_error < 1e-6
        and image_check.get("passed") is True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True)[:6000])
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
