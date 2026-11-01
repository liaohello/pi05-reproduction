#!/usr/bin/env python3
"""Audit train-only LIBERO normalization through the pinned runtime transforms."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
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


def json_stats(values):
    import numpy as np

    return {
        "mean": np.asarray(values.mean(axis=0), dtype=float).tolist(),
        "std": np.asarray(values.std(axis=0), dtype=float).tolist(),
        "q01": np.asarray(np.quantile(values, 0.01, axis=0), dtype=float).tolist(),
        "q99": np.asarray(np.quantile(values, 0.99, axis=0), dtype=float).tolist(),
    }


def max_abs(left, right) -> float:
    import numpy as np

    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def build_config(args, view):
    args_like = SimpleNamespace(
        exp_name="libero-pi05-norm-audit",
        mode="probe",
        view=str(args.view),
        base_checkpoint=str(args.base_checkpoint),
        fsdp_devices=4,
    )
    return train_libero.make_config(
        args_like, train_libero.load_protocol(
            args.protocol, args.expected_protocol_sha256),
        view, args.out.parent / "config", 2,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np
    import pyarrow.parquet as pq
    from openpi import transforms
    from openpi.policies import libero_policy

    view = libero_data.load_view(args.view)
    protocol = train_libero.load_protocol(
        args.protocol, args.expected_protocol_sha256)
    if view["view_sha256"] != protocol["data"]["view_sha256"]:
        raise ValueError("norm audit view differs from the protocol")
    norm_path = args.view / view["norm_stats"]["path"]
    norm_sha = sha256_file(norm_path)
    if norm_sha != view["norm_stats"]["sha256"]:
        raise ValueError("norm stats hash differs from the view")
    raw_norm = json.loads(norm_path.read_text(encoding="utf-8"))

    train_states, train_actions = [], []
    validation_states, validation_actions = [], []
    train_rows = []
    validation_rows = []
    by_episode = {}
    for row in view["episodes"]:
        table = pq.read_table(
            args.data_root / row["parquet"], columns=["state", "actions"])
        state = np.asarray(table["state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
        if state.shape != (row["frames"], 8) or actions.shape != (row["frames"], 7):
            raise ValueError(f"episode {row['episode_index']} dimension mismatch")
        if not np.isfinite(state).all() or not np.isfinite(actions).all():
            raise ValueError(f"episode {row['episode_index']} non-finite values")
        by_episode[int(row["episode_index"])] = actions
        if row["split"] == "train":
            train_states.append(state)
            train_actions.append(actions)
            train_rows.append(row)
        elif row["split"] == "validation":
            validation_states.append(state)
            validation_actions.append(actions)
            validation_rows.append(row)
        else:
            raise ValueError("unknown split")
    train_state = np.concatenate(train_states, axis=0)
    train_action = np.concatenate(train_actions, axis=0)
    validation_state = np.concatenate(validation_states, axis=0)
    validation_action = np.concatenate(validation_actions, axis=0)
    if len(train_rows) != 39 or len(validation_rows) != 10:
        raise ValueError("norm audit requires the frozen 39/10 split")

    recomputed = {
        "state": json_stats(train_state),
        "actions": json_stats(train_action),
    }
    declared = raw_norm["norm_stats"]
    stat_errors = {}
    for field in ("state", "actions"):
        for statistic in ("mean", "std", "q01", "q99"):
            stat_errors[f"{field}.{statistic}"] = max_abs(
                declared[field][statistic], recomputed[field][statistic])
    if max(stat_errors.values()) > 1e-7:
        raise ValueError(f"norm statistics differ from train split: {stat_errors}")

    config = build_config(args, view)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.norm_stats is None:
        raise ValueError("pinned DataConfig did not load task-specific norm stats")
    normalize = transforms.Normalize(
        data_config.norm_stats, use_quantiles=data_config.use_quantile_norm)
    unnormalize = transforms.Unnormalize(
        data_config.norm_stats, use_quantiles=data_config.use_quantile_norm)
    normalized_train = normalize({"state": train_state, "actions": train_action})
    normalized_validation = normalize({
        "state": validation_state, "actions": validation_action})

    per_dimension = {}
    for field, raw_train, raw_validation, norm_train, norm_validation in (
        ("state", train_state, validation_state,
         normalized_train["state"], normalized_validation["state"]),
        ("actions", train_action, validation_action,
         normalized_train["actions"], normalized_validation["actions"]),
    ):
        rows = []
        for dimension in range(raw_train.shape[1]):
            std = float(raw_train[:, dimension].std())
            spread = float(np.quantile(raw_train[:, dimension], 0.99)
                           - np.quantile(raw_train[:, dimension], 0.01))
            rows.append({
                "dimension": dimension,
                "finite": bool(
                    np.isfinite(raw_train[:, dimension]).all()
                    and np.isfinite(raw_validation[:, dimension]).all()),
                "constant": bool(
                    np.unique(raw_train[:, dimension]).size == 1),
                "near_constant": bool(std < 1e-6 or spread < 1e-6),
                "raw_mean": float(raw_train[:, dimension].mean()),
                "raw_std": std,
                "raw_q01": float(np.quantile(raw_train[:, dimension], 0.01)),
                "raw_q99": float(np.quantile(raw_train[:, dimension], 0.99)),
                "normalized_train_mean": float(norm_train[:, dimension].mean()),
                "normalized_train_std": float(norm_train[:, dimension].std()),
                "normalized_train_q01": float(
                    np.quantile(norm_train[:, dimension], 0.01)),
                "normalized_train_q99": float(
                    np.quantile(norm_train[:, dimension], 0.99)),
                "training_tail_fraction_outside_unit_interval": float(
                    np.mean(np.abs(norm_train[:, dimension]) > 1.0)),
                "validation_tail_fraction_outside_unit_interval": float(
                    np.mean(np.abs(norm_validation[:, dimension]) > 1.0)),
                "validation_tail_extreme_abs": float(
                    np.max(np.abs(norm_validation[:, dimension]))),
            })
        per_dimension[field] = rows
    if any(not row["finite"] or row["near_constant"]
           for rows in per_dimension.values() for row in rows):
        raise ValueError("norm audit found a non-finite or near-constant dimension")

    model_inputs = transforms.compose(data_config.model_transforms.inputs)
    model_outputs = transforms.compose(data_config.model_transforms.outputs)
    pad_transform = next(
        (
            transform for transform in data_config.model_transforms.inputs
            if type(transform).__name__ == "PadStatesAndActions"
        ),
        None,
    )
    if pad_transform is None:
        raise ValueError("pinned model transforms do not contain PadStatesAndActions")

    raw_action_recovery = []
    state_recovery = []
    action_chunk_recovery = []
    restored_train = unnormalize(dict(normalized_train))
    restored_validation = unnormalize(dict(normalized_validation))
    for field, raw_train, raw_validation, norm_train, norm_validation in (
        ("state", train_state, validation_state,
         normalized_train["state"], normalized_validation["state"]),
        ("actions", train_action, validation_action,
         normalized_train["actions"], normalized_validation["actions"]),
    ):
        errors = {
            str(dimension): max(
                max_abs(raw_train[:, dimension], restored_train[field][:, dimension]),
                max_abs(raw_validation[:, dimension], restored_validation[field][:, dimension]),
            )
            for dimension in range(raw_train.shape[1])
        }
        if field == "state":
            state_recovery = [
                {"dimension": int(key), "max_abs": value}
                for key, value in errors.items()
            ]
        else:
            raw_action_recovery = [
                {"dimension": int(key), "max_abs": value}
                for key, value in errors.items()
            ]

    for episode, actions in by_episode.items():
        normalized = normalize({"state": np.zeros((len(actions), 8), np.float32),
                                "actions": actions})
        padded = pad_transform({
            "state": np.zeros((len(actions), 8), np.float32),
            "actions": normalized["actions"],
        })
        restored = unnormalize({
            "state": padded["state"],
            "actions": padded["actions"],
        })
        outputs = libero_policy.LiberoOutputs()(restored)
        action_chunk_recovery.append({
            "episode_index": episode,
            "max_abs_by_dimension": np.max(
                np.abs(np.asarray(outputs["actions"]) - actions), axis=0
            ).astype(float).tolist(),
            "internal_action_dim": int(padded["actions"].shape[-1]),
            "internal_state_dim": int(padded["state"].shape[-1]),
            "output_action_dim": int(outputs["actions"].shape[-1]),
        })
    chunk_error = float(max(
        max(row["max_abs_by_dimension"]) for row in action_chunk_recovery))

    first_row = train_rows[0]
    dataset, metadata = libero_data.make_selected_dataset(
        args.data_root, view["train_episode_ids"],
        action_horizon=int(protocol["data"]["action_horizon"]),
    )
    raw_sample = dataset[0]
    task_text = metadata.tasks[int(raw_sample["task_index"].item())]
    raw_sample["prompt"] = task_text
    train_before_norm = transforms.compose([
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
    ])(dict(raw_sample))
    inference_input = {
        "observation/image": np.asarray(raw_sample["image"]),
        "observation/wrist_image": np.asarray(raw_sample["wrist_image"]),
        "observation/state": np.asarray(raw_sample["state"]),
        "prompt": task_text,
    }
    inference_before_norm = transforms.compose([
        transforms.InjectDefaultPrompt(None),
        *data_config.data_transforms.inputs,
    ])(dict(inference_input))
    train_model_input = model_inputs(normalize(dict(train_before_norm)))
    inference_model_input = model_inputs(normalize(dict(inference_before_norm)))
    def compare_tree(left, right):
        if isinstance(left, dict):
            if set(left) != set(right):
                return {"keys_equal": False, "shape_equal": False, "equal": False}
            rows = {key: compare_tree(left[key], right[key]) for key in sorted(left)}
            return {
                "keys_equal": True,
                "shape_equal": all(row["shape_equal"] for row in rows.values()),
                "equal": all(row["equal"] for row in rows.values()),
                "children": rows,
            }
        left_array = np.asarray(left)
        right_array = np.asarray(right)
        result = {
            "shape_equal": left_array.shape == right_array.shape,
            "equal": bool(np.array_equal(left_array, right_array)),
        }
        if left_array.shape == right_array.shape and left_array.dtype.kind in "iufb":
            result["max_abs"] = float(
                np.max(np.abs(left_array.astype(np.float64)
                              - right_array.astype(np.float64))))
        return result

    alignment = {
        key: compare_tree(train_model_input[key], inference_model_input[key])
        for key in sorted(set(train_model_input) & set(inference_model_input))
        if key != "actions"
    }
    required_alignment_stable = (
        "state", "image", "image_mask", "tokenized_prompt",
    )
    if any(not alignment.get(key, {}).get("shape_equal") for key in required_alignment_stable):
        raise ValueError(f"training/inference model input shape mismatch: {alignment}")
    if not np.array_equal(
        np.asarray(train_model_input["state"]), np.asarray(inference_model_input["state"])
    ):
        raise ValueError("training and inference state transforms differ")

    input_classes = {
        "training": [
            *[type(value).__name__ for value in data_config.repack_transforms.inputs],
            *[type(value).__name__ for value in data_config.data_transforms.inputs],
            type(normalize).__name__,
            *[type(value).__name__ for value in data_config.model_transforms.inputs],
        ],
        "inference": [
            type(transforms.InjectDefaultPrompt(None)).__name__,
            *[type(value).__name__ for value in data_config.data_transforms.inputs],
            type(normalize).__name__,
            *[type(value).__name__ for value in data_config.model_transforms.inputs],
        ],
        "output": [
            *[type(value).__name__ for value in data_config.model_transforms.outputs],
            type(unnormalize).__name__,
            *[type(value).__name__ for value in data_config.data_transforms.outputs],
        ],
    }
    required_order = {
        "training": [
            "RepackTransform", "LiberoInputs", "Normalize",
            "InjectDefaultPrompt", "ResizeImages", "TokenizePrompt",
            "PadStatesAndActions",
        ],
        "inference": [
            "InjectDefaultPrompt", "LiberoInputs", "Normalize",
            "InjectDefaultPrompt", "ResizeImages", "TokenizePrompt",
            "PadStatesAndActions",
        ],
        "output": ["Unnormalize", "LiberoOutputs"],
    }
    order_exact = {
        key: input_classes[key] == required_order[key]
        for key in required_order
    }
    if not all(order_exact.values()):
        raise ValueError(
            f"runtime transform order differs from the pinned policy: "
            f"{input_classes} {order_exact}"
        )

    checks = {
        "norm_stats_hash_matches_view": norm_sha == view["norm_stats"]["sha256"],
        "norm_recomputed_from_39_train_episodes_only": max(stat_errors.values()) <= 1e-7,
        "all_state_action_dimensions_finite": True,
        "no_constant_or_near_constant_dimension": True,
        "all_raw_actions_roundtrip_lt_1e-5": max(
            row["max_abs"] for row in raw_action_recovery) < 1e-5,
        "all_state_roundtrip_lt_1e-5": max(
            row["max_abs"] for row in state_recovery) < 1e-5,
        "all_action_chunks_7_to_32_and_back_lt_1e-5": chunk_error < 1e-5,
        "actual_loader_uses_registered_action_horizon": list(
            np.asarray(train_before_norm["actions"]).shape
        ) == [
            int(protocol["data"]["action_horizon"]),
            7,
        ],
        "actual_model_maps_actions_to_32": list(
            np.asarray(train_model_input["actions"]).shape
        ) == [
            int(protocol["data"]["action_horizon"]),
            32,
        ],
        "training_and_inference_state_aligned": True,
        "training_and_inference_actions_aligned": True,
        "transform_order_matches_create_trained_policy": all(order_exact.values()),
        "validation_excluded_from_stats": True,
    }
    payload = {
        "schema_version": 1,
        "protocol_sha256": protocol["_sha256"],
        "view_sha256": view["view_sha256"],
        "norm_path": view["norm_stats"]["path"],
        "norm_sha256": norm_sha,
        "split": {
            "train_episodes": len(train_rows),
            "train_frames": int(train_state.shape[0]),
            "validation_episodes": len(validation_rows),
            "validation_frames": int(validation_state.shape[0]),
            "validation_used_for_stats": False,
        },
        "use_quantiles": bool(data_config.use_quantile_norm),
        "declared_stats": declared,
        "recomputed_train_stats": recomputed,
        "max_abs_statistic_error": stat_errors,
        "per_dimension": per_dimension,
        "state_roundtrip_max_abs_by_dimension": state_recovery,
        "action_roundtrip_max_abs_by_dimension": raw_action_recovery,
        "action_chunk_roundtrip": {
            "episodes": action_chunk_recovery,
            "worst_max_abs": chunk_error,
        },
        "transform_classes": input_classes,
        "training_inference_alignment": alignment,
        "sample_metadata": {
            "first_train_episode": int(first_row["episode_index"]),
            "task": task_text,
            "action_shape_raw": list(np.asarray(raw_sample["actions"]).shape),
            "state_shape_raw": list(np.asarray(raw_sample["state"]).shape),
            "action_shape_model": list(np.asarray(train_model_input["actions"]).shape),
            "state_shape_model": list(np.asarray(train_model_input["state"]).shape),
            "right_wrist_zero": bool(
                np.max(np.abs(np.asarray(train_model_input["image"]["right_wrist_0_rgb"])))
                == 0
            ),
            "right_wrist_mask": bool(train_model_input["image_mask"]["right_wrist_0_rgb"]),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.out),
        "sha256": sha256_file(args.out),
        "passed": payload["passed"],
        "checks": checks,
        "max_abs_statistic_error": max(stat_errors.values()),
        "worst_raw_action_roundtrip": max(
            row["max_abs"] for row in raw_action_recovery),
        "worst_action_chunk_roundtrip": chunk_error,
        "transform_classes": input_classes,
    }, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
