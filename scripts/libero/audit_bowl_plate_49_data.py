#!/usr/bin/env python3
"""Audit episode identity, chunk boundaries, prompts, and all-49 Norm Stats."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bowl_plate_49  # noqa: E402


def json_stats(values) -> dict:
    import numpy as np

    values = np.asarray(values, dtype=np.float32)
    return {
        "mean": np.asarray(values.mean(axis=0), dtype=float).tolist(),
        "std": np.asarray(values.std(axis=0), dtype=float).tolist(),
        "q01": np.asarray(np.quantile(values, 0.01, axis=0), dtype=float).tolist(),
        "q99": np.asarray(np.quantile(values, 0.99, axis=0), dtype=float).tolist(),
    }


def max_abs_stat_error(left: dict, right: dict) -> float:
    import numpy as np

    return float(max(
        np.max(np.abs(np.asarray(left[field][stat]) - np.asarray(right[field][stat])))
        for field in ("state", "actions")
        for stat in ("mean", "std", "q01", "q99")
    ))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import jax
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from openpi import transforms

    protocol = bowl_plate_49.load_protocol(
        args.protocol, args.expected_protocol_sha256)
    view = bowl_plate_49.load_view(args.view, protocol)
    norm_path = args.view / view["norm_stats"]["path"]
    declared_norm = json.loads(norm_path.read_text(encoding="utf-8"))["norm_stats"]

    states_all = []
    actions_all = []
    by_episode = {}
    raw_rows = {}
    for row in view["episodes"]:
        episode = int(row["episode_index"])
        parquet = args.data_root / row["parquet"]
        table = pq.read_table(
            parquet,
            columns=["episode_index", "frame_index", "task_index", "state", "actions"],
        )
        if set(table["episode_index"].to_pylist()) != {episode}:
            raise ValueError(f"episode ID mismatch in Parquet for {episode}")
        if set(table["task_index"].to_pylist()) != {protocol["dataset_task_index"]}:
            raise ValueError(f"task index mismatch in Parquet for {episode}")
        state = np.asarray(table["state"].to_pylist(), dtype=np.float32)
        action = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
        if state.shape != (int(row["frames"]), 8):
            raise ValueError(f"state shape mismatch for episode {episode}")
        if action.shape != (int(row["frames"]), 7):
            raise ValueError(f"action shape mismatch for episode {episode}")
        states_all.append(state)
        actions_all.append(action)
        by_episode[episode] = action
        raw_rows[episode] = {
            "state": state,
            "action": action,
            "frames": int(row["frames"]),
        }
    all_state = np.concatenate(states_all, axis=0)
    all_action = np.concatenate(actions_all, axis=0)
    recomputed_norm = {
        "state": json_stats(all_state),
        "actions": json_stats(all_action),
    }
    statistic_error = max_abs_stat_error(declared_norm, recomputed_norm)
    if statistic_error > 1e-7:
        raise ValueError(
            f"Norm Stats do not match all 49 selected episodes: {statistic_error}")

    config_args = SimpleNamespace(
        exp_name="libero-bowl-plate-audit",
        mode="probe",
        view=str(args.view),
        base_checkpoint=str(args.base_checkpoint),
        batch_size=32,
        fsdp_devices=4,
    )
    config = bowl_plate_49.make_config(
        config_args, protocol, view, args.out.parent / "audit_checkpoints", 1)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.norm_stats is None:
        raise ValueError("OpenPI did not load the all-49 Norm Stats")
    if not data_config.use_quantile_norm:
        raise ValueError("pi05 must use quantile normalization")

    dataset, metadata = bowl_plate_49.make_selected_dataset(
        args.data_root,
        bowl_plate_49.EXPECTED_EPISODES,
        action_horizon=10,
    )
    if len(dataset) != 4563:
        raise ValueError(f"selected dataset length is {len(dataset)}, expected 4563")
    if [int(value) for value in dataset.episodes] != list(
        bowl_plate_49.EXPECTED_EPISODES
    ):
        raise ValueError("LeRobot changed the selected episode order")
    expected_indices = dataset.episode_data_index
    if len(expected_indices["from"]) <= max(bowl_plate_49.EXPECTED_EPISODES):
        raise ValueError("episode_data_index was not expanded to original episode IDs")
    hf_episode_indices = np.asarray([
        int(value) for value in torch.stack(dataset.hf_dataset["episode_index"])
    ])
    hf_frame_indices = np.asarray([
        int(value) for value in torch.stack(dataset.hf_dataset["frame_index"])
    ])
    hf_task_indices = np.asarray([
        int(value) for value in torch.stack(dataset.hf_dataset["task_index"])
    ])
    if set(hf_episode_indices) != set(bowl_plate_49.EXPECTED_EPISODES):
        raise ValueError("actual dataset contains episodes outside the requested subset")
    if set(hf_task_indices) != {protocol["dataset_task_index"]}:
        raise ValueError("actual dataset contains a different task")
    if not np.array_equal(
        np.asarray(dataset.delta_indices["actions"]), np.arange(10)
    ):
        raise ValueError("actual LeRobot action delta indices are not 0..9")

    row_starts = np.asarray(expected_indices["from"])[hf_episode_indices]
    row_ends = np.asarray(expected_indices["to"])[hf_episode_indices]
    global_rows = np.arange(len(dataset), dtype=np.int64)
    queried_indices = np.maximum(
        row_starts[:, None],
        np.minimum(
            global_rows[:, None] + np.arange(10)[None, :],
            row_ends[:, None] - 1,
        ),
    )
    query_within_owner = (
        (queried_indices >= row_starts[:, None])
        & (queried_indices < row_ends[:, None])
    )
    expected_padding = np.ones((len(dataset), 10), dtype=bool)
    for local_delta in range(10):
        raw_index = global_rows + local_delta
        expected_pad = (raw_index < row_starts) | (raw_index >= row_ends)
        expected_padding[:, local_delta] = expected_pad
    sampled_padding_matches = True
    for index in (0, len(dataset) // 2, len(dataset) - 1):
        query, padding = dataset._get_query_indices(
            index, int(hf_episode_indices[index])
        )
        query = np.asarray(query["actions"], dtype=np.int64)
        start = int(expected_indices["from"][hf_episode_indices[index]].item())
        end = int(expected_indices["to"][hf_episode_indices[index]].item())
        if np.any(query < start) or np.any(query >= end):
            raise ValueError("LeRobot action query escaped its source episode")
        if not np.array_equal(
            np.asarray(padding["actions_is_pad"], dtype=bool),
            expected_padding[index],
        ):
            raise ValueError("LeRobot action padding differs from episode bounds")
        sampled_padding_matches &= bool(
            np.array_equal(
                np.asarray(padding["actions_is_pad"], dtype=bool),
                expected_padding[index],
            )
        )
    all_action_queries_within_episode = bool(np.all(query_within_owner))

    task_text = metadata.tasks[protocol["dataset_task_index"]]
    if task_text != protocol["task"]:
        raise ValueError("dataset prompt differs from the registered task")
    episode_checks = []
    sampled_chunk_checks = []
    for episode in bowl_plate_49.EXPECTED_EPISODES:
        start = int(expected_indices["from"][episode].item())
        end = int(expected_indices["to"][episode].item())
        expected_frames = raw_rows[episode]["frames"]
        if end - start != expected_frames:
            raise ValueError(f"episode data range mismatch for {episode}")
        positions = np.flatnonzero(hf_episode_indices == episode)
        if len(positions) != expected_frames:
            raise ValueError(f"HF dataset row count mismatch for episode {episode}")
        if int(positions[0]) != start or int(positions[-1]) != end - 1:
            raise ValueError(f"HF dataset rows are not contiguous for episode {episode}")
        if not np.array_equal(
            hf_frame_indices[start:end], np.arange(expected_frames)
        ):
            raise ValueError(f"HF frame indices are not contiguous for {episode}")
        raw = raw_rows[episode]
        for local_index in sorted({0, expected_frames // 2, expected_frames - 1}):
            global_index = start + local_index
            item = dataset[global_index]
            if int(item["episode_index"]) != episode:
                raise ValueError(f"dataset row remapped episode {episode}")
            if int(item["frame_index"]) != local_index:
                raise ValueError(f"dataset row remapped frame in episode {episode}")
            if item["task"] != task_text:
                raise ValueError(f"dataset prompt differs in episode {episode}")
            if not np.array_equal(
                np.asarray(item["state"], dtype=np.float32), raw["state"][local_index]
            ):
                raise ValueError(f"state changed in episode {episode}")
            action_chunk = np.asarray(item["actions"], dtype=np.float32)
            query = np.clip(
                np.arange(local_index, local_index + 10),
                0,
                expected_frames - 1,
            )
            expected_chunk = raw["action"][query]
            if not np.array_equal(action_chunk, expected_chunk):
                raise ValueError(f"action chunk mismatch in episode {episode}")
            action_pad = np.asarray(item["actions_is_pad"], dtype=bool)
            expected_pad = np.arange(local_index, local_index + 10) >= expected_frames
            if not np.array_equal(action_pad, expected_pad):
                raise ValueError(f"action padding mismatch in episode {episode}")
            if np.any(query < 0) or np.any(query >= expected_frames):
                raise ValueError(f"action chunk crossed episode {episode}")
            sampled_chunk_checks.append({
                "episode_index": episode,
                "local_frame_index": local_index,
                "last_queried_frame": int(query[-1]),
                "cross_episode": False,
                "padding": expected_pad.tolist(),
            })
        episode_checks.append({
            "episode_index": episode,
            "dataset_start": start,
            "dataset_end_exclusive": end,
            "frames": expected_frames,
            "original_id_preserved": True,
            "task_index": protocol["dataset_task_index"],
            "prompt": task_text,
        })

    first = dataset[0]
    first["prompt"] = task_text
    pre_norm = transforms.compose([
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
    ])(dict(first))
    normalize = transforms.Normalize(
        data_config.norm_stats, use_quantiles=data_config.use_quantile_norm)
    unnormalize = transforms.Unnormalize(
        data_config.norm_stats, use_quantiles=data_config.use_quantile_norm)
    model_inputs = transforms.compose(data_config.model_transforms.inputs)
    normalized = normalize(dict(pre_norm))
    model_input = model_inputs(dict(normalized))
    restored = unnormalize(dict(normalized))
    normalized_action = np.asarray(jax.device_get(normalized["actions"]))
    model_action = np.asarray(jax.device_get(model_input["actions"]))
    model_state = np.asarray(jax.device_get(model_input["state"]))
    restored_action = np.asarray(jax.device_get(restored["actions"]))
    roundtrip_error = float(np.max(np.abs(restored_action - np.asarray(first["actions"]))))
    image_shapes = {
        name: list(np.asarray(value).shape)
        for name, value in model_input["image"].items()
    }
    mask_values = {
        name: bool(np.asarray(value))
        for name, value in model_input["image_mask"].items()
    }
    checks = {
        "exact_49_original_episode_ids": (
            tuple(int(value) for value in dataset.episodes)
            == bowl_plate_49.EXPECTED_EPISODES
        ),
        "no_unrequested_episode_rows": (
            set(hf_episode_indices) == set(bowl_plate_49.EXPECTED_EPISODES)
        ),
        "dataset_length_4563": len(dataset) == 4563,
        "all_episode_ranges_contiguous": all(
            row["dataset_end_exclusive"] - row["dataset_start"] == row["frames"]
            for row in episode_checks
        ),
        "action_horizon_10": int(
            np.asarray(dataset.delta_indices["actions"]).max()
        ) == 9,
        "all_action_queries_confined_to_owner_episode": (
            all_action_queries_within_episode
        ),
        "sample_padding_matches_episode_bounds": sampled_padding_matches,
        "sampled_chunks_never_cross_episode": all(
            not row["cross_episode"] for row in sampled_chunk_checks
        ),
        "single_task_prompt": task_text == protocol["task"],
        "norm_stats_all_49": statistic_error <= 1e-7,
        "norm_stats_sha256_matches_view": (
            bowl_plate_49.sha256_file(norm_path)
            == protocol["data"]["norm_stats_sha256"]
        ),
        "quantile_normalization_active": bool(data_config.use_quantile_norm),
        "unused_wrist_masked": mask_values["right_wrist_0_rgb"] is False,
        "model_action_shape_10x32": model_action.shape == (10, 32),
        "model_state_shape_32": model_state.shape == (32,),
        "raw_action_shape_10x7": normalized_action.shape == (10, 7),
        "raw_action_roundtrip_lt_1e-5": roundtrip_error < 1e-5,
    }
    payload = {
        "schema_version": 1,
        "protocol_sha256": protocol["_sha256"],
        "view_sha256": view["view_sha256"],
        "dataset": {
            "repo_id": "local/libero-goal8",
            "lerobot_version": "0.1.0",
            "format": "v2.0",
            "declared_total_episodes": len(metadata.episodes),
            "selected_episodes": len(dataset.episodes),
            "selected_frames": len(dataset),
            "fps": metadata.fps,
            "task_index": protocol["dataset_task_index"],
            "prompt": task_text,
            "episode_data_index_length": len(expected_indices["from"]),
            "episodes": episode_checks,
        },
        "action_chunk_audit": {
            "method": (
                "For first, middle, and final frames of every episode, compare the "
                "loaded 10-action chunk to Parquet actions with LeRobot's documented "
                "clamp-to-episode-end behavior, and check actions_is_pad."
            ),
            "samples": sampled_chunk_checks,
        },
        "norm_stats": {
            "path": view["norm_stats"]["path"],
            "sha256": bowl_plate_49.sha256_file(norm_path),
            "use_quantiles": bool(data_config.use_quantile_norm),
            "episode_scope": list(bowl_plate_49.EXPECTED_EPISODES),
            "frames": len(all_state),
            "declared": declared_norm,
            "recomputed": recomputed_norm,
            "max_abs_statistic_error": statistic_error,
        },
        "transform_audit": {
            "raw_action_shape": list(normalized_action.shape),
            "model_action_shape": list(model_action.shape),
            "model_state_shape": list(model_state.shape),
            "image_shapes": image_shapes,
            "image_masks": mask_values,
            "action_roundtrip_max_abs": roundtrip_error,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.out),
        "sha256": bowl_plate_49.sha256_file(args.out),
        "passed": payload["passed"],
        "checks": checks,
        "norm_stats_sha256": payload["norm_stats"]["sha256"],
        "episodes": len(episode_checks),
        "frames": len(dataset),
    }, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
