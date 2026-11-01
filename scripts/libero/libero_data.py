#!/usr/bin/env python3
"""Task-filtered LeRobot loading and validation for the LIBERO training line."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence


def load_view(view_path: Path) -> dict:
    view_path = Path(view_path)
    document = json.loads((view_path / "view.json").read_text(encoding="utf-8"))
    claimed = document.get("view_sha256")
    payload = {key: value for key, value in document.items() if key != "view_sha256"}
    actual = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if not claimed or actual != claimed:
        raise ValueError("LIBERO view SHA256 mismatch")
    if document.get("suite") != "libero_goal" or int(document.get("task_id", -1)) != 8:
        raise ValueError("view is not the selected libero_goal task 8")
    if int(document.get("dataset_task_index", -1)) != 10:
        raise ValueError("view has the wrong LeRobot task index")
    train = [int(value) for value in document.get("train_episode_ids", [])]
    validation = [int(value) for value in document.get("validation_episode_ids", [])]
    if len(train) != 39 or len(validation) != 10 or set(train) & set(validation):
        raise ValueError("view must contain disjoint 39/10 episode splits")
    norm_path = view_path / document["norm_stats"]["path"]
    if not norm_path.is_file():
        raise FileNotFoundError("view norm stats are missing")
    if hashlib.sha256(norm_path.read_bytes()).hexdigest() != document["norm_stats"]["sha256"]:
        raise ValueError("view norm stats SHA256 mismatch")
    return document


def _patch_selected_episode_index(dataset, episodes: Sequence[int]) -> None:
    """Work around the pinned LeRobot v2.0 selected-episode offset bug."""
    import torch

    original = dataset.episode_data_index
    if dataset.episodes is None or len(dataset.episodes) != len(episodes):
        raise ValueError("LeRobot selected-episode count mismatch")
    maximum = max(episodes)
    from_values = torch.zeros(maximum + 1, dtype=original["from"].dtype)
    to_values = torch.zeros(maximum + 1, dtype=original["to"].dtype)
    for position, episode in enumerate(dataset.episodes):
        from_values[int(episode)] = original["from"][position]
        to_values[int(episode)] = original["to"][position]
    dataset.episode_data_index = {"from": from_values, "to": to_values}


def make_selected_dataset(
    data_root: Path,
    episodes: Sequence[int],
    *,
    action_horizon: int,
    fps: int = 10,
):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    metadata = LeRobotDatasetMetadata(
        "local/libero-goal8", root=Path(data_root)
    )
    dataset = LeRobotDataset(
        "local/libero-goal8",
        root=Path(data_root),
        episodes=[int(value) for value in episodes],
        delta_timestamps={
            "actions": [tick / float(fps) for tick in range(int(action_horizon))]
        },
        download_videos=False,
    )
    _patch_selected_episode_index(dataset, episodes)
    return dataset, metadata


def fixed_validation_sample_plan(
    sample_count: int, global_batch_size: int, seed: int
) -> dict:
    """Freeze validation row order and the final-batch padding used for exact averaging."""
    if sample_count < 1:
        raise ValueError("validation sample count must be positive")
    if global_batch_size < 1:
        raise ValueError("validation batch size must be positive")
    if seed < 0:
        raise ValueError("validation seed must be non-negative")
    batch_count = (sample_count + global_batch_size - 1) // global_batch_size
    padded_count = batch_count * global_batch_size
    padding = padded_count - sample_count
    unique_indices = list(range(sample_count))
    sampler_indices = unique_indices + unique_indices[:padding]
    valid_per_batch = [
        min(global_batch_size, sample_count - batch * global_batch_size)
        for batch in range(batch_count)
    ]
    order_hash = hashlib.sha256(
        json.dumps(unique_indices, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    sampler_hash = hashlib.sha256(
        json.dumps(sampler_indices, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "seed": int(seed),
        "sample_order": "episode_index_then_frame_index",
        "sample_count": int(sample_count),
        "global_batch_size": int(global_batch_size),
        "batch_count": int(batch_count),
        "padded_count": int(padded_count),
        "padding": int(padding),
        "valid_samples_per_batch": valid_per_batch,
        "sample_order_sha256": order_hash,
        "sampler_indices_sha256": sampler_hash,
        "sample_weighting": "uniform_per_valid_sample",
        "_sampler_indices": sampler_indices,
    }


def create_filtered_data_loader(
    config,
    *,
    view_path: Path,
    data_root: Path,
    sharding=None,
    shuffle: bool = False,
    num_batches: int | None = None,
    split: str = "train",
    sampler_indices: Sequence[int] | None = None,
    expected_samples: int | None = None,
):
    """Build the exact OpenPI transform chain over a task-filtered episode view."""
    from openpi import transforms as _transforms
    from openpi.training import data_loader

    view = load_view(Path(view_path))
    episodes = (
        view["train_episode_ids"]
        if split == "train"
        else sorted(view["validation_episode_ids"])
    )
    dataset, metadata = make_selected_dataset(
        Path(data_root), episodes, action_horizon=config.model.action_horizon
    )
    if expected_samples is not None and len(dataset) != int(expected_samples):
        raise ValueError(
            f"{split} dataset has {len(dataset)} samples, expected {expected_samples}")
    if sampler_indices is not None:
        if shuffle:
            raise ValueError("a fixed validation sample order cannot shuffle")
        if len(sampler_indices) % config.batch_size:
            raise ValueError("fixed validation sampler must fill complete global batches")
        if any(value < 0 or value >= len(dataset) for value in sampler_indices):
            raise ValueError("fixed validation sampler contains an out-of-range index")
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.prompt_from_task:
        dataset = data_loader.TransformedDataset(
            dataset, [_transforms.PromptFromLeRobotTask(metadata.tasks)]
        )
    if data_config.norm_stats is None:
        raise ValueError("OpenPI did not load the task-specific norm stats")
    transformed = data_loader.transform_dataset(dataset, data_config)
    loader = data_loader.TorchDataLoader(
        transformed,
        local_batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        framework="jax",
        sampler=sampler_indices,
    )
    return data_loader.DataLoaderImpl(data_config, loader)


def create_fixed_validation_loader(
    config,
    *,
    view_path: Path,
    data_root: Path,
    sharding=None,
    seed: int,
    expected_samples: int,
):
    """Build a finite, ordered validation loader with explicit final-batch padding."""
    plan = fixed_validation_sample_plan(expected_samples, config.batch_size, seed)
    loader = create_filtered_data_loader(
        config,
        view_path=view_path,
        data_root=data_root,
        sharding=sharding,
        shuffle=False,
        num_batches=plan["batch_count"],
        split="validation",
        sampler_indices=plan["_sampler_indices"],
        expected_samples=expected_samples,
    )
    return loader, plan
