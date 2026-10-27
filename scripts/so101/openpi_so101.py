"""π0.5 SO101 data config, training config and training-view dataset.

This module is imported inside the pinned ``openpi-server`` image.  It does not
modify the upstream tree: it defines an original DataConfigFactory and a
PyTorch Dataset, and the training launcher injects the loader into OpenPI's
public ``train.main`` entry point.
"""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from flax import nnx
from typing_extensions import override

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms

CAMERAS = {
    "base_0_rgb": "observation.images.central",
    "left_wrist_0_rgb": "observation.images.left_wrist",
    "right_wrist_0_rgb": "observation.images.right_wrist",
}
CAMERA_FILES = {
    "observation.images.central": "central",
    "observation.images.left_wrist": "left_wrist",
    "observation.images.right_wrist": "right_wrist",
}
ACTION_DIM = 12
TASK = "Sort the red cubes into the red tray and the blue cubes into the blue tray."


def _parse_image(value) -> np.ndarray:
    image = np.asarray(value)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"image must be HWC RGB, got {image.shape}")
    return np.ascontiguousarray(image)


@dataclasses.dataclass(frozen=True)
class SO101Inputs(_transforms.DataTransformFn):
    """Map raw SO101 rows/observations to the model input dictionary."""

    def __call__(self, data: dict) -> dict:
        missing = [key for key in CAMERAS.values() if key not in data]
        if missing:
            raise ValueError(f"missing SO101 camera keys: {missing}")
        base = _parse_image(data[CAMERAS["base_0_rgb"]])
        left = _parse_image(data[CAMERAS["left_wrist_0_rgb"]])
        right = _parse_image(data[CAMERAS["right_wrist_0_rgb"]])
        inputs = {
            "state": np.asarray(data["observation.state"], dtype=np.float32),
            "image": {
                "base_0_rgb": base,
                "left_wrist_0_rgb": left,
                "right_wrist_0_rgb": right,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        actions = data.get("actions", data.get("action"))
        if actions is not None:
            inputs["actions"] = np.asarray(actions, dtype=np.float32)
        prompt = data.get("prompt")
        if prompt is None:
            prompt = data.get("task")
        if prompt is not None:
            inputs["prompt"] = prompt
        return inputs


@dataclasses.dataclass(frozen=True)
class SO101Outputs(_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.shape[-1] < ACTION_DIM:
            raise ValueError(f"policy returned {actions.shape[-1]} action dims, need {ACTION_DIM}")
        return {"actions": actions[..., :ACTION_DIM]}


@dataclasses.dataclass(frozen=True)
class SO101DataConfig(_config.DataConfigFactory):
    @override
    def create(self, assets_dirs: Path, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=_transforms.Group(),
            data_transforms=_transforms.Group(
                inputs=[SO101Inputs()],
                outputs=[SO101Outputs()],
            ),
            model_transforms=_config.ModelTransformFactory()(model_config),
            action_sequence_keys=("action",),
        )


def make_model_config(state_mode: str = "off") -> pi0_config.Pi0Config:
    if state_mode not in {"off", "on"}:
        raise ValueError(f"unknown state_mode {state_mode!r}")
    return pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=state_mode == "on",
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )


def make_freeze_filter():
    """Frozen vision+LLM backbone, trainable LoRA adapters and projections.

    OpenPI's convenience LoRA filter freezes only ``.*llm.*`` and therefore
    leaves the SigLIP vision encoder fully trainable.  SO101 E2 freezes the
    whole backbone (``.*llm.*`` and ``PaliGemma/img``) so the trainable target
    is exactly the 20 LoRA tensors plus the action/time projections.
    """
    return nnx.All(
        nnx.Param,
        nnx.Any(
            nnx_utils.PathRegex(".*llm.*"),
            nnx_utils.PathRegex(".*PaliGemma/img.*"),
        ),
        nnx.Not(nnx_utils.PathRegex(".*lora.*")),
    )


def make_train_config(*, mode: str, run_root: Path, train_view: Path,
                      base_checkpoint: Path, exp_name: str,
                      state_mode: str = "off", steps_override: int | None = None,
                      save_interval: int | None = None, batch_size: int = 1,
                      lr_warmup_steps: int = 10, peak_lr: float = 5e-5,
                      decay_steps: int = 100, decay_lr: float = 5e-5,
                      dataset_label: str = "so101_e1_20",
                      phase: str = "E2", num_workers: int = 0) -> _config.TrainConfig:
    model_config = make_model_config(state_mode)
    if mode not in {"initial", "resume", "serve", "smoke", "probe"}:
        raise ValueError(f"unknown mode {mode}")
    steps = {"initial": 100, "resume": 101, "serve": 0, "smoke": 1, "probe": 1}[mode]
    if steps_override is not None:
        steps = int(steps_override)
    if mode == "resume" and steps_override is None:
        steps = 101
    return _config.TrainConfig(
        name="pi05_so101_e2",
        project_name="so101",
        exp_name=exp_name,
        model=model_config,
        weight_loader=_weight_loaders.CheckpointWeightLoader(str(Path(base_checkpoint) / "params")),
        data=SO101DataConfig(
            repo_id="so101_e1",
            assets=_config.AssetsConfig(assets_dir=str(Path(train_view) / "assets"),
                                        asset_id="so101"),
        ),
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        num_train_steps=steps,
        log_interval=1,
        save_interval=int(save_interval if save_interval is not None else 50),
        keep_period=1,
        overwrite=mode in {"initial", "smoke", "probe"},
        resume=mode == "resume",
        wandb_enabled=False,
        ema_decay=None,
        fsdp_devices=1,
        seed=42,
        assets_base_dir=str(Path(train_view) / "assets"),
        checkpoint_base_dir=str(Path(run_root) / "checkpoints"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=int(lr_warmup_steps), peak_lr=float(peak_lr),
            decay_steps=int(decay_steps), decay_lr=float(decay_lr)),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=make_freeze_filter(),
        policy_metadata={
            "model": "pi05_so101_lora",
            "dataset": dataset_label,
            "phase": phase,
            "state_action_dim": ACTION_DIM,
            "action_horizon": 10,
            "cameras": list(CAMERAS.values()),
            "control_hz": 50,
            "engineering_chain_only": True,
            "state_mode": state_mode,
        },
    )


class SO101EpisodeDataset:
    """Training-view reader with chunk-safe episode boundaries."""

    def __init__(self, view_root: Path, *, split: str, action_horizon: int,
                 episode_ids: Sequence[int] | None = None):
        self.view_root = Path(view_root)
        self.split = split
        self.action_horizon = int(action_horizon)
        if self.action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        meta = json.loads((self.view_root / "meta.json").read_text(encoding="utf-8"))
        self.meta = meta
        self.feature_names = meta["feature_names"]
        self.episodes: dict[int, dict] = {}
        self.samples: list[tuple[int, int]] = []
        self.sample_metadata: list[dict] = []
        wanted = None if episode_ids is None else {int(value) for value in episode_ids}
        available = {int(episode["episode_index"]) for episode in meta["episodes"]
                     if episode["split"] == split}
        if wanted is not None and not wanted <= available:
            raise ValueError(f"requested episodes {sorted(wanted)} are not all in split {split}")
        for episode in meta["episodes"]:
            if episode["split"] != split:
                continue
            episode_id = int(episode["episode_index"])
            if wanted is not None and episode_id not in wanted:
                continue
            directory = self.view_root / "episodes" / f"episode_{episode_id:03d}"
            with np.load(directory / "episode.npz", allow_pickle=False) as data:
                arrays = {key: np.asarray(data[key]) for key in
                          ("state", "action", "timestamp", "frame_index", "task")}
            state, action = arrays["state"], arrays["action"]
            if state.shape != (len(state), ACTION_DIM) or action.shape != (len(action), ACTION_DIM):
                raise ValueError(f"episode {episode_id}: bad state/action shape")
            if not np.isfinite(state).all() or not np.isfinite(action).all():
                raise ValueError(f"episode {episode_id}: non-finite state/action")
            self.episodes[episode_id] = {
                "arrays": arrays,
                "directory": directory,
                "task": str(arrays["task"]),
                "frames": int(state.shape[0]),
            }
            usable = int(state.shape[0]) - self.action_horizon + 1
            if usable <= 0:
                continue
            perturbation_frame = episode.get("perturbation_frame")
            event = episode.get("event") or {}
            if perturbation_frame is None:
                perturbation_frame = event.get("control_frame")
            excluded_cross = 0
            for frame in range(usable):
                if perturbation_frame is not None:
                    trigger = int(perturbation_frame)
                    end = frame + self.action_horizon - 1
                    if frame <= trigger <= end:
                        excluded_cross += 1
                        continue
                self.samples.append((episode_id, frame))
                self.sample_metadata.append({
                    "sample_index": len(self.samples) - 1, "episode_id": episode_id, "frame": frame,
                    "category": str(episode.get("category", "clean")),
                    "subgroup": str(episode.get("subgroup", episode.get("category", "clean"))),
                    "split": split,
                    "perturbation_frame": (int(perturbation_frame) if perturbation_frame is not None else None),
                })
            self.episodes[episode_id]["excluded_cross_event_windows"] = excluded_cross
        if not self.samples:
            raise ValueError(f"split {split} has no usable samples")
        if len(self.samples) != len(self.sample_metadata):
            raise ValueError("sample metadata desynchronised from samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        episode_id, frame = self.samples[index]
        episode = self.episodes[episode_id]
        arrays = episode["arrays"]
        camera_arrays = {}
        for key, name in CAMERA_FILES.items():
            path = episode["directory"] / name / f"{frame:06d}.jpg"
            from PIL import Image

            with Image.open(path) as image:
                camera_arrays[key] = np.asarray(image.convert("RGB"))
        return {
            **camera_arrays,
            "observation.state": arrays["state"][frame].copy(),
            "action": arrays["action"][frame:frame + self.action_horizon].copy(),
            "task": episode["task"],
        }


    def sampling_summary(self) -> dict:
        groups: dict[str, dict] = {}
        for row in self.sample_metadata:
            key = row["subgroup"]
            entry = groups.setdefault(key, {"episodes": set(), "samples": 0})
            entry["episodes"].add(row["episode_id"])
            entry["samples"] += 1
        return {
            "split": self.split,
            "total_eligible_windows": len(self.samples),
            "excluded_cross_event_windows": sum(int(row.get("excluded_cross_event_windows", 0))
                                                for row in self.episodes.values()),
            "groups": {key: {"episodes": len(value["episodes"]), "samples": value["samples"]}
                       for key, value in groups.items()},
        }

    def weighted_sample_probabilities(self, policy: dict) -> tuple[np.ndarray, dict]:
        mode = str(policy.get("mode", "balanced_categories"))
        if mode != "balanced_categories":
            raise ValueError(f"unknown sampling mode {mode}")
        group_weights = {str(k): float(v) for k, v in policy["group_weights"].items()}
        present = sorted({row["subgroup"] for row in self.sample_metadata})
        missing = [group for group in present if group not in group_weights]
        if missing:
            raise ValueError(f"sampling plan has no weight for present groups {missing}")
        unused = [group for group in group_weights if group not in present]
        normalizer = sum(group_weights[group] for group in present)
        if normalizer <= 0:
            raise ValueError("sampling plan group weights sum to zero")
        by_group: dict[str, dict[int, int]] = {}
        for row in self.sample_metadata:
            by_group.setdefault(row["subgroup"], {}).setdefault(row["episode_id"], 0)
            by_group[row["subgroup"]][row["episode_id"]] += 1
        weights = np.zeros(len(self.sample_metadata), dtype=float)
        for group, episodes in by_group.items():
            group_mass = group_weights[group] / normalizer
            for episode_id, episode_samples in episodes.items():
                for row in self.sample_metadata:
                    if row["subgroup"] == group and row["episode_id"] == episode_id:
                        weights[row["sample_index"]] = group_mass / len(episodes) / episode_samples
        if not np.isclose(weights.sum(), 1.0, atol=1e-9):
            raise ValueError(f"sampling weights sum to {weights.sum()}, expected 1.0")
        actual = {}
        for group, episodes in by_group.items():
            actual[group] = {
                "configured_mass": group_weights[group] / normalizer,
                "episodes": len(episodes),
                "eligible_windows": sum(episodes.values()),
            }
        return weights, {"mode": mode, "groups": actual, "unused_groups": unused,
                         "episode_equal_within_group": True}

    def precomputed_sample_indices(self, weights: np.ndarray, count: int, *, seed: int) -> list[int]:
        rng = np.random.default_rng(int(seed))
        values = rng.choice(len(weights), size=int(count), replace=True, p=weights)
        return [int(value) for value in values]


class _ListSampler(torch.utils.data.Sampler):
    """Finite sampler that replays the exact precomputed update sequence."""

    def __init__(self, indices: Sequence[int]):
        self.indices = [int(value) for value in indices]
        if not self.indices:
            raise ValueError("sample-index sequence is empty")

    def __iter__(self):
        while True:
            yield from self.indices

    def __len__(self) -> int:
        return len(self.indices)


def create_so101_data_loader(config: _config.TrainConfig, *, sharding=None,
                             shuffle: bool = False, num_batches: int | None = None,
                             skip_norm_stats: bool = False, framework: str = "jax"):
    data_config = config.data.create(config.assets_dirs, config.model)
    view_root = Path(os.environ.get("SO101_TRAIN_VIEW", ""))
    if not view_root.is_dir():
        raise FileNotFoundError("SO101_TRAIN_VIEW must point at the exported training view")
    requested = os.environ.get("SO101_TRAIN_EPISODES", "").strip()
    episode_ids = [int(value) for value in requested.split(",") if value.strip()] if requested else None
    raw_dataset = SO101EpisodeDataset(view_root, split="train",
                                      action_horizon=config.model.action_horizon,
                                      episode_ids=episode_ids)
    sampler = None
    sample_ids_path = os.environ.get("SO101_TRAINING_SAMPLE_IDS", "").strip()
    if sample_ids_path:
        path = Path(sample_ids_path)
        if path.suffix == ".gz":
            import gzip
            opener = gzip.open
        else:
            opener = Path.open
        with opener(path, "rt", encoding="utf-8") as handle:
            indices = [int(line) for line in handle if line.strip()]
        sampler = _ListSampler(indices)
    dataset = _data_loader.transform_dataset(raw_dataset, data_config, skip_norm_stats=skip_norm_stats)
    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size,
        sharding=sharding,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        framework=framework,
    )
    return _data_loader.DataLoaderImpl(data_config, loader)
