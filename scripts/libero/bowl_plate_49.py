"""Configuration and data loading for the all-49 LIBERO bowl-to-plate run."""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Sequence

EXPECTED_EPISODES = (
    379, 422, 426, 431, 433, 447, 448, 451, 459, 466,
    481, 483, 488, 507, 511, 513, 522, 532, 537, 549,
    551, 563, 568, 582, 607, 615, 620, 621, 626, 634,
    639, 642, 646, 653, 655, 670, 679, 708, 716, 718,
    726, 727, 749, 750, 768, 770, 801, 803, 806,
)
PROTOCOL_FIELDS = {
    "schema_version", "protocol_id", "experiment_prefix", "policy_config",
    "suite", "task_id", "dataset_task_index", "task", "base", "data",
    "runtime", "model", "training", "checkpoint", "resource", "evaluation",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> dict:
    path = Path(path)
    if path.is_file():
        data = path.read_bytes()
        return {
            "files": 1,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    files = sorted(
        item for item in path.rglob("*")
        if item.is_file()
        and "__pycache__" not in item.parts
        and item.suffix not in {".pyc", ".pyo"}
    )
    if not files:
        raise ValueError(f"empty tree: {path}")
    digest = hashlib.sha256()
    total = 0
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        file_digest = hashlib.sha256()
        size = 0
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                size += len(block)
                file_digest.update(block)
        total += size
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(file_digest.digest())
    return {"files": len(files), "bytes": total, "sha256": digest.hexdigest()}


def canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8")
    ).hexdigest()


def effective_schedule(protocol: dict, batch_size: int) -> dict:
    """Scale every step-based setting to preserve the batch-32 sample budget."""
    training = protocol["training"]
    base_batch = int(training["configured_batch_size"])
    batch_size = int(batch_size)
    if batch_size not in [int(value) for value in training["allowed_batch_sizes"]]:
        raise ValueError("batch size is not registered")
    if base_batch % batch_size:
        raise ValueError("batch-size reduction must use an integer divisor")
    scale = base_batch // batch_size
    steps = int(training["optimizer_updates"]) * scale
    checkpoint_steps = [
        int(value) * scale for value in protocol["checkpoint"]["save_steps"]
    ]
    schedule = {
        "selected_batch_size": batch_size,
        "batch_scale": scale,
        "sample_budget": int(training["optimizer_updates"]) * base_batch,
        "optimizer_updates": steps,
        "warmup_steps": int(training["warmup_steps"]) * scale,
        "cosine_decay_steps": int(training["cosine_decay_steps"]) * scale,
        "log_interval": int(training["log_interval"]) * scale,
        "save_interval": int(protocol["checkpoint"]["save_interval"]) * scale,
        "keep_period": int(protocol["checkpoint"]["keep_period"]) * scale,
        "checkpoint_steps": checkpoint_steps,
    }
    expected_checkpoints = [
        int(value) * scale for value in (500, 1000, 1500)
    ]
    if checkpoint_steps != expected_checkpoints:
        raise ValueError("checkpoint cadence does not match the scaled horizon")
    if schedule["cosine_decay_steps"] != steps:
        raise ValueError("LR decay horizon does not match scaled updates")
    if schedule["optimizer_updates"] * batch_size != schedule["sample_budget"]:
        raise ValueError("scaled schedule changes the sample budget")
    return schedule


def load_protocol(path: Path, expected_sha256: str) -> dict:
    path = Path(path).resolve()
    actual = sha256_file(path)
    if not expected_sha256 or actual != expected_sha256:
        raise ValueError(f"protocol SHA256 mismatch: {actual}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if set(document) != PROTOCOL_FIELDS:
        raise ValueError("protocol has an unexpected field set")
    expected = {
        "schema_version": 1,
        "protocol_id": "pi05-libero-bowl-plate-49-lora-v1",
        "policy_config": "pi05_libero",
        "suite": "libero_goal",
        "task_id": 8,
        "dataset_task_index": 10,
        "task": "put the bowl on the plate",
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise ValueError(f"protocol {key} differs from the experiment")
    if document["base"]["name"] != "pi05_base":
        raise ValueError("protocol must initialize from pi05_base")
    model = document["model"]
    if (
        model.get("pi05") is not True
        or model.get("paligemma_variant") != "gemma_2b_lora"
        or model.get("action_expert_variant") != "gemma_300m_lora"
        or model.get("discrete_state_input") is not False
    ):
        raise ValueError("protocol model is not the requested pi05 LoRA variant")
    training = document["training"]
    exact_training = {
        "optimizer_updates": 1500,
        "optimizer": "AdamW",
        "warmup_steps": 100,
        "peak_lr": 2.5e-5,
        "cosine_decay_steps": 1500,
        "final_lr": 2.5e-6,
        "ema_decay": None,
        "gradient_clip_norm": 1.0,
        "wandb_enabled": False,
    }
    for key, value in exact_training.items():
        if training.get(key) != value:
            raise ValueError(f"protocol training.{key} differs from the experiment")
    if int(training.get("configured_batch_size", 32)) != 32:
        raise ValueError("configured batch size must be 32")
    allowed = [int(value) for value in training.get("allowed_batch_sizes", [])]
    if allowed != [32, 16, 8, 4]:
        raise ValueError("unexpected batch-size fallback set")
    scaling = training.get("step_scaling", {})
    if scaling != {
        "mode": "preserve_sample_budget",
        "basis_batch_size": 32,
        "scale_by": "basis_batch_size / selected_batch_size",
        "scaled_fields": [
            "optimizer_updates",
            "warmup_steps",
            "cosine_decay_steps",
            "log_interval",
            "save_interval",
            "keep_period",
            "checkpoint_steps",
        ],
    }:
        raise ValueError("protocol step scaling differs from the experiment")
    if [int(value) for value in training.get("fsdp_devices", [])] != [4, 2]:
        raise ValueError("FSDP devices must be selected from 4 or 2")
    checkpoint = document["checkpoint"]
    if checkpoint != {
        "save_steps": [500, 1000, 1500],
        "save_interval": 500,
        "keep_period": 500,
        "full_train_state": True,
        "retain_all_saved_steps": True,
        "step_semantics": "actual_train_state_step",
    }:
        raise ValueError("protocol checkpoint settings differ from the experiment")
    for batch_size in allowed:
        effective_schedule(document, batch_size)
    return {**document, "_path": str(path), "_sha256": actual}


def load_view(path: Path, protocol: dict) -> dict:
    path = Path(path)
    document = json.loads((path / "view.json").read_text(encoding="utf-8"))
    claimed = document.get("view_sha256")
    payload = {key: value for key, value in document.items() if key != "view_sha256"}
    if not claimed or canonical_sha256(payload) != claimed:
        raise ValueError("view SHA256 is invalid")
    if claimed != protocol["data"]["view_sha256"]:
        raise ValueError("view differs from the protocol")
    ids = tuple(int(value) for value in document.get("train_episode_ids", []))
    if ids != EXPECTED_EPISODES:
        raise ValueError("view does not contain exactly the requested 49 episodes")
    if document.get("validation_episode_ids") != []:
        raise ValueError("this experiment must not declare a validation split")
    if int(document.get("action_horizon", -1)) != 10:
        raise ValueError("view action horizon must be 10")
    if sum(int(row["frames"]) for row in document["episodes"]) != 4563:
        raise ValueError("view frame count differs from 4563")
    if any(row.get("split") != "train" for row in document["episodes"]):
        raise ValueError("all selected episodes must be training episodes")
    norm_path = path / document["norm_stats"]["path"]
    if not norm_path.is_file():
        raise ValueError("task-specific norm stats are missing")
    norm_sha = sha256_file(norm_path)
    if norm_sha != document["norm_stats"]["sha256"]:
        raise ValueError("view norm stats SHA256 mismatch")
    if norm_sha != protocol["data"]["norm_stats_sha256"]:
        raise ValueError("norm stats differ from the protocol")
    return document


def _patch_selected_episode_index(dataset, episodes: Sequence[int]) -> None:
    """Map selected-episode positions back to the original LeRobot v2.0 IDs."""
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
    from lerobot.common.datasets.lerobot_dataset import (
        LeRobotDataset,
        LeRobotDatasetMetadata,
    )

    expected = [int(value) for value in episodes]
    if expected != list(EXPECTED_EPISODES):
        raise ValueError("selected dataset request differs from the 49 episodes")
    metadata = LeRobotDatasetMetadata("local/libero-goal8", root=Path(data_root))
    dataset = LeRobotDataset(
        "local/libero-goal8",
        root=Path(data_root),
        episodes=expected,
        delta_timestamps={
            "actions": [tick / float(fps) for tick in range(int(action_horizon))]
        },
        download_videos=False,
    )
    _patch_selected_episode_index(dataset, expected)
    return dataset, metadata


def create_train_loader(
    config,
    *,
    view_path: Path,
    data_root: Path,
    sharding,
):
    from openpi import transforms
    from openpi.training import data_loader

    dataset, metadata = make_selected_dataset(
        Path(data_root),
        EXPECTED_EPISODES,
        action_horizon=config.model.action_horizon,
    )
    if len(dataset) != 4563:
        raise ValueError(f"selected dataset has {len(dataset)} samples, expected 4563")
    dataset = data_loader.TransformedDataset(
        dataset, [transforms.PromptFromLeRobotTask(metadata.tasks)])
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.norm_stats is None:
        raise ValueError("OpenPI did not load the all-49 norm stats")
    transformed = data_loader.transform_dataset(dataset, data_config)
    loader = data_loader.TorchDataLoader(
        transformed,
        local_batch_size=config.batch_size,
        sharding=sharding,
        shuffle=True,
        num_batches=None,
        num_workers=config.num_workers,
        seed=config.seed,
        framework="jax",
    )
    return data_loader.DataLoaderImpl(data_config, loader)


def make_config(args, protocol: dict, view: dict, run_root: Path, target_steps: int):
    import openpi.training.config as config_lib
    import openpi.training.optimizer as optimizer_lib
    from openpi.training import weight_loaders

    base = config_lib.get_config(protocol["policy_config"])
    model_settings = protocol["model"]
    model = dataclasses.replace(
        base.model,
        pi05=True,
        paligemma_variant=model_settings["paligemma_variant"],
        action_expert_variant=model_settings["action_expert_variant"],
        action_horizon=int(protocol["data"]["action_horizon"]),
        discrete_state_input=False,
        dtype=model_settings["dtype"],
    )
    data = dataclasses.replace(
        base.data,
        repo_id="local/libero-goal8",
        assets=config_lib.AssetsConfig(
            assets_dir=str(Path(args.view) / "assets"),
            asset_id="libero_goal8",
        ),
        base_config=config_lib.DataConfig(prompt_from_task=True),
        extra_delta_transform=False,
    )
    training = protocol["training"]
    schedule = effective_schedule(protocol, int(args.batch_size))
    freeze_filter = model.get_freeze_filter()
    return dataclasses.replace(
        base,
        name="pi05_libero_bowl_plate_49_lora",
        project_name="libero-bowl-plate",
        exp_name=args.exp_name,
        model=model,
        data=data,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            str(Path(args.base_checkpoint) / "params")
        ),
        freeze_filter=freeze_filter,
        batch_size=int(args.batch_size),
        num_train_steps=int(target_steps),
        num_workers=int(training["data_loader_workers"]),
        save_interval=int(schedule["save_interval"]),
        keep_period=int(schedule["keep_period"]),
        seed=int(training["seed"]),
        ema_decay=None,
        fsdp_devices=int(args.fsdp_devices),
        assets_base_dir=str(Path(args.view) / "assets"),
        checkpoint_base_dir=str(Path(run_root) / "checkpoints"),
        lr_schedule=optimizer_lib.CosineDecaySchedule(
            warmup_steps=int(schedule["warmup_steps"]),
            peak_lr=float(training["peak_lr"]),
            decay_steps=int(schedule["cosine_decay_steps"]),
            decay_lr=float(training["final_lr"]),
        ),
        optimizer=optimizer_lib.AdamW(
            b1=float(training["adamw_b1"]),
            b2=float(training["adamw_b2"]),
            eps=float(training["adamw_eps"]),
            weight_decay=float(training["adamw_weight_decay"]),
            clip_gradient_norm=float(training["gradient_clip_norm"]),
        ),
        overwrite=args.mode in {"full", "probe"},
        resume=False,
        wandb_enabled=False,
        log_interval=int(schedule["log_interval"]),
        policy_metadata={
            "suite": protocol["suite"],
            "task_id": protocol["task_id"],
            "dataset_task_index": protocol["dataset_task_index"],
            "task": protocol["task"],
            "view_sha256": view["view_sha256"],
            "norm_stats_sha256": view["norm_stats"]["sha256"],
            "base": protocol["base"]["name"],
            "scope": protocol["model"]["scope"],
            "protocol_sha256": protocol["_sha256"],
            "batch_size": int(args.batch_size),
            "fsdp_devices": int(args.fsdp_devices),
            "effective_schedule": schedule,
        },
    )


def runtime_hashes() -> dict:
    return {
        "openpi_src": sha256_tree(Path("/opt/openpi/src/openpi")),
        "openpi_scripts": sha256_tree(Path("/opt/openpi/scripts")),
        "libero_source": sha256_tree(Path("/opt/openpi/third_party/libero")),
    }


def load_train_module():
    path = Path("/opt/openpi/scripts/train.py")
    if not path.is_file():
        raise FileNotFoundError("pinned OpenPI scripts/train.py is unavailable")
    spec = importlib.util.spec_from_file_location("openpi_train_entry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path
