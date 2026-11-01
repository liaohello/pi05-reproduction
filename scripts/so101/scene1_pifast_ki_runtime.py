"""Runtime configuration for Scene1 PI-FAST plus Flow Matching and KI."""
from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import numpy as np
from typing_extensions import override

import openpi.models.model as model_lib
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as config_lib
import openpi.training.data_loader as data_loader
import openpi.training.optimizer as optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as transforms
from scene1_openpi_so101 import SO101Inputs, SO101Outputs
from scene1_pifast_ki_fast_data import Scene1FASTTokenizer
from scene1_pifast_ki_model import JointPi05Config

ACTION_DIM = 32
INNER_ACTION_DIM = 12
ACTION_HORIZON = 16
MAX_TOKEN_LEN = 128


@dataclasses.dataclass(frozen=True)
class StrictJointWeightLoader(weight_loaders.WeightLoader):
    """Load every inherited base tensor and initialize only LoRA plus FAST rows."""

    params_path: str

    def load(self, params):
        reference = traverse_util.flatten_dict(params, sep="/")
        loaded = traverse_util.flatten_dict(
            model_lib.restore_params(
                self.params_path,
                restore_type=np.ndarray,
            ),
            sep="/",
        )
        unexpected = sorted(set(loaded) - set(reference))
        if unexpected:
            raise ValueError(
                f"base checkpoint has unexpected parameters: {unexpected[:8]}")
        missing = sorted(set(reference) - set(loaded))
        allowed = re.compile(r"(?:.*lora.*|fast_token_residual/table)")
        disallowed = [path for path in missing if not allowed.fullmatch(path)]
        if disallowed:
            raise ValueError(
                f"base checkpoint is missing inherited parameters: {disallowed[:8]}")
        result = {}
        for path, value in loaded.items():
            target = reference[path]
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f"shape mismatch for {path}: {value.shape} != {target.shape}")
            result[path] = np.asarray(value).astype(np.dtype(target.dtype))
        for path in missing:
            result[path] = reference[path]
        return traverse_util.unflatten_dict(result, sep="/")


def joint_freeze_filter():
    """Freeze all original Gemma language/expert tensors, keep every LoRA."""
    return nnx.All(
        nnx.Param,
        nnx_utils.PathRegex(".*PaliGemma/llm.*"),
        nnx.Not(nnx_utils.PathRegex(".*lora.*")),
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class JointDataConfig(config_lib.DataConfigFactory):
    fast_tokenizer_dir: str
    paligemma_tokenizer_path: str

    @override
    def create(
        self,
        assets_dirs: Path,
        model_config: model_lib.BaseModelConfig,
    ) -> config_lib.DataConfig:
        tokenizer = Scene1FASTTokenizer(
            self.fast_tokenizer_dir,
            self.paligemma_tokenizer_path,
            max_len=MAX_TOKEN_LEN,
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=transforms.Group(),
            data_transforms=transforms.Group(
                inputs=[SO101Inputs()],
                outputs=[SO101Outputs()],
            ),
            model_transforms=transforms.Group(
                inputs=[
                    transforms.ResizeImages(224, 224),
                    transforms.TokenizeFASTInputs(tokenizer),
                    transforms.PadStatesAndActions(ACTION_DIM),
                ],
            ),
            action_sequence_keys=("action",),
        )


def make_train_config(
    *,
    mode: str,
    run_root: Path,
    train_view: Path,
    base_checkpoint: Path,
    exp_name: str,
    steps: int,
    batch_size: int,
    save_interval: int,
    warmup_steps: int,
    peak_lr: float,
    decay_steps: int,
    decay_lr: float,
    seed: int,
    fsdp_devices: int,
    num_workers: int,
    fast_ce_weight: float,
    fast_tokenizer_dir: Path,
    paligemma_tokenizer_path: Path,
    dataset_label: str,
) -> config_lib.TrainConfig:
    if mode not in {"initial", "resume", "smoke", "probe"}:
        raise ValueError(f"unsupported mode {mode!r}")
    if steps < 1 or batch_size < 1 or save_interval < 1:
        raise ValueError("steps, batch size, and save interval must be positive")
    if fsdp_devices < 1:
        raise ValueError("fsdp_devices must be positive")
    if not 0.0 < peak_lr < 1e-2 or not 0.0 < decay_lr <= peak_lr:
        raise ValueError("invalid learning-rate schedule")
    if not 0.0 < fast_ce_weight < 100.0:
        raise ValueError("fast_ce_weight must be explicitly selected")

    model_config = JointPi05Config(
        pi05=True,
        discrete_state_input=True,
        action_dim=ACTION_DIM,
        action_horizon=ACTION_HORIZON,
        max_token_len=MAX_TOKEN_LEN,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora_r64",
        dtype="bfloat16",
        fast_ce_weight=fast_ce_weight,
    )
    return config_lib.TrainConfig(
        name="pi05_so101_pifast",
        project_name="so101",
        exp_name=exp_name,
        model=model_config,
        weight_loader=StrictJointWeightLoader(
            str(Path(base_checkpoint) / "params")),
        data=JointDataConfig(
            repo_id="so101_scene1",
            assets=config_lib.AssetsConfig(
                assets_dir=str(Path(train_view) / "assets"),
                asset_id="so101",
            ),
            fast_tokenizer_dir=str(fast_tokenizer_dir),
            paligemma_tokenizer_path=str(paligemma_tokenizer_path),
        ),
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        num_train_steps=int(steps),
        log_interval=1,
        save_interval=int(save_interval),
        keep_period=1,
        overwrite=mode in {"initial", "smoke", "probe"},
        resume=mode == "resume",
        wandb_enabled=False,
        ema_decay=None,
        fsdp_devices=int(fsdp_devices),
        seed=int(seed),
        assets_base_dir=str(Path(train_view) / "assets"),
        checkpoint_base_dir=str(Path(run_root) / "checkpoints"),
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=int(warmup_steps),
            peak_lr=float(peak_lr),
            decay_steps=int(decay_steps),
            decay_lr=float(decay_lr),
        ),
        optimizer=optimizer.AdamW(
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=1e-10,
            clip_gradient_norm=1.0,
        ),
        freeze_filter=joint_freeze_filter(),
        policy_metadata={
            "model": "pi05_scene1_pifast_flow_ki",
            "paligemma_variant": "gemma_2b_lora",
            "action_expert_variant": "gemma_300m_lora_r64",
            "freeze_rule": (
                "original Gemma language and action-expert tensors frozen; "
                "attention and MLP LoRA trainable; SigLIP full trainable"),
            "dataset": dataset_label,
            "phase": "scene1-pifast-ki",
            "physical_action_dim": INNER_ACTION_DIM,
            "internal_action_dim": ACTION_DIM,
            "action_horizon": ACTION_HORIZON,
            "n_action_steps": ACTION_HORIZON,
            "fast_ce_weight": float(fast_ce_weight),
            "knowledge_insulation": True,
            "ema_decay": None,
            "seed": int(seed),
            "dtype": "bfloat16",
            "state_mode": "on",
            "control_hz": 50,
        },
    )
