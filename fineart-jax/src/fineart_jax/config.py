# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from LeRobot FineART-VLA configuration for the JAX migration.

"""FineART-VLA configuration for the JAX migration.

The defaults here are the frozen facts of ``migration/contract.json``; the
validation rules mirror the reference so that an unsupported combination fails
loudly instead of silently changing behaviour. Nothing in this module touches a
GPU or imports a model: the data layer must be usable on its own.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, fields
from typing import Any

# Reference default recipe: a two-stream blend, 0.30 high-level / 0.70 low-level.
DEFAULT_RECIPE: dict[str, Any] = {
    "blend": {
        "high_level_subtask": {
            "weight": 0.30,
            "messages": [
                {"role": "user", "content": "${task}", "stream": "high_level"},
                {
                    "role": "assistant",
                    "content": "${subtask}",
                    "stream": "high_level",
                    "target": True,
                    "if_present": "subtask",
                },
            ],
        },
        "low_level_execution": {
            "weight": 0.70,
            "messages": [
                {
                    "role": "user",
                    "content": "${subtask}",
                    "stream": "low_level",
                    "if_present": "subtask",
                }
            ],
        },
    }
}

SUPPORTED_DTYPES = ("float32",)
UNSUPPORTED_IN_THIS_PROFILE = (
    "bfloat16",
    "float16",
    "multi_device",
    "distributed_sharding",
    "rtc_enabled_training",
    "custom_cuda_kernels",
    "cpu_fallback",
)
KNOWN_VARIANTS = ("gemma_2b", "gemma_300m")


def default_recipe() -> dict[str, Any]:
    return copy.deepcopy(DEFAULT_RECIPE)


@dataclass
class FineARTJaxConfig:
    """The subset of the FineART-VLA contract the data layer depends on.

    Fields carry the reference defaults verbatim. ``recipe`` is the serialized
    training/runtime contract; ``None`` selects the plain PI0.5 prompt, which is
    a supported default-API boundary rather than an error.
    """

    # Model identity -----------------------------------------------------
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"

    # Action / sequence shape -------------------------------------------
    chunk_size: int = 50
    n_action_steps: int = 50
    n_obs_steps: int = 1
    max_state_dim: int = 32
    max_action_dim: int = 32
    num_inference_steps: int = 10

    # Text and FAST tokenization ----------------------------------------
    tokenizer_max_length: int = 200
    text_tokenizer_name: str = "google/paligemma-3b-pt-224"
    action_tokenizer_name: str = "physical-intelligence/fast"
    max_action_tokens: int = 256
    fast_skip_tokens: int = 1152
    auto_fit_fast_tokenizer: bool = True
    fast_tokenizer_cache_dir: str = "~/.cache/lerobot/fast_tokenizers"
    fast_tokenizer_fit_samples: int = 1024
    fast_tokenizer_validation_samples: int = 256
    fast_tokenizer_max_reconstruction_rmse: float = 0.10
    fast_tokenizer_max_dim_rmse: float = 0.20

    # Losses -------------------------------------------------------------
    text_loss_weight: float = 1.0
    flow_loss_weight: float = 10.0
    fast_action_loss_weight: float = 1.0
    enable_fast_action_loss: bool = True
    text_ce_z_loss_weight: float = 1e-4
    knowledge_insulation: bool = True
    flow_num_repeats: int = 5

    # Recipe / language --------------------------------------------------
    recipe_path: str | None = None
    recipe: dict[str, Any] | None = field(default_factory=default_recipe)
    memory_scratchpad: bool = False
    unfreeze_lm_head: bool = True  # deprecated and ignored by the reference

    # Data ---------------------------------------------------------------
    image_resolution: tuple[int, int] = (224, 224)
    empty_cameras: int = 0
    use_relative_actions: bool = False
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    action_feature_names: list[str] | None = None
    normalization_mapping: dict[str, str] = field(
        default_factory=lambda: {"VISUAL": "IDENTITY", "STATE": "QUANTILES", "ACTION": "QUANTILES"}
    )
    relative_action_mask: list[bool] | None = None

    # Training -----------------------------------------------------------
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 30000
    scheduler_decay_lr: float = 2.5e-6
    lm_head_lr_scale: float = 1.0
    backbone_lr_scale: float = 1.0
    action_expert_lr_scale: float = 1.0
    conditioning_lr_scale: float = 1.0
    freeze_vision_encoder: bool = False
    train_expert_only: bool = False

    # Opt-in backends; defaults keep the eager path ------------------------
    use_liger_kernels: bool = False
    use_flashrt_adarms: bool = False
    use_compiled_text_ce: bool = False
    use_compiled_vision: bool = False
    use_flex_attention: bool = False
    use_manual_attention: bool = False
    manual_attention_scope: str = "all"
    optimizer_foreach: bool | None = False
    optimizer_fused: bool | None = True

    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Reject combinations the frozen contract does not support."""
        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError(
                f"dtype {self.dtype!r} is not supported in the fp32 default profile; "
                f"supported: {list(SUPPORTED_DTYPES)}"
            )
        if self.paligemma_variant not in KNOWN_VARIANTS:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")
        if self.action_expert_variant not in KNOWN_VARIANTS:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )
        if self.enable_fast_action_loss and self.recipe is None:
            raise ValueError(
                "FineART-VLA FAST action loss requires recipe_path to build action supervision."
            )
        if self.train_expert_only and (self.text_loss_weight > 0 or self.enable_fast_action_loss):
            raise ValueError(
                "train_expert_only=true freezes the VLM that the text and FAST losses train. "
                "Set train_expert_only=false, or disable both (text_loss_weight=0, "
                "enable_fast_action_loss=false) to train only the action expert."
            )
        if self.flow_num_repeats < 1:
            raise ValueError(f"flow_num_repeats must be >= 1, got {self.flow_num_repeats}")
        if self.fast_tokenizer_validation_samples < 1:
            raise ValueError("fast_tokenizer_validation_samples must be >= 1")
        if self.fast_tokenizer_max_reconstruction_rmse <= 0 or self.fast_tokenizer_max_dim_rmse <= 0:
            raise ValueError("FAST tokenizer reconstruction thresholds must be positive")
        if not math.isfinite(self.conditioning_lr_scale) or self.conditioning_lr_scale < 0:
            raise ValueError("conditioning_lr_scale must be finite and nonnegative")
        if self.manual_attention_scope not in {"all", "action"}:
            raise ValueError(
                f"manual_attention_scope must be 'all' or 'action', got {self.manual_attention_scope!r}"
            )
        if self.use_flex_attention and self.use_manual_attention:
            raise ValueError("use_flex_attention and use_manual_attention are mutually exclusive")
        if self.use_flex_attention and self.flow_num_repeats == 1:
            raise ValueError("use_flex_attention requires flow_num_repeats > 1")
        if not self.knowledge_insulation and (
            self.use_flex_attention or self.use_manual_attention or self.use_flashrt_adarms
        ):
            raise ValueError("KI attention and AdaRMS optimizations require knowledge_insulation=True")
        if self.recipe is not None:
            # Fail at configuration time, not on the first batch.
            validate_recipe(self.recipe)

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration, including the resolved recipe.

        The recipe is embedded so a checkpoint keeps the contract it trained
        with; ``recipe_path`` is cleared for the same reason the reference
        clears it.
        """
        data = {item.name: getattr(self, item.name) for item in fields(self)}
        data["recipe_path"] = None
        data["image_resolution"] = list(self.image_resolution)
        data["optimizer_betas"] = list(self.optimizer_betas)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FineARTJaxConfig:
        known = {item.name for item in fields(cls)}
        payload = {key: value for key, value in data.items() if key in known}
        if "image_resolution" in payload:
            payload["image_resolution"] = tuple(payload["image_resolution"])
        if "optimizer_betas" in payload:
            payload["optimizer_betas"] = tuple(payload["optimizer_betas"])
        return cls(**payload)

    @property
    def relative_mask(self) -> list[bool]:
        """Per-dimension relative-action mask, gripper excluded by name."""
        return build_relative_mask(
            action_dim=self.max_action_dim,
            action_names=self.action_feature_names,
            exclude_joints=self.relative_exclude_joints,
            explicit=self.relative_action_mask,
        )


def build_relative_mask(
    action_dim: int,
    action_names: list[str] | None = None,
    exclude_joints: list[str] | None = None,
    explicit: list[bool] | None = None,
) -> list[bool]:
    """Which action dimensions become relative, mirroring the reference rule.

    ``explicit`` wins when supplied; otherwise a dimension is made relative
    unless its name contains an excluded joint token (case-insensitive). With no
    names every dimension is relative.
    """
    if explicit is not None:
        mask = list(explicit)
        if len(mask) < action_dim:
            mask.extend([True] * (action_dim - len(mask)))
        return mask[:action_dim]
    if not exclude_joints or action_names is None:
        return [True] * action_dim

    tokens = [str(name).lower() for name in exclude_joints if name]
    if not tokens:
        return [True] * action_dim

    mask = []
    for name in list(action_names)[:action_dim]:
        lowered = str(name).lower()
        excluded = any(token == lowered or token in lowered for token in tokens)
        mask.append(not excluded)
    if len(mask) < action_dim:
        mask.extend([True] * (action_dim - len(mask)))
    return mask


def validate_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
    """Validate a serialized recipe and return it unchanged.

    Mirrors the reference's structural rules: exactly one of ``messages`` or
    ``blend``, weighted blend components, a stream on every turn, and at least
    one supervised turn per message recipe.
    """
    if not isinstance(recipe, dict):
        raise TypeError("recipe must be a mapping")
    has_messages = recipe.get("messages") is not None
    has_blend = recipe.get("blend") is not None
    if has_messages and has_blend:
        raise ValueError("TrainingRecipe must set only one of messages or blend.")
    if not has_messages and not has_blend:
        raise ValueError("TrainingRecipe must set one of messages or blend.")
    if recipe.get("route") is not None:
        if recipe["route"] not in ("vqa",):
            raise ValueError(f"Unsupported recipe route: {recipe['route']!r}")
        if has_blend:
            raise ValueError("TrainingRecipe.route may only be set on a message recipe inside a blend.")

    if has_blend:
        blend = recipe["blend"]
        if not isinstance(blend, dict) or not blend:
            raise ValueError("Blend recipes must contain at least one component.")
        for name, component in blend.items():
            if component.get("blend") is not None:
                raise ValueError(f"Blend component {name!r} cannot itself define a blend.")
            if component.get("messages") is None:
                raise ValueError(f"Blend component {name!r} must define messages.")
            weight = component.get("weight")
            if weight is None:
                raise ValueError(f"Blend component {name!r} must define weight.")
            if weight <= 0:
                raise ValueError(f"Blend component {name!r} must have a positive weight.")
            validate_recipe({**component, "blend": None, "weight": None})
        return recipe

    turns = recipe["messages"]
    if not isinstance(turns, list) or not turns:
        raise ValueError("Message recipes need at least one turn.")
    has_target = any(turn.get("target") for turn in turns)
    has_low_level = any(turn.get("stream") == "low_level" for turn in turns)
    for turn in turns:
        if turn.get("stream") is None:
            raise ValueError(
                f"MessageTurn(role={turn.get('role')!r}) is missing a stream — "
                "every turn must declare one of ['high_level', 'low_level']."
            )
        if turn.get("stream") not in ("high_level", "low_level"):
            raise ValueError(f"Unsupported message stream: {turn.get('stream')!r}")
        if turn.get("role") not in ("user", "assistant", "system", "tool"):
            raise ValueError(f"Unsupported message role: {turn.get('role')!r}")
        if turn.get("content") is None and turn.get("tool_calls_from") is None:
            raise ValueError("MessageTurn.content is required unless tool_calls_from is set.")
    if not (has_target or has_low_level):
        raise ValueError(
            "Message recipes must contain at least one supervised turn — "
            "either ``target: true`` (text CE) or ``stream: low_level`` (flow/action loss)."
        )
    return recipe
