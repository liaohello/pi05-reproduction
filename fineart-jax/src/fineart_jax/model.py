"""Full FineART-VLA / PI0.5 stack: vision tower, prefix assembly, 18 joint layers,
final norms and the flow-matching action head.

The stack is a thin composition of operators already certified by G1/G2 (vision tower,
embeddings, RoPE attention, RMSNorm/AdaRMS) plus the single joint layer in ``nn.joint``.
No new numerical rule is introduced here; every block maps one-to-one onto the read-only
reference forward (``PI05Pytorch.embed_prefix``/``embed_suffix``/``forward`` and
``PaliGemmaWithExpertModel.forward``).

Tensor names are the ones produced by ``fineart_jax.weights.map_to_registered``; the
builder below only re-groups them, it never renames a source tensor.
"""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .nn.joint import (embed_suffix, flow_mse_loss, joint_layer, sinusoidal_time_embedding)
from .nn.masking import combined_masks
from .nn.norms import ada_rms
from .nn.vision import project, vision_tower

VLM = "model.paligemma_with_expert.paligemma.model.language_model."
VISION = "model.paligemma_with_expert.paligemma.model.vision_tower."
PROJECTOR = "model.paligemma_with_expert.paligemma.model.multi_modal_projector.linear."
EXPERT = "model.paligemma_with_expert.gemma_expert.model."

VLM_WIDTH = 2048
EXPERT_WIDTH = 1024
VISION_WIDTH = 1152


def _layer_params(registered: dict[str, np.ndarray], prefix: str) -> dict[str, Any]:
    """Group one layer's tensors into the shape ``nn.joint`` expects."""
    out: dict[str, Any] = {
        "input_norm": {},
        "post_norm": {},
        "attention": {},
        "mlp": {},
    }
    for name, value in registered.items():
        if not name.startswith(prefix):
            continue
        tail = name[len(prefix):]
        if tail == "input_layernorm.weight":
            out["input_norm"]["weight"] = value
        elif tail == "input_layernorm.dense.weight":
            out["input_norm"]["dense_weight"] = value
        elif tail == "input_layernorm.dense.bias":
            out["input_norm"]["dense_bias"] = value
        elif tail == "post_attention_layernorm.weight":
            out["post_norm"]["weight"] = value
        elif tail == "post_attention_layernorm.dense.weight":
            out["post_norm"]["dense_weight"] = value
        elif tail == "post_attention_layernorm.dense.bias":
            out["post_norm"]["dense_bias"] = value
        elif tail.startswith("self_attn.") and tail.endswith("_proj.weight"):
            out["attention"][tail.split(".")[1]] = value
        elif tail.startswith("mlp.") and tail.endswith("_proj.weight"):
            out["mlp"][tail.split(".")[1]] = value
    return out


def build_params(registered: dict[str, np.ndarray]) -> dict[str, Any]:
    """Registered flat dict -> nested parameter tree for :func:`forward`."""
    vlm_layers = sorted(
        {
            int(name[len(VLM + "layers."):].split(".")[0])
            for name in registered
            if name.startswith(VLM + "layers.")
        }
    )
    expert_layers = sorted(
        {
            int(name[len(EXPERT + "layers."):].split(".")[0])
            for name in registered
            if name.startswith(EXPERT + "layers.")
        }
    )
    if vlm_layers != expert_layers:
        raise ValueError("VLM and action expert must expose the same layer indices")
    if vlm_layers != list(range(len(vlm_layers))):
        raise ValueError("layer indices must be contiguous from zero")

    vision = {
        name[len(VISION):]: value
        for name, value in registered.items()
        if name.startswith(VISION)
    }
    return {
        "vision": vision,
        "projector": {
            "weight": registered[PROJECTOR + "weight"],
            "bias": registered[PROJECTOR + "bias"],
        },
        "embed": registered[VLM + "embed_tokens.weight"],
        "prefix_layers": [_layer_params(registered, f"{VLM}layers.{i}.") for i in vlm_layers],
        "expert_layers": [_layer_params(registered, f"{EXPERT}layers.{i}.") for i in expert_layers],
        "prefix_norm": {"weight": registered[VLM + "norm.weight"]},
        "expert_norm": {
            "dense_weight": registered[EXPERT + "norm.dense.weight"],
            "dense_bias": registered[EXPERT + "norm.dense.bias"],
        },
        "action_in": {
            "weight": registered["model.action_in_proj.weight"],
            "bias": registered["model.action_in_proj.bias"],
        },
        "action_out": {
            "weight": registered["model.action_out_proj.weight"],
            "bias": registered["model.action_out_proj.bias"],
        },
        "time_in": {
            "weight": registered["model.time_mlp_in.weight"],
            "bias": registered["model.time_mlp_in.bias"],
        },
        "time_out": {
            "weight": registered["model.time_mlp_out.weight"],
            "bias": registered["model.time_mlp_out.bias"],
        },
    }


def _linear(x, params: dict):
    return x @ params["weight"].T + params["bias"]


def prefix_embeddings(params: dict, images, image_masks, tokens, token_masks):
    """Reference ``embed_prefix``: image patches, then language tokens scaled by sqrt(width)."""
    per_camera = []
    for image in images:
        patches = vision_tower(image, params["vision"], layers=27)
        per_camera.append(project(patches, params["projector"]["weight"], params["projector"]["bias"]))
    image_tokens = jnp.concatenate(per_camera, axis=1)

    pads = []
    for image, mask in zip(images, image_masks, strict=True):
        mask = jnp.asarray(mask, dtype=bool)
        # Reference rule: a [B, frames] mask contributes its last frame.
        current = mask[:, -1] if mask.ndim == 2 else mask
        pads.append(jnp.broadcast_to(current[:, None], (image.shape[0], per_camera[0].shape[1])))
    image_pad = jnp.concatenate(pads, axis=1)

    language = params["embed"][jnp.asarray(tokens)] * jnp.asarray(VLM_WIDTH**0.5, dtype=jnp.float32)
    prefix = jnp.concatenate([image_tokens, language], axis=1)
    pad = jnp.concatenate([image_pad, jnp.asarray(token_masks, dtype=bool)], axis=1)
    return prefix, pad


def suffix_embeddings(params: dict, x_t, time, *, chunk_size: int = 50):
    """Reference ``embed_suffix``: action projection plus the AdaRMS time conditioning."""
    action = _linear(jnp.asarray(x_t, dtype=jnp.float32), params["action_in"])
    raw_time = sinusoidal_time_embedding(time, params["time_in"]["weight"].shape[0])
    cond = jax.nn.silu(_linear(jax.nn.silu(_linear(raw_time, params["time_in"])), params["time_out"]))
    return action, cond


def _pad_action_width(value, width: int):
    value = jnp.asarray(value, dtype=jnp.float32)
    if value.shape[-1] < width:
        padding = [(0, 0)] * (value.ndim - 1) + [(0, width - value.shape[-1])]
        return jnp.pad(value, padding)
    return value[..., :width]


def forward(
    params: dict,
    images,
    image_masks,
    tokens,
    token_masks,
    actions,
    noise,
    time,
    *,
    knowledge_insulation: bool = True,
    chunk_size: int = 50,
    action_dim: int = 32,
    draws: int = 1,
    trace: bool = False,
):
    """One full FP32 training forward.

    ``actions``/``noise`` may carry fewer than the projection width (12 for SO-101, 14 for
    the published dataset). The base ``action_in_proj`` is a fixed 32-wide matrix, so the
    inputs are **zero-padded to the projection width** and only the loss reduction is
    restricted to the real dimensions. Slicing the input instead would feed a 12-wide vector
    into a 32-wide projection.

    ``draws > 1`` follows the reference's amortised path: one prefix, ``draws`` action
    chunks in one sequence, block-diagonal visibility and a per-draw position restart.

    Returns ``velocity`` shaped ``[B, chunk, pad]`` for a single draw and
    ``[B, draws, chunk, pad]`` otherwise.
    """
    pad_dim = int(params["action_in"]["weight"].shape[1])
    actions = _pad_action_width(actions, pad_dim)
    noise = _pad_action_width(noise, pad_dim)
    time = jnp.asarray(time, dtype=jnp.float32)

    prefix, prefix_pad = prefix_embeddings(params, images, image_masks, tokens, token_masks)
    batch = prefix.shape[0]
    prefix_len = prefix.shape[1]

    if draws == 1:
        t = time.reshape(-1, 1, 1)
        x_t = t * noise + (1.0 - t) * actions
        suffix, cond = suffix_embeddings(params, x_t, time.reshape(-1), chunk_size=chunk_size)
        total = chunk_size
    else:
        if time.size != batch * draws:
            raise ValueError(
                f"time must carry one value per draw ({batch * draws}), got {time.size}")
        t = time.reshape(batch, draws, 1, 1)
        # Each draw may carry its own noise/action chunk: accept [B, draws, chunk, pad]
        # as well as a single [B, chunk, pad] reused across draws.
        draw_noise = noise if noise.ndim == 4 else noise[:, None]
        draw_actions = actions if actions.ndim == 4 else actions[:, None]
        x_t = t * draw_noise + (1.0 - t) * draw_actions
        suffix, cond = suffix_embeddings(
            params, x_t.reshape(batch, draws * chunk_size, pad_dim),
            time.reshape(-1), chunk_size=chunk_size,
        )
        # Each draw owns its conditioning vector; repeat it across that draw's tokens.
        cond = jnp.repeat(cond.reshape(batch, draws, 1, -1), chunk_size, axis=2).reshape(
            batch, draws * chunk_size, -1)
        total = draws * chunk_size

    # ``combined_masks`` expects one draw's worth of suffix (it tiles the positions itself
    # for ``draws > 1``), so this stays chunk-sized even when the sequence is longer.
    suffix_pad = jnp.ones((batch, chunk_size), dtype=bool)
    suffix_att = jnp.concatenate(
        [jnp.ones((batch, 1), dtype=jnp.int32), jnp.zeros((batch, chunk_size - 1), dtype=jnp.int32)],
        axis=1,
    )
    mask, positions = combined_masks(
        prefix_pad,
        jnp.zeros((batch, prefix_len), dtype=jnp.int32),
        suffix_pad,
        suffix_att,
        non_fast_len=prefix_len,
        fast_len=0,
        draws=draws,
        chunk=chunk_size,
    )

    hidden = []
    for layer_index, (vlm_layer, expert_layer) in enumerate(
        zip(params["prefix_layers"], params["expert_layers"], strict=True)
    ):
        prefix, suffix = joint_layer(
            {"prefix": vlm_layer, "expert": expert_layer},
            prefix,
            suffix,
            mask,
            positions,
            cond,
            num_heads=8,
            num_kv_heads=1,
            head_dim=VLM_WIDTH // 8,
            knowledge_insulation=knowledge_insulation,
        )
        if trace:
            hidden.append((prefix, suffix))

    prefix = _norm_final(prefix, params["prefix_norm"])
    suffix, _ = ada_rms(suffix, params["expert_norm"]["dense_weight"],
                        params["expert_norm"]["dense_bias"], cond)

    velocity = _linear(suffix[:, -total:].astype(jnp.float32), params["action_out"])
    if draws > 1:
        velocity = velocity.reshape(batch, draws, chunk_size, pad_dim)
    return {
        "velocity": velocity,
        "x_t": x_t,
        "prefix": prefix,
        "suffix": suffix,
        "layer_hidden": hidden,
        "time_cond": cond,
        "mask": mask,
        "positions": positions,
    }


def sample_time(uniform, *, alpha: float = 1.5, beta: float = 1.0,
                scale: float = 0.999, offset: float = 0.001):
    """Reference time sampling: ``Beta(alpha, beta).icdf(u) * scale + offset``.

    The reference draws the Beta sample itself; the inverse-CDF form lets a fixed uniform
    drive both frameworks, which is the only way to compare a sampled quantity across
    different RNG implementations.

    jax 0.5 exposes the regularised incomplete beta but not its inverse, so the contract's
    ``beta == 1`` case uses its closed form: ``CDF(x) = x**alpha``, hence
    ``icdf(u) = u**(1/alpha)``. Any other ``beta`` fails loudly instead of silently
    falling back to a different distribution.
    """
    if beta != 1.0:
        raise ValueError(
            "closed-form inverse is implemented for beta == 1 (the locked contract); "
            f"got beta={beta}"
        )
    quantile = jnp.power(jnp.asarray(uniform, dtype=jnp.float32),
                         jnp.asarray(1.0 / alpha, dtype=jnp.float32))
    return quantile * jnp.asarray(scale, dtype=jnp.float32) + jnp.asarray(offset, dtype=jnp.float32)


def _norm_final(x, params: dict):
    variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    return x * (1.0 / jnp.sqrt(variance + 1e-6)) * (1.0 + params["weight"].astype(jnp.float32))


def loss_fn(result: dict, actions, noise, *, action_dim: int, predict_actions=None,
            reduction: str = "mean"):
    """MSE against ``noise - action``, with ``[B,T,D]`` inputs shared across draws."""
    velocity = result["velocity"]
    if velocity.ndim not in (3, 4):
        raise ValueError("velocity must be [B,T,D] or [B,K,T,D]")

    def align_input(name, value):
        value = jnp.asarray(value, dtype=jnp.float32)
        if value.ndim not in (3, velocity.ndim):
            raise ValueError(f"{name} must be [B,T,D] or match the velocity rank")
        if value.shape[0] != velocity.shape[0] or value.shape[-2] != velocity.shape[-2]:
            raise ValueError(f"{name} batch and chunk must match velocity")
        if value.shape[-1] < action_dim:
            raise ValueError(f"{name} must contain all {action_dim} valid action dimensions")
        if velocity.ndim == 4:
            if value.ndim == 3:
                value = value[:, None]
            elif value.shape[1] not in (1, velocity.shape[1]):
                raise ValueError(f"{name} draw count must be one or match velocity")
        return value[..., :action_dim]

    target = align_input("noise", noise) - align_input("actions", actions)
    return flow_mse_loss(velocity, target, action_dim,
                         predict_actions=predict_actions, reduction=reduction)
