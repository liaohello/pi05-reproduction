"""Full FineART-VLA prefix and joint-Flow assembly helpers.

The functions in this module adapt the strictly registered G2 parameter tree
to the G3 operators. They do not load or initialize weights; callers must pass
the complete, independently validated registered mapping.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax.numpy as jnp

from .embedding import embed_tokens
from .vision import assemble_prefix, project, vision_tower


_VLM = "model.paligemma_with_expert.paligemma.model.language_model."
_EXPERT = "model.paligemma_with_expert.gemma_expert.model."
_VISION = "model.paligemma_with_expert.paligemma.model.vision_tower."
_PROJECTOR = "model.paligemma_with_expert.paligemma.model.multi_modal_projector.linear."


def _array(parameters: Mapping[str, object], name: str):
    if name not in parameters:
        raise ValueError("missing registered parameter: " + name)
    value = jnp.asarray(parameters[name])
    if value.dtype != jnp.float32:
        raise ValueError(f"registered parameter is not float32: {name} ({value.dtype})")
    return value


def _attention(parameters: Mapping[str, object], base: str) -> dict:
    return {
        f"{name}_proj": _array(parameters, base + f"self_attn.{name}_proj.weight")
        for name in ("q", "k", "v", "o")
    }


def _mlp(parameters: Mapping[str, object], base: str) -> dict:
    return {
        f"{name}_proj": _array(parameters, base + f"mlp.{name}_proj.weight")
        for name in ("gate", "up", "down")
    }


def joint_parameters_from_registered(parameters: Mapping[str, object]) -> dict:
    """Build full 18-pair joint-stack and Flow projection parameters.

    The returned layout is consumed by :func:`joint_flow_forward`. The expert
    LM head and VLM text LM head are intentionally not copied: this Flow path
    ends at the action velocity projection. The VLM token embedding is read
    separately by :func:`prefix_embeddings_from_registered`.
    """
    layers = []
    for index in range(18):
        vlm = _VLM + f"layers.{index}."
        expert = _EXPERT + f"layers.{index}."
        layers.append({
            "prefix": {
                "input_norm": {"weight": _array(parameters, vlm + "input_layernorm.weight")},
                "attention": _attention(parameters, vlm),
                "post_norm": {"weight": _array(parameters, vlm + "post_attention_layernorm.weight")},
                "mlp": _mlp(parameters, vlm),
            },
            "expert": {
                "input_norm": {
                    "dense_weight": _array(parameters, expert + "input_layernorm.dense.weight"),
                    "dense_bias": _array(parameters, expert + "input_layernorm.dense.bias"),
                },
                "attention": _attention(parameters, expert),
                "post_norm": {
                    "dense_weight": _array(parameters, expert + "post_attention_layernorm.dense.weight"),
                    "dense_bias": _array(parameters, expert + "post_attention_layernorm.dense.bias"),
                },
                "mlp": _mlp(parameters, expert),
            },
        })

    return {
        "layers": layers,
        "final_norms": {
            "prefix": {"weight": _array(parameters, _VLM + "norm.weight")},
            "expert": {
                "dense_weight": _array(parameters, _EXPERT + "norm.dense.weight"),
                "dense_bias": _array(parameters, _EXPERT + "norm.dense.bias"),
            },
        },
        "suffix_embedding": {
            name: {
                "weight": _array(parameters, f"model.{name}.weight"),
                "bias": _array(parameters, f"model.{name}.bias"),
            }
            for name in ("action_in_proj", "time_mlp_in", "time_mlp_out")
        },
        "action_out_proj": {
            "weight": _array(parameters, "model.action_out_proj.weight"),
            "bias": _array(parameters, "model.action_out_proj.bias"),
        },
    }


def _vision_parameters(parameters: Mapping[str, object]) -> dict:
    prefix = _VISION
    result = {
        name[len(prefix):]: _array(parameters, name)
        for name in parameters
        if name.startswith(prefix)
    }
    expected_layers = {
        f"encoder.layers.{index}.self_attn.q_proj.weight" for index in range(27)
    }
    if not expected_layers.issubset(result):
        raise ValueError("registered mapping does not contain all 27 SigLIP layers")
    return result


def prefix_embeddings_from_registered(
    parameters: Mapping[str, object],
    images: Sequence[object],
    image_masks: Sequence[object],
    text_tokens,
    text_mask,
    *,
    fast_tokens=None,
    fast_mask=None,
) -> tuple:
    """Replay the reference prefix builder from images and token IDs.

    Images are already preprocessed as FP32 NCHW tensors in ``[-1, 1]``. The
    optional FAST block is appended after ordinary prompt tokens and remains
    separately identifiable by the returned ``non_fast_len`` and ``fast_len``.
    """
    if len(images) != len(image_masks) or not images:
        raise ValueError("at least one image and its matching mask are required")
    vision = _vision_parameters(parameters)
    projector_weight = _array(parameters, _PROJECTOR + "weight")
    projector_bias = _array(parameters, _PROJECTOR + "bias")
    image_embeddings, image_padding = [], []
    batch = int(jnp.asarray(text_tokens).shape[0])
    for image, mask in zip(images, image_masks, strict=True):
        image = jnp.asarray(image, dtype=jnp.float32)
        mask = jnp.asarray(mask, dtype=jnp.bool_).reshape(batch)
        patches = vision_tower(image, vision)
        image_embeddings.append(project(patches, projector_weight, projector_bias))
        image_padding.append(jnp.broadcast_to(mask[:, None], patches.shape[:2]))

    text_table = _array(parameters, _VLM + "embed_tokens.weight")
    text = embed_tokens(jnp.asarray(text_tokens, dtype=jnp.int32), text_table)
    text_mask = jnp.asarray(text_mask, dtype=jnp.bool_)
    non_fast, non_fast_mask = assemble_prefix(
        image_embeddings, image_padding, text, text_mask, image_token_id=257152
    )
    non_fast_len = int(non_fast.shape[1])

    if fast_tokens is None:
        if fast_mask is not None:
            raise ValueError("fast_mask cannot be supplied without fast_tokens")
        fast = jnp.zeros((batch, 0, text.shape[-1]), dtype=jnp.float32)
        fast_padding = jnp.zeros((batch, 0), dtype=jnp.bool_)
        fast_len = 0
    else:
        fast = embed_tokens(jnp.asarray(fast_tokens, dtype=jnp.int32), text_table)
        if fast_mask is None:
            fast_padding = jnp.ones(fast.shape[:2], dtype=jnp.bool_)
        else:
            fast_padding = jnp.asarray(fast_mask, dtype=jnp.bool_)
            if fast_padding.shape != fast.shape[:2]:
                raise ValueError("fast_mask must match fast_tokens")
        fast_len = int(fast.shape[1])

    prefix = jnp.concatenate([non_fast, fast], axis=1)
    prefix_padding = jnp.concatenate([non_fast_mask, fast_padding], axis=1)
    prefix_attention = jnp.concatenate([
        jnp.zeros(non_fast_mask.shape, dtype=jnp.int32),
        jnp.ones(fast_padding.shape, dtype=jnp.int32),
    ], axis=1)
    if prefix.shape[:2] != prefix_padding.shape or prefix_padding.shape != prefix_attention.shape:
        raise AssertionError("assembled prefix values and masks do not align")
    return prefix, prefix_padding, prefix_attention, non_fast_len, fast_len
