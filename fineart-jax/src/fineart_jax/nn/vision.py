"""Base operators certified at G2."""
from __future__ import annotations

import jax
import jax.numpy as jnp


def patch_embed(images, weight, bias, patch: int = 14):
    """images [B,C,H,W] (FP32) -> [B,num_patches,width] (reference transpose)."""
    batch, channels, height, width = images.shape
    if height % patch or width % patch:
        raise ValueError("image size must be divisible by the patch size")
    cells = height // patch
    tiles = images.reshape(batch, channels, cells, patch, cells, patch)
    tiles = tiles.transpose(0, 2, 4, 1, 3, 5).reshape(batch, cells * cells, channels * patch * patch)
    kernel = weight.reshape(weight.shape[0], channels * patch * patch)
    result = tiles @ kernel.T + bias[None, :]
    return result


def add_position(tokens, position):
    if tokens.shape[1] != position.shape[0] or tokens.shape[2] != position.shape[1]:
        raise ValueError("position embedding shape does not match the patch tokens")
    return tokens + position[None, ...]


def project(tokens, weight, bias):
    return tokens @ weight.T + bias[None, None, :]


def layer_norm(x, weight, bias, eps: float = 1e-6):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(variance + eps) * weight + bias


def gelu_tanh(x):
    return jax.nn.gelu(x, approximate=True)


def attention(x, params, heads: int = 16):
    """SigLIP self-attention: q/k/v/out projections with bias, pre-scaled q."""
    batch, seq, width = x.shape
    head_dim = width // heads
    q = (x @ params["q_proj.weight"].T + params["q_proj.bias"]).reshape(batch, seq, heads, head_dim)
    k = (x @ params["k_proj.weight"].T + params["k_proj.bias"]).reshape(batch, seq, heads, head_dim)
    v = (x @ params["v_proj.weight"].T + params["v_proj.bias"]).reshape(batch, seq, heads, head_dim)
    q = q * jnp.asarray(head_dim ** -0.5, dtype=q.dtype)
    scores = jnp.einsum("bthd,bshd->bhts", q, k)
    probs = jax.nn.softmax(scores, axis=-1)
    context = jnp.einsum("bhts,bshd->bthd", probs, v).reshape(batch, seq, width)
    return context @ params["out_proj.weight"].T + params["out_proj.bias"]


def encoder_layer(x, params):
    h = layer_norm(x, params["layer_norm1.weight"], params["layer_norm1.bias"])
    x = x + attention(h, params["self_attn"])
    h = layer_norm(x, params["layer_norm2.weight"], params["layer_norm2.bias"])
    h = gelu_tanh(h @ params["mlp.fc1.weight"].T + params["mlp.fc1.bias"])
    h = h @ params["mlp.fc2.weight"].T + params["mlp.fc2.bias"]
    return x + h


def vision_tower(images, params, layers: int = 27, return_hidden_states: bool = False):
    """Full SigLIP tower from pinned checkpoint tensors: [B,3,224,224] -> [B,256,1152]."""
    x = patch_embed(images, params["embeddings.patch_embedding.weight"],
                    params["embeddings.patch_embedding.bias"])
    x = add_position(x, params["embeddings.position_embedding.weight"])
    hidden = []
    for index in range(layers):
        prefix = f"encoder.layers.{index}."
        layer_params = {
            "layer_norm1.weight": params[prefix + "layer_norm1.weight"],
            "layer_norm1.bias": params[prefix + "layer_norm1.bias"],
            "layer_norm2.weight": params[prefix + "layer_norm2.weight"],
            "layer_norm2.bias": params[prefix + "layer_norm2.bias"],
            "self_attn": {
                f"{name}_proj.weight": params[prefix + f"self_attn.{name}_proj.weight"]
                for name in ("q", "k", "v", "out")
            } | {
                f"{name}_proj.bias": params[prefix + f"self_attn.{name}_proj.bias"]
                for name in ("q", "k", "v", "out")
            },
            "mlp.fc1.weight": params[prefix + "mlp.fc1.weight"],
            "mlp.fc1.bias": params[prefix + "mlp.fc1.bias"],
            "mlp.fc2.weight": params[prefix + "mlp.fc2.weight"],
            "mlp.fc2.bias": params[prefix + "mlp.fc2.bias"],
        }
        x = encoder_layer(x, layer_params)
        if return_hidden_states:
            hidden.append(x)
    x = layer_norm(x, params["post_layernorm.weight"], params["post_layernorm.bias"])
    return (x, hidden) if return_hidden_states else x


def assemble_prefix(image_tokens, image_masks, text_tokens, text_masks, image_token_id: int):
    """The reference prefix is image tokens followed by language tokens.

    Both streams keep their own padding mask; text right-padding must not be
    dropped just because the image stream has a mask.
    """
    del image_token_id  # kept for interface compatibility; no id lookup at this layer
    images = jnp.concatenate(image_tokens, axis=1)
    masks = jnp.concatenate(image_masks, axis=1).astype(jnp.bool_)
    both = jnp.concatenate([images, text_tokens], axis=1)
    masks = jnp.concatenate([masks, text_masks.astype(jnp.bool_)], axis=1)
    if both.shape[:2] != masks.shape:
        raise ValueError(f"prefix/mask shape mismatch: {both.shape[:2]} vs {masks.shape}")
    return both, masks
