"""Pure JAX implementation of one joint VLM/action-expert transformer layer.

The parameter layout mirrors one pair of reference Gemma layers. This module is
also usable for a compact operator fixture; its dimensions come from the
provided weight arrays, while ``num_heads`` and ``head_dim`` remain explicit.
"""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .attention import rope
from .norms import ada_rms, rms_norm


def _linear(x, weight, bias=None):
    value = jnp.einsum("...i,oi->...o", x, weight)
    return value if bias is None else value + bias


def _norm(x, params: dict, cond=None):
    if "dense_weight" in params:
        if cond is None:
            raise ValueError("AdaRMS parameters require a conditioning tensor")
        return ada_rms(x, params["dense_weight"], params["dense_bias"], cond)
    if cond is not None:
        raise ValueError("conditioning was supplied to a non-adaptive RMSNorm")
    return rms_norm(x, params.get("weight")), None


def _project_qkv(x, params: dict, positions, num_heads: int, num_kv_heads: int, head_dim: int):
    q = _linear(x, params["q_proj"])
    k = _linear(x, params["k_proj"])
    v = _linear(x, params["v_proj"])
    batch, length = x.shape[:2]
    q = q.reshape(batch, length, num_heads, head_dim)
    k = k.reshape(batch, length, num_kv_heads, head_dim)
    v = v.reshape(batch, length, num_kv_heads, head_dim)
    return rope(q, positions), rope(k, positions), v


def _attention(q, k, v, mask, num_heads: int, num_kv_heads: int):
    """Gemma GQA with an FP32 softmax and a boolean ``[B,Q,K]`` mask."""
    if num_heads % num_kv_heads:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))
    repeats = num_heads // num_kv_heads
    if repeats != 1:
        k = jnp.repeat(k, repeats, axis=1)
        v = jnp.repeat(v, repeats, axis=1)
    scale = jnp.asarray(q.shape[-1] ** -0.5, dtype=jnp.float32)
    logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
    minimum = jnp.asarray(jnp.finfo(jnp.float32).min, dtype=logits.dtype)
    logits = jnp.where(jnp.asarray(mask, dtype=bool)[:, None, :, :], logits, minimum)
    probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(q.dtype)
    value = jnp.einsum("bhqk,bhkd->bhqd", probs, v)
    return jnp.transpose(value, (0, 2, 1, 3))


def _feed_forward(x, params: dict):
    gate = _linear(x, params["gate_proj"])
    up = _linear(x, params["up_proj"])
    # Gemma uses gelu_pytorch_tanh for the VLM/action expert in this contract.
    coeff = jnp.asarray(0.7978845608028654, dtype=jnp.float32)
    activated = 0.5 * gate * (1.0 + jnp.tanh(coeff * (gate + 0.044715 * gate**3)))
    return _linear(activated * up, params["down_proj"])


def _residual(x, value, gate):
    return x + value if gate is None else x + value * gate


def _finish_stream(x, attn, params: dict, gate):
    value = _linear(
        attn.reshape(attn.shape[0], attn.shape[1], -1),
        params["attention"]["o_proj"],
    )
    after_attention = _residual(x, value, gate)
    normalized, mlp_gate = _norm(after_attention, params["post_norm"], params.get("condition"))
    output = _feed_forward(normalized, params["mlp"])
    return _residual(after_attention, output, mlp_gate)


def _stream_layer(x, params: dict, positions, mask, cond, num_heads, num_kv_heads, head_dim,
                  key_cache=None, value_cache=None, detach_prefix=False):
    normalized, gate = _norm(x, params["input_norm"], cond)
    q, k, v = _project_qkv(normalized, params["attention"], positions,
                           num_heads, num_kv_heads, head_dim)
    if key_cache is None:
        all_k, all_v = k, v
    else:
        prefix_k = jax.lax.stop_gradient(key_cache) if detach_prefix else key_cache
        prefix_v = jax.lax.stop_gradient(value_cache) if detach_prefix else value_cache
        all_k = jnp.concatenate([prefix_k, k], axis=1)
        all_v = jnp.concatenate([prefix_v, v], axis=1)
    attended = _attention(q, all_k, all_v, mask, num_heads, num_kv_heads)
    return _finish_stream(x, attended, params, gate), (k, v)


def joint_layer(
    params: dict,
    prefix,
    suffix,
    attention_mask,
    position_ids,
    suffix_condition,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    knowledge_insulation: bool,
):
    """Run one fused prefix/action layer and return the two hidden streams.

    Knowledge insulation only detaches the prefix K/V used by action queries.
    It does not alter masks, values, or the forward computation.
    """
    prefix_len = prefix.shape[1]
    total_mask = jnp.asarray(attention_mask, dtype=bool)
    if total_mask.shape != (prefix.shape[0], prefix_len + suffix.shape[1], prefix_len + suffix.shape[1]):
        raise ValueError("joint attention mask must cover the concatenated prefix and suffix")
    if position_ids.shape != total_mask.shape[:2]:
        raise ValueError("position_ids must match the concatenated sequence")

    pnorm, pgate = _norm(prefix, params["prefix"]["input_norm"], None)
    snorm, sgate = _norm(suffix, params["expert"]["input_norm"], suffix_condition)
    pq, pk, pv = _project_qkv(pnorm, params["prefix"]["attention"], position_ids[:, :prefix_len],
                             num_heads, num_kv_heads, head_dim)
    sq, sk, sv = _project_qkv(snorm, params["expert"]["attention"], position_ids[:, prefix_len:],
                             num_heads, num_kv_heads, head_dim)
    keys = jnp.concatenate([pk, sk], axis=1)
    values = jnp.concatenate([pv, sv], axis=1)

    prefix_attn = _attention(pq, keys, values, total_mask[:, :prefix_len, :],
                             num_heads, num_kv_heads)
    action_keys = jnp.concatenate([
        jax.lax.stop_gradient(pk) if knowledge_insulation else pk,
        sk,
    ], axis=1)
    action_values = jnp.concatenate([
        jax.lax.stop_gradient(pv) if knowledge_insulation else pv,
        sv,
    ], axis=1)
    suffix_attn = _attention(sq, action_keys, action_values, total_mask[:, prefix_len:, :],
                             num_heads, num_kv_heads)
    prefix_out = _finish_stream(prefix, prefix_attn, params["prefix"], pgate)
    # The suffix AdaRMS condition is needed for the post-attention norm as well.
    expert_params = dict(params["expert"])
    expert_params["post_norm"] = params["expert"]["post_norm"]
    expert_params["condition"] = suffix_condition
    suffix_out = _finish_stream(suffix, suffix_attn, expert_params, sgate)
    return prefix_out, suffix_out


def prefix_layer_with_cache(params: dict, prefix, position_ids, prefix_mask, *,
                            num_heads: int, num_kv_heads: int, head_dim: int):
    """Run the VLM layer once and return post-layer hidden plus RoPE-applied K/V."""
    p = params["prefix"]
    normalized, gate = _norm(prefix, p["input_norm"], None)
    q, k, v = _project_qkv(normalized, p["attention"], position_ids,
                           num_heads, num_kv_heads, head_dim)
    attended = _attention(q, k, v, prefix_mask, num_heads, num_kv_heads)
    return _finish_stream(prefix, attended, p, gate), k, v


def expert_layer_with_cache(params: dict, suffix, position_ids, suffix_mask, suffix_condition,
                             prefix_key, prefix_value, *, num_heads: int, num_kv_heads: int,
                             head_dim: int, knowledge_insulation: bool):
    """Run the action layer against a prefix cache from this same layer."""
    p = params["expert"]
    normalized, gate = _norm(suffix, p["input_norm"], suffix_condition)
    q, k, v = _project_qkv(normalized, p["attention"], position_ids,
                           num_heads, num_kv_heads, head_dim)
    if knowledge_insulation:
        prefix_key = jax.lax.stop_gradient(prefix_key)
        prefix_value = jax.lax.stop_gradient(prefix_value)
    all_key = jnp.concatenate([prefix_key, k], axis=1)
    all_value = jnp.concatenate([prefix_value, v], axis=1)
    attended = _attention(q, all_key, all_value, suffix_mask, num_heads, num_kv_heads)
    expert_params = dict(p)
    expert_params["condition"] = suffix_condition
    return _finish_stream(suffix, attended, expert_params, gate)


def flow_matching_inputs(actions, noise, time):
    """Return contract x_t and velocity target using the noise-at-one convention."""
    time = jnp.asarray(time, dtype=jnp.float32)
    while time.ndim < actions.ndim:
        time = time[..., None]
    return time * noise + (1.0 - time) * actions, noise - actions


def flow_mse_loss(velocity, target, actual_action_dim: int, predict_actions=None,
                  reduction: str = "mean"):
    """MSE over real action dimensions, then reference per-sample routing/reduction."""
    if reduction not in {"mean", "none"}:
        raise ValueError(f"unsupported action loss reduction: {reduction!r}")
    if actual_action_dim < 1 or actual_action_dim > velocity.shape[-1]:
        raise ValueError("actual_action_dim must select a non-empty prefix of velocity dimensions")
    per_dim = jnp.square(velocity[..., :actual_action_dim] - target[..., :actual_action_dim])
    axes = tuple(range(1, per_dim.ndim))
    per_sample = jnp.mean(per_dim, axis=axes)
    if predict_actions is None:
        return per_sample if reduction == "none" else jnp.mean(per_sample)
    mask = jnp.asarray(predict_actions, dtype=per_sample.dtype)
    if mask.shape != per_sample.shape:
        raise ValueError("predict_actions must have one value per sample")
    if reduction == "none":
        return per_sample * mask
    return jnp.sum(per_sample * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def prefix_cache_matches(cache: dict[str, Any], prefix, position_ids, prefix_mask) -> bool:
    """Host-side exact identity check; do not reuse KV after an observation changes."""
    return bool(
        np.array_equal(np.asarray(cache["prefix"]), np.asarray(prefix))
        and np.array_equal(np.asarray(cache["position_ids"]), np.asarray(position_ids))
        and np.array_equal(np.asarray(cache["prefix_mask"]), np.asarray(prefix_mask))
    )


def make_prefix_cache(prefix, position_ids, prefix_mask, key, value) -> dict[str, Any]:
    """Package layer KV with the exact input identity that produced it."""
    return {
        "prefix": jnp.asarray(prefix),
        "position_ids": jnp.asarray(position_ids),
        "prefix_mask": jnp.asarray(prefix_mask, dtype=bool),
        "key": key,
        "value": value,
    }


def sinusoidal_time_embedding(time, dimension: int, min_period: float = 0.004,
                              max_period: float = 4.0):
    """PI0.5 sine/cosine timestep features, with periods spaced geometrically."""
    if dimension % 2:
        raise ValueError("time embedding dimension must be even")
    time = jnp.asarray(time, dtype=jnp.float32)
    fraction = jnp.linspace(0.0, 1.0, dimension // 2, dtype=jnp.float32)
    period = min_period * (max_period / min_period) ** fraction
    phase = time[..., None] * (2.0 * jnp.pi / period)
    return jnp.concatenate([jnp.sin(phase), jnp.cos(phase)], axis=-1)


def embed_suffix(x_t, timestep, params: dict, *, chunk_size: int,
                 min_period: float = 0.004, max_period: float = 4.0):
    """Embed noisy actions and timestep as the PI0.5 action-expert suffix."""
    if x_t.ndim != 3 or x_t.shape[1] != chunk_size:
        raise ValueError("noisy actions must be [batch, configured chunk, action_dim]")
    if timestep.ndim not in (1, 2):
        raise ValueError("timestep must be [batch] or [batch, token]")
    if timestep.shape[0] != x_t.shape[0]:
        raise ValueError("timestep batch must match noisy actions")
    if timestep.ndim == 2 and timestep.shape[1] not in (1, chunk_size):
        raise ValueError("per-token timestep must have one or chunk_size entries")
    width = params["action_in_proj"]["weight"].shape[0]
    time_emb = sinusoidal_time_embedding(timestep, width, min_period, max_period)
    action_emb = _linear(
        x_t,
        params["action_in_proj"]["weight"],
        params["action_in_proj"].get("bias"),
    )
    time_hidden = jax.nn.silu(_linear(
        time_emb, params["time_mlp_in"]["weight"], params["time_mlp_in"].get("bias")
    ))
    condition = jax.nn.silu(_linear(
        time_hidden, params["time_mlp_out"]["weight"], params["time_mlp_out"].get("bias")
    ))
    pad = jnp.ones(x_t.shape[:2], dtype=bool)
    att = jnp.zeros(x_t.shape[:2], dtype=action_emb.dtype).at[:, 0].set(1.0)
    return action_emb, pad, att, condition


def _apply_final_norms(prefix, suffix, condition, final_norms: dict | None):
    if final_norms is None:
        return prefix, suffix
    prefix, _ = _norm(prefix, final_norms["prefix"], None)
    suffix, _ = _norm(suffix, final_norms["expert"], condition)
    return prefix, suffix


def joint_stack(layers: list[dict], prefix, suffix, attention_mask, position_ids,
                suffix_condition, *, num_heads: int, num_kv_heads: int, head_dim: int,
                knowledge_insulation: bool, final_norms: dict | None = None):
    """Run paired VLM/expert layers and return every layer output plus final norms."""
    prefix_rows, suffix_rows = [], []
    for layer in layers:
        prefix, suffix = joint_layer(
            layer, prefix, suffix, attention_mask, position_ids, suffix_condition,
            num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
            knowledge_insulation=knowledge_insulation,
        )
        prefix_rows.append(prefix)
        suffix_rows.append(suffix)
    prefix, suffix = _apply_final_norms(prefix, suffix, suffix_condition, final_norms)
    return prefix, suffix, tuple(prefix_rows), tuple(suffix_rows)


def joint_stack_with_prefix_cache(layers: list[dict], prefix, suffix, attention_mask,
                                  position_ids, suffix_condition, *, num_heads: int,
                                  num_kv_heads: int, head_dim: int,
                                  knowledge_insulation: bool,
                                  final_norms: dict | None = None):
    """Run the same paired stack with layerwise prefix-only KV construction."""
    prefix_len = prefix.shape[1]
    prefix_rows, suffix_rows, layer_cache = [], [], []
    prefix_mask = jnp.asarray(attention_mask, dtype=bool)[:, :prefix_len, :prefix_len]
    suffix_mask = jnp.asarray(attention_mask, dtype=bool)[:, prefix_len:, :]
    for layer in layers:
        prefix, key, value = prefix_layer_with_cache(
            layer, prefix, position_ids[:, :prefix_len], prefix_mask,
            num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
        )
        suffix = expert_layer_with_cache(
            layer, suffix, position_ids[:, prefix_len:], suffix_mask, suffix_condition,
            key, value, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
            knowledge_insulation=knowledge_insulation,
        )
        prefix_rows.append(prefix)
        suffix_rows.append(suffix)
        layer_cache.append((key, value))
    prefix, suffix = _apply_final_norms(prefix, suffix, suffix_condition, final_norms)
    return prefix, suffix, tuple(prefix_rows), tuple(suffix_rows), tuple(layer_cache)


def joint_flow_forward(params: dict, prefix, actions, noise, time, attention_mask, position_ids,
                       *, actual_action_dim: int, draws: int, num_heads: int,
                       num_kv_heads: int, head_dim: int, chunk_size: int,
                       knowledge_insulation: bool, predict_actions=None,
                       reduction: str = "mean", min_period: float = 0.004,
                       max_period: float = 4.0, use_prefix_cache: bool = False):
    """Run PI0.5 action/time embedding, joint stack, velocity head and routed Flow MSE.

    ``noise`` is ``[B,T,D]`` for K=1 and ``[K,B,T,D]`` for K>1; ``time`` uses
    matching ``[B]`` or ``[K,B]`` layout. The K dimension is folded into one
    suffix sequence after each draw receives its own timestep embedding.
    """
    if actions.ndim != 3 or actions.shape[1] != chunk_size:
        raise ValueError("actions must be [batch, configured chunk, action_dim]")
    batch = actions.shape[0]
    if draws == 1:
        if noise.shape != actions.shape or time.shape != (batch,):
            raise ValueError("K=1 noise/time must match [B,T,D] actions and [B] time")
        x_t, target = flow_matching_inputs(actions, noise, time)
        suffix, suffix_pad, suffix_att, condition = embed_suffix(
            x_t, time, params["suffix_embedding"], chunk_size=chunk_size,
            min_period=min_period, max_period=max_period,
        )
    else:
        if noise.shape != (draws, batch, *actions.shape[1:]) or time.shape != (draws, batch):
            raise ValueError("K>1 noise/time must be [K,B,T,D] and [K,B]")
        action_draws = jnp.broadcast_to(actions[None, ...], noise.shape)
        times = jnp.asarray(time, dtype=jnp.float32)[..., None, None]
        x_draws = times * noise + (1.0 - times) * action_draws
        targets = noise - action_draws
        x_flat = x_draws.reshape(draws * batch, chunk_size, actions.shape[-1])
        time_flat = jnp.asarray(time, dtype=jnp.float32).reshape(draws * batch)
        suffix_flat, suffix_pad_flat, suffix_att_flat, condition_flat = embed_suffix(
            x_flat, time_flat, params["suffix_embedding"], chunk_size=chunk_size,
            min_period=min_period, max_period=max_period,
        )
        width = suffix_flat.shape[-1]
        suffix = suffix_flat.reshape(draws, batch, chunk_size, width).transpose(1, 0, 2, 3)
        suffix = suffix.reshape(batch, draws * chunk_size, width)
        suffix_pad = suffix_pad_flat.reshape(draws, batch, chunk_size).transpose(1, 0, 2)
        suffix_pad = suffix_pad.reshape(batch, draws * chunk_size)
        suffix_att = suffix_att_flat.reshape(draws, batch, chunk_size).transpose(1, 0, 2)
        suffix_att = suffix_att.reshape(batch, draws * chunk_size)
        cond_width = condition_flat.shape[-1]
        condition = condition_flat.reshape(draws, batch, cond_width).transpose(1, 0, 2)
        condition = jnp.broadcast_to(
            condition[:, :, None, :], (batch, draws, chunk_size, cond_width)
        ).reshape(batch, draws * chunk_size, cond_width)
        target = targets.transpose(1, 0, 2, 3)

    if prefix.shape[0] != batch or attention_mask.shape[0] != batch:
        raise ValueError("prefix and joint masks must match the action batch")
    if suffix.shape[1] != draws * chunk_size:
        raise AssertionError("suffix sequence must include every draw exactly once")
    final_norms = params.get("final_norms")
    if use_prefix_cache:
        prefix_out, suffix_out, prefix_layers, suffix_layers, cache = joint_stack_with_prefix_cache(
            params["layers"], prefix, suffix, attention_mask, position_ids, condition,
            num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
            knowledge_insulation=knowledge_insulation, final_norms=final_norms,
        )
    else:
        prefix_out, suffix_out, prefix_layers, suffix_layers = joint_stack(
            params["layers"], prefix, suffix, attention_mask, position_ids, condition,
            num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
            knowledge_insulation=knowledge_insulation, final_norms=final_norms,
        )
        cache = None
    velocity_flat = _linear(
        suffix_out, params["action_out_proj"]["weight"], params["action_out_proj"].get("bias")
    )
    if draws > 1:
        velocity = velocity_flat.reshape(batch, draws, chunk_size, -1)
    else:
        velocity = velocity_flat.reshape(batch, chunk_size, -1)
    flow_loss = flow_mse_loss(velocity, target, actual_action_dim, predict_actions, reduction)
    return {
        "x_t": x_t if draws == 1 else x_draws.transpose(1, 0, 2, 3),
        "target": target,
        "prefix": prefix_out,
        "suffix": suffix_out,
        "suffix_embeddings": suffix,
        "condition": condition,
        "velocity": velocity,
        "flow_loss": flow_loss,
        "prefix_layer_outputs": prefix_layers,
        "suffix_layer_outputs": suffix_layers,
        "prefix_cache": cache,
        "suffix_pad": suffix_pad,
        "suffix_att": suffix_att,
    }
