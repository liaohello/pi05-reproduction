"""Base operators certified at G2."""
from __future__ import annotations

import jax.numpy as jnp


def split_qkv(x, w_q, w_kv, heads: int, kv_heads: int, head_dim: int):
    """Reference einsums: BTD,NDH->BTNH and BSD,KDH->BSKH."""
    q = jnp.einsum("BTD,NDH->BTNH", x, w_q)
    k = jnp.einsum("BSD,KDH->BSKH", x, w_kv[0])
    v = jnp.einsum("BSD,KDH->BSKH", x, w_kv[1])
    return q, k, v


def rope(values, positions, max_wavelength: float = 10000.0):
    dim = values.shape[-1]
    exponent = (2.0 / dim) * jnp.arange(dim // 2, dtype=jnp.float32)
    timescale = jnp.power(jnp.asarray(max_wavelength, dtype=jnp.float32), exponent)
    radians = positions[..., None] / timescale[None, None, :]
    radians = radians[..., None, :]
    sin, cos = jnp.sin(radians), jnp.cos(radians)
    x1, x2 = jnp.split(values, 2, axis=-1)
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def prepare_qkv(q, k, v, positions, head_dim: int):
    q = rope(q, positions) * jnp.asarray(head_dim ** -0.5, dtype=jnp.float32)
    k = rope(k, positions)
    return q, k, v


def output_projection(encoded, w_o):
    """Reference o_proj: keep the token axis, BTD,OD->BTO."""
    return jnp.einsum("BTD,OD->BTO", encoded, w_o)


# --------------------------------------------------------------------------- #
# RMSNorm / AdaRMS
# --------------------------------------------------------------------------- #
