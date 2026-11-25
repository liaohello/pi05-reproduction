"""Base operators certified at G2."""
from __future__ import annotations

import jax.numpy as jnp


def rms_norm(x, weight=None, eps: float = 1e-6):
    variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    normed = x * (1.0 / jnp.sqrt(variance + eps))
    if weight is not None:
        normed = normed * (1.0 + weight.astype(jnp.float32))
    return normed


def ada_rms(x, dense_weight, dense_bias, cond, eps: float = 1e-6):
    variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    normed = x * (1.0 / jnp.sqrt(variance + eps))
    modulation = jnp.einsum("...i,oi->...o", cond, dense_weight) + dense_bias
    if x.ndim == 3 and modulation.ndim == 2:
        modulation = modulation[:, None, :]
    scale, shift, gate = jnp.split(modulation, 3, axis=-1)
    value = normed * (1.0 + scale.astype(jnp.float32)) + shift.astype(jnp.float32)
    return value, gate


# --------------------------------------------------------------------------- #
# Vision patch/position/projector and prefix assembly
# --------------------------------------------------------------------------- #
