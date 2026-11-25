"""Base operators certified at G2."""
from __future__ import annotations

import jax.numpy as jnp


def embed_tokens(ids, table):
    width = table.shape[-1]
    return table[ids] * jnp.asarray(width ** 0.5, dtype=jnp.float32)


def head_logits(hidden, head_weight):
    return hidden @ head_weight.T
