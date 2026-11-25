"""Attention masks and RoPE positions for the joint prefix/action stack.

Every rule here is transcribed from the read-only reference, not invented:

* ``make_att_2d_masks`` — ``policies/common/vla_utils.py`` (cumulative block mask).
* suffix-to-FAST blocking and the FAST-parallel suffix position flow —
  ``fineart_vla/modeling_fineart_vla.py::_combined_prefix_and_flow``.
* per-draw block diagonal plus per-draw position restart —
  ``fineart_vla/modeling_fineart_vla.py::_amortized_prefix_and_flow``.

The two reference paths are **not** identical: the single-draw path makes the
action chunk causal (``att_masks=[1, 0, ...]``), while the multi-draw path gives
each draw full visibility inside its own block. Both are reproduced here as-is.
"""

from __future__ import annotations

import jax.numpy as jnp


def make_att_2d_masks(pad_masks, att_masks):
    """bool[B, N, N]; query i sees key j iff cumsum_j <= cumsum_i and both are valid."""
    cumsum = jnp.cumsum(jnp.asarray(att_masks, dtype=jnp.int32), axis=1)
    att = cumsum[:, None, :] <= cumsum[:, :, None]
    pad = jnp.asarray(pad_masks, dtype=bool)[:, None, :] & jnp.asarray(pad_masks, dtype=bool)[:, :, None]
    return att & pad


def position_ids(pad_masks):
    """Reference position ids: cumulative count of valid tokens minus one."""
    return jnp.cumsum(jnp.asarray(pad_masks, dtype=jnp.int32), axis=1) - 1


def flow_block_diag(draws: int, chunk: int):
    """``torch.block_diag(ones(chunk, chunk) x draws)`` as a boolean mask."""
    index = jnp.arange(draws * chunk) // chunk
    return index[:, None] == index[None, :]


def combined_masks(prefix_pad, prefix_att, suffix_pad, suffix_att, non_fast_len: int,
                   fast_len: int, draws: int, chunk: int):
    """Masks and positions for one prefix plus ``draws`` action chunks.

    ``draws == 1`` follows the fused single-draw path; ``draws > 1`` follows the
    amortised path (prefix computed once, draws mutually invisible).
    """
    prefix_pad = jnp.asarray(prefix_pad, dtype=bool)
    prefix_att = jnp.asarray(prefix_att, dtype=jnp.int32)
    suffix_pad = jnp.asarray(suffix_pad, dtype=bool)
    batch, prefix_len = prefix_pad.shape
    fast_end = non_fast_len + fast_len

    if draws == 1:
        pad = jnp.concatenate([prefix_pad, suffix_pad], axis=1)
        att = jnp.concatenate([prefix_att, jnp.asarray(suffix_att, dtype=jnp.int32)], axis=1)
        mask = make_att_2d_masks(pad, att)
        if fast_len > 0:
            mask = mask.at[:, fast_end:, non_fast_len:fast_end].set(False)
        positions = position_ids(pad)
        if fast_len > 0:
            non_fast_valid = jnp.sum(prefix_pad[:, :non_fast_len], axis=1, keepdims=True)
            suffix_positions = non_fast_valid + jnp.cumsum(
                jnp.asarray(suffix_pad, dtype=jnp.int32), axis=1) - 1
            positions = jnp.concatenate([positions[:, :prefix_len], suffix_positions], axis=1)
        return mask, positions

    total = draws * chunk
    prefix_2d = make_att_2d_masks(prefix_pad, prefix_att)
    prefix_rows = jnp.concatenate(
        [prefix_2d, jnp.zeros((batch, prefix_len, total), dtype=bool)], axis=2)
    action_to_prefix = jnp.broadcast_to(
        prefix_pad[:, None, :], (batch, total, prefix_len))
    if fast_len > 0:
        action_to_prefix = action_to_prefix.at[:, :, non_fast_len:fast_end].set(False)
    action_rows = jnp.concatenate(
        [action_to_prefix, jnp.broadcast_to(flow_block_diag(draws, chunk), (batch, total, total))],
        axis=2)
    mask = jnp.concatenate([prefix_rows, action_rows], axis=1)

    if fast_len > 0:
        offsets = jnp.sum(prefix_pad[:, :non_fast_len], axis=1, keepdims=True)
    else:
        offsets = jnp.sum(prefix_pad, axis=1, keepdims=True)
    block_positions = offsets + jnp.cumsum(jnp.asarray(suffix_pad, dtype=jnp.int32), axis=1) - 1
    repeated = jnp.tile(block_positions, (1, draws))
    positions = jnp.concatenate([position_ids(prefix_pad), repeated], axis=1)
    return mask, positions
