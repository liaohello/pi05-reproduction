"""Numerical data layer for the FineART-VLA JAX migration.

Pure NumPy: this is CPU data processing, which is the declared boundary of the
contract. It is deliberately independent of the reference implementation: every
formula is re-expressed here and then checked against reference arrays exported
by a separate read-only PyTorch process.

Reference semantics reproduced:

* normalisation follows the reference's four supported modes with ``eps=1e-8``
  substituting a zero denominator;
* relative actions subtract the *current* state from every step of the chunk,
  for the masked dimensions only, and never touch excluded joints;
* images accept channels-first or channels-last layouts, are resized without
  distortion (bilinear, ``align_corners=False``), centre-padded with zeros,
  clamped to ``[0, 1]`` and finally mapped to ``[-1, 1]``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-8
NORMALIZATION_MODES = ("IDENTITY", "MEAN_STD", "MIN_MAX", "QUANTILES", "QUANTILE10")


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
def _stat(stats: dict, name: str, fallback) -> np.ndarray:
    value = stats.get(name) if stats else None
    if value is None:
        value = fallback()
    return np.asarray(value, dtype=np.float32)


def normalize(
    values: np.ndarray,
    mode: str,
    stats: dict | None = None,
    *,
    inverse: bool = False,
    eps: float = EPS,
) -> np.ndarray:
    """Normalise (or denormalise) an array with the reference's exact arithmetic.

    ``QUANTILES`` uses the 1st/99th percentile pair, ``QUANTILE10`` the 10th/90th,
    ``MIN_MAX`` the observed extremes and ``MEAN_STD`` mean/std. A zero spread is
    replaced by ``eps`` so a constant dimension maps to a stable value instead of
    dividing by zero.
    """
    mode = str(mode).upper()
    if mode not in NORMALIZATION_MODES:
        raise ValueError(f"Unsupported normalization mode: {mode}")
    array = np.asarray(values, dtype=np.float32)
    if mode == "IDENTITY":
        return array.copy()

    flat = array.reshape(-1, array.shape[-1])
    stats = stats or {}

    if mode == "MEAN_STD":
        mean = _stat(stats, "mean", lambda: flat.mean(axis=0))
        std = _stat(stats, "std", lambda: flat.std(axis=0))
        if inverse:
            return (array * std + mean).astype(np.float32)
        return ((array - mean) / (std + np.float32(eps))).astype(np.float32)

    if mode == "MIN_MAX":
        low = _stat(stats, "min", lambda: flat.min(axis=0))
        high = _stat(stats, "max", lambda: flat.max(axis=0))
    elif mode == "QUANTILES":
        low = _stat(stats, "q01", lambda: np.quantile(flat, 0.01, axis=0))
        high = _stat(stats, "q99", lambda: np.quantile(flat, 0.99, axis=0))
    else:  # QUANTILE10
        low = _stat(stats, "q10", lambda: np.quantile(flat, 0.10, axis=0))
        high = _stat(stats, "q90", lambda: np.quantile(flat, 0.90, axis=0))

    spread = high - low
    spread = np.where(spread == 0, np.float32(eps), spread).astype(np.float32)
    if inverse:
        return ((array + np.float32(1.0)) * spread / np.float32(2.0) + low).astype(np.float32)
    return (np.float32(2.0) * (array - low) / spread - np.float32(1.0)).astype(np.float32)


def unnormalize(values: np.ndarray, mode: str, stats: dict | None = None, *, eps: float = EPS) -> np.ndarray:
    return normalize(values, mode, stats, inverse=True, eps=eps)


# --------------------------------------------------------------------------- #
# Relative actions
# --------------------------------------------------------------------------- #
def relative_mask(
    action_dim: int,
    action_names: list[str] | None = None,
    exclude_joints: list[str] | None = None,
    explicit: list[bool] | None = None,
) -> list[bool]:
    """Per-dimension mask of dimensions converted to relative actions."""
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


def _state_offset(actions: np.ndarray, state: np.ndarray, mask: list[bool]) -> np.ndarray:
    mask_array = np.asarray(mask, dtype=actions.dtype)
    dims = mask_array.shape[0]
    state = np.asarray(state, dtype=actions.dtype)
    # A temporally stacked observation collapses to the current frame.
    if state.ndim == 3:
        state = state[:, 0]
    offset = state[..., :dims] * mask_array
    if actions.ndim == 3:
        offset = np.expand_dims(offset, -2)
    return offset


def to_relative_actions(actions: np.ndarray, state: np.ndarray, mask: list[bool]) -> np.ndarray:
    """``relative = action - state`` over the masked dimensions."""
    actions = np.asarray(actions)
    dims = len(mask)
    if dims > actions.shape[-1]:
        raise ValueError(
            f"relative mask has {dims} entries but the action has {actions.shape[-1]} dimensions"
        )
    out = actions.astype(np.float32, copy=True)
    out[..., :dims] -= _state_offset(out, state, mask)
    return out


def to_absolute_actions(actions: np.ndarray, state: np.ndarray, mask: list[bool]) -> np.ndarray:
    """``absolute = relative + state`` over the masked dimensions."""
    actions = np.asarray(actions)
    dims = len(mask)
    if dims > actions.shape[-1]:
        raise ValueError(
            f"relative mask has {dims} entries but the action has {actions.shape[-1]} dimensions"
        )
    out = actions.astype(np.float32, copy=True)
    out[..., :dims] += _state_offset(out, state, mask)
    return out


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #
def _source_axis(in_size: int, out_size: int):
    """Indices and weights of the two taps on one axis (``align_corners=False``)."""
    scale = np.float32(in_size) / np.float32(out_size)
    positions = (np.arange(out_size, dtype=np.float32) + np.float32(0.5)) * scale - np.float32(0.5)
    positions = np.maximum(positions, np.float32(0.0))
    lower = np.floor(positions).astype(np.int64)
    lower = np.minimum(lower, in_size - 1)
    upper = np.where(lower < in_size - 1, lower + 1, lower)
    weight = (positions - lower.astype(np.float32)).astype(np.float32)
    return lower, upper, weight


def resize_bilinear(images: np.ndarray, height: int, width: int) -> np.ndarray:
    """Bilinear resize following PyTorch's four-tap accumulation order.

    A separable two-pass resize rounds differently from ``F.interpolate``, so this
    builds the same four weighted taps and sums them in the same order. The
    remaining last-bit kernel differences are *not* claimed to meet the 1e-6
    data-preprocessing tolerance: they are measured and recorded separately by
    the G1 image test, while resampling-free cases are compared at 1e-6.
    """
    batch, channels, cur_h, cur_w = images.shape
    if (cur_h, cur_w) == (height, width):
        return images.astype(np.float32, copy=True)

    y0, y1, wy = _source_axis(cur_h, height)
    x0, x1, wx = _source_axis(cur_w, width)
    wy = wy.reshape(1, 1, height, 1).astype(np.float32)
    wx = wx.reshape(1, 1, 1, width).astype(np.float32)

    source = images.astype(np.float32)
    # One advanced index per axis, then the second axis: indexing both at once would
    # fuse their broadcast dimensions into a higher-rank result.
    rows0 = source[:, :, y0, :]
    rows1 = source[:, :, y1, :]
    top = rows0[:, :, :, x0] * (np.float32(1.0) - wx) + rows0[:, :, :, x1] * wx
    bottom = rows1[:, :, :, x0] * (np.float32(1.0) - wx) + rows1[:, :, :, x1] * wx
    return (top * (np.float32(1.0) - wy) + bottom * wy).astype(np.float32)


def resize_with_pad(images: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize preserving aspect ratio, clamp to [0, 1], then centre-pad with zeros."""
    array = np.asarray(images, dtype=np.float32)
    batch, channels, cur_h, cur_w = array.shape
    ratio = max(cur_w / width, cur_h / height)
    resized_h = int(cur_h / ratio)
    resized_w = int(cur_w / ratio)
    resized = resize_bilinear(array, resized_h, resized_w)
    resized = np.clip(resized, np.float32(0.0), np.float32(1.0)).astype(np.float32)

    pad_h0, remainder_h = divmod(height - resized_h, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_w, 2)
    pad_w1 = pad_w0 + remainder_w
    if any((pad_h0, pad_h1, pad_w0, pad_w1)):
        resized = np.pad(
            resized,
            ((0, 0), (0, 0), (pad_h0, pad_h1), (pad_w0, pad_w1)),
            mode="constant",
            constant_values=0.0,
        )
    return resized.astype(np.float32)


@dataclass
class ProcessedImages:
    """Images in the layout the model expects, with their validity masks.

    ``keys`` follows the same order as ``images``/``masks``: the cameras that were
    actually present first, then one entry per missing camera. ``present_keys``
    records which of those entries came from the batch; the trailing entries are
    the reference's all-``-1`` padding and must carry a false mask.
    """

    images: list[np.ndarray]
    masks: list[np.ndarray]
    keys: list[str]
    present_keys: list[str]

    def stacked(self) -> np.ndarray:
        return np.stack(self.images, axis=1)


def preprocess_images(
    batch: dict[str, np.ndarray],
    image_features: list[str],
    resolution: tuple[int, int] = (224, 224),
) -> ProcessedImages:
    """Replicate ``PI05Policy._preprocess_images`` for one batch.

    The reference processes present cameras in configured order, then appends one
    all-``-1`` image and one all-false mask per missing camera. Only the case with
    *no* present camera raises; a partially missing batch is valid and the
    padding position is at the end (not in the missing camera's original slot).
    """
    present = [key for key in image_features if key in batch]
    missing = [key for key in image_features if key not in batch]
    if not present:
        raise ValueError(
            "All image features are missing from the batch. At least one expected. "
            f"(batch: {sorted(batch)}) (image_features: {image_features})"
        )

    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for key in present:
        image = np.asarray(batch[key], dtype=np.float32)
        is_video = image.ndim == 5
        channel_dim = 2 if is_video else 1
        channels_first = image.shape[channel_dim] == 3
        if channels_first:
            work = np.transpose(image, (0, 1, 3, 4, 2)) if is_video else np.transpose(image, (0, 2, 3, 1))
        else:
            work = image

        if is_video:
            # [B, T, H, W, C] -> [B*T, C, H, W] -> resize -> back to [B, T, H, W, C].
            batch_size, frames = work.shape[0], work.shape[1]
            flat = work.reshape(batch_size * frames, *work.shape[2:])
            flat = np.transpose(flat, (0, 3, 1, 2))
            resized = resize_with_pad(flat, *resolution)
            resized = np.transpose(resized, (0, 2, 3, 1))
            normalised = np.float32(2.0) * resized - np.float32(1.0)
            normalised = normalised.reshape(batch_size, frames, *normalised.shape[1:])
            if channels_first:
                normalised = np.transpose(normalised, (0, 1, 4, 2, 3))
            images.append(normalised.astype(np.float32))
            pad_key = f"{key}_is_pad"
            if pad_key in batch:
                masks.append(~np.asarray(batch[pad_key], dtype=bool))
            else:
                masks.append(np.ones((batch_size, frames), dtype=bool))
            continue

        if work.shape[1:3] != tuple(resolution):
            work = np.transpose(work, (0, 3, 1, 2))
            resized = resize_with_pad(work, *resolution)
            resized = np.transpose(resized, (0, 2, 3, 1))
        else:
            resized = work

        normalised = np.float32(2.0) * resized - np.float32(1.0)
        if channels_first:
            normalised = np.transpose(normalised, (0, 3, 1, 2))
        images.append(normalised.astype(np.float32))
        masks.append(np.ones((normalised.shape[0],), dtype=bool))

    # The reference reuses the last processed image/mask as the template for every
    # missing camera: `torch.ones_like(img) * -1` and `torch.zeros_like(mask)`.
    for _ in missing:
        images.append(np.full_like(images[-1], -1.0))
        masks.append(np.zeros_like(masks[-1], dtype=bool))
    return ProcessedImages(
        images=images, masks=masks, keys=present + missing, present_keys=present
    )
