#!/usr/bin/env python3
"""Tiny CPU contract test for Scene1 joint loss and KI loss isolation."""
from __future__ import annotations

import dataclasses
import json

import flax.linen as linen
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from openpi.models import gemma, lora, model as model_lib, pi0, pi0_config
from scene1_pifast_ki_model import (
    FAST_TOKEN_START,
    JointPi05Config,
    flow_mse_12d,
)


class TinyVision(linen.Module):
    width: int

    @linen.compact
    def __call__(self, image, *, train=False):
        value = jnp.mean(image, axis=(1, 2))
        return linen.Dense(self.width, name="projection")(value)[:, None, :], {}


def setup_model():
    old_get_config = gemma.get_config

    def tiny_config(variant):
        base = old_get_config("dummy")
        if variant == "gemma_2b_lora":
            return dataclasses.replace(
                base,
                lora_configs={
                    "attn": lora.LoRAConfig(rank=2, alpha=2),
                    "ffn": lora.LoRAConfig(rank=2, alpha=2),
                },
            )
        return base

    gemma.get_config = tiny_config
    pi0._siglip.Module = lambda **kwargs: TinyVision(
        width=kwargs["num_classes"])
    config = JointPi05Config(
        pi05=True,
        action_dim=32,
        action_horizon=2,
        max_token_len=8,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora_r64",
        dtype="float32",
        fast_ce_weight=0.1,
    )
    return config, config.create(jax.random.key(7))


def make_obs(batch_size=2):
    ids = np.repeat(np.array([[
        2, 3, 4, FAST_TOKEN_START + 10,
        FAST_TOKEN_START + 11, 1, 0, 0,
    ]], dtype=np.int32), batch_size, axis=0)
    valid = np.repeat(np.array([[
        True, True, True, True, True, True, False, False,
    ]], dtype=bool), batch_size, axis=0)
    ar = np.repeat(np.array([[
        0, 0, 0, 1, 1, 1, 0, 0,
    ]], dtype=np.int32), batch_size, axis=0)
    action_region = ar.astype(bool)
    image = jnp.full(
        (batch_size, 224, 224, 3), 0.3, dtype=jnp.float32)
    return model_lib.Observation(
        images={"base_0_rgb": image},
        image_masks={
            "base_0_rgb": jnp.ones((batch_size,), dtype=bool)},
        state=jnp.zeros((batch_size, 32)),
        tokenized_prompt=jnp.asarray(ids),
        tokenized_prompt_mask=jnp.asarray(valid),
        token_ar_mask=jnp.asarray(ar),
        token_loss_mask=jnp.asarray(action_region),
    )


def main() -> int:
    config, model = setup_model()
    obs = make_obs()
    actions = jnp.repeat(
        jnp.arange(2 * 32, dtype=jnp.float32).reshape(1, 2, 32) / 100.0,
        2,
        axis=0,
    )
    key = jax.random.key(123)
    total = model.compute_loss(key, obs, actions, train=False)
    flow = model.compute_flow_loss(key, obs, actions, train=False, ki=True)
    fast = model.compute_fast_loss(key, obs, actions, train=False)
    if not bool(
            jnp.isfinite(total).all()
            & jnp.isfinite(flow).all()
            & jnp.isfinite(fast).all()):
        raise AssertionError("joint loss is non-finite")

    velocity = jnp.arange(
        2 * 32, dtype=jnp.float32).reshape(1, 2, 32) / 10.0
    target = velocity + 0.25
    padded_velocity = velocity.at[..., 12:].set(1e6)
    padded_target = target.at[..., 12:].set(-1e6)
    if not np.array_equal(
            np.asarray(flow_mse_12d(velocity, target)),
            np.asarray(flow_mse_12d(padded_velocity, padded_target))):
        raise AssertionError("padded dimensions entered Flow loss")

    modified = dataclasses.replace(
        obs,
        tokenized_prompt=obs.tokenized_prompt.at[:, 3:5].set(
            jnp.asarray([
                [FAST_TOKEN_START + 20, FAST_TOKEN_START + 21],
                [FAST_TOKEN_START + 30, FAST_TOKEN_START + 31],
            ], dtype=jnp.int32)),
    )
    flow_modified = model.compute_flow_loss(
        key, modified, actions, train=False, ki=True)
    fast_modified = model.compute_fast_loss(
        key, modified, actions, train=False)
    if not np.array_equal(
            np.asarray(flow), np.asarray(flow_modified)):
        raise AssertionError(
            "teacher action changed Flow loss with KI enabled")
    if np.allclose(
            np.asarray(fast), np.asarray(fast_modified), atol=1e-8):
        raise AssertionError(
            "teacher action did not change FAST loss")

    missing_weight_rejected = False
    try:
        dataclasses.replace(config, fast_ce_weight=None).create(
            jax.random.key(8))
    except ValueError:
        missing_weight_rejected = True

    result = {
        "status": "pass",
        "backend": jax.default_backend(),
        "missing_fast_weight_rejected": missing_weight_rejected,
        "flow_mse_12d_ignores_padded_20_dimensions": True,
        "teacher_action_does_not_change_flow_loss_with_ki": True,
        "teacher_action_changes_fast_loss": True,
        "losses": {
            "total": float(jnp.mean(total)),
            "flow": float(jnp.mean(flow)),
            "fast": float(jnp.mean(fast)),
        },
        "claim": (
            "Tiny CPU contract only; no optimizer update or checkpoint"),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
