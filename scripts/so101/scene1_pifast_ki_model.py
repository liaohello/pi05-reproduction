"""Scene1 PI05 trunk with pure-FAST and continuous Flow Matching branches."""
from __future__ import annotations

import dataclasses
import math

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp

import openpi.models.gemma as gemma
import openpi.models.model as model_lib
import openpi.models.pi0_config as pi0_config
from openpi.models.pi0 import Pi0, make_attn_mask

PALIGEMMA_VOCAB_SIZE = 257_152
FAST_SKIP_TOKENS = 128
FAST_TOKEN_COUNT = 2_048
FAST_TOKEN_START = 254_976
FAST_TOKEN_STOP = 257_024


def flow_mse_12d(
    velocity: jax.Array, target_velocity: jax.Array
) -> jax.Array:
    """Return one Flow-MSE value per sample over exactly 12 dimensions."""
    if velocity.shape[-1] < 12 or target_velocity.shape[-1] < 12:
        raise ValueError("Flow tensors must contain at least 12 action dimensions")
    return jnp.mean(
        jnp.square(velocity[..., :12] - target_velocity[..., :12]),
        axis=(-2, -1),
    )


class FASTTokenResidual(nnx.Module):
    """Shared zero-initialized input/output compensation for FAST tokens."""

    def __init__(self, width: int):
        self.table = nnx.Param(
            jnp.zeros((FAST_TOKEN_COUNT, width), dtype=jnp.float32))

    def token_mask(self, tokens):
        return (
            (tokens >= FAST_TOKEN_START)
            & (tokens < FAST_TOKEN_STOP)
        )

    def embed_delta(self, tokens):
        mask = self.token_mask(tokens)
        indices = jnp.clip(
            tokens - FAST_TOKEN_START, 0, FAST_TOKEN_COUNT - 1)
        rows = jnp.take(self.table.value, indices, axis=0)
        return rows * mask[..., None]

    def logits_delta(self, hidden):
        return jnp.einsum(
            "btd,nd->btn", hidden.astype(jnp.float32), self.table.value)

    def add_logits(self, hidden, base_logits):
        if base_logits.shape[-1] != PALIGEMMA_VOCAB_SIZE:
            raise ValueError("base logits do not span the PaliGemma vocabulary")
        delta = self.logits_delta(hidden)
        return base_logits.at[
            ..., FAST_TOKEN_START:FAST_TOKEN_STOP
        ].add(delta)


@dataclasses.dataclass(frozen=True)
class JointPi05Config(pi0_config.Pi0Config):
    fast_ce_weight: float | None = None

    def create(self, rng):
        if self.fast_ce_weight is None:
            raise ValueError("fast_ce_weight must be selected before model creation")
        return JointPi05V3(
            self,
            rngs=nnx.Rngs(rng),
            fast_ce_weight=self.fast_ce_weight,
        )


class JointPi05V3(Pi0):
    """Shared PI05 parameters with independent FAST and Flow objectives."""

    def __init__(
        self,
        config: JointPi05Config,
        rngs: nnx.Rngs,
        *,
        fast_ce_weight: float,
    ):
        super().__init__(config, rngs=rngs)
        width = gemma.get_config(config.paligemma_variant).width
        self.fast_token_residual = FASTTokenResidual(width)
        weight = float(fast_ce_weight)
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("fast_ce_weight must be finite and positive")
        self.fast_ce_weight = weight

    def _require_token_inputs(self, obs):
        missing = [
            name for name in (
                "tokenized_prompt",
                "tokenized_prompt_mask",
                "token_ar_mask",
                "token_loss_mask",
            )
            if getattr(obs, name) is None
        ]
        if missing:
            raise ValueError(f"joint model requires token fields: {missing}")

    def _embed_with_ar(self, obs):
        self._require_token_inputs(obs)
        embeddings, masks, ar_masks = [], [], []
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(
                obs.images[name], train=False)
            image_mask = einops.repeat(
                obs.image_masks[name],
                "b -> b s",
                s=image_tokens.shape[1],
            )
            embeddings.append(image_tokens)
            masks.append(image_mask)
            ar_masks.append(jnp.zeros_like(image_mask, dtype=jnp.bool_))
        token_embeddings = self.PaliGemma.llm(
            obs.tokenized_prompt, method="embed")
        token_embeddings = (
            token_embeddings
            + self.fast_token_residual.embed_delta(
                obs.tokenized_prompt
            ).astype(token_embeddings.dtype)
        )
        embeddings.append(token_embeddings)
        masks.append(obs.tokenized_prompt_mask)
        ar_masks.append(obs.token_ar_mask.astype(jnp.bool_))
        return (
            jnp.concatenate(embeddings, axis=1),
            jnp.concatenate(masks, axis=1),
            jnp.concatenate(ar_masks, axis=1),
        )

    def _token_logits(self, hidden):
        base = self.PaliGemma.llm(hidden, method="deembed").astype(jnp.float32)
        return self.fast_token_residual.add_logits(hidden, base)

    def _validate_and_preprocess(self, rng, observation, actions, train):
        if not self.pi05:
            raise ValueError("joint model requires a pi05 configuration")
        if actions.ndim != 3 or actions.shape[-1] < 12:
            raise ValueError("actions must have shape (batch, horizon, 32)")
        obs = model_lib.preprocess_observation(
            rng,
            observation,
            train=train,
            image_keys=list(observation.images.keys()),
        )
        return obs

    def _fast_loss(self, obs):
        prefix, pmask, ar = self._embed_with_ar(obs)
        positions = jnp.cumsum(pmask, axis=1) - 1
        (out, _), _ = self.PaliGemma.llm(
            [prefix, None],
            mask=make_attn_mask(pmask, ar),
            positions=positions,
            adarms_cond=[None, None],
        )
        target = obs.tokenized_prompt[:, 1:]
        hidden = out[:, -obs.tokenized_prompt.shape[1]:-1]
        logp = jax.nn.log_softmax(self._token_logits(hidden), axis=-1)
        target_logp = jnp.take_along_axis(
            logp, target[..., None], axis=-1)[..., 0]
        loss_mask = obs.token_loss_mask[:, 1:].astype(jnp.float32)
        token_count = jnp.sum(loss_mask, axis=-1)
        fast_loss = -jnp.sum(
            target_logp * loss_mask, axis=-1) / token_count
        return fast_loss

    def _flow_loss(self, obs, actions, rng_noise, rng_time, *, ki):
        condition_mask = (
            obs.tokenized_prompt_mask & ~obs.token_ar_mask.astype(jnp.bool_)
        )
        flow_obs = dataclasses.replace(
            obs,
            tokenized_prompt=jnp.where(
                condition_mask, obs.tokenized_prompt, 0),
            tokenized_prompt_mask=condition_mask,
        )
        condition_tokens, condition_mask, condition_ar = self.embed_prefix(
            flow_obs)
        condition_positions = jnp.cumsum(condition_mask, axis=1) - 1
        (_, _), kv = self.PaliGemma.llm(
            [condition_tokens, None],
            mask=make_attn_mask(condition_mask, condition_ar),
            positions=condition_positions,
            adarms_cond=[None, None],
        )
        if ki:
            kv = jax.tree.map(jax.lax.stop_gradient, kv)

        noise = jax.random.normal(rng_noise, actions.shape)
        time = (
            jax.random.beta(rng_time, 1.5, 1, actions.shape[:-2])
            * 0.999
            + 0.001
        )
        time_expanded = time[..., None, None]
        noisy_actions = (
            time_expanded * noise + (1 - time_expanded) * actions)
        target_velocity = noise - actions
        suffix, suffix_mask, suffix_ar, adarms_cond = self.embed_suffix(
            flow_obs, noisy_actions, time)
        all_mask = jnp.concatenate(
            [condition_mask, suffix_mask], axis=1)
        attention = make_attn_mask(
            all_mask,
            jnp.concatenate([condition_ar, suffix_ar], axis=0),
        )[:, -suffix.shape[1]:, :]
        suffix_positions = (
            jnp.cumsum(all_mask, axis=1) - 1
        )[:, -suffix.shape[1]:]
        (_, expert), _ = self.PaliGemma.llm(
            [None, suffix],
            mask=attention,
            positions=suffix_positions,
            kv_cache=kv,
            adarms_cond=[None, adarms_cond],
        )
        velocity = self.action_out_proj(
            expert[:, -self.action_horizon:])
        flow_loss = flow_mse_12d(velocity, target_velocity)
        return flow_loss

    def compute_losses(
        self,
        rng,
        observation,
        actions,
        *,
        train=False,
        ki=True,
    ):
        """Return per-sample FM-MSE and FAST-CE values."""
        rng_obs, rng_noise, rng_time = jax.random.split(rng, 3)
        obs = self._validate_and_preprocess(
            rng_obs, observation, actions, train)
        fast_loss = self._fast_loss(obs)
        flow_loss = self._flow_loss(
            obs, actions, rng_noise, rng_time, ki=ki)
        return flow_loss, fast_loss

    def compute_flow_loss(
        self, rng, observation, actions, *, train=False, ki=True
    ):
        """Compute only the fixed Flow-MSE validation objective."""
        rng_obs, rng_noise, rng_time = jax.random.split(rng, 3)
        obs = self._validate_and_preprocess(
            rng_obs, observation, actions, train)
        return self._flow_loss(
            obs, actions, rng_noise, rng_time, ki=ki)

    def compute_fast_loss(
        self, rng, observation, actions, *, train=False
    ):
        """Compute only the FAST cross-entropy objective."""
        rng_obs, _, _ = jax.random.split(rng, 3)
        obs = self._validate_and_preprocess(
            rng_obs, observation, actions, train)
        return self._fast_loss(obs)

    def compute_loss(
        self, rng, observation, actions, *, train=False
    ):
        """Training objective used by the pinned OpenPI optimizer step."""
        flow_loss, fast_loss = self.compute_losses(
            rng, observation, actions, train=train, ki=True)
        return flow_loss + self.fast_ce_weight * fast_loss
