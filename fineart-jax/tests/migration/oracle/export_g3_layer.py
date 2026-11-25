"""Export a compact G3 operator fixture from the pinned PyTorch reference.

This is debugging evidence only: it exercises the real PiGemma decoder layer,
RoPE, AdaRMS, joint-attention function and Flow loss reduction with a compact
single-layer configuration. It is not a substitute for the frozen full model.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.gemma.configuration_gemma import GemmaConfig
from transformers.models.gemma import modeling_gemma

from lerobot.policies.common.flow_matching import FlowConvention, make_flow_matching_inputs
from lerobot.policies.common.vla_utils import make_att_2d_masks
from lerobot.policies.fineart_vla.modeling_fineart_vla import FineARTVLAPolicy, _reduce_action_loss
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch, compute_layer_complete
from lerobot.policies.pi_gemma import PiGemmaModel


HEADS = 8
KV_HEADS = 2
HEAD_DIM = 16
WIDTH = HEADS * HEAD_DIM
CHUNK = 4
PREFIX_LEN = 7


def _sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _configuration(adaptive: bool) -> GemmaConfig:
    config = GemmaConfig(
        vocab_size=128,
        hidden_size=WIDTH,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
    )
    config.use_adarms = adaptive
    config.adarms_cond_dim = WIDTH if adaptive else None
    return config


def _set_nonzero_adarms(layer) -> None:
    with torch.no_grad():
        for norm_index, norm in enumerate((layer.input_layernorm, layer.post_attention_layernorm)):
            indices = np.arange(norm.dense.weight.numel(), dtype=np.float32).reshape(tuple(norm.dense.weight.shape))
            weight = np.asarray(0.002 * np.sin(indices * np.float32(0.013) + np.float32(norm_index * 0.2)),
                                dtype=np.float32)
            bias = np.zeros(tuple(norm.dense.bias.shape), dtype=np.float32)
            dims = np.arange(WIDTH, dtype=np.float32)
            bias[:WIDTH] = 0.015 * np.sin(dims * np.float32(0.11))
            bias[WIDTH : 2 * WIDTH] = 0.012 * np.cos(dims * np.float32(0.07))
            bias[2 * WIDTH :] = 0.22 + 0.03 * np.sin(dims * np.float32(0.09))
            norm.dense.weight.copy_(torch.from_numpy(weight))
            norm.dense.bias.copy_(torch.from_numpy(bias))


def _named_parameters(prefix_layer, expert_layer):
    return {
        **{f"prefix__{name}": p for name, p in prefix_layer.named_parameters()},
        **{f"expert__{name}": p for name, p in expert_layer.named_parameters()},
    }


def _parameter_archive_key(name: str) -> str:
    if name.startswith("prefix__"):
        stream, remainder = "prefix", name[len("prefix__") :]
    else:
        stream, remainder = "expert", name[len("expert__") :]
    parts = remainder.split(".")
    if parts[0] == "input_layernorm":
        group = "input_norm"
        tail = parts[1:]
        if tail == ["dense", "weight"]:
            tail = ["dense_weight"]
        elif tail == ["dense", "bias"]:
            tail = ["dense_bias"]
    elif parts[0] == "post_attention_layernorm":
        group = "post_norm"
        tail = parts[1:]
        if tail == ["dense", "weight"]:
            tail = ["dense_weight"]
        elif tail == ["dense", "bias"]:
            tail = ["dense_bias"]
    elif parts[0] == "self_attn":
        group = "attention"
        tail = parts[1:]
        if tail[-1] == "weight":
            tail = tail[:-1]
    elif parts[0] == "mlp":
        group = "mlp"
        tail = parts[1:]
        if tail[-1] == "weight":
            tail = tail[:-1]
    else:
        raise ValueError(f"unmapped reference parameter: {name}")
    return "p__" + "__".join([stream, group, *tail])


def _attn_mask(prefix_pad, prefix_att, suffix_pad, suffix_att, draws: int) -> tuple[torch.Tensor, torch.Tensor]:
    prefix_len = prefix_pad.shape[1]
    prefix_mask = make_att_2d_masks(prefix_pad, prefix_att)
    prefix_pos = torch.cumsum(prefix_pad.to(torch.int32), dim=1) - 1
    if draws == 1:
        pad = torch.cat([prefix_pad, suffix_pad], dim=1)
        att = torch.cat([prefix_att, suffix_att], dim=1)
        mask = make_att_2d_masks(pad, att)
        positions = torch.cumsum(pad.to(torch.int32), dim=1) - 1
        return mask, positions
    action_length = draws * suffix_pad.shape[1]
    prefix_rows = torch.cat(
        [prefix_mask, torch.zeros(1, prefix_len, action_length, dtype=torch.bool)], dim=-1
    )
    action_to_prefix = prefix_pad[:, None, :].expand(1, action_length, prefix_len)
    block = torch.block_diag(*[torch.ones(suffix_pad.shape[1], suffix_pad.shape[1], dtype=torch.bool)
                               for _ in range(draws)])[None]
    action_rows = torch.cat([action_to_prefix, block], dim=-1)
    mask = torch.cat([prefix_rows, action_rows], dim=1)
    one_draw_pos = prefix_pad.sum(dim=1, keepdim=True) + torch.cumsum(
        suffix_pad.to(torch.int32), dim=1
    ) - 1
    positions = torch.cat([prefix_pos, one_draw_pos.repeat(1, draws)], dim=1)
    return mask, positions


def _run_layer(prefix_layer, expert_layer, rotary, prefix, suffix, condition,
               attention_mask, position_ids, ki: bool):
    additive = torch.zeros(attention_mask.shape, dtype=torch.float32)
    additive.masked_fill_(~attention_mask, torch.finfo(torch.float32).min)
    additive = additive[:, None, :, :]
    return compute_layer_complete(
        [prefix, suffix], additive, position_ids, [None, condition],
        layers=(prefix_layer, expert_layer), rotary_emb=rotary,
        fp32_attention=True, knowledge_insulation=ki,
    )


def _gradient_arrays(prefix_layer, expert_layer, rotary, prefix, suffix, condition,
                     attention_mask, position_ids, target, arrays, state_rows, mode: str):
    parameters = _named_parameters(prefix_layer, expert_layer)
    outputs = _run_layer(prefix_layer, expert_layer, rotary, prefix, suffix, condition,
                         attention_mask, position_ids, mode == "ki_on")
    loss = torch.square(outputs[1] - target).mean()
    grads = torch.autograd.grad(loss, tuple(parameters.values()), allow_unused=True)
    for (name, parameter), grad in zip(parameters.items(), grads, strict=True):
        if grad is None:
            state = "none"
            value = np.zeros(tuple(parameter.shape), dtype=np.float32)
        else:
            value = grad.detach().float().cpu().numpy()
            state = "numeric_zero" if not np.any(value) else "nonzero"
        arrays[f"grad__{mode}__{name}"] = value
        state_rows[f"{mode}__{name}"] = state
    return float(loss.detach().cpu())


def _check_pi05_fallback_dispatch() -> dict:
    """Exercise the real FineART wrapper's unannotated PI05 delegation branch."""
    batch = {"debug_unannotated_batch": torch.tensor([17], dtype=torch.int64)}
    sentinel = object()
    captured: dict = {}
    original = PI05Policy.forward

    def capture_parent(self, received_batch, reduction="mean"):
        captured["same_batch"] = received_batch is batch
        captured["reduction"] = reduction
        return sentinel

    try:
        PI05Policy.forward = capture_parent
        model = FineARTVLAPolicy.__new__(FineARTVLAPolicy)
        torch.nn.Module.__init__(model)
        model.config = SimpleNamespace(enable_fast_action_loss=False)
        result = FineARTVLAPolicy.forward(model, batch, reduction="none")
        if result is not sentinel or captured != {"same_batch": True, "reduction": "none"}:
            raise AssertionError(f"unannotated PI05 fallback dispatch changed: {captured}")
    finally:
        PI05Policy.forward = original
    return {"same_batch": captured["same_batch"], "reduction": captured["reduction"],
            "scope": "wrapper delegation only; base PI05 full-model forward not run"}


def _integrated_flow_gradient_arrays(prefix_layer, expert_layer, rotary, prefix, x_t, time,
                                     target, attention_mask, position_ids, suffix_embedding,
                                     prefix_model, expert_model, action_out_proj, arrays,
                                     state_rows, mode: str) -> float:
    """Reference gradients for the compact PI0.5 suffix→velocity→Flow path."""
    parameters = _named_parameters(prefix_layer, expert_layer)
    embedded, _, _, condition = PI05Pytorch.embed_suffix(suffix_embedding, x_t, time)
    outputs = _run_layer(prefix_layer, expert_layer, rotary, prefix, embedded, condition,
                         attention_mask, position_ids, mode == "flow_ki_on")
    suffix = expert_model.norm(outputs[1], cond=condition)[0]
    velocity = action_out_proj(suffix)
    per_sample = F.mse_loss(velocity, target, reduction="none")[..., :6].mean(dim=(1, 2))
    loss = _reduce_action_loss(per_sample, torch.ones_like(per_sample, dtype=torch.bool), "mean")
    grads = torch.autograd.grad(loss, tuple(parameters.values()), allow_unused=True)
    for (name, parameter), grad in zip(parameters.items(), grads, strict=True):
        if grad is None:
            state = "none"
            value = np.zeros(tuple(parameter.shape), dtype=np.float32)
        else:
            value = grad.detach().float().cpu().numpy()
            state = "numeric_zero" if not np.any(value) else "nonzero"
        arrays[f"grad__{mode}__{name}"] = value
        state_rows[f"{mode}__{name}"] = state
    return float(loss.detach().cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261122)
    fallback_dispatch = _check_pi05_fallback_dispatch()

    prefix_model = PiGemmaModel(_configuration(False)).eval()
    expert_model = PiGemmaModel(_configuration(True)).eval()
    prefix_layer = prefix_model.layers[0]
    expert_layer = expert_model.layers[0]
    _set_nonzero_adarms(expert_layer)
    rotary = prefix_model.rotary_emb

    generator = torch.Generator(device="cpu").manual_seed(20261123)
    prefix = torch.randn((1, PREFIX_LEN, WIDTH), generator=generator, dtype=torch.float32) * 0.2
    suffix_k1 = torch.randn((1, CHUNK, WIDTH), generator=generator, dtype=torch.float32) * 0.2
    condition = torch.randn((1, WIDTH), generator=generator, dtype=torch.float32) * 0.4
    prefix_pad = torch.ones((1, PREFIX_LEN), dtype=torch.bool)
    prefix_att = torch.tensor([[0, 0, 0, 0, 1, 0, 0]], dtype=torch.int32)
    suffix_pad = torch.ones((1, CHUNK), dtype=torch.bool)
    suffix_att = torch.tensor([[1, 0, 0, 0]], dtype=torch.int32)
    mask_k1, positions_k1 = _attn_mask(prefix_pad, prefix_att, suffix_pad, suffix_att, 1)

    suffix_k5 = torch.randn((1, 5 * CHUNK, WIDTH), generator=generator, dtype=torch.float32) * 0.2
    condition_k5 = (
        condition[:, None, None, :]
        .expand(1, 5, CHUNK, WIDTH)
        .reshape(1, 5 * CHUNK, WIDTH)
    )
    mask_k5, positions_k5 = _attn_mask(prefix_pad, prefix_att, suffix_pad, suffix_att, 5)

    arrays: dict[str, np.ndarray] = {}
    states: dict[str, str] = {}
    for name, parameter in _named_parameters(prefix_layer, expert_layer).items():
        arrays[_parameter_archive_key(name)] = parameter.detach().float().cpu().numpy()
    arrays.update({
        "input__prefix": prefix.numpy(),
        "input__suffix_k1": suffix_k1.numpy(),
        "input__condition": condition.numpy(),
        "input__mask_k1": mask_k1.numpy(),
        "input__positions_k1": positions_k1.numpy(),
        "input__suffix_k5": suffix_k5.numpy(),
        "input__condition_k5": condition_k5.numpy(),
        "input__mask_k5": mask_k5.numpy(),
        "input__positions_k5": positions_k5.numpy(),
    })

    ada_value, ada_gate = expert_layer.input_layernorm(suffix_k1, cond=condition)
    arrays["adarms__input_value"] = ada_value.detach().float().cpu().numpy()
    arrays["adarms__input_gate"] = ada_gate.detach().float().cpu().numpy()
    indices = torch.arange(6 * HEADS * HEAD_DIM, dtype=torch.float32).reshape(1, 6, HEADS, HEAD_DIM)
    q = torch.sin(indices * 0.017)
    k = torch.cos(indices * 0.023)
    rope_positions = torch.tensor([[0, 2, 5, 9, 12, 17]], dtype=torch.long)
    dummy = torch.zeros((1, rope_positions.shape[1], HEAD_DIM), dtype=torch.float32)
    cos, sin = rotary(dummy, rope_positions)
    q_ref, k_ref = modeling_gemma.apply_rotary_pos_emb(
        q.transpose(1, 2), k.transpose(1, 2), cos, sin, unsqueeze_dim=1
    )
    arrays.update({
        "rope__q_input": q.numpy(),
        "rope__k_input": k.numpy(),
        "rope__positions": rope_positions.numpy(),
        "rope__q_expected": q_ref.transpose(1, 2).numpy(),
        "rope__k_expected": k_ref.transpose(1, 2).numpy(),
    })

    with torch.no_grad():
        out_on_k1 = _run_layer(prefix_layer, expert_layer, rotary, prefix, suffix_k1, condition,
                               mask_k1, positions_k1, True)
        out_off_k1 = _run_layer(prefix_layer, expert_layer, rotary, prefix, suffix_k1, condition,
                                mask_k1, positions_k1, False)
        out_k5 = _run_layer(prefix_layer, expert_layer, rotary, prefix, suffix_k5, condition_k5,
                             mask_k5, positions_k5, True)
        independent = []
        for draw in range(5):
            start = draw * CHUNK
            current = _run_layer(prefix_layer, expert_layer, rotary, prefix,
                                 suffix_k5[:, start : start + CHUNK], condition,
                                 mask_k1, positions_k1, True)
            independent.append(current[1].numpy())
    arrays.update({
        "expected__k1_prefix": out_on_k1[0].numpy(),
        "expected__k1_suffix_ki_on": out_on_k1[1].numpy(),
        "expected__k1_prefix_ki_off": out_off_k1[0].numpy(),
        "expected__k1_suffix_ki_off": out_off_k1[1].numpy(),
        "expected__k5_prefix": out_k5[0].numpy(),
        "expected__k5_suffix": out_k5[1].numpy(),
        "expected__k5_independent_suffix": np.stack(independent, axis=1),
    })

    target = torch.cos(torch.arange(CHUNK * WIDTH, dtype=torch.float32).reshape(1, CHUNK, WIDTH) * 0.009)
    arrays["input__flow_target_suffix"] = target.numpy()
    loss_on = _gradient_arrays(prefix_layer, expert_layer, rotary, prefix, suffix_k1, condition,
                               mask_k1, positions_k1, target, arrays, states, "ki_on")
    loss_off = _gradient_arrays(prefix_layer, expert_layer, rotary, prefix, suffix_k1, condition,
                                mask_k1, positions_k1, target, arrays, states, "ki_off")

    flow_generator = torch.Generator(device="cpu").manual_seed(20261124)
    actions = torch.randn((2, 5, 3, 32), generator=flow_generator, dtype=torch.float32) * 0.3
    noise = torch.randn((2, 5, 3, 32), generator=flow_generator, dtype=torch.float32) * 0.4
    times = torch.tensor([[0.001, 0.5, 1.0, 0.25, 0.75], [0.5, 0.001, 0.75, 1.0, 0.25]], dtype=torch.float32)
    flat_actions = actions.reshape(2 * 5, 3, 32)
    flat_noise = noise.reshape(2 * 5, 3, 32)
    flat_times = times.reshape(2 * 5)
    x_t, target_flow, _ = make_flow_matching_inputs(
        flat_actions, flat_noise, flat_times, convention=FlowConvention.NOISE_AT_ONE
    )
    x_t = x_t.reshape_as(actions)
    target_flow = target_flow.reshape_as(actions)
    velocity = torch.sin(torch.arange(2 * 5 * 3 * 32, dtype=torch.float32).reshape(2, 5, 3, 32) * 0.003)
    per_dim = F.mse_loss(target_flow, velocity, reduction="none")[..., :6]
    per_sample = per_dim.mean(dim=(1, 2, 3))
    predict = torch.tensor([True, False])
    arrays.update({
        "flow__actions": actions.numpy(),
        "flow__noise": noise.numpy(),
        "flow__time": times.numpy(),
        "flow__x_t_expected": x_t.numpy(),
        "flow__target_expected": target_flow.numpy(),
        "flow__velocity": velocity.numpy(),
        "flow__per_sample_expected": per_sample.numpy(),
        "flow__mean_expected": _reduce_action_loss(per_sample, predict, "mean").numpy(),
        "flow__none_expected": _reduce_action_loss(per_sample, predict, "none").numpy(),
        "flow__predict_actions": predict.numpy(),
    })
    for action_dim in (6, 12, 14, 32):
        error = F.mse_loss(target_flow, velocity, reduction="none")[..., :action_dim]
        routed_per_sample = error.mean(dim=(1, 2, 3))
        arrays[f"flow__routed_mean_expected_d{action_dim}"] = _reduce_action_loss(
            routed_per_sample, predict, "mean"
        ).numpy()
        arrays[f"flow__routed_none_expected_d{action_dim}"] = _reduce_action_loss(
            routed_per_sample, predict, "none"
        ).numpy()

    # Reference PI0.5 suffix embedding and a complete one-pair Flow forward.
    suffix_embedding = SimpleNamespace(
        config=SimpleNamespace(chunk_size=CHUNK, min_period=0.004, max_period=4.0),
        action_in_proj=torch.nn.Linear(32, WIDTH),
        time_mlp_in=torch.nn.Linear(WIDTH, WIDTH),
        time_mlp_out=torch.nn.Linear(WIDTH, WIDTH),
        use_on_device_suffix_mask=True,
        _apply_checkpoint=lambda function, *items, **kwargs: function(*items, **kwargs),
    )
    action_out_proj = torch.nn.Linear(WIDTH, 32)
    forward_actions = torch.randn((1, CHUNK, 32), generator=flow_generator, dtype=torch.float32) * 0.3
    forward_noise = torch.randn((5, 1, CHUNK, 32), generator=flow_generator, dtype=torch.float32) * 0.4
    forward_time = torch.tensor([[0.001], [0.5], [1.0], [0.25], [0.75]], dtype=torch.float32)
    forward_action_draws = forward_actions.repeat(5, 1, 1)
    forward_noise_flat = forward_noise.reshape(5, CHUNK, 32)
    forward_time_flat = forward_time.reshape(5)
    forward_x_flat, forward_target_flat, _ = make_flow_matching_inputs(
        forward_action_draws, forward_noise_flat, forward_time_flat,
        convention=FlowConvention.NOISE_AT_ONE,
    )
    forward_x_draws = forward_x_flat.reshape(5, 1, CHUNK, 32)
    forward_target_draws = forward_target_flat.reshape(5, 1, CHUNK, 32)
    with torch.no_grad():
        embedded_flat, embedded_pad_flat, embedded_att_flat, cond_flat = PI05Pytorch.embed_suffix(
            suffix_embedding, forward_x_flat, forward_time_flat
        )
    embedded_k5 = embedded_flat.reshape(5, 1, CHUNK, WIDTH).permute(1, 0, 2, 3).reshape(1, 5 * CHUNK, WIDTH)
    cond_k5 = cond_flat.reshape(5, 1, WIDTH).permute(1, 0, 2)
    cond_k5 = cond_k5[:, :, None, :].expand(1, 5, CHUNK, WIDTH).reshape(1, 5 * CHUNK, WIDTH)
    with torch.no_grad():
        out_flow_k5 = _run_layer(prefix_layer, expert_layer, rotary, prefix, embedded_k5,
                                 cond_k5, mask_k5, positions_k5, True)
        prefix_flow_k5 = prefix_model.norm(out_flow_k5[0])[0]
        suffix_flow_k5 = expert_model.norm(out_flow_k5[1], cond=cond_k5)[0]
        velocity_k5 = action_out_proj(suffix_flow_k5).reshape(1, 5, CHUNK, 32)
        independent_velocity = []
        for draw in range(5):
            time_one = forward_time[draw]
            emb_one, _, _, cond_one = PI05Pytorch.embed_suffix(
                suffix_embedding, forward_x_draws[draw], time_one
            )
            out_one = _run_layer(prefix_layer, expert_layer, rotary, prefix, emb_one,
                                 cond_one, mask_k1, positions_k1, True)
            suffix_one = expert_model.norm(out_one[1], cond=cond_one)[0]
            independent_velocity.append(action_out_proj(suffix_one).numpy())
    independent_velocity = np.stack(independent_velocity, axis=1)
    flow_error_k5 = F.mse_loss(
        forward_target_draws.permute(1, 0, 2, 3), velocity_k5, reduction="none"
    )[..., :6]
    per_sample_k5 = flow_error_k5.mean(dim=(1, 2, 3))
    flow_loss_k5 = _reduce_action_loss(per_sample_k5, torch.tensor([True]), "mean")
    time_k1 = forward_time[0]
    with torch.no_grad():
        embedded_k1, pad_k1, att_k1, cond_k1 = PI05Pytorch.embed_suffix(
            suffix_embedding, forward_x_draws[0], time_k1
        )
    with torch.no_grad():
        out_flow_k1 = _run_layer(prefix_layer, expert_layer, rotary, prefix, embedded_k1,
                                 cond_k1, mask_k1, positions_k1, True)
        prefix_flow_k1 = prefix_model.norm(out_flow_k1[0])[0]
        suffix_flow_k1 = expert_model.norm(out_flow_k1[1], cond=cond_k1)[0]
        velocity_k1 = action_out_proj(suffix_flow_k1)
    error_k1 = F.mse_loss(
        forward_target_draws[0], velocity_k1, reduction="none"
    )[..., :6]
    flow_loss_k1 = error_k1.mean()

    flow_gradient_states: dict[str, str] = {}
    flow_loss_grad_on = _integrated_flow_gradient_arrays(
        prefix_layer, expert_layer, rotary, prefix, forward_x_draws[0], time_k1,
        forward_target_draws[0], mask_k1, positions_k1, suffix_embedding,
        prefix_model, expert_model, action_out_proj, arrays, flow_gradient_states,
        "flow_ki_on",
    )
    flow_loss_grad_off = _integrated_flow_gradient_arrays(
        prefix_layer, expert_layer, rotary, prefix, forward_x_draws[0], time_k1,
        forward_target_draws[0], mask_k1, positions_k1, suffix_embedding,
        prefix_model, expert_model, action_out_proj, arrays, flow_gradient_states,
        "flow_ki_off",
    )

    arrays.update({
        "se__action_in_proj__weight": suffix_embedding.action_in_proj.weight.detach().numpy(),
        "se__action_in_proj__bias": suffix_embedding.action_in_proj.bias.detach().numpy(),
        "se__time_mlp_in__weight": suffix_embedding.time_mlp_in.weight.detach().numpy(),
        "se__time_mlp_in__bias": suffix_embedding.time_mlp_in.bias.detach().numpy(),
        "se__time_mlp_out__weight": suffix_embedding.time_mlp_out.weight.detach().numpy(),
        "se__time_mlp_out__bias": suffix_embedding.time_mlp_out.bias.detach().numpy(),
        "out__weight": action_out_proj.weight.detach().numpy(),
        "out__bias": action_out_proj.bias.detach().numpy(),
        "fn__prefix__weight": prefix_model.norm.weight.detach().numpy(),
        "fn__expert__dense_weight": expert_model.norm.dense.weight.detach().numpy(),
        "fn__expert__dense_bias": expert_model.norm.dense.bias.detach().numpy(),
        "forward__actions": forward_actions.numpy(),
        "forward__noise_k5": forward_noise.numpy(),
        "forward__time_k5": forward_time.numpy(),
        "forward__x_t_expected_k5": forward_x_draws.permute(1, 0, 2, 3).numpy(),
        "forward__target_expected_k5": forward_target_draws.permute(1, 0, 2, 3).numpy(),
        "forward__suffix_expected_k5": embedded_k5.numpy(),
        "forward__condition_expected_k5": cond_k5.numpy(),
        "forward__prefix_expected_k5": prefix_flow_k5.numpy(),
        "forward__suffix_expected_k5_final": suffix_flow_k5.numpy(),
        "forward__velocity_expected_k5": velocity_k5.numpy(),
        "forward__velocity_expected_independent": independent_velocity,
        "forward__flow_loss_expected_k5": flow_loss_k5.numpy(),
        "forward__x_t_expected_k1": forward_x_draws[0].numpy(),
        "forward__target_expected_k1": forward_target_draws[0].numpy(),
        "forward__suffix_expected_k1": embedded_k1.numpy(),
        "forward__condition_expected_k1": cond_k1.numpy(),
        "forward__prefix_expected_k1": prefix_flow_k1.numpy(),
        "forward__suffix_expected_k1_final": suffix_flow_k1.numpy(),
        "forward__velocity_expected_k1": velocity_k1.numpy(),
        "forward__flow_loss_expected_k1": flow_loss_k1.numpy(),
        "forward__suffix_pad_k1": pad_k1.numpy(),
        "forward__suffix_att_k1": att_k1.numpy(),
        "forward__suffix_pad_k5": embedded_pad_flat.reshape(5, 1, CHUNK).permute(1, 0, 2).reshape(1, 5 * CHUNK).numpy(),
        "forward__suffix_att_k5": embedded_att_flat.reshape(5, 1, CHUNK).permute(1, 0, 2).reshape(1, 5 * CHUNK).numpy(),
    })

    np.savez(args.out / "g3_layer_fixture.npz", **arrays)
    source_path = Path(inspect.getsourcefile(compute_layer_complete))
    reference_source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    manifest = {
        "schema": "fineart-jax-g3-layer-debug-fixture/v1",
        "evidence_class": "debug_only_not_full_architecture_gate_evidence",
        "reference_components": [
            "lerobot.policies.pi05.modeling_pi05.compute_layer_complete",
            "lerobot.policies.pi05.modeling_pi05.PI05Pytorch.embed_suffix",
            "lerobot.policies.pi_gemma.PiGemmaRMSNorm",
            "transformers.models.gemma.modeling_gemma.apply_rotary_pos_emb",
            "lerobot.policies.fineart_vla.modeling_fineart_vla._reduce_action_loss",
            "lerobot.policies.common.flow_matching.make_flow_matching_inputs",
        ],
        "reference_source_sha256": reference_source_sha,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "seed": 20261122,
        "shape": {
            "prefix_width": WIDTH,
            "expert_width": WIDTH,
            "heads": HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "layers": 1,
            "mlp_dim": 256,
            "prefix_tokens": PREFIX_LEN,
            "chunk": CHUNK,
            "draws": 5,
            "dtype": "float32",
        },
        "nondegenerate_adarms": {
            "input_gate_max_abs": float(ada_gate.abs().max()),
            "input_gate_min_abs": float(ada_gate.abs().min()),
            "input_gate_nonzero_count": int((ada_gate != 0).sum()),
        },
        "reference_flow_losses": {"ki_on": loss_on, "ki_off": loss_off},
        "integrated_flow_gradient_losses": {
            "ki_on": flow_loss_grad_on,
            "ki_off": flow_loss_grad_off,
        },
        "pi05_fallback_dispatch": fallback_dispatch,
        "gradient_states": states,
        "flow_gradient_states": flow_gradient_states,
        "arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": _sha(value)}
            for name, value in sorted(arrays.items())
        },
    }
    (args.out / "g3_layer_fixture.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "arrays": len(arrays),
        "reference_source_sha256": reference_source_sha,
        "ada_gate_max_abs": manifest["nondegenerate_adarms"]["input_gate_max_abs"],
        "ki_on_flow_loss": loss_on,
        "ki_off_flow_loss": loss_off,
        "fixture_sha256": hashlib.sha256((args.out / "g3_layer_fixture.npz").read_bytes()).hexdigest(),
    }, indent=2))


if __name__ == "__main__":
    main()
