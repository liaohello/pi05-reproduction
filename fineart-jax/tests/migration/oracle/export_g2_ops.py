#!/usr/bin/env python3
"""Reference-only exporter for G2 base operators.

Runs under the locked REF_PYTHON. Every expected value comes from an independent
PyTorch implementation of the read-only reference semantics. The candidate never
imports this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import SiglipVisionConfig, SiglipVisionModel


def sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def dump(path: Path, arrays: dict, manifest: dict) -> None:
    arrays = {name: np.asarray(value) for name, value in arrays.items()}
    np.savez(str(path), **arrays)
    record = {
        "arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": sha(value)}
            for name, value in sorted(arrays.items())
        },
        **manifest,
    }
    path.with_suffix(".json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def rope(values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    dim = values.shape[-1]
    exponent = (2.0 / dim) * torch.arange(dim // 2, dtype=torch.float32)
    timescale = torch.pow(torch.tensor(10000.0), exponent)
    radians = positions[..., None] / timescale[None, None, :]
    radians = radians[..., None, :]
    sin, cos = torch.sin(radians), torch.cos(radians)
    x1, x2 = torch.split(values, dim // 2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def group_qkv(out: Path) -> None:
    torch.manual_seed(20261116)
    width, heads, kv_heads, head_dim, seq = 2048, 8, 1, 256, 5
    x = torch.randn(1, seq, width, dtype=torch.float32)
    w_q = torch.randn(heads, width, head_dim, dtype=torch.float32)
    w_kv = torch.randn(2, kv_heads, width, head_dim, dtype=torch.float32)
    w_o = torch.randn(heads * head_dim, width, dtype=torch.float32)
    positions = torch.tensor([[0, 2, 3, 7, 11]], dtype=torch.float32)

    q = torch.einsum("BTD,NDH->BTNH", x, w_q)
    k = torch.einsum("BSD,KDH->BSKH", x, w_kv[0])
    v = torch.einsum("BSD,KDH->BSKH", x, w_kv[1])
    q = rope(q, positions) * (head_dim ** -0.5)
    k = rope(k, positions)
    # The output projection is exercised with its own pinned 2-D input: under GQA
    # q and k/v have different head cardinalities and are never concatenated.
    torch.manual_seed(20261120)
    encoded = torch.randn(1, seq, heads * head_dim, dtype=torch.float32)
    projected = torch.nn.functional.linear(encoded, w_o)

    arrays = {
        "x": x.numpy(), "w_q": w_q.numpy(), "w_kv": w_kv.numpy(), "w_o": w_o.numpy(),
        "positions": positions.numpy(), "q_after_rope_scale": q.numpy(),
        "k_after_rope": k.numpy(), "v": v.numpy(), "encoded": encoded.numpy(),
        "projected": projected.numpy(),
    }
    dump(out / "qkv.npz", arrays, {
        "width": width, "heads": heads, "kv_heads": kv_heads, "head_dim": head_dim,
        "scale": head_dim ** -0.5, "positions": positions[0].tolist(),
    })


def group_norm(out: Path) -> None:
    torch.manual_seed(20261117)
    dim, cond_dim = 1024, 1024
    x = torch.randn(2, 4, dim, dtype=torch.float32)

    plain = torch.nn.Parameter(torch.randn(dim))
    normed = x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    plain_out = normed * (1.0 + plain.float())

    dense = torch.nn.Linear(cond_dim, dim * 3, bias=True)
    arrays = {
        "x": x.numpy(), "plain_weight": plain.detach().numpy(), "plain_out": plain_out.detach().numpy(),
        "dense_weight": dense.weight.detach().numpy(), "dense_bias": dense.bias.detach().numpy(),
    }
    manifest = {"cases": {}, "eps": 1e-6}
    for name, cond_shape in (("per_sample", (2, cond_dim)), ("per_token", (2, 4, cond_dim))):
        cond = torch.randn(*cond_shape, dtype=torch.float32)
        modulation = dense(cond)
        if x.ndim == 3 and modulation.ndim == 2:
            modulation = modulation.unsqueeze(1)
        scale, shift, gate = modulation.chunk(3, dim=-1)
        value = normed * (1 + scale) + shift
        arrays[f"{name}__cond"] = cond.numpy()
        arrays[f"{name}__out"] = value.detach().numpy()
        arrays[f"{name}__gate"] = gate.detach().numpy()
        manifest["cases"][name] = {"cond_shape": list(cond_shape)}
    dump(out / "norm.npz", arrays, manifest)


def _load_vision_weights():
    """Load the pinned pi05_base vision subtree exported as SigLIP-named safetensors."""
    from safetensors.numpy import load_file

    path = Path(os.environ["G2_VISION_WEIGHTS"])
    return load_file(str(path))


def group_vision(out: Path) -> None:
    width, projected_width = 1152, 2048
    weights = _load_vision_weights()
    config = SiglipVisionConfig(
        hidden_size=width, intermediate_size=4304, num_hidden_layers=27,
        num_attention_heads=16, image_size=224, patch_size=14, num_channels=3,
        layer_norm_eps=1e-6, hidden_act="gelu_pytorch_tanh",
    )
    vision = SiglipVisionModel(config)
    state = vision.state_dict()
    state.update({
        key: torch.from_numpy(np.ascontiguousarray(value)).to(torch.float32)
        for key, value in weights.items() if not key.startswith("projector.")
    })
    vision.load_state_dict(state, strict=True)
    vision.eval()
    projector = nn.Linear(width, projected_width, bias=True)
    with torch.no_grad():
        projector.weight.copy_(torch.from_numpy(np.ascontiguousarray(weights["projector.weight"])))
        projector.bias.copy_(torch.from_numpy(np.ascontiguousarray(weights["projector.bias"])))
    projector.eval()

    arrays = {}
    generator = torch.Generator().manual_seed(20261121)

    def images_for(count, offset=0):
        batch = []
        size = 224
        for camera in range(count):
            base = (torch.arange(size * size, dtype=torch.float32) % 251.0) / 251.0
            base = (base + 0.07 * (camera + offset)).reshape(size, size)
            batch.append(torch.stack([base, base.roll(1, 0), base.roll(2, 1)])[None])
        return torch.cat(batch, dim=0)

    for count in (1, 2, 3):
        pixels = images_for(count)
        with torch.no_grad():
            patch = vision.embeddings.patch_embedding(pixels)
            patch = patch.flatten(2).transpose(1, 2)
            with_pos = patch + vision.embeddings.position_embedding.weight[None]
            outputs = vision(pixel_values=pixels, output_hidden_states=True)
            final = outputs.last_hidden_state
            projected = projector(final)
        arrays[f"image_c{count}"] = pixels.numpy()
        arrays[f"patches_c{count}"] = patch.numpy()
        arrays[f"with_pos_c{count}"] = with_pos.numpy()
        arrays[f"final_c{count}"] = final.numpy()
        arrays[f"projected_c{count}"] = projected.numpy()
        arrays[f"image_mask_c{count}"] = torch.ones(count, 256, dtype=torch.bool).numpy()
        text = torch.randn(count, 6, projected_width, generator=generator)
        text_mask = torch.ones(count, 6, dtype=torch.bool)
        text_mask[:, 4:] = False
        prefix_tokens = torch.cat([projected, text], dim=1)
        prefix_mask = torch.cat([torch.ones(count, 256, dtype=torch.bool), text_mask], dim=1)
        arrays[f"text_c{count}"] = text.numpy()
        arrays[f"text_mask_c{count}"] = text_mask.numpy()
        arrays[f"prefix_tokens_c{count}"] = prefix_tokens.numpy()
        arrays[f"prefix_mask_c{count}"] = prefix_mask.numpy()
    # One layer-0 and one final hidden-state reference for the full tower.
    pixels = images_for(1)
    with torch.no_grad():
        outputs = vision(pixel_values=pixels, output_hidden_states=True)
    arrays["hidden0_c1"] = outputs.hidden_states[1].numpy()
    arrays["hidden_last_c1"] = outputs.last_hidden_state.numpy()

    # B=2, C=3: camera axis must be concatenated, batch axis preserved.
    batch = torch.stack([images_for(3, 0), images_for(3, 6)], dim=0)  # [2,3,3,224,224]
    flat_pixels = batch.reshape(6, 3, 224, 224)
    with torch.no_grad():
        outputs = vision(pixel_values=flat_pixels)
        final = outputs.last_hidden_state.reshape(2, 3, 256, width)
        projected = projector(final)
    image_mask = torch.ones(2, 3, 256, dtype=torch.bool)
    image_mask[:, 0, 100:] = False
    text = torch.randn(2, 6, projected_width, generator=generator)
    text_mask = torch.ones(2, 6, dtype=torch.bool)
    text_mask[:, 4:] = False
    prefix_tokens = torch.cat([projected.reshape(2, 3 * 256, projected_width), text], dim=1)
    prefix_mask = torch.cat([image_mask.reshape(2, 3 * 256), text_mask], dim=1)
    arrays["image_b2c3"] = batch.numpy()
    arrays["projected_b2c3"] = projected.numpy()
    arrays["image_mask_b2c3"] = image_mask.numpy()
    arrays["text_b2c3"] = text.numpy()
    arrays["text_mask_b2c3"] = text_mask.numpy()
    arrays["prefix_tokens_b2c3"] = prefix_tokens.numpy()
    arrays["prefix_mask_b2c3"] = prefix_mask.numpy()
    dump(out / "vision.npz", arrays, {
        "checkpoint": "pi05_base vision subtree",
        "width": width,
        "projected_width": projected_width,
        "depth": 27,
    })


def group_embedding(out: Path) -> None:
    torch.manual_seed(20261119)
    vocab, width = 257152, 2048
    table = torch.randn(vocab, width, dtype=torch.float32)
    token_ids = torch.tensor([[0, 1, 255999, 256000, 257151]], dtype=torch.int64)
    embedded = table[token_ids] * np.sqrt(width)
    logits = embedded @ table.T
    arrays = {
        "table": table.numpy(), "token_ids": token_ids.numpy(),
        "embedded": embedded.numpy(), "logits": logits.numpy(),
        "head": table.clone().numpy(),
        "head_small": table[:8].clone().numpy(),
        "logits_small": (embedded @ table[:8].T).numpy(),
    }
    dump(out / "embedding.npz", arrays, {
        "vocab": vocab, "width": width, "sqrt_width": float(np.sqrt(width)),
        "token_ids": token_ids[0].tolist(),
    })


GROUPS = {"qkv": group_qkv, "norm": group_norm, "vision": group_vision, "embedding": group_embedding}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True, choices=sorted(GROUPS))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    GROUPS[args.group](out)
    print(json.dumps({"group": args.group, "out": str(out)}))


if __name__ == "__main__":
    main()
