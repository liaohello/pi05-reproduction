#!/usr/bin/env python3
"""G3 K=1 oracle: run the read-only reference over the fixed fixture.

Runs under the *reference* interpreter (torch, no JAX). Loads the same weights the
candidate restores -- the pinned pi05_base checkpoint, adapted to the reference's own
state-dict names -- verifies every tensor against the export manifest, then records:

* per joint layer, the post-layer prefix and suffix hidden states,
* the final velocity from ``action_out_proj``,
* the elementwise flow loss (``reduction="none"``).

The oracle never imports candidate code: it only reads a safetensors file and a JSON
manifest, so an error in the candidate cannot shape the expected values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.fineart_vla.configuration_fineart_vla import FineARTVLAConfig
from lerobot.policies.fineart_vla.modeling_fineart_vla import FineARTVLAPolicy
from lerobot.utils.constants import ACTION
import lerobot.policies.pi05.modeling_pi05 as pi05_modeling


def sha(tensor) -> str:
    array = tensor.detach().to("cpu", torch.float32).contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="safetensors in reference names")
    parser.add_argument("--manifest", required=True, help="per-tensor sha256 record")
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--knowledge-insulation", action="store_true", default=True)
    parser.add_argument("--no-knowledge-insulation", dest="knowledge_insulation",
                        action="store_false")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text())

    t0 = time.perf_counter()
    # Built from the default contract, not from any published checkpoint config: the flow
    # semantics under test are the contract's, and the weights arrive explicitly.
    config = FineARTVLAConfig(
        device="cpu",
        dtype=torch.float32,
        input_features={
            "observation.images.base": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(32,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(32,))},
    )
    with torch.device("cpu"):
        policy = FineARTVLAPolicy(config)

    from safetensors.torch import load_file

    tensors = load_file(args.weights)
    # Weight identity is checked before use: the manifest was written when the pinned
    # Orbax checkpoint was adapted, so a swapped or edited file fails here, not in a
    # silently different comparison. The manifest is keyed by the exported names, which
    # still carry the reference's own ``model.`` prefix.
    if len(tensors) != manifest["tensors"]:
        raise SystemExit(f"weight count {len(tensors)} != manifest {manifest['tensors']}")
    for name, value in tensors.items():
        recorded = manifest["sha256_by_tensor"].get(name)
        if recorded is None:
            raise SystemExit("tensor missing from manifest: " + name)
        digest = hashlib.sha256(np.ascontiguousarray(value.numpy()).tobytes()).hexdigest()
        if digest != recorded:
            raise SystemExit("tensor digest mismatch: " + name)
    normalized = {key.removeprefix("model."): value for key, value in tensors.items()}
    del tensors
    policy.model.load_state_dict(normalized, strict=True)
    del normalized
    policy.model.eval()
    policy.model.paligemma_with_expert.knowledge_insulation = args.knowledge_insulation
    load_seconds = time.perf_counter() - t0

    records: list[tuple[torch.Tensor, torch.Tensor]] = []
    original = pi05_modeling.compute_layer_complete

    def wrapped(inputs_embeds, *inner_args, **kwargs):
        output = original(inputs_embeds, *inner_args, **kwargs)
        records.append((output[0].detach().clone(), output[1].detach().clone()))
        return output

    pi05_modeling.compute_layer_complete = wrapped
    captured: dict[str, torch.Tensor] = {}
    handle = policy.model.action_out_proj.register_forward_hook(
        lambda module, inputs, output: captured.update(velocity=output.detach().clone())
    )

    fixture = np.load(args.fixture)
    images = [torch.from_numpy(fixture["images"][i][None]) for i in range(fixture["images"].shape[0])]
    img_masks = [torch.ones(1, dtype=torch.bool) for _ in images]
    tokens = torch.from_numpy(fixture["tokens"]).long()
    masks = torch.from_numpy(fixture["token_mask"])
    actions = torch.from_numpy(fixture["actions"])
    noise = torch.from_numpy(fixture["noise"])
    time_t = torch.from_numpy(fixture["time"])

    started = time.perf_counter()
    with torch.no_grad():
        loss_elements = policy.model(images, img_masks, tokens, masks, actions, noise, time_t)
    forward_seconds = time.perf_counter() - started
    handle.remove()
    pi05_modeling.compute_layer_complete = original

    payload = {
        "velocity": captured["velocity"].numpy(),
        "loss_elements": loss_elements.numpy(),
        "x_t": (time_t[:, None, None] * noise + (1 - time_t[:, None, None]) * actions).numpy(),
    }

    # Time sampling rule: the reference draws Beta(alpha, 1.0) and applies scale/offset.
    # torch has no Beta.icdf, so the inverse CDF is the closed form for b == 1:
    # CDF(x) = x**alpha  =>  icdf(u) = u**(1/alpha). Exporting it at fixed quantiles makes
    # the rule comparable without requiring two RNGs to agree.
    from lerobot.policies.common.flow_matching import sample_time_beta

    alpha = float(config.time_sampling_beta_alpha)
    beta = float(config.time_sampling_beta_beta)
    scale = float(config.time_sampling_scale)
    offset = float(config.time_sampling_offset)
    if beta != 1.0:
        raise SystemExit("closed-form inverse requires beta == 1, got " + str(beta))
    beta_u = np.asarray([0.001, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999], dtype=np.float32)
    beta_time = (
        torch.from_numpy(beta_u) ** (1.0 / alpha) * scale + offset
    ).numpy().astype(np.float32)
    # One real draw as a sanity bound: the reference's own sampler must land in (offset, scale+offset).
    drawn = float(sample_time_beta(4, torch.device("cpu"), alpha=alpha, beta=beta,
                                   scale=scale, offset=offset)[0])
    payload["beta_u"] = beta_u
    payload["beta_time"] = beta_time
    time_sampling = {"alpha": alpha, "beta": beta, "scale": scale, "offset": offset,
                     "sample": drawn}
    for index, (prefix_hidden, suffix_hidden) in enumerate(records):
        payload[f"layer{index:02d}.prefix"] = prefix_hidden.numpy()
        payload[f"layer{index:02d}.suffix"] = suffix_hidden.numpy()
    np.savez(out / "torch_k1.npz", **payload)

    record = {
        "knowledge_insulation": args.knowledge_insulation,
        "weights": args.weights,
        "weights_manifest": args.manifest,
        "fixture": args.fixture,
        "load_seconds": round(load_seconds, 3),
        "forward_seconds": round(forward_seconds, 3),
        "max_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "loss": float(loss_elements.mean()),
        "velocity_sha256": sha(captured["velocity"]),
        "layer_count": len(records),
        "time_sampling": time_sampling,
        "arrays": {
            name: {"sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}
            for name, value in payload.items()
        },
    }
    (out / "torch_k1.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({"ok": True, "loss": record["loss"], "layer_count": record["layer_count"],
                      "out": str(out)}))


if __name__ == "__main__":
    main()
