#!/usr/bin/env python3
"""Export the G0 reference fixture from the read-only PyTorch tree.

Runs under REF_PYTHON with the reference source on PYTHONPATH. It never imports
candidate code, so a candidate defect cannot become an expected value.

Scope of this fixture: the plain PI0.5 fallback path (`recipe=None`, no text
labels, FAST disabled) of the small real variant (gemma_300m x2) with knowledge
insulation on. Text and FAST supervision are exported by their own stages; G0
only has to prove the oracle is deterministic before anything is trusted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="directory for the fixture")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Determinism first: a fixture that cannot be replayed is not evidence.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.fineart_vla.configuration_fineart_vla import FineARTVLAConfig
    from lerobot.policies.fineart_vla.modeling_fineart_vla import FineARTVLAPolicy
    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("reference export requires a CUDA device; CPU fallback is not allowed")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    batch_size, seq_len, chunk, action_dim, state_dim = 2, 16, 8, 8, 8
    config = FineARTVLAConfig(
        device=device,
        dtype=torch.float32,
        chunk_size=chunk,
        n_action_steps=chunk,
        # The joint model hardcodes the SigLIP projection to 2048, which only matches a
        # gemma_2b language tower; gemma_300m as the VLM fails in embed_prefix with a
        # 2048-vs-1024 concat. The smallest *valid* pairing is therefore the default
        # 2b + 300m pair, used here with a small batch instead of a shrunken model.
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        recipe=None,
        enable_fast_action_loss=False,
        text_loss_weight=0.0,
        knowledge_insulation=True,
        flow_num_repeats=1,
        input_features={
            "observation.images.base": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
    )

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    cpu_batch = {
        "observation.images.base": torch.rand(batch_size, 3, 224, 224, generator=generator),
        "observation.state": torch.randn(batch_size, state_dim, generator=generator),
        ACTION: torch.randn(batch_size, chunk, action_dim, generator=generator),
        OBS_LANGUAGE_TOKENS: torch.randint(0, 1000, (batch_size, seq_len), generator=generator),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(batch_size, seq_len, dtype=torch.bool),
    }
    batch = {key: value.to(device) for key, value in cpu_batch.items()}

    with torch.device(device):
        policy = FineARTVLAPolicy(config)
    policy = policy.to(device)
    policy.train()
    policy.zero_grad(set_to_none=True)

    flow_seed_before = torch.initial_seed()
    loss, loss_dict = policy.forward(batch)
    if not torch.isfinite(loss).all():
        raise SystemExit("reference loss is not finite")
    loss.backward()

    groups = {"backbone": [], "action_expert": [], "conditioning": [], "lm_head": []}
    for name, parameter in policy.named_parameters():
        if parameter.grad is None:
            continue
        if "lm_head" in name:
            key = "lm_head"
        elif "action_expert" in name or "action_in_proj" in name or "action_out_proj" in name:
            key = "action_expert"
        elif "time_mlp" in name or "adaRMS" in name or "adarms" in name:
            key = "conditioning"
        else:
            key = "backbone"
        groups[key].append(parameter.grad.detach().float())

    grad_norms = {
        key: (
            float(torch.sqrt(sum((tensor.pow(2).sum() for tensor in tensors), torch.zeros((), device=device))))
            if tensors
            else None
        )
        for key, tensors in groups.items()
    }
    # A group that matched no parameter is reported as null, never as a computed zero:
    # "0.0" would claim a gradient was measured and found to vanish.

    # A few real parameter slices: cheap to compare, sensitive to weight drift.
    probes = {}
    for name, parameter in list(policy.named_parameters())[:3]:
        flat = parameter.detach().float().reshape(-1)[:64].cpu().numpy()
        probes[name] = flat

    payload_arrays = {
        "input.images": cpu_batch["observation.images.base"].numpy(),
        "input.state": cpu_batch["observation.state"].numpy(),
        "input.action": cpu_batch[ACTION].numpy(),
        "input.tokens": cpu_batch[OBS_LANGUAGE_TOKENS].numpy(),
        "input.attention_mask": cpu_batch[OBS_LANGUAGE_ATTENTION_MASK].numpy(),
        "output.loss": np.asarray(float(loss.detach()), dtype=np.float64),
    }
    for key, value in loss_dict.items():
        payload_arrays["output." + key] = np.asarray(value, dtype=np.float64).reshape(-1)
    for key, value in probes.items():
        payload_arrays["param." + key] = value
    for key, value in grad_norms.items():
        # A group that matched nothing stays null in the provenance and is absent from
        # the array payload: None would serialize as NaN and look like a broken gradient.
        if value is None:
            continue
        payload_arrays["grad_norm." + key] = np.asarray(value, dtype=np.float64)

    np.savez(str(out / "reference.npz"), **payload_arrays)

    provenance = {
        "schema": "fineart-jax-oracle-fixture/v1",
        "oracle": "reference PyTorch FineART-VLA, read-only tree",
        "seed": args.seed,
        "flow_rng_state_before_forward": int(flow_seed_before),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "platform": platform.platform(),
        "config": {
            "paligemma_variant": config.paligemma_variant,
            "action_expert_variant": config.action_expert_variant,
            "chunk_size": config.chunk_size,
            "dtype": "float32",
            "knowledge_insulation": config.knowledge_insulation,
            "flow_num_repeats": config.flow_num_repeats,
            "text_loss_weight": config.text_loss_weight,
            "enable_fast_action_loss": config.enable_fast_action_loss,
            "recipe": config.recipe,
        },
        "batch": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "chunk_size": chunk,
            "action_dim": action_dim,
            "state_dim": state_dim,
        },
        "loss_terms": {key: float(np.asarray(value, dtype=np.float64).reshape(-1)[0]) for key, value in loss_dict.items()},
        "grad_norms": grad_norms,
        "array_sha256": {},
        "degenerate_check": {},
    }

    for name, array in sorted(payload_arrays.items()):
        provenance["array_sha256"][name] = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()

    # Non-degeneracy: constant or zero outputs would make several gates vacuous.
    for name, array in sorted(payload_arrays.items()):
        if array.dtype.kind != "f" or array.size < 2:
            continue
        provenance["degenerate_check"][name] = {
            "std": float(np.std(array.astype(np.float64))),
            "finite": bool(np.isfinite(array).all()),
        }
    (out / "reference.provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    summary = {
        "ok": True,
        "out": str(out),
        "loss": float(loss.detach()),
        "loss_terms": {key: float(np.asarray(value, dtype=np.float64).reshape(-1)[0]) for key, value in loss_dict.items()},
        "grad_norms": grad_norms,
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
