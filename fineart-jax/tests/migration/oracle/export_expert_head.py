#!/usr/bin/env python3
"""Export the reference-initialised expert LM head that has no checkpoint tensor.

The 51 logical checkpoint tensors explain 812 of the 813 registered FP32
parameters. The remaining tensor is the action expert's LM head: the read-only
reference registers it, its default configuration does not tie it, no forward
path in the migrated flow uses it, and the pinned checkpoint has no source for it.
It is therefore exported once from the read-only reference under an explicit
seed and treated as a pinned reference-initialisation fixture, never as a
checkpoint value and never as a random fill inside the candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.fineart_vla.configuration_fineart_vla import FineARTVLAConfig
from lerobot.policies.fineart_vla.modeling_fineart_vla import FineARTVLAPolicy
from lerobot.utils.constants import ACTION

SEED = 20261116
KEY = "model.paligemma_with_expert.gemma_expert.lm_head.weight"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
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
    value = policy.state_dict()[KEY].detach().cpu().numpy().astype(np.float32)
    np.savez(str(out / "expert_head.npz"), head=value)
    manifest = {
        "seed": SEED,
        "registered_name": KEY,
        "source": "read-only reference default initialisation",
        "checkpoint_source": None,
        "arrays": {
            "head": {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
            }
        },
    }
    (out / "expert_head.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(out), "sha256": manifest["arrays"]["head"]["sha256"]}))


if __name__ == "__main__":
    main()
