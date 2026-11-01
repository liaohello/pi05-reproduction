#!/usr/bin/env python3
"""Re-derive the freeze scope from a measured OpenPI parameter tree.

The input is the JSON emitted by ``parameter_tree.py`` inside the pinned image.
Nothing here trusts a configured count; every number is summed from tensor rows.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

EXPERT = re.compile(r".*llm.*_1.*")
LORA = re.compile(r".*lora.*")
LLM = re.compile(r".*llm.*")
PROJECTIONS = ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out")


def group(path):
    if LLM.fullmatch(path) and EXPERT.fullmatch(path) and not LORA.fullmatch(path):
        return "action_expert"
    if "/img/" in path or "/img." in path:
        return "vision_encoder"
    if LORA.fullmatch(path):
        return "language_lora"
    if LLM.fullmatch(path):
        return "language_backbone"
    for name in PROJECTIONS:
        if path.startswith(name + "/") or path == name:
            return f"projection:{name}"
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameter-tree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tree = json.loads(args.parameter_tree.read_text())
    tensors = tree["tensors"]
    totals = {name: {"tensors": 0, "parameters": 0, "trainable": 0, "frozen": 0}
              for name in ("language_backbone", "language_lora", "action_expert",
                           "vision_encoder", "projections", "other")}
    projection_components = {}
    lora_ranks = set()
    for row in tensors:
        key = group(row["path"])
        bucket = "projections" if key.startswith("projection:") else key
        entry = totals[bucket]
        entry["tensors"] += 1
        entry["parameters"] += int(row["parameters"])
        if row["trainable"]:
            entry["trainable"] += int(row["parameters"])
        else:
            entry["frozen"] += int(row["parameters"])
        if key.startswith("projection:"):
            name = key.split(":", 1)[1]
            component = row["path"].rsplit("/", 1)[-1]
            projection_components.setdefault(name, set()).add(component)
        if LORA.fullmatch(row["path"]) and row["path"].endswith("lora_a"):
            lora_ranks.add(int(row["shape"][-1]))
    lora_tensors = totals["language_lora"]["tensors"]
    projection_ok = all({"kernel", "bias"} <= projection_components.get(name, set())
                        for name in PROJECTIONS)
    checks = {
        "language_backbone_trainable_parameters": totals["language_backbone"]["trainable"],
        "language_lora_tensors": lora_tensors,
        "language_lora_ranks": sorted(lora_ranks),
        "action_expert_trainable_parameters": totals["action_expert"]["trainable"],
        "vision_encoder_trainable_parameters": totals["vision_encoder"]["trainable"],
        "projection_trainable_parameters": totals["projections"]["trainable"],
        "projection_components": {name: sorted(values)
                                  for name, values in sorted(projection_components.items())},
        "total_parameters": int(tree["total_parameters"]),
        "trainable_parameters": int(tree["trainable_parameters"]),
        "frozen_parameters": int(tree["frozen_parameters"]),
        "total_tensors": int(tree["total_tensors"]),
    }
    passed = bool(
        checks["language_backbone_trainable_parameters"] == 0
        and lora_tensors == 10
        and checks["language_lora_ranks"] == [16]
        and checks["action_expert_trainable_parameters"] > 0
        and checks["vision_encoder_trainable_parameters"] > 0
        and projection_ok
        and checks["trainable_parameters"] + checks["frozen_parameters"] == checks["total_parameters"]
        and checks["total_tensors"] == len(tensors)
    )
    payload = {"schema_version": 1, "source": "pinned_image_parameter_tree",
               "checks": checks, "groups": totals, "passed": passed,
               "claim": "freeze scope re-measured in this run; not inherited from clean60"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
