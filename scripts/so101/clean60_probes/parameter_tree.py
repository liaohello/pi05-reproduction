"""Enumerate the real clean60 parameter tree inside the pinned OpenPI runtime.

Runs with no GPU (meta-device init) and writes a JSON audit that names every
tensor, its dtype and whether the round's freeze filter keeps it trainable.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, "/work/scripts/so101")

import jax
import jax.numpy as jnp
import flax.nnx as nnx

import openpi_so101
import openpi.training.config as _config
import openpi.models.gemma as _gemma

RUN = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("/work/runs/so101/clean60_runtime")
VARIANTS = {"paligemma_variant": "gemma_2b_lora", "action_expert_variant": "gemma_300m"}
STATE_MODE = "on"


import re

EXPERT_RE = re.compile(".*llm.*_1.*")      # the pinned runtime's action-expert regex
LORA_RE = re.compile(".*lora.*")
LLM_RE = re.compile(".*llm.*")


def group_of(path: str) -> str:
    if LLM_RE.fullmatch(path) and EXPERT_RE.fullmatch(path) and not LORA_RE.fullmatch(path):
        return "action_expert"
    if "/img/" in path or "/img." in path:
        return "vision_encoder"
    if "lora" in path:
        return "language_lora"
    if "llm" in path:
        return "language_backbone"
    for name in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out",
                 "state_proj", "action_time_mlp_in", "action_time_mlp_out"):
        if name in path:
            return f"projection:{name}"
    return "other"


def _create(model_config):
    """Call the pinned runtime's own model factory, whatever its argument names."""
    # Pi0Config.create wraps its argument: create(seed) -> nnx.Rngs(seed).
    return model_config.create(0)


def _init(model_config):
    """Build the tree on the meta device when this flax supports it, else on CPU."""
    rngs = nnx.Rngs(0)
    for name in ("set_and_restore_meta", "set_meta"):
        context = getattr(nnx, name, None)
        if context is not None:
            with context(True):
                return _create(model_config), f"nnx.{name}"
    return _create(model_config), "cpu-arrays"


def _name(path) -> str:
    parts = []
    for item in path:
        parts.append(str(getattr(item, "key", getattr(item, "name", item))))
    return "/".join(parts)


def _array(node):
    value = getattr(node, "value", node)
    return value if hasattr(value, "shape") else None


def _flatten(sub) -> dict:
    """name -> array, walking the nested raw_mapping of a filtered nnx State."""
    out: dict[str, object] = {}

    def walk(node, prefix: str):
        array = getattr(node, "value", node)
        if hasattr(array, "shape") and hasattr(array, "dtype"):
            out[prefix] = array
            return
        items = getattr(node, "items", None)
        if callable(items):
            for key, child in node.items():
                walk(child, f"{prefix}/{key}" if prefix else str(key))

    walk(sub.raw_mapping, "")
    return out


def _query(state, filter_):
    try:
        sub = state.filter(matched=filter_)["matched"]
    except TypeError:
        sub = state.filter(filter_)
    return _flatten(sub)


def main() -> int:
    model_config = openpi_so101.make_model_config(STATE_MODE, **VARIANTS)
    freeze_filter = openpi_so101.make_freeze_filter(model_config)
    gemma_config = _gemma.get_config(VARIANTS["paligemma_variant"])
    lora_runtime = {
        name: {"rank": int(config.rank), "alpha": float(config.alpha),
               "axes": list(config.axes), "rslora": bool(config.rslora)}
        for name, config in (gemma_config.lora_configs or {}).items()
    }
    model, init_mode = _init(model_config)
    graphdef, state = nnx.split(model)
    frozen_leaves = _query(state, freeze_filter)
    trainable_leaves = _query(state, nnx.Not(freeze_filter))
    all_leaves = {**trainable_leaves, **frozen_leaves}
    groups: dict[str, dict] = {}
    tensors: list[dict] = []
    trainable = frozen = 0
    projection_kernel_bias = {
        name: sorted(path for path in all_leaves
                     if path.startswith(name + "/") and path.split("/")[-1] in {"kernel", "bias"})
        for name in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out")
    }
    for name, value in sorted(all_leaves.items()):
        entry = groups.setdefault(group_of(name), {
            "tensors": 0, "parameters": 0, "dtypes": set(), "trainable": 0, "frozen": 0,
            "sample_paths": []})
        size = int(jnp.size(value))
        entry["tensors"] += 1
        entry["parameters"] += size
        entry["dtypes"].add(str(value.dtype))
        is_frozen = name in frozen_leaves
        entry["frozen" if is_frozen else "trainable"] += size
        if is_frozen:
            frozen += size
        else:
            trainable += size
        if len(entry["sample_paths"]) < 4:
            entry["sample_paths"].append(name + f" {tuple(value.shape)} {value.dtype}")
        tensors.append({"path": name, "shape": list(value.shape), "dtype": str(value.dtype),
                        "parameters": size, "trainable": not is_frozen,
                        "matches_action_expert_regex": bool(EXPERT_RE.fullmatch(name)),
                        "is_lora": bool(LORA_RE.fullmatch(name))})
    required_scope = {
        "language_backbone_tensors": sum(1 for row in tensors if group_of(row["path"]) ==
                                         "language_backbone" and row["trainable"]),
        "language_lora_tensors": sum(1 for row in tensors if row["trainable"] and row["is_lora"]),
        "action_expert_tensors": sum(1 for row in tensors if group_of(row["path"]) ==
                                     "action_expert" and row["trainable"]),
        "vision_encoder_tensors": sum(1 for row in tensors if group_of(row["path"]) ==
                                      "vision_encoder" and row["trainable"]),
        "projection_kernel_bias": projection_kernel_bias,
        "missing_projection_components": {
            name: [component for component in ("kernel", "bias")
                   if not any(path.endswith("/" + component) for path in paths)]
            for name, paths in projection_kernel_bias.items()
        },
    }
    scope_passed = bool(
        required_scope["language_backbone_tensors"] == 0
        and required_scope["language_lora_tensors"] == 10
        and required_scope["action_expert_tensors"] > 0
        and required_scope["vision_encoder_tensors"] > 0
        and not any(required_scope["missing_projection_components"].values())
        and all(set(lora_runtime) == {"attn", "ffn"}
                and {"rank", "alpha"} <= set(lora_runtime[name])
                for name in ("attn", "ffn")))
    payload = {
        "schema_version": 1,
        "init_mode": init_mode,
        "model": {"pi05": True, "action_horizon": 10, "state_mode": STATE_MODE, **VARIANTS,
                  "freeze_filter": repr(freeze_filter)},
        "total_tensors": len(all_leaves),
        "total_parameters": trainable + frozen,
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
        "groups": {key: {**value, "dtypes": sorted(value["dtypes"])}
                   for key, value in sorted(groups.items())},
        "lora_runtime": lora_runtime,
        "required_scope": required_scope,
        "scope_passed": scope_passed,
        "tensors": tensors,
        "passed": bool(scope_passed and tensors),
        "note": "counts come from the pinned runtime parameter tree, not from the JSON config",
    }
    RUN.mkdir(parents=True, exist_ok=True)
    out = RUN / "parameter_tree_audit.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
