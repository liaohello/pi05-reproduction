"""Strict Orbax -> registered-name weight adaptation for FineART-VLA.

The checkpoint owns 51 logical tensors. The reference model registers 813 FP32
parameters: 812 come from those tensors through the reversible rules below; the
remaining expert LM head has no checkpoint source and is loaded from a pinned
reference-initialisation fixture. The VLM LM head is a clone of the VLM token
embedding, as the read-only FineART loader explicitly specifies.

No parameter is dropped, no missing value is filled randomly, and no load uses
``strict=False``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .mapping import SOURCE_RULES, inverse_source, source_key

V = "model.paligemma_with_expert.paligemma.model.vision_tower."
L = "model.paligemma_with_expert.paligemma.model.language_model."
P = "model.paligemma_with_expert.paligemma.model.multi_modal_projector."
E = "model.paligemma_with_expert.gemma_expert.model."
VLM_HEAD = "model.paligemma_with_expert.paligemma.lm_head.weight"
EXPERT_HEAD = "model.paligemma_with_expert.gemma_expert.lm_head.weight"


def load_checkpoint(path: str | Path) -> dict[str, np.ndarray]:
    """Restore every logical tensor without changing dtype or bytes."""
    import jax
    import orbax.checkpoint as ocp
    from etils import epath

    directory = epath.Path(str(path))
    handler = ocp.PyTreeCheckpointHandler()
    # Checkpoint I/O is host-only: the stored sharding points at the GPU, but a
    # weight audit must not occupy training memory. No model computation uses
    # this device choice.
    metadata = handler.metadata(directory)
    restore_args = jax.tree_util.tree_map(
        lambda _: ocp.RestoreArgs(restore_type=np.ndarray), metadata.tree
    )
    tree = handler.restore(directory, args=ocp.args.PyTreeRestore(restore_args=restore_args))
    out = {}
    for keypath, value in jax.tree_util.tree_leaves_with_path(tree):
        name = ".".join(key.key for key in keypath)
        array = np.asarray(value)
        if array.dtype != np.float32:
            raise ValueError("unexpected checkpoint dtype " + str(array.dtype) + " for " + name)
        out[name] = np.ascontiguousarray(array)
    return out


def map_to_registered(checkpoint: dict[str, np.ndarray], expert_head: np.ndarray) -> dict[str, np.ndarray]:
    """Source tensors -> the exact registered parameter names of the reference."""
    result: dict[str, np.ndarray] = {}
    for name, array in checkpoint.items():
        key = source_key(name)
        if key not in SOURCE_RULES:
            raise ValueError("unmapped checkpoint tensor: " + name)
        for target, transform in SOURCE_RULES[key]:
            if target in result:
                raise ValueError("duplicate target: " + target)
            result[target] = np.ascontiguousarray(transform(array) if transform else array)
    if VLM_HEAD not in result:
        raise ValueError("the VLM head clone rule was not applied")
    if EXPERT_HEAD in result:
        raise ValueError("the expert head must come from the pinned fixture")
    result[EXPERT_HEAD] = np.ascontiguousarray(expert_head, dtype=np.float32)
    return result


def map_to_checkpoint(registered: dict[str, np.ndarray], checkpoint_shapes: dict[str, tuple]) -> dict[str, np.ndarray]:
    """Registered parameters -> source tensors, reversing every declared axis rule."""
    out = {}
    for name in checkpoint_shapes:
        key = source_key(name)
        targets = [target for target, _ in SOURCE_RULES[key]]
        values = []
        for target in targets:
            if target not in registered:
                raise ValueError("missing registered parameter for " + name)
            values.append(registered[target])
        out[name] = np.ascontiguousarray(inverse_source(key, values, tuple(checkpoint_shapes[name])))
    return out


def strict_manifest(checkpoint: dict[str, np.ndarray], expert_head: np.ndarray) -> dict:
    """Name/shape/dtype manifest plus provenance, never a partial summary."""
    registered = map_to_registered(checkpoint, expert_head)
    rows = []
    for name in sorted(registered):
        array = registered[name]
        rows.append({
            "name": name,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "numel": int(array.size),
            "source": EXPERT_HEAD if name == EXPERT_HEAD else None,
        })
    return {"count": len(rows), "parameters": rows}


def validate_registered(registered: dict[str, np.ndarray], expected_rows: list[dict]) -> None:
    """Production strict check against the independent reference audit schema."""
    expected = {row["name"]: row for row in expected_rows}
    got = set(registered)
    missing = sorted(set(expected) - got)
    unexpected = sorted(got - set(expected))
    if missing or unexpected:
        raise ValueError(f"strict parameter set mismatch: missing={missing[:4]} unexpected={unexpected[:4]}")
    for name, row in expected.items():
        array = registered[name]
        if list(array.shape) != row["shape"]:
            raise ValueError(f"shape mismatch for {name}: {array.shape} vs {row['shape']}")
        if str(array.dtype) != "float32":
            raise ValueError(f"dtype mismatch for {name}: {array.dtype}")
        if int(array.size) != int(row["numel"]):
            raise ValueError(f"numel mismatch for {name}: {array.size} vs {row['numel']}")


def validate_expert_head(head: np.ndarray, manifest: dict) -> None:
    """The fixture is a pinned reference-initialised value, not a checkpoint value."""
    if list(head.shape) != [257152, 1024]:
        raise ValueError(f"expert head shape mismatch: {head.shape}")
    if str(head.dtype) != "float32":
        raise ValueError(f"expert head dtype mismatch: {head.dtype}")
    digest = hashlib.sha256(np.ascontiguousarray(head).tobytes()).hexdigest()
    expected = manifest["arrays"]["head"]["sha256"]
    if digest != expected:
        raise ValueError("expert head bytes do not match the pinned fixture manifest")
