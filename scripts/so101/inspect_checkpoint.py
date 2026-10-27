#!/usr/bin/env python3
"""Inspect and compare SO101 π0.5 checkpoints inside the pinned OpenPI image.

``--base`` verifies that every non-LoRA tensor of pi05_libero has a matching
shape in the configured π0.5 LoRA model and that every missing tensor is a LoRA
adapter.  ``--trained`` additionally reports the parameter delta between the
base checkpoint and one trained Orbax step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

sys.path.insert(0, str(Path(__file__).resolve().parent))

import openpi_so101  # noqa: E402
import openpi.models.model as _model  # noqa: E402

LORA = re.compile(r".*lora.*")


def _flatten(value) -> dict:
    return traverse_util.flatten_dict(value, sep="/")


def _model_shapes():
    model = openpi_so101.make_model_config()
    state = jax.eval_shape(lambda: nnx.state(model.create(jax.random.key(0))))
    return _flatten(state.to_pure_dict())


def _tree_sha(flat: dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(flat):
        array = np.asarray(flat[key])
        digest.update(key.encode())
        digest.update(b"\0")
        digest.update(str(array.dtype).encode())
        digest.update(b"\0")
        digest.update(str(array.shape).encode())
        digest.update(b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def _array_stats(array: np.ndarray) -> dict:
    array = np.asarray(array)
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "float32_sha256": hashlib.sha256(
                np.asarray(array, dtype=np.float32).tobytes()).hexdigest()}


def _base_report(base: Path) -> tuple[dict, dict]:
    params = _model.restore_params(base / "params", restore_type=np.ndarray)
    flat = _flatten(params)
    shapes = _model_shapes()
    common = sorted(set(flat) & set(shapes))
    only_base = sorted(set(flat) - set(shapes))
    only_model = sorted(set(shapes) - set(flat))
    mismatched = []
    for key in common:
        expected = tuple(shapes[key].shape)
        got = tuple(np.asarray(flat[key]).shape)
        if expected != got:
            mismatched.append({"key": key, "base": got, "model": expected})
    unexpected_missing = [key for key in only_model if not LORA.fullmatch(key)]
    report = {
        "base": str(base),
        "tensors": len(flat),
        "params_sha256": _tree_sha(flat),
        "model_tensors": len(shapes),
        "common_tensors": len(common),
        "only_in_base": only_base,
        "only_in_model": only_model,
        "missing_lora_tensors": [key for key in only_model if LORA.fullmatch(key)],
        "missing_non_lora_tensors": unexpected_missing,
        "shape_mismatches": mismatched,
        "action_in_proj": _array_stats(flat["action_in_proj/kernel"]),
        "action_out_proj": _array_stats(flat["action_out_proj/kernel"]),
        "internal_action_dim": int(np.asarray(flat["action_in_proj/kernel"]).shape[0]),
        "lora_init": "normal(stddev=0.01), one adapter pair per attention/FFN projection",
        "passed": not only_base and not unexpected_missing and not mismatched,
    }
    return report, flat


def _orbax_steps(checkpoint_root: Path) -> list[int]:
    if not checkpoint_root.is_dir():
        return []
    return sorted(int(path.name) for path in checkpoint_root.iterdir()
                  if path.is_dir() and path.name.isdigit())


def _delta(flat: dict, reference: dict) -> dict:
    common = sorted(set(flat) & set(reference))
    changed = []
    unchanged = 0
    for key in common:
        current = jnp.asarray(flat[key])
        old = jnp.asarray(reference[key]).astype(current.dtype)
        if bool(jnp.array_equal(current, old)):
            unchanged += 1
            continue
        delta = np.asarray(current.astype(jnp.float32) - old.astype(jnp.float32))
        changed.append({"key": key, "max_abs": float(np.max(np.abs(delta))),
                        "lora": bool(LORA.fullmatch(key))})
    return {
        "common_tensors": len(common),
        "unchanged_common_tensors": unchanged,
        "changed_tensors": len(changed),
        "changed_lora_tensors": sum(item["lora"] for item in changed),
        "changed_non_lora_tensors": sum(not item["lora"] for item in changed),
        "max_abs_delta": max((item["max_abs"] for item in changed), default=0.0),
        "changed_lora_examples": [item for item in changed if item["lora"]][:5],
        "changed_non_lora_examples": [item for item in changed if not item["lora"]][:5],
    }


def _trained_report(trained: Path, checkpoint_root: Path, base_flat: dict,
                    train_state_step: int | None, previous: Path | None) -> dict:
    flat = _flatten(_model.restore_params(trained / "params", restore_type=np.ndarray))
    lora_keys = sorted(set(flat) - set(base_flat))
    report = {
        "trained": str(trained),
        "tensors": len(flat),
        "params_sha256": _tree_sha(flat),
        "lora_tensors": len(lora_keys),
        "train_state_step": train_state_step,
        "orbax_step_directories": _orbax_steps(checkpoint_root),
        "against_base": _delta(flat, base_flat),
        "note": "Orbax directory names are loop indices; train_state_step is authoritative",
    }
    if previous is not None:
        previous_flat = _flatten(_model.restore_params(previous / "params",
                                                       restore_type=np.ndarray))
        report["against_previous"] = _delta(flat, previous_flat)
        report["previous"] = str(previous)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--trained", type=Path, default=None,
                        help="one Orbax step directory (contains params/ and train_state/)")
    parser.add_argument("--checkpoint-root", type=Path, default=None,
                        help="Orbax checkpoint root (parent of step directories)")
    parser.add_argument("--train-state-step", type=int, default=None,
                        help="authoritative TrainState.step read from the train_state item")
    parser.add_argument("--previous", type=Path, default=None,
                        help="optional earlier step directory for an update delta")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    base_report, base_flat = _base_report(args.base)
    result = {"base": base_report}
    if args.trained:
        root = args.checkpoint_root or args.trained.parent
        result["trained"] = _trained_report(args.trained, root, base_flat,
                                            args.train_state_step, args.previous)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if base_report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
