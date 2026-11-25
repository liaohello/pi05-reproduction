#!/usr/bin/env python3
"""Adapt the pinned pi05_base Orbax checkpoint to the reference's state-dict names.

Both G3 sides must replay *the same weights*. The candidate can read Orbax directly; the
reference cannot (no JAX in its environment). This exporter applies the G2-certified
mapping once, writes a single safetensors file, and records a per-tensor SHA-256 so the
oracle can prove the file it loads is the file that was exported.

Runs under the JAX interpreter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from fineart_jax.weights import load_checkpoint, map_to_registered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="pi05_base params (Orbax)")
    parser.add_argument("--expert-head", required=True, help="pinned expert-head npz")
    parser.add_argument("--out", required=True, help="destination safetensors")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    head = np.load(args.expert_head)["head"].astype(np.float32)

    started = time.perf_counter()
    registered = map_to_registered(load_checkpoint(args.checkpoint), head)
    digests = {
        name: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        for name, value in registered.items()
    }
    save_file({name: np.ascontiguousarray(value) for name, value in registered.items()}, str(out))

    manifest = {
        "output": str(out),
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "bytes": out.stat().st_size,
        "tensors": len(registered),
        "checkpoint": args.checkpoint,
        "expert_head": args.expert_head,
        "sha256_by_tensor": digests,
        "seconds": round(time.perf_counter() - started, 1),
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"ok": True, "tensors": manifest["tensors"], "sha256": manifest["sha256"],
                      "manifest": str(out.with_suffix(".manifest.json"))}))


if __name__ == "__main__":
    main()
