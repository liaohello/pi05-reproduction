#!/usr/bin/env python3
"""Collect the G0 measurements that the lock records.

Runs under a neutral interpreter and shells out to the pinned reference and
candidate interpreters. The probes here are written independently of the ones
inside `test_g0_preflight.py`: the lock holds measured facts, and the tests
re-measure them rather than trusting the file.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

BUDGET_PROBE = r"""
import json
import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.fineart_vla.configuration_fineart_vla import FineARTVLAConfig
from lerobot.policies.fineart_vla.modeling_fineart_vla import FineARTVLAPolicy
from lerobot.utils.constants import ACTION

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
counts = {}
for name, parameter in policy.named_parameters():
    key = "vision" if ("vision_tower" in name or "multi_modal_projector" in name) else (
        "action_expert" if ("action_expert" in name or "action_in_proj" in name or "action_out_proj" in name) else (
            "backbone" if name.startswith("model.paligemma_with_expert.paligemma") else "other"))
    counts[key] = counts.get(key, 0) + parameter.numel()
print(json.dumps({"total": sum(counts.values()), "groups": counts}))
"""

TORCH_PROBE = r"""
import json
import torch
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
record = {
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "available": torch.cuda.is_available(),
}
if record["available"]:
    record["device_name"] = torch.cuda.get_device_name(0)
    record["capability"] = list(torch.cuda.get_device_capability(0))
    x = torch.randn(256, 256, device="cuda", requires_grad=True)
    y = torch.randn(256, 256, device="cuda")
    loss = (x @ y).sin().sum()
    loss.backward()
    torch.cuda.synchronize()
    record["finite"] = bool(torch.isfinite(loss).item() and torch.isfinite(x.grad).all().item())
print(json.dumps(record))
"""

JAX_PROBE = r"""
import json
import numpy as np
import jax
import jax.numpy as jnp
record = {"jax": jax.__version__, "backend": jax.default_backend()}
try:
    record["jaxlib"] = jax.lib.__version__
except Exception:
    record["jaxlib"] = None
record["devices"] = [{"platform": d.platform, "kind": d.device_kind} for d in jax.devices()]
a = jnp.asarray(np.random.RandomState(0).randn(256, 256).astype("float32"))
b = jnp.asarray(np.random.RandomState(1).randn(256, 256).astype("float32"))
out = jax.jit(lambda p, q: jnp.sin(p @ q).sum())(a, b)
out.block_until_ready()
grad = jax.grad(lambda p: jnp.sin(p @ b).sum())(a)
record["finite"] = bool(jnp.isfinite(out) and jnp.isfinite(grad).all())
record["loss"] = float(out)
print(json.dumps(record))
"""


def run(python: str, code: str, extra_env: dict | None = None, timeout: int = 2400) -> dict:
    environment = dict(os.environ)
    # PyTorch deterministic algorithms require a fixed cuBLAS workspace.
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if extra_env:
        environment.update(extra_env)
    result = subprocess.run(
        [python, "-c", code], capture_output=True, text=True, timeout=timeout, env=environment, check=False
    )
    if result.returncode != 0:
        raise SystemExit(
            "probe failed under " + python + "\n--- stdout ---\n" + result.stdout[-6000:]
            + "\n--- stderr ---\n" + result.stderr[-6000:]
        )
    for line in reversed(result.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise SystemExit("probe printed no JSON: " + result.stdout[-2000:])


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-python", required=True)
    parser.add_argument("--jax-python", required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--base-python", required=True, help="archive that both environments were created from")
    parser.add_argument("--base-python-url", required=True)
    parser.add_argument("--base-python-sha256", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reference_env = {"PYTHONPATH": str(Path(args.reference_root) / "src")}

    budget = run(args.ref_python, BUDGET_PROBE, reference_env)
    torch_probe = run(args.ref_python, TORCH_PROBE, reference_env)
    jax_probe = run(args.jax_python, JAX_PROBE)

    archive = Path(args.base_python)
    if not archive.is_file():
        raise SystemExit("base interpreter archive is missing: " + str(archive))
    import hashlib

    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != args.base_python_sha256:
        raise SystemExit("base interpreter hash mismatch: " + actual + " != " + args.base_python_sha256)

    base = {
        "version": "3.12.16",
        "sha256": actual,
        "source_url": args.base_python_url,
        "verified": True,
        "verified_sha256": True,
        "archive_bytes": archive.stat().st_size,
        "note": "python-build-standalone install_only archive; hash compared against the published digest after download",
    }

    (out / "budget.json").write_text(json.dumps(budget, indent=2) + "\n", encoding="utf-8")
    (out / "cuda.json").write_text(
        json.dumps({"torch": torch_probe, "jax": jax_probe}, indent=2) + "\n", encoding="utf-8"
    )
    (out / "base_python.json").write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"budget_total": budget["total"], "torch": torch_probe, "jax_backend": jax_probe["backend"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
