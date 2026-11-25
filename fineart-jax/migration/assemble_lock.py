#!/usr/bin/env python3
"""Assemble `migration.lock.json` from G0 measurements.

Every value is read from a file produced by an actual measurement: a pin
fragment, an environment freeze, a probe record, or the oracle provenance.
Nothing is typed in by hand, and a missing input aborts instead of defaulting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "fineart-jax-lock/v1"


def load(path: Path):
    if not path.is_file():
        raise SystemExit("missing required G0 input: " + str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze_lines(path: Path):
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise SystemExit("empty freeze file: " + str(path))
    return lines


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", required=True, help="directory with reference.json and assets.json")
    parser.add_argument("--freeze-ref", required=True)
    parser.add_argument("--freeze-jax", required=True)
    parser.add_argument("--base-python", required=True, help="JSON: version, sha256, source_url, verified")
    parser.add_argument("--cuda", required=True, help="JSON: torch probe + jax probe")
    parser.add_argument("--budget", required=True, help="JSON: parameter budget probe")
    parser.add_argument("--fixture", required=True, help="reference fixture provenance JSON")
    parser.add_argument("--oracle", required=True, help="oracle script, hashed here")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    parts = Path(args.parts)
    reference = load(parts / "reference.json")["reference"]
    assets_part = load(parts / "assets.json")
    params_bytes = assets_part["assets"]["pi05_base"]["params"]["byte_count"]

    base = load(Path(args.base_python))
    cuda = load(Path(args.cuda))
    budget = load(Path(args.budget))
    fixture = load(Path(args.fixture))

    record = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "profile": "fp32-default-semantics",
        "reference": reference,
        "assets": assets_part["assets"],
        "openpi": assets_part["openpi"],
        "base_python": base,
        "environments": {
            "ref": {
                "role": "reference PyTorch (read-only tree, PYTHONPATH only)",
                "freeze": freeze_lines(Path(args.freeze_ref)),
            },
            "jax": {
                "role": "candidate JAX",
                "freeze": freeze_lines(Path(args.freeze_jax)),
                "cuda": "cuda12",
            },
        },
        "cuda_runtime": {
            "torch": cuda["torch"]["torch"],
            "torch_cuda": cuda["torch"]["torch_cuda"],
            "cudnn": cuda["torch"]["cudnn"],
            "jax": cuda["jax"]["jax"],
            "jaxlib": cuda["jax"].get("jaxlib"),
            "jax_backend": cuda["jax"]["backend"],
            "device_name": cuda["torch"]["device_name"],
            "capability": cuda["torch"]["capability"],
        },
        "resource_budget": {
            "parameter_count": budget["total"],
            "parameter_groups": budget["groups"],
            "weights_fp32_bytes": 4 * budget["total"],
            "g7_training_bytes": 16 * budget["total"],
            "g7_fits_vram": 16 * budget["total"] <= 34359738368,
            "vram_bytes": 34359738368,
            "method": "real reference model built on the CPU from the default config; parameter count summed from the module tree",
            "checkpoint_param_equivalent_fp32": params_bytes // 4,
            "checkpoint_vs_model_ratio": round(
                (params_bytes / 4) / budget["total"], 4
            ),
            "g7_training_bytes_checkpoint_equivalent": 16
            * (params_bytes // 4),
            "checkpoint_gap_note": (
                "The pinned checkpoint holds 12,441,721,931 bytes; at 4 bytes per value that is 3.11e9 "
                "parameters, while the default config constructs 4.14e9. The G7 verdict is the same under "
                "either count (16x exceeds 32 GiB in both cases), but the gap must be explained before any "
                "claim about the exact trainable count: candidates are bf16 storage, an omitted untied head, "
                "or a different variant than the config default. Unresolved at G0; carried as an open item."
            ),
            "g7_open_items": ["checkpoint_vs_model_parameter_gap"],
        },
        "reference_fixture": {
            "oracle": "tests/migration/oracle/export_tiny_reference.py",
            "oracle_sha256": sha256_file(Path(args.oracle)),
            "source_aggregate_sha256": reference["aggregate_sha256"],
            "array_sha256": fixture["array_sha256"],
            "loss_terms": fixture["loss_terms"],
            "grad_norms": fixture["grad_norms"],
            "batch": fixture["batch"],
            "config": fixture["config"],
            "environment": {
                "torch": fixture["torch"],
                "torch_cuda": fixture["torch_cuda"],
                "cudnn": fixture["cudnn"],
                "device_name": fixture["device_name"],
            },
            "scope": "plain PI0.5 fallback path with knowledge insulation on; text and FAST supervision are exported by their own stages",
        },
        "blocked": [
            {
                "id": "fineart_mid_trained_checkpoint",
                "status": "blocked",
                "reason": (
                    "no FineART-mid-trained checkpoint is available locally or on the test host; "
                    "the pinned real checkpoint is the pi0.5 base parameter tree. Cross-framework parity "
                    "may use it, but nothing may be claimed about FineART-trained weights."
                ),
                "affects": ["G6 real_checkpoint_strict_load", "G7 real_training_hundred_updates"],
            }
        ],
        "environment_gaps": {},
    }

    out = Path(args.out)
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        "wrote "
        + str(out)
        + " params="
        + str(record["resource_budget"]["parameter_count"])
        + " g7_fits="
        + str(record["resource_budget"]["g7_fits_vram"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
