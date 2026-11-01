#!/usr/bin/env python3
"""Compare complete restored LIBERO state digests without holding two GPU states."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_documents(reference: dict, restored: dict) -> dict:
    mismatches = []
    sections = {}
    for section in ("params", "opt_state", "ema_params"):
        left = reference["sections"].get(section, {})
        right = restored["sections"].get(section, {})
        if set(left) != set(right):
            mismatches.append({
                "section": section,
                "kind": "path_set",
                "only_reference": sorted(set(left) - set(right))[:20],
                "only_restored": sorted(set(right) - set(left))[:20],
            })
        section_mismatches = 0
        compared = 0
        for name in sorted(set(left) & set(right)):
            compared += 1
            first, second = left[name], right[name]
            fields = [
                field for field in ("shape", "dtype", "finite", "sha256")
                if first.get(field) != second.get(field)
            ]
            if fields:
                section_mismatches += 1
                mismatches.append({
                    "section": section,
                    "path": name,
                    "fields": fields,
                    "reference": {field: first.get(field) for field in fields},
                    "restored": {field: second.get(field) for field in fields},
                })
        sections[section] = {
            "compared_tensors": compared,
            "mismatched_tensors": section_mismatches,
            "hashes_exact": section_mismatches == 0 and set(left) == set(right),
        }
    passed = (
        reference["train_state_step"] == restored["train_state_step"]
        and reference["ema_decay"] == restored["ema_decay"]
        and not mismatches
    )
    return {
        "schema_version": 1,
        "passed": passed,
        "train_state_step": {
            "reference": reference["train_state_step"],
            "restored": restored["train_state_step"],
            "equal": reference["train_state_step"] == restored["train_state_step"],
        },
        "ema_decay": {
            "reference": reference["ema_decay"],
            "restored": restored["ema_decay"],
            "equal": reference["ema_decay"] == restored["ema_decay"],
        },
        "sections": sections,
        "exact_tensor_hashes_equal": not mismatches,
        "cross_process_max_abs": 0.0 if not mismatches else None,
        "mismatches": mismatches,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--restored", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    restored = json.loads(args.restored.read_text(encoding="utf-8"))
    payload = compare_documents(reference, restored)
    payload.update({
        "reference": str(args.reference),
        "restored": str(args.restored),
        "reference_sha256": sha256_file(args.reference),
        "restored_sha256": sha256_file(args.restored),
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.out),
        "sha256": sha256_file(args.out),
        "passed": payload["passed"],
        "train_state_step": payload["train_state_step"],
        "ema_decay": payload["ema_decay"],
        "sections": payload["sections"],
        "exact_tensor_hashes_equal": payload["exact_tensor_hashes_equal"],
        "cross_process_max_abs": payload["cross_process_max_abs"],
        "mismatch_count": len(payload["mismatches"]),
    }, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
