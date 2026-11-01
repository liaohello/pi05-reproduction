#!/usr/bin/env python3
"""Create an immutable, hash-indexed copy of the Scene1 PI-FAST runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNTIME_FILES = (
    "scene1_train_so101.py",
    "scene1_openpi_so101.py",
    "scene1_validation.py",
    "scene1_model_scope.py",
    "clean60_contracts.py",
    "scene1_pifast_ki_model.py",
    "scene1_pifast_ki_runtime.py",
    "scene1_pifast_ki_fast_data.py",
    "scene1_pifast_ki_data_evidence.py",
    "scene1_pifast_ki_gpu_probe.py",
    "scene1_pifast_ki_batch_audit.py",
    "scene1_pifast_ki_launch.py",
    "scene1_pifast_ki_contract.py",
    "scene1_pifast_ki_select_weight.py",
    "scene1_pifast_ki_scheduler_audit.py",
    "scene1_pifast_ki_resource_gate.py",
    "scene1_pifast_ki_freeze_source.py",
)
TEST_FILES = (
    "tests/test_scene1_pifast_ki_plan_v2.py",
    "tests/test_scene1_pifast_ki_runtime_contract.py",
)
PATCH_FILES = (
    "patches/so101/scene1-pifast-ki-v2-requirements.patch",
    "patches/so101/scene1-pifast-ki-v2-runtime.patch",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-overlay", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--data-audit", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--fast-tokenizer-dir", type=Path, required=True)
    parser.add_argument("--paligemma-tokenizer", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise ValueError("source snapshot output already exists")
    output.mkdir(parents=True)
    runtime_out = output / "runtime"
    runtime_out.mkdir()
    for name in RUNTIME_FILES:
        shutil.copy2(REPO / "scripts/so101" / name, runtime_out / name)
    tests_out = output / "tests"
    tests_out.mkdir()
    for name in TEST_FILES:
        shutil.copy2(REPO / name, tests_out / Path(name).name)
    patches_out = output / "patches"
    patches_out.mkdir()
    for name in PATCH_FILES:
        shutil.copy2(REPO / name, patches_out / Path(name).name)
    shutil.copytree(args.runtime_overlay, output / "overlay")
    shutil.copytree(
        args.fast_tokenizer_dir, output / "fast-tokenizer-source")
    shutil.copy2(
        args.paligemma_tokenizer, output / "paligemma_tokenizer.model")
    shutil.copy2(args.plan, output / "scene1_pifast_ki_training_v2.json")
    shutil.copy2(args.data_audit, output / "pifast_data_audit_v2.json")
    shutil.copy2(args.admission, output / "pifast_data_admission.json")

    files = {}
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = path.relative_to(output).as_posix()
        files[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest = {
        "schema_version": 1,
        "claim": (
            "Independent source/config/tokenizer snapshot for one Scene1 "
            "PI-FAST execution"),
        "files": files,
    }
    manifest_path = output / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "pass",
        "files": len(files),
        "manifest_sha256": sha256_file(manifest_path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
