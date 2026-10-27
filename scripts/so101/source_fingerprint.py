#!/usr/bin/env python3
"""Fingerprint the external dual-SO101 source tree used by E0-E3.

The hash definitions match the external project's own
``scripts/source_fingerprint.py`` so a dataset's ``provenance.json`` can be
compared directly.  Only source and configuration are hashed; generated model
caches and recordings are not part of the binding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

EXPERT_PACKAGES = ("control", "experts", "tasks", "robots", "sim")
SOURCE_DIR = "src/bimanual"


def _digest(base: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(base).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_text(encoding="utf-8").encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _config_digest(root: Path) -> dict:
    files = sorted((root / "configs").rglob("*.yaml"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {"config_sha256": digest.hexdigest(), "config_files": len(files)}


def _package_versions(python: str | None) -> dict:
    if not python:
        return {}
    script = (
        "import importlib.metadata as m, json\n"
        "names=['mujoco','numpy','PyYAML','Pillow','lerobot','torch','av','pyarrow']\n"
        "out={}\n"
        "for name in names:\n"
        "    try: out[name]=m.version(name)\n"
        "    except Exception: pass\n"
        "print(json.dumps(out))\n"
    )
    result = subprocess.run([python, "-c", script], capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        return {"error": (result.stderr or result.stdout)[-400:]}
    return json.loads(result.stdout)


def fingerprint(root: Path, python: str | None = None) -> dict:
    base = root / SOURCE_DIR
    if not base.is_dir():
        raise FileNotFoundError(f"external SO101 source not found under {base}")
    files = sorted(base.rglob("*.py"))
    expert = [path for path in files if path.relative_to(base).parts[0] in EXPERT_PACKAGES]
    model = root / "assets/robots/so101/so101.xml"
    return {
        "schema_version": 1,
        "external_root": str(root),
        "source_sha256": _digest(base, files),
        "source_files": len(files),
        "expert_sha256": _digest(base, expert),
        "expert_files": len(expert),
        "expert_packages": list(EXPERT_PACKAGES),
        **_config_digest(root),
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest() if model.is_file() else None,
        "python": os.path.basename(python) if python else None,
        "packages": _package_versions(python),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="external SO101 checkout (default: $SO101_SOURCE_ROOT)")
    parser.add_argument("--python", help="external Python (default: $SO101_PYTHON)")
    parser.add_argument("--expect-source-sha256")
    parser.add_argument("--expect-expert-sha256")
    parser.add_argument("--write", type=Path, help="write the fingerprint JSON to this path")
    args = parser.parse_args()

    root_value = args.root or os.environ.get("SO101_SOURCE_ROOT")
    if not root_value:
        print("missing --root or SO101_SOURCE_ROOT", file=sys.stderr)
        return 2
    root = Path(root_value).expanduser().resolve()
    python = args.python or os.environ.get("SO101_PYTHON")
    result = fingerprint(root, python)

    if args.expect_source_sha256 and result["source_sha256"] != args.expect_source_sha256:
        print("SOURCE FINGERPRINT MISMATCH", file=sys.stderr)
        return 1
    if args.expect_expert_sha256 and result["expert_sha256"] != args.expect_expert_sha256:
        print("EXPERT FINGERPRINT MISMATCH", file=sys.stderr)
        return 1
    text = json.dumps(result, indent=2) + "\n"
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
