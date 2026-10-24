#!/usr/bin/env python3
"""Verify third-party source against the locked SHA-256 manifest.

The upstream OpenPI tree used here was exported from a pinned Docker image and
has no verifiable upstream commit, so the lock is content-based: per-file size
plus SHA-256, together with the image digests. The manifest itself is not
published (it is machine-specific bookkeeping); regenerate it from the images
and compare aggregates instead.

Usage:
    python3 scripts/verify/source_lock.py                 # local tree only
    python3 scripts/verify/source_lock.py --server        # also read the images over SSH
    PI05_OPENPI_SRC=/path/to/openpi python3 scripts/verify/source_lock.py
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "deploy"))

import common  # noqa: E402

SKIP_DIRS = {".git", "__pycache__", ".venv", ".pytest_cache", ".mypy_cache"}

CONTAINER_PROBE = r'''
import hashlib, json, os, pathlib
trees = {"openpi": pathlib.Path("/opt/openpi"),
         "libero": pathlib.Path("/opt/openpi/third_party/libero")}
skip = {".git", "__pycache__", ".venv", ".pytest_cache", ".mypy_cache"}
out = {}
for name, base in trees.items():
    files = {}
    if not base.is_dir():
        out[name] = {"files": files}
        continue
    for current, dirs, names in os.walk(base):
        dirs[:] = [d for d in dirs if d not in skip and not d.endswith(".egg-info")]
        for fname in names:
            if fname.endswith(".pyc"):
                continue
            path = pathlib.Path(current) / fname
            rel = path.relative_to(base).as_posix()
            data = path.read_bytes()
            files[rel] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    out[name] = {"files": files}
print(json.dumps(out))
'''


def manifest_path() -> Path:
    env = os.environ.get("PI05_SOURCE_MANIFEST")
    return Path(env) if env else ROOT / "third_party" / "lock" / "manifest.json"


def openpi_src() -> Path:
    env = os.environ.get("PI05_OPENPI_SRC")
    return Path(env) if env else ROOT / "third_party" / "openpi"


def verify(tree: Path, manifest: dict, ignore_top: tuple[str, ...] = ()) -> tuple[int, list[tuple[str, str]]]:
    bad: list[tuple[str, str]] = []
    total = 0
    for rel, expected in manifest.items():
        path = tree / rel
        if not path.is_file():
            bad.append(("missing", rel))
            continue
        size = path.stat().st_size
        total += size
        if size != expected["size"]:
            bad.append(("size", rel))
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
            bad.append(("sha256", rel))
    for current, dirs, files in os.walk(tree):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.endswith(".egg-info")
                   and not (Path(current) == tree and d in ignore_top)]
        for fname in files:
            if fname.endswith(".pyc"):
                continue
            rel = (Path(current) / fname).relative_to(tree).as_posix()
            if rel not in manifest:
                bad.append(("extra", rel))
    return total, bad


def aggregate(manifest: dict) -> str:
    """Stable digest over a file manifest, so the lock can be published as one number."""
    lines = [f"{rel}\0{meta['size']}\0{meta['sha256']}" for rel, meta in sorted(manifest.items())]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def read_from_images(cfg: dict) -> dict:
    images = cfg["images"]
    server = next(tag for tag in images if "server" in tag)
    client = next(tag for tag in images if "client" in tag)
    payload = base64.b64encode(CONTAINER_PROBE.encode("utf-8")).decode("ascii")
    remote = (
        f"docker run --rm --network none -e OPENPI_PROBE=1 --entrypoint python {images[server]} "
        f"-c \"import base64;exec(base64.b64decode('{payload}'))\""
    )
    result = common.run(common.ssh_argv(cfg) + [remote], timeout=600)
    if result.returncode != 0:
        raise SystemExit(f"server image probe failed: {result.stderr[-800:]}")
    data = json.loads(result.stdout)
    data["images"] = {"server": images[server], "client": images[client]}
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true", help="re-read the pinned images over SSH")
    args = parser.parse_args()

    lock_file = manifest_path()
    if not lock_file.is_file():
        print(f"no manifest at {lock_file}\n"
              "  generate one from the pinned images: PI05_SOURCE_MANIFEST=... "
              "python3 scripts/verify/source_lock.py --server > manifest.json")
        return 2

    locked = json.loads(lock_file.read_text(encoding="utf-8"))
    current = read_from_images(common.load_config()) if args.server else locked
    failed = False
    for name, ignore_top in (("openpi", ("third_party",)), ("libero", ())):
        files = current[name]["files"]
        tree = openpi_src() if name == "openpi" else openpi_src() / "third_party" / "libero"
        total, bad = verify(tree, files, ignore_top)
        if args.server and files != locked[name]["files"]:
            print(f"{name}: IMAGES DO NOT MATCH LOCK")
            failed = True
        print(f"{name}: files={len(files)} bytes={total} mismatches={len(bad)} "
              f"aggregate={aggregate(files)[:16]}...")
        for status, path in bad[:20]:
            print(" ", status, path)
        failed |= bool(bad)
    if args.server:
        print(f"images: {current.get('images', {})}")
    print("SOURCE LOCK PASSED" if not failed else "SOURCE LOCK FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
