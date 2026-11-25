#!/usr/bin/env python3
"""Produce the G0 source/asset pin record.

Run on the machine that holds the bytes. The same script runs on the editing
host (reference tree) and on the test host (assets), each writing a fragment;
`--merge` combines fragments into `migration.lock.json`.

Nothing here writes into a read-only tree. Unobtainable inputs are recorded as
`blocked` with a reason; they are never given a fabricated identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "fineart-jax-lock/v1"
LOCK = Path(__file__).with_name("migration.lock.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def walk(root: Path, skip_names=()):
    """Every regular file under root, sorted, with size and SHA-256."""
    entries = []
    for current, directories, names in os.walk(str(root), followlinks=False):
        current_path = Path(current)
        directories.sort()
        for name in sorted(directories):
            candidate = current_path / name
            if candidate.is_symlink():
                raise SystemExit("symlink directory in pinned tree: " + str(candidate))
        for name in sorted(names):
            path = current_path / name
            if name in skip_names:
                continue
            relative = str(path.relative_to(root))
            if path.is_symlink():
                # A symlink is part of the tree identity: record the target and, when the
                # target lies inside the tree, the bytes it resolves to.
                target = os.readlink(path)
                entry = {"path": relative, "symlink_to": target}
                resolved = path.resolve()
                if resolved.is_file() and str(resolved).startswith(str(root.resolve())):
                    entry["resolves_to"] = str(resolved.relative_to(root.resolve()))
                    entry["bytes"] = resolved.stat().st_size
                    entry["sha256"] = sha256_file(resolved)
                entries.append(entry)
                continue
            if not path.is_file():
                raise SystemExit("non-regular file in pinned tree: " + str(path))
            entries.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return entries


def aggregate(entries):
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: item["path"]):
        token = entry.get("sha256", "")
        if "symlink_to" in entry:
            token = "symlink:" + entry["symlink_to"] + ":" + token
        digest.update(entry["path"].encode() + b"\0" + token.encode() + b"\0")
    return digest.hexdigest(), len(entries), sum(entry.get("bytes", 0) for entry in entries)


def pin_tree(root: Path, role: str, acquisition: dict, skip_names=()) -> dict:
    entries = walk(root, skip_names=skip_names)
    aggregate_sha256, count, size = aggregate(entries)
    return {
        "role": role,
        "acquisition": acquisition,
        "file_count": count,
        "byte_count": size,
        "aggregate_sha256": aggregate_sha256,
        "files": entries,
    }


def parse_sha256sums(path: Path) -> dict:
    """Read `<hex>  <name>` lines into name -> hex. Used for large weight trees."""
    mapping = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        digest, name = parts[0], parts[-1].lstrip("*")
        mapping[str(Path(name))] = digest
    return mapping


def fragment_reference(root: Path) -> dict:
    return {
        "reference": pin_tree(
            root,
            "REFERENCE_ROOT",
            {
                "kind": "local_readonly_export",
                "vcs": None,
                "public_pin": "blocked",
                "public_pin_reason": "exported tree carries no verifiable upstream revision; the public acquisition path is unresolved, so this is a local byte pin only",
            },
        )
    }


def fragment_assets(asset_root: Path) -> dict:
    assets = {}
    base = asset_root / "assets" / "pi05_base"
    if base.is_dir():
        sums = base / "SHA256SUMS.params"
        params = base / "params"
        entry = {
            "role": "ASSET_ROOT/assets/pi05_base",
            "acquisition": {
                "kind": "local_readonly_copy",
                "public_pin": "blocked",
                "public_pin_reason": "weight tree obtained outside a verifiable public release; redistribution terms not established",
            },
        }
        if sums.is_file() and params.is_dir():
            declared = parse_sha256sums(sums)
            on_disk = []
            for path in sorted(params.rglob("*")):
                if path.is_file():
                    on_disk.append(path)
            verified = []
            mismatched = []
            for path in on_disk:
                relative = str(path.relative_to(params))
                actual = sha256_file(path)
                expected = declared.get(relative)
                verified.append({"path": relative, "bytes": path.stat().st_size, "sha256": actual})
                if expected is not None and expected != actual:
                    mismatched.append({"path": relative, "declared": expected, "actual": actual})
            entry.update(
                {
                    "declared_manifest": {"file": "SHA256SUMS.params", "sha256": sha256_file(sums), "entries": len(declared)},
                    "params": {
                        "file_count": len(verified),
                        "byte_count": sum(item["bytes"] for item in verified),
                        "aggregate_sha256": aggregate(verified)[0],
                        "files": verified,
                    },
                    "declared_mismatches": mismatched,
                    "verification": "every parameter file re-hashed from bytes and compared with the declared manifest",
                }
            )
        else:
            entry["status"] = "blocked"
            entry["reason"] = "pi05_base present without params/ and SHA256SUMS.params"
        other = [p for p in sorted(base.iterdir()) if p.name not in {"params", "SHA256SUMS.params"}]
        entry["auxiliary"] = [
            {"path": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}
            for p in other
            if p.is_file()
        ]
        assets["pi05_base"] = entry

    tokenizers = asset_root / "tokenizers"
    if tokenizers.is_dir():
        assets["tokenizers"] = pin_tree(
            tokenizers,
            "ASSET_ROOT/tokenizers",
            {
                "kind": "local_readonly_copy",
                "public_pin": "blocked",
                "public_pin_reason": "tokenizer files exported from a Hub model without a pinned upstream revision",
            },
        )
    return {"assets": assets}


def fragment_openpi(openpi_root: Path, subpaths=("src/openpi/models", "src/openpi/models_pytorch", "src/openpi/policies", "pyproject.toml")) -> dict:
    if not openpi_root.is_dir():
        return {
            "openpi": {
                "status": "blocked",
                "reason": "OPENPI_ROOT not present on this host",
                "acquisition": {"kind": "absent_here", "public_pin": "blocked"},
            }
        }
    pinned = {}
    for sub in subpaths:
        target = openpi_root / sub
        if target.is_file():
            pinned[sub] = {"bytes": target.stat().st_size, "sha256": sha256_file(target)}
        elif target.is_dir():
            # Bytecode caches are generated artefacts of whoever imported the tree;
            # they are excluded and the exclusion is recorded rather than hidden.
            entries = [item for item in walk(target) if "__pycache__" not in item["path"]]
            aggregate_sha256, count, size = aggregate(entries)
            pinned[sub] = {
                "file_count": count,
                "byte_count": size,
                "aggregate_sha256": aggregate_sha256,
                "excluded": ["__pycache__"],
            }
        else:
            pinned[sub] = {"status": "missing"}
    return {
        "openpi": {
            "status": "pinned_subset",
            "acquisition": {
                "kind": "local_readonly_copy",
                "public_pin": "blocked",
                "public_pin_reason": "the public OpenPI acquisition path is unresolved; only the local fixed snapshot is described",
            },
            "subpaths": pinned,
        }
    }


def merge(parts, output: Path):
    merged = {"schema": SCHEMA, "generated_utc": datetime.now(timezone.utc).isoformat()}
    for part in parts:
        data = json.loads(Path(part).read_text(encoding="utf-8"))
        for key, value in data.items():
            if key == "schema":
                continue
            merged[key] = value
    output.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote " + str(output))


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference")
    parser.add_argument("--assets")
    parser.add_argument("--openpi")
    parser.add_argument("--out", required=True)
    parser.add_argument("--merge", nargs="+")
    args = parser.parse_args(argv)

    if args.merge:
        merge(args.merge, LOCK if args.out == "-" else Path(args.out))
        return 0

    fragment = {}
    if args.reference:
        fragment.update(fragment_reference(Path(args.reference)))
    if args.assets:
        fragment.update(fragment_assets(Path(args.assets)))
    if args.openpi:
        fragment.update(fragment_openpi(Path(args.openpi)))
    if not fragment:
        raise SystemExit("nothing to pin: pass --reference, --assets or --openpi")
    Path(args.out).write_text(json.dumps(fragment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote fragment " + args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
