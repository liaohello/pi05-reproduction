"""Shared helpers for the migration gates.

Every path here is resolved from the private environment, never from a constant
in the tree. Anything that must reconcile with a pinned record recomputes the
bytes rather than trusting the record.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
MIGRATION_DIR = PROJECT / "migration"
CONTRACT_PATH = MIGRATION_DIR / "contract.json"
LOCK_PATH = MIGRATION_DIR / "migration.lock.json"

ROOT_VARS = (
    "REFERENCE_ROOT",
    "OPENPI_ROOT",
    "ASSET_ROOT",
    "FINEART_JAX_ROOT",
    "JAX_PYTHON",
    "REF_PYTHON",
)


class MissingRoot(AssertionError):
    pass


def roots() -> dict:
    """Private roots, or a hard failure. A missing root must never degrade to a skip."""
    values, missing = {}, []
    for name in ROOT_VARS:
        value = os.environ.get(name, "").strip()
        if value:
            values[name] = value
        else:
            missing.append(name)
    if missing:
        raise MissingRoot(
            "private configuration is incomplete; unset: " + ", ".join(missing)
        )
    return values


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def contract() -> dict:
    return read_json(CONTRACT_PATH)


def lock() -> dict:
    if not LOCK_PATH.exists():
        raise AssertionError("migration.lock.json is missing; G0 pins are not recorded")
    return read_json(LOCK_PATH)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def walk_tree(root: Path) -> list:
    """Independent re-implementation of the pinning walk.

    Kept deliberately separate from `migration/pin_sources.py` so that the pin and
    its verification are not the same code path. Semantics must stay identical:
    sorted relative paths, symlinks recorded with their target, regular files only.
    """
    root = Path(root).resolve()
    entries = []
    for current, directories, names in os.walk(str(root), followlinks=False):
        current_path = Path(current)
        directories.sort()
        for name in sorted(directories):
            if (current_path / name).is_symlink():
                raise AssertionError("symlink directory inside a pinned tree: " + str(current_path / name))
        for name in sorted(names):
            path = current_path / name
            relative = str(path.relative_to(root))
            if path.is_symlink():
                target = os.readlink(path)
                entry = {"path": relative, "symlink_to": target}
                resolved = path.resolve()
                if resolved.is_file() and str(resolved).startswith(str(root)):
                    entry["resolves_to"] = str(resolved.relative_to(root))
                    entry["bytes"] = resolved.stat().st_size
                    entry["sha256"] = sha256_file(resolved)
                entries.append(entry)
                continue
            if not path.is_file():
                raise AssertionError("non-regular file inside a pinned tree: " + relative)
            entries.append(
                {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
    return entries


def aggregate(entries) -> tuple:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: item["path"]):
        token = entry.get("sha256", "")
        if "symlink_to" in entry:
            token = "symlink:" + entry["symlink_to"] + ":" + token
        digest.update(entry["path"].encode() + b"\0" + token.encode() + b"\0")
    return digest.hexdigest(), len(entries), sum(entry.get("bytes", 0) for entry in entries)


def evidence_dir(stage: str) -> Path:
    """Evidence must live outside the candidate: the gate hashes the tree mid-run."""
    base = os.environ.get("FINEART_EVIDENCE_DIR", "").strip()
    root = Path(base) if base else PROJECT.parent / "fineart-jax-evidence"
    target = root / stage
    target.mkdir(parents=True, exist_ok=True)
    return target


def reference_env(roots: dict) -> dict:
    """Environment that lets a subprocess import the read-only reference tree.

    PYTHONPATH only: the reference is never installed, so the bytes that run are
    the bytes that were pinned.
    """
    return {
        "PYTHONPATH": str(Path(roots["REFERENCE_ROOT"]) / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }


def run_python(python: str, code: str, timeout: int = 1800, env_extra: dict | None = None) -> dict:
    """Run a snippet in another interpreter and parse the JSON object it prints.

    Real models load in separate processes: one framework resident at a time.
    """
    environment = dict(os.environ)
    environment.setdefault("PYTHONHASHSEED", "0")
    # Never write bytecode into a read-only reference tree.
    environment.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Deterministic cuBLAS needs a fixed workspace configuration.
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if env_extra:
        environment.update(env_extra)
    result = subprocess.run(
        [python, "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=environment,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            "subprocess failed (" + python + "), rc=" + str(result.returncode) + "\n"
            + "--- stdout ---\n" + result.stdout[-4000:]
            + "\n--- stderr ---\n" + result.stderr[-4000:]
        )
    payload = None
    for line in reversed(result.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            payload = json.loads(line)
            break
    if payload is None:
        raise AssertionError(
            "subprocess produced no JSON record\n--- stdout ---\n" + result.stdout[-4000:]
        )
    return payload


def pip_freeze(python: str) -> list:
    result = subprocess.run(
        [python, "-m", "pip", "freeze", "--disable-pip-version-check"],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError("pip freeze failed for " + python + ": " + result.stderr[-2000:])
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def freeze_map(freeze_lines) -> dict:
    """name (lowercased, normalised) -> version, for comparison across environments."""
    mapping = {}
    for line in freeze_lines:
        if " @ " in line:
            name, rest = line.split(" @ ", 1)
            mapping[name.lower()] = "@" + rest
            continue
        if "==" not in line:
            continue
        name, version = line.split("==", 1)
        mapping[name.lower()] = version
    return mapping


def assert_no_placeholders(value, path="contract"):
    """A locked contract may not carry TODO markers or null thresholds."""
    if isinstance(value, dict):
        for key, item in value.items():
            assert_no_placeholders(item, path + "." + str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_no_placeholders(item, path + "[" + str(index) + "]")
    elif isinstance(value, str):
        lowered = value.lower()
        for marker in ("todo", "tbd", "fixme", "xxx"):
            if marker in lowered:
                raise AssertionError("unlocked placeholder at " + path + ": " + value)
    elif value is None:
        raise AssertionError("null value at " + path + " in a locked contract")


def python_identity(python: str) -> dict:
    code = (
        "import json,sys,platform\n"
        "print(json.dumps({'version':platform.python_version(),'executable':sys.executable,"
        "'prefix':sys.prefix,'base_prefix':getattr(sys,'base_prefix',sys.prefix)}))\n"
    )
    result = subprocess.run(
        [python, "-c", code], capture_output=True, text=True, timeout=120, check=False
    )
    if result.returncode != 0:
        raise AssertionError("cannot run " + python + ": " + result.stderr[-2000:])
    return json.loads(result.stdout.strip().splitlines()[-1])


__all__ = [
    "PROJECT",
    "MIGRATION_DIR",
    "CONTRACT_PATH",
    "LOCK_PATH",
    "ROOT_VARS",
    "roots",
    "contract",
    "lock",
    "read_json",
    "sha256_file",
    "walk_tree",
    "aggregate",
    "evidence_dir",
    "run_python",
    "reference_env",
    "pip_freeze",
    "freeze_map",
    "assert_no_placeholders",
    "python_identity",
]
