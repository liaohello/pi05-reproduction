"""Shared bindings for the SO101 E0-E3 engineering chain.

Everything here is deliberately path-agnostic: the external checkout and the
run directory come from environment variables or CLI arguments, and no private
absolute path is written into a tracked file.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = "configs/profiles/collection_50hz_three_camera.yaml"
COLLECTION_REPO_ID = "local/so101_pi05_e1_20"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value, *, sort_keys: bool = True) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=sort_keys, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def sha256_tree(root: Path, pattern: str = "*") -> dict:
    root = Path(root)
    digest = hashlib.sha256()
    files = 0
    total = 0
    for path in sorted(p for p in root.rglob(pattern) if p.is_file()):
        data = path.read_bytes()
        files += 1
        total += len(data)
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(data).hexdigest().encode())
        digest.update(b"\n")
    return {"files": files, "bytes": total, "sha256": digest.hexdigest()}


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8")


def setup_external(root: str | Path) -> Path:
    """Bind the external SO101 checkout before importing its modules."""
    root = Path(root).expanduser().resolve()
    if not (root / "configs/robot.yaml").is_file():
        raise FileNotFoundError(f"external SO101 checkout not found at {root}")
    os.environ["BIMANUAL_ROOT"] = str(root)
    os.environ.setdefault("BIMANUAL_EGL_DEVICE_ID", os.environ.get("SO101_EGL_DEVICE", "2"))
    os.environ.setdefault("MUJOCO_GL", "egl")
    source = str(root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    return root


def apply_collection_protocol(config: dict, protocol: dict) -> dict:
    """Return a config whose collection block follows configs/so101/protocol.json."""
    config = copy.deepcopy(config)
    collection = copy.deepcopy(config.get("collection") or {})
    settings = protocol["collection"]
    collection.update({
        "schema_version": 1,
        "protocol": "so101_pi05_e1_v1",
        "direct_seeds": list(settings["direct_seed_range"]),
        "target_successes_per_profile": int(settings["target_successes"]),
        "maximum_attempts_per_profile": int(settings["maximum_attempts"]),
        "fps": int(protocol["control"]["recording_fps"]),
        "repo_id": COLLECTION_REPO_ID,
        "diagnostics": bool(settings["diagnostics"]),
        "pilot_seeds": [],
        "pair_stride": 0,
        "pair_fraction": 0.0,
    })
    config["collection"] = collection
    config["recording"]["fps"] = int(protocol["control"]["recording_fps"])
    config["runtime"]["control_hz"] = int(protocol["control"]["control_hz"])
    config["recording"]["camera_keys"] = list(protocol["observation"]["cameras"])
    return config


def load_external_config(root: str | Path, protocol: dict) -> dict:
    root = setup_external(root)
    from bimanual.config import load_config

    config = load_config(root=root, profile=DEFAULT_PROFILE)
    return apply_collection_protocol(config, protocol)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    if total <= 0:
        return None
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [round(max(0.0, centre - half), 6), round(min(1.0, centre + half), 6)]
