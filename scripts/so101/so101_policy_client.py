"""Pi0.5 SO101 adapter for the external PolicyRunner.

Runs in the external SO101 Python environment.  The wire format matches
OpenPI's ``msgpack_numpy`` protocol; actions are requested as a 10-step chunk
and one absolute 12-D target is returned per 50 Hz control frame.

The ``_pack_array``/``_unpack_array`` wire helpers are adapted from OpenPI's
``openpi_client.msgpack_numpy`` (Apache-2.0); see ``THIRD_PARTY_LICENSES.md``.
"""
from __future__ import annotations

import time
from pathlib import Path

import msgpack
import numpy as np
import websockets.sync.client

from bimanual.types import FEATURE_NAMES


def _pack_array(obj):
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"unsupported dtype {obj.dtype}")
        return {b"__ndarray__": True, b"data": obj.tobytes(),
                b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]),
                          shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


class SO101PolicyClient:
    """PolicyRunner adapter; never steps the environment and never clips."""

    action_steps = 10
    camera_features = {
        "observation.images.central": (3, 480, 640),
        "observation.images.left_wrist": (3, 480, 640),
        "observation.images.right_wrist": (3, 480, 640),
    }

    def __init__(self, host: str, port: int, *, instruction: str,
                 timeout_s: float = 120.0, connect_timeout_s: float = 15.0):
        self.uri = f"ws://{host}:{port}"
        self.instruction = instruction
        self.timeout_s = float(timeout_s)
        self._packer = msgpack.Packer(default=_pack_array)
        self._connection = ws = websockets.sync.client.connect(
            self.uri, compression=None, max_size=None,
            open_timeout=float(connect_timeout_s))
        self.metadata = msgpack.unpackb(ws.recv(timeout=float(connect_timeout_s)),
                                        object_hook=_unpack_array)
        self._queue: list[np.ndarray] = []
        self.requests = 0
        self.inference_ms: list[float] = []
        self.last_telemetry: dict | None = None
        self.feature_names = list(FEATURE_NAMES)
        self.reset_seed = 0

    def reset(self) -> None:
        seed = int(self.reset_seed)
        self._connection.send(self._packer.pack({"__reset__": True, "seed": seed}))
        response = msgpack.unpackb(self._connection.recv(timeout=self.timeout_s),
                                   object_hook=_unpack_array)
        if not response.get("reset"):
            raise RuntimeError(f"policy server did not acknowledge reset: {response}")
        self._queue.clear()
        self.last_telemetry = None

    def _observation(self, observation: dict) -> dict:
        state = np.asarray(observation["observation.state"], dtype=np.float32)
        if state.shape != (12,) or not np.isfinite(state).all():
            raise ValueError("SO101 observation.state must be 12 finite values")
        payload = {"observation.state": state, "prompt": self.instruction}
        for key in self.camera_features:
            image = observation.get(key)
            if image is None:
                raise ValueError(f"SO101 observation is missing {key}")
            image = np.asarray(image)
            if image.dtype != np.uint8 or image.shape != (480, 640, 3):
                raise ValueError(f"{key} must be uint8 HWC 480x640x3, got {image.dtype} {image.shape}")
            payload[key] = image
        return payload

    def act(self, observation: dict) -> np.ndarray:
        if not self._queue:
            started = time.perf_counter()
            self._connection.send(self._packer.pack(self._observation(observation)))
            response = msgpack.unpackb(self._connection.recv(timeout=self.timeout_s),
                                       object_hook=_unpack_array)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            actions = np.asarray(response.get("actions"), dtype=np.float32)
            if actions.ndim != 2 or actions.shape[0] < self.action_steps or actions.shape[1] != 12:
                raise ValueError(f"policy returned actions with shape {actions.shape}, expected (>=10,12)")
            if not np.isfinite(actions).all():
                raise ValueError("policy returned non-finite actions")
            self.requests += 1
            self.inference_ms.append(elapsed_ms)
            self.last_telemetry = {
                "client_roundtrip_ms": elapsed_ms,
                "server_timing": response.get("server_timing"),
                "policy_timing": response.get("policy_timing"),
                "returned_horizon": int(actions.shape[0]),
                "executed_horizon": self.action_steps,
            }
            self._queue = [row.copy() for row in actions[:self.action_steps]]
        return self._queue.pop(0)

    def warmup(self, timeout_s: float = 600.0) -> dict:
        """Compile the server graph once with a zero observation; not an episode."""
        saved = self.timeout_s
        self.timeout_s = float(timeout_s)
        observation = {
            "observation.state": np.zeros(12, dtype=np.float32),
            **{key: np.zeros((480, 640, 3), dtype=np.uint8) for key in self.camera_features},
        }
        self._queue.clear()
        try:
            self.act(observation)
        finally:
            self.timeout_s = saved
            self._queue.clear()
        result = self.latency_summary()
        self.inference_ms.clear()
        self.requests = 0
        self.last_telemetry = None
        return result

    def latency_summary(self) -> dict:
        values = np.asarray(self.inference_ms, dtype=float)
        if not values.size:
            return {"calls": 0}
        return {"calls": int(values.size), "mean_ms": float(values.mean()),
                "p50_ms": float(np.percentile(values, 50)),
                "p95_ms": float(np.percentile(values, 95)),
                "max_ms": float(values.max())}

    def close(self) -> None:
        try:
            self._connection.close()
        except Exception:
            pass
