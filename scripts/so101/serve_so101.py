#!/usr/bin/env python3
"""E3 policy server: serve the E2 SO101 π0.5 checkpoint over OpenPI's wire protocol.

Runs inside the pinned openpi-server image.  The raw request is the SO101
contract (three uint8 RGB cameras, 12-D state, prompt); the server applies the
same SO101 transforms and train-only normalisation used during training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

import jax
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import openpi_so101  # noqa: E402
sys.path.insert(0, str(HERE))
import so101_common as common  # noqa: E402
import openpi.training.checkpoints as _checkpoints  # noqa: E402
from openpi.policies import policy_config as _policy_config  # noqa: E402
from openpi.serving import websocket_policy_server  # noqa: E402


class ResettablePolicy:
    """OpenPI Policy with an explicit episode reset for deterministic rollouts."""

    def __init__(self, policy, *, server_metadata: dict):
        self._policy = policy
        self._server_metadata = server_metadata

    def infer(self, obs: dict) -> dict:
        if obs.get("__reset__"):
            seed = int(obs.get("seed", 0))
            self._policy._rng = jax.random.key(seed)
            return {"reset": True, "seed": seed}
        return self._policy.infer(obs)

    @property
    def metadata(self) -> dict:
        return {**self._policy.metadata, **self._server_metadata}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Orbax step directory containing params/ and assets/")
    parser.add_argument("--train-view", type=Path, default=None)
    parser.add_argument("--state-mode", choices=("off", "on"), default=None)
    parser.add_argument("--run-config", type=Path, default=None,
                        help="saved training run config; state mode and train view come from it")
    parser.add_argument("--skip-metadata-write", action="store_true",
                        help="serve without writing a sidecar into the checkpoint directory")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    checkpoint = args.checkpoint.expanduser().resolve()
    run_config = {}
    if args.run_config is not None:
        run_config = common.load_json(args.run_config)
    state_mode = run_config.get("state_mode", args.state_mode or "off")
    if args.state_mode is not None and args.state_mode != state_mode:
        raise SystemExit("explicit --state-mode differs from the saved run config")
    train_view_value = run_config.get("train_view") or args.train_view
    if train_view_value is None:
        raise SystemExit("provide --train-view or --run-config")
    train_view = Path(train_view_value).expanduser().resolve()
    exp_name = str(run_config.get("exp_name", "serve"))
    config = openpi_so101.make_train_config(
        mode="serve", run_root=checkpoint.parent, train_view=train_view,
        base_checkpoint=checkpoint, exp_name=exp_name, state_mode=state_mode,
    )
    data_config = config.data.create(config.assets_dirs, config.model)
    norm_stats = _checkpoints.load_norm_stats(checkpoint / "assets", data_config.asset_id)
    norm_path = checkpoint / "assets" / str(data_config.asset_id) / "norm_stats.json"
    norm_file_sha = hashlib.sha256(norm_path.read_bytes()).hexdigest()
    canonical_payload = {
        split: {field: np.asarray(getattr(stats, field)).tolist()
                for field in ("mean", "std", "q01", "q99")}
        for split, stats in norm_stats.items()
    }
    norm_sha = hashlib.sha256(
        json.dumps(canonical_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    policy = _policy_config.create_trained_policy(
        config, checkpoint, norm_stats=norm_stats,
        default_prompt=openpi_so101.TASK,
    )
    metadata = {
        "model": "pi05_so101_lora",
        "checkpoint": str(checkpoint),
        "checkpoint_name": checkpoint.name,
        "norm_stats_sha256": norm_sha,
        "norm_stats_file_sha256": norm_file_sha,
        "action_dim": openpi_so101.ACTION_DIM,
        "action_horizon": 10,
        "control_hz": 50,
        "state_key": "observation.state",
        "cameras": list(openpi_so101.CAMERAS.values()),
        "engineering_chain_only": True,
        "state_mode": state_mode,
        "run_config": str(args.run_config) if args.run_config else None,
        "run_config_sha256": (hashlib.sha256(Path(args.run_config).read_bytes()).hexdigest()
                              if args.run_config else None),
        "expected_config_sha256": run_config.get("config_sha256"),
        "expected_norm_stats_canonical_sha256": run_config.get("norm_stats_canonical_sha256"),
    }
    wrapped = ResettablePolicy(policy, server_metadata=metadata)
    if not args.skip_metadata_write:
        (checkpoint / "server_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    logging.info("serving SO101 policy on %s:%s", args.host, args.port)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=wrapped, host=args.host, port=args.port, metadata=metadata,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
