#!/usr/bin/env python3
"""Serve one all-49 bowl-to-plate LoRA checkpoint over the OpenPI protocol."""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bowl_plate_49  # noqa: E402


class ReadyWebsocketPolicyServer:
    """Bind the socket before announcing readiness."""

    def __init__(self, policy, *, host: str, port: int, metadata: dict) -> None:
        from openpi.serving import websocket_policy_server

        self._base = websocket_policy_server.WebsocketPolicyServer(
            policy=policy,
            host=host,
            port=port,
            metadata=metadata,
        )
        self._host = host
        self._port = port
        self._ready = metadata

    def serve_forever(self) -> None:
        from openpi.serving import websocket_policy_server as server_module
        import websockets.asyncio.server as server_api

        async def run() -> None:
            async with server_api.serve(
                self._base._handler,
                self._host,
                self._port,
                compression=None,
                max_size=None,
                process_request=server_module._health_check,
            ):
                print(
                    "POLICY_SERVER_READY "
                    + json.dumps(self._ready, sort_keys=True),
                    flush=True,
                )
                await asyncio.Event().wait()

        asyncio.run(run())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    from openpi.policies import policy_config
    from openpi.training import checkpoints

    logging.basicConfig(level=logging.INFO, force=True)
    protocol = bowl_plate_49.load_protocol(
        args.protocol, args.expected_protocol_sha256)
    view = bowl_plate_49.load_view(args.view, protocol)
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint step directory is missing: {checkpoint}")
    if checkpoint.name != str(args.expected_step):
        raise ValueError("checkpoint directory does not match expected step")
    if not (checkpoint / "params").is_dir():
        raise ValueError("checkpoint has no inference params directory")
    assets = checkpoint / "assets" / "libero_goal8"
    norm_path = assets / "norm_stats.json"
    if not norm_path.is_file():
        raise FileNotFoundError("checkpoint does not contain saved Norm Stats")
    norm_sha = bowl_plate_49.sha256_file(norm_path)
    if norm_sha != protocol["data"]["norm_stats_sha256"]:
        raise ValueError("checkpoint Norm Stats differ from the training protocol")

    config_args = SimpleNamespace(
        exp_name="bowl_plate_49_lora",
        mode="probe",
        view=str(args.view),
        base_checkpoint="/not-used-for-serving",
        batch_size=8,
        fsdp_devices=1,
    )
    config = bowl_plate_49.make_config(
        config_args, protocol, view, checkpoint.parent.parent, args.expected_step)
    norm_stats = checkpoints.load_norm_stats(
        checkpoint / "assets", "libero_goal8")
    metadata = {
        "model_id": "pi05_libero_bowl_plate_49_lora",
        "checkpoint_step": int(args.expected_step),
        "base_checkpoint_tree_sha256": protocol["base"]["tree_sha256"],
        "protocol_sha256": protocol["_sha256"],
        "view_sha256": view["view_sha256"],
        "norm_stats_sha256": norm_sha,
        "suite": protocol["suite"],
        "task_id": protocol["task_id"],
        "task": protocol["task"],
        "action_dim": protocol["data"]["action_dim"],
        "action_horizon": protocol["data"]["action_horizon"],
        "action_semantics": protocol["data"]["action_semantics"],
        "state_dim": protocol["data"]["state_dim"],
        "training_episodes": protocol["data"]["train_episodes"],
        "training_frames": protocol["data"]["train_frames"],
        "eval_only": True,
    }
    config = dataclasses.replace(config, policy_metadata=metadata)
    inference_policy = policy_config.create_trained_policy(
        config,
        checkpoint,
        default_prompt=protocol["task"],
        norm_stats=norm_stats,
    )
    server = ReadyWebsocketPolicyServer(
        policy=inference_policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
