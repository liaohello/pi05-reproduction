from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/so101/scene1_single_random_protocol.json"
TRAINING = ROOT / "configs/so101/scene1_training.json"
BASE_V2_TRAINING = ROOT / "configs/so101/scene1_training_base_v2.json"
PATCH = ROOT / "patches/so101_scene1_single_random.patch"
PREFLIGHT = ROOT / "experiments/so101/scene1-preflight"


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_protocol_binds_task_counts_and_split():
    protocol = _json(PROTOCOL)
    assert protocol["task"]["name"] == "single_block_place"
    assert protocol["task"]["active_arm_rule"]["red"] == "left"
    assert protocol["task"]["active_arm_rule"]["blue"] == "right"
    assert protocol["scene_layout"]["object_count"] == 1
    assert protocol["scene_layout"]["color_balance"]["red"] == 32
    assert protocol["scene_layout"]["color_balance"]["blue"] == 32
    assert protocol["scene_layout"]["block_yaw"]["domain_degrees"] == [0.0, 72.0]
    assert protocol["collection"]["formal"]["target_successes"] == 64
    assert protocol["collection"]["formal"]["train"] == 52
    assert protocol["collection"]["formal"]["validation"] == 12
    assert protocol["collection"]["formal"]["maximum_attempts"] == 128
    assert protocol["training"]["updates"] == 20000
    assert protocol["training"]["executed"] is False


def test_protocol_seed_blocks_are_disjoint():
    registry = _json(PROTOCOL)["collection"]["seed_registry"]
    pilot = registry["pilot"]
    collection = registry["collection"]
    evaluation = registry["evaluation_reserved"]
    assert pilot[0] < pilot[1] < collection[0] < collection[1] < evaluation[0] < evaluation[1]


def test_training_config_is_not_claiming_an_executed_run():
    config = _json(TRAINING)
    assert config["executed"] is False
    assert config["optimisation"]["planned_updates"] == 20000
    assert config["devices"]["global_batch_size"] == 8
    assert config["devices"]["fsdp_devices_candidate"] == 4
    assert config["initialisation"]["base_checkpoint"] == "official pi05_libero"
    assert "clean60 checkpoint" in _json(PROTOCOL)["training"]["initialisation"]["forbidden"]
    assert config["data"]["train"] == 52
    assert config["data"]["validation"] == 12


def test_pi05_base_v2_hyperparameters_are_bound():
    config = _json(BASE_V2_TRAINING)
    assert config["executed"] is False
    assert config["runtime_binding"]["base_checkpoint"] == "pi05_base"
    assert config["data_binding"]["view_sha256"] == (
        "3c4911b5dd59430288433de8edbff2de1ee88c39da186fe4d59a1c36310031e9")
    assert config["interface_contract"]["state_dim"] == 12
    assert config["interface_contract"]["action_dim"] == 12
    assert config["interface_contract"]["action_horizon"] == 10
    assert config["interface_contract"]["state_mode"] == "on"
    assert config["model_scope"]["vision_encoder"] == "frozen"
    assert config["model_scope"]["action_expert"] == "full"
    assert config["optimisation"]["optimizer"] == "AdamW"
    assert config["optimisation"]["global_batch_size"] == 8
    assert config["optimisation"]["updates"] == 25000
    assert config["optimisation"]["warmup_updates"] == 1000
    assert config["optimisation"]["ema_decay"] == 0.999
    assert config["devices"]["fsdp_devices"] == 4
    assert config["devices"]["minimum_memory_margin_mib"] == 10
    assert config["checkpointing"]["save_every_updates"] == 3000
    assert config["checkpointing"]["checkpoint_steps"][-1] == 25000


def test_patch_hash_matches_protocol():
    protocol = _json(PROTOCOL)
    assert PATCH.is_file()
    actual = hashlib.sha256(PATCH.read_bytes()).hexdigest()
    assert actual == protocol["source_binding"]["source_patch"]["sha256"]


@pytest.mark.skipif(not (PREFLIGHT / "data_audit.json").is_file(),
                    reason="preflight artifacts are local run evidence")
def test_preflight_artifacts_are_consistent():
    audit = _json(PREFLIGHT / "data_audit.json")
    train = _json(PREFLIGHT / "train_stats.json")
    norm = _json(PREFLIGHT / "action_norm_audit.json")
    assert audit["episodes"] == 64
    assert audit["train"] == 52
    assert audit["validation"] == 12
    assert audit["substeps_all_full"] is True
    assert audit["incomplete_frames"] == 0
    assert audit["max_state_error_vs_trace"] == 0.0
    assert audit["max_action_error_vs_trace"] == 0.0
    assert audit["camera_sync"]["passed"] is True
    assert audit["video_timestamps_aligned_across_cameras"] is True
    assert audit["contact_monitor_errors"] == 0
    assert audit["soft_contacts"] == 0
    assert train["constant_dimensions"] == {"state": [], "actions": []}
    assert norm["passed"] is True
    assert norm["loader_consistency"]["episodes_covered"] == 52
    assert norm["image_pipeline"]["passed"] is True
