from __future__ import annotations

import ast
import argparse
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.so101.model_scope import resolve_model_scope
from scripts.so101 import e4t_launch_training, e4t_run_training


ROOT = Path(__file__).resolve().parents[1]


class ModelScopeTest(unittest.TestCase):
    def test_historical_snapshot_recovers_lora_and_frozen_vision(self):
        snapshot = {
            "config": {
                "model": {
                    "paligemma_variant": "gemma_2b_lora",
                    "action_expert_variant": "gemma_300m_lora",
                },
                "freeze_filter": (
                    "All(Any(PathRegex('.*llm.*'), "
                    "PathRegex('.*PaliGemma/img.*')), Not(PathRegex('.*lora.*'))))"
                ),
            }
        }
        self.assertEqual(resolve_model_scope(snapshot), {
            "paligemma_variant": "gemma_2b_lora",
            "action_expert_variant": "gemma_300m_lora",
            "freeze_vision_encoder": True,
        })

    def test_old_protocol_is_a_fallback_for_small_historical_run_config(self):
        run_config = {"phase": "E4-T", "state_mode": "on"}
        protocol = json.loads((ROOT / "configs/so101/e4t_training.json").read_text())
        self.assertEqual(resolve_model_scope(run_config, protocol), {
            "paligemma_variant": "gemma_2b_lora",
            "action_expert_variant": "gemma_300m_lora",
            "freeze_vision_encoder": True,
        })

    def test_clean60_protocol_selects_full_action_and_trainable_vision(self):
        protocol = json.loads((ROOT / "configs/so101/clean60_training.json").read_text())
        self.assertEqual(resolve_model_scope({}, protocol), {
            "paligemma_variant": "gemma_2b_lora",
            "action_expert_variant": "gemma_300m",
            "freeze_vision_encoder": False,
        })

    def test_explicit_scope_in_run_config_overrides_protocol(self):
        run_config = {"model_scope": {
            "paligemma_variant": "gemma_2b",
            "action_expert_variant": "gemma_300m_lora",
            "freeze_vision_encoder": True,
        }}
        self.assertEqual(resolve_model_scope(run_config, {
            "initialisation": {
                "paligemma_variant": "gemma_2b_lora",
                "action_expert_variant": "gemma_300m",
                "freeze_rule": "SigLIP image tower fully trainable",
            }
        }), {
            "paligemma_variant": "gemma_2b",
            "action_expert_variant": "gemma_300m_lora",
            "freeze_vision_encoder": True,
        })

    def test_snapshot_alone_reads_freeze_filter_even_when_model_is_present(self):
        snapshot = {"config": {
            "model": {
                "paligemma_variant": "gemma_2b_lora",
                "action_expert_variant": "gemma_300m_lora",
            },
            "freeze_filter": "All(PathRegex(pattern=re.compile('.*llm.*'), sep='/'))",
        }}
        self.assertEqual(resolve_model_scope(snapshot), {
            "paligemma_variant": "gemma_2b_lora",
            "action_expert_variant": "gemma_300m_lora",
            "freeze_vision_encoder": False,
        })

    def test_saved_snapshot_vision_setting_overrides_old_protocol(self):
        snapshot = {"config": {
            "model": {
                "paligemma_variant": "gemma_2b_lora",
                "action_expert_variant": "gemma_300m_lora",
            },
            "freeze_filter": "All(PathRegex(pattern=re.compile('.*llm.*'), sep='/'))",
        }}
        old_protocol = json.loads((ROOT / "configs/so101/e4t_training.json").read_text())
        self.assertTrue(resolve_model_scope({}, old_protocol)["freeze_vision_encoder"])
        self.assertFalse(resolve_model_scope(snapshot, old_protocol)["freeze_vision_encoder"])

    def test_complete_explicit_scope_skips_malformed_lower_priority_fallback(self):
        saved_scope = {"model_scope": {
            "paligemma_variant": "gemma_2b_lora",
            "action_expert_variant": "gemma_300m",
            "freeze_vision_encoder": False,
        }}
        malformed_protocol = {"initialisation": {
            "freeze_rule": "scope unknown from this unrecognized description",
        }}
        self.assertEqual(resolve_model_scope(saved_scope, malformed_protocol), {
            "paligemma_variant": "gemma_2b_lora",
            "action_expert_variant": "gemma_300m",
            "freeze_vision_encoder": False,
        })

    def test_protocol_freeze_flag_reaches_the_generated_train_command(self):
        cases = (
            ("e4t_training.json", "--freeze-vision-encoder", True),
            ("clean60_training.json", "--no-freeze-vision-encoder", False),
        )
        for filename, expected_flag, expected_value in cases:
            with self.subTest(protocol=filename):
                protocol = json.loads((ROOT / "configs/so101" / filename).read_text())
                launcher_scope = e4t_launch_training.model_scope_cli_args(protocol)

                # Parse exactly the scope flags accepted by the real runner CLI.
                parser = argparse.ArgumentParser()
                e4t_run_training.add_model_scope_arguments(parser)
                parsed_scope = parser.parse_args(launcher_scope)
                self.assertIs(parsed_scope.freeze_vision_encoder, expected_value)

                args = SimpleNamespace(
                    **vars(parsed_scope), mode="initial", exp_name="scope-test",
                    state_mode="on", batch_size=8, steps=25, save_interval=5,
                    data_loader_workers=0, lr_warmup_steps=1, peak_lr=5e-5,
                    decay_steps=25, decay_lr=5e-6, fsdp_devices=1,
                    seed=42, action_horizon=10, optimizer="AdamW",
                    gradient_clip_norm=1.0, ema_decay=None, dtype="bfloat16",
                    run_id="scope-test-run", training_config_sha256="a" * 64,
                    expected_groups=1, sampling_plan=None, data_admission=None,
                    dataset_label="scope-test", expected_total_episodes=60,
                    expected_train_episodes=48, expected_validation_episodes=12,
                )
                command = e4t_run_training.build_training_command(
                    args, "/work/runs/scope-test", "/work/views/scope-test")
                self.assertIn(expected_flag, command)
                self.assertNotIn(
                    "--no-freeze-vision-encoder" if expected_value else "--freeze-vision-encoder",
                    command,
                )
                self.assertEqual(command[command.index("--paligemma-variant") + 1],
                                 parsed_scope.paligemma_variant)
                self.assertEqual(command[command.index("--action-expert-variant") + 1],
                                 parsed_scope.action_expert_variant)

    def test_protocol_defaults_keep_legacy_e4t_fsdp_and_registered_clean60_topology(self):
        old_protocol = json.loads((ROOT / "configs/so101/e4t_training.json").read_text())
        clean60 = json.loads((ROOT / "configs/so101/clean60_training.json").read_text())
        self.assertEqual(e4t_launch_training.default_fsdp_devices(old_protocol, 4), 1)
        self.assertEqual(e4t_launch_training.default_fsdp_devices(clean60, 4), 4)

    def test_corrupt_explicit_scope_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_model_scope({"model_scope": {"paligemma_variant": "unknown"}})

    def test_serving_and_evaluation_forward_resolved_scope(self):
        for filename in ("serve_so101.py", "evaluate_e4t_openloop.py"):
            tree = ast.parse((ROOT / "scripts/so101" / filename).read_text())
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute)
                     and node.func.attr == "make_train_config"]
            self.assertTrue(calls, filename)
            self.assertTrue(any(
                any(keyword.arg is None and isinstance(keyword.value, ast.Name)
                    and keyword.value.id == "scope" for keyword in call.keywords)
                for call in calls), filename)
            self.assertTrue(any(isinstance(node, ast.Call)
                                and isinstance(node.func, ast.Attribute)
                                and node.func.attr == "resolve_model_scope"
                                for node in ast.walk(tree)), filename)


if __name__ == "__main__":
    unittest.main()
