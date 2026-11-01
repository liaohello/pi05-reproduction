from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/so101/scene1_pifast_ki_training_v2.json"


class Scene1PiFastPlanV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = json.loads(PLAN.read_text(encoding="utf-8"))

    def test_version_status_and_review_only_state(self) -> None:
        self.assertEqual(self.plan["plan_id"], "scene1-pifast-ki-v2")
        self.assertEqual(self.plan["status"], "ready_for_training")
        self.assertTrue(self.plan["execution_authorized"])
        self.assertFalse(self.plan["formal_training_started"])
        self.assertEqual(
            self.plan["supersedes"],
            "configs/so101/scene1_pifast_ki_training_v1.json",
        )
        self.assertTrue(self.plan["implementation_state"]["runtime_implemented"])
        self.assertTrue(
            self.plan["implementation_state"]["runtime_integration_verified"])
        self.assertTrue(
            self.plan["implementation_state"]["runtime_parameter_tree_verified"])
        self.assertTrue(
            self.plan["implementation_state"]["flow12_loss_executed"])
        self.assertTrue(
            self.plan["implementation_state"]["ki_teacher_isolation_verified"])
        self.assertTrue(
            self.plan["implementation_state"]["ki_gradient_matrix_verified"])
        self.assertTrue(
            self.plan["implementation_state"]["joint_optimizer_update_verified"])
        self.assertTrue(
            self.plan["implementation_state"][
                "checkpoint_restore_continuation_verified"])
        patch = self.plan["implementation_patch"]
        patch_path = ROOT / patch["path"]
        self.assertFalse(patch["applied_to_shared_source"])
        self.assertTrue(patch["applied_to_independent_cpu_overlay"])
        self.assertEqual(
            hashlib.sha256(patch_path.read_bytes()).hexdigest(),
            patch["sha256"],
        )

    def test_scope_uses_action_expert_rank64_and_freezes_non_lora(self) -> None:
        scope = self.plan["model_scope"]
        self.assertEqual(scope["language"]["rank"], 16)
        self.assertEqual(scope["language"]["alpha"], 16.0)
        self.assertEqual(scope["action_expert"]["rank"], 64)
        self.assertEqual(scope["action_expert"]["alpha"], 64.0)
        self.assertEqual(scope["action_expert"]["non_lora"], "frozen")
        self.assertEqual(scope["vision_encoder"], "full_trainable")
        self.assertEqual(scope["tied_input_embedding"]["state"], "frozen")
        self.assertFalse(
            scope["legacy_parameter_counts"]["valid_for_v2"])
        self.assertEqual(scope["v2_parameter_counts"]["total"], 3429734160)
        self.assertEqual(scope["v2_parameter_counts"]["trainable"], 493269776)
        self.assertEqual(scope["v2_parameter_counts"]["frozen"], 2936464384)
        self.assertEqual(scope["v2_parameter_counts"]["optimizer_tensors"], 106)
        self.assertEqual(
            scope["v2_parameter_counts"]["optimizer_elements"], 986539554)

    def test_combined_norm_and_twelve_dimension_loss(self) -> None:
        norm = self.plan["normalization"]
        interface = self.plan["interface"]
        self.assertEqual(norm["grouping"], "combined_train_red_blue_single_group")
        self.assertTrue(norm["state_and_action_separate"])
        self.assertTrue(norm["red_blue_use_same_stats"])
        self.assertFalse(norm["overlapping_chunks_reused_for_stats"])
        self.assertEqual(interface["internal_action_dim"], 32)
        self.assertEqual(interface["flow_loss_dimensions"], 12)
        self.assertFalse(interface["padded_dimensions_in_flow_loss"])
        self.assertEqual(interface["state_text_dimensions"], 12)
        self.assertEqual(interface["action_horizon"], 16)
        self.assertEqual(interface["n_action_steps"], 16)
        fast = self.plan["fast"]
        self.assertEqual(fast["fast_token_id_range"], [254976, 257024])
        self.assertTrue(fast["residual_shared_between_embedding_and_output"])
        self.assertEqual(fast["residual_scale"], 1.0)
        self.assertEqual(fast["vocabulary_size"], 2048)
        self.assertEqual(fast["dct_scale"], 10)
        self.assertEqual(fast["min_token"], -354)
        self.assertEqual(fast["padded_language_vocabulary"], 257152)
        self.assertEqual(fast["language_pad_id"], 0)
        self.assertEqual(fast["language_eos_id"], 1)
        self.assertEqual(fast["max_token_len"], 128)

    def test_ki_and_explicit_fast_weight_contract(self) -> None:
        ki = self.plan["knowledge_insulation"]
        self.assertTrue(ki["enabled"])
        self.assertFalse(ki["flow_prefix_contains_teacher_forced_action_suffix"])
        self.assertFalse(ki["fast_and_flow_action_tokens_cross_attend"])
        self.assertTrue(ki["stop_gradient_flow_condition_keys"])
        self.assertTrue(ki["stop_gradient_flow_condition_values"])
        loss = self.plan["loss"]
        self.assertEqual(loss["flow_weight"], 1.0)
        self.assertEqual(loss["fast_ce_weight"], 0.018)
        self.assertFalse(loss["implicit_default_allowed"])
        self.assertTrue(loss["formal_weights_frozen"])

    def test_training_schedule_and_saved_steps(self) -> None:
        optimization = self.plan["optimization"]
        checkpointing = self.plan["checkpointing"]
        self.assertEqual(optimization["optimizer_updates"], 40000)
        self.assertEqual(optimization["global_batch_size"], 8)
        self.assertEqual(optimization["per_device_batch"], 2)
        self.assertEqual(self.plan["runtime"]["device_count"], 4)
        self.assertEqual(self.plan["runtime"]["fsdp_devices"], 4)
        self.assertEqual(self.plan["runtime"]["data_parallel_replicas"], 1)
        self.assertEqual(optimization["warmup_updates"], 1000)
        self.assertEqual(optimization["peak_lr"], 2.5e-5)
        self.assertEqual(optimization["final_lr"], 2.5e-6)
        self.assertFalse(optimization["ema_enabled"])
        self.assertEqual(
            checkpointing["save_steps"], list(range(2000, 40001, 2000)))
        self.assertTrue(checkpointing["retain_all"])

    def test_validation_uses_valid_tail_mask_and_fixed_metrics(self) -> None:
        validation = self.plan["validation"]
        self.assertEqual(validation["full_samples"], 8228)
        self.assertEqual(validation["full_batches"], 515)
        self.assertEqual(validation["full_last_batch_valid_samples"], 4)
        self.assertEqual(validation["full_padding_rows"], 12)
        self.assertTrue(validation["train_false"])
        self.assertEqual(validation["gradient_updates"], 0)
        self.assertTrue(validation["fixed_flow_time_and_noise"])

    def test_resource_values_are_not_misrepresented(self) -> None:
        resource = self.plan["resource"]
        self.assertEqual(resource["minimum_memory_margin_mib"], 100)
        self.assertFalse(
            resource["margin_conflict"]["formal_gpu_execution_started"])
        self.assertEqual(resource["wall_limit_s"], 432000)

if __name__ == "__main__":
    unittest.main()
