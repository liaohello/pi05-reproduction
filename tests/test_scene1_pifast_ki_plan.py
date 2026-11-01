from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/so101/scene1_pifast_ki_training_v1.json"


class Scene1PiFastPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = json.loads(PLAN.read_text(encoding="utf-8"))

    def test_interface_and_training_budget(self) -> None:
        interface = self.plan["interface"]
        training = self.plan["optimization"]
        self.assertEqual(interface["action_dim"], 12)
        self.assertEqual(interface["internal_action_dim"], 32)
        self.assertEqual(interface["action_horizon"], 16)
        self.assertEqual(interface["n_action_steps"], 16)
        self.assertEqual(training["optimizer_updates"], 40000)
        self.assertEqual(training["global_batch_size"], 8)
        self.assertEqual(training["warmup_updates"], 1000)
        self.assertEqual(training["peak_lr"], 2.5e-5)
        self.assertEqual(training["final_lr"], 2.5e-6)
        self.assertFalse(training["ema_enabled"])
        self.assertIsNone(training["ema_decay"])

    def test_scope_requires_action_expert_lora_and_keeps_embedding_frozen(self) -> None:
        scope = self.plan["model_scope"]
        self.assertEqual(scope["gemma_2b"]["rank"], 16)
        self.assertEqual(scope["action_expert"]["rank"], 64)
        self.assertEqual(scope["action_expert"]["alpha"], 64.0)
        self.assertEqual(scope["action_expert"]["non_lora"], "frozen")
        self.assertEqual(scope["vision_encoder"], "full_trainable")
        self.assertEqual(scope["tied_input_embedding"]["state"], "frozen")
        self.assertIsNone(scope["parameter_counts_after_change"])

    def test_ki_and_unfrozen_loss_weight_are_explicit(self) -> None:
        ki = self.plan["knowledge_insulation"]
        self.assertTrue(ki["enabled"])
        self.assertFalse(ki["fast_and_flow_action_tokens_may_attend"])
        self.assertTrue(ki["stop_gradient_flow_prefix_keys"])
        self.assertTrue(ki["stop_gradient_flow_prefix_values"])
        self.assertFalse(self.plan["loss"]["formal_weights_frozen"])
        self.assertIsNone(self.plan["loss"]["fast_ce_weight"])

    def test_2000_step_saving_and_horizon16_sample_counts(self) -> None:
        checkpoints = self.plan["checkpointing"]
        self.assertEqual(checkpoints["save_interval_updates"], 2000)
        self.assertEqual(
            checkpoints["save_steps"],
            list(range(2000, 40001, 2000)),
        )
        data = self.plan["data"]
        self.assertEqual(data["train_windows_action_horizon_16"], 35843)
        self.assertEqual(data["validation_windows_action_horizon_16"], 8228)
        self.assertEqual(
            self.plan["validation_loss"]["full_validation_samples"], 8228)


if __name__ == "__main__":
    unittest.main()
