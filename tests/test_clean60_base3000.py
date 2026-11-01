"""Contracts for the pi05_base run with 3000-update checkpoints."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/so101"))
import clean60_contracts as contracts  # noqa: E402


class Clean60Base3000ContractTest(unittest.TestCase):
    def test_registered_variant_passes_without_relaxing_core_limits(self):
        import json

        config = json.loads(
            (ROOT / "configs/so101/clean60_training_base3000.json").read_text(
                encoding="utf-8"))
        self.assertEqual(contracts.training_config_problems(config), [])
        self.assertEqual(config["initialisation"]["base_checkpoint"], "pi05_base")
        self.assertEqual(config["checkpointing"]["save_interval"], 3000)
        self.assertEqual(config["optimisation"]["primary_batch_size"], 8)
        self.assertEqual(config["optimisation"]["planned_updates"], 25000)
        self.assertEqual(config["checkpointing"]["eval_checkpoints"],
                         [0, 3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 25000])

    def test_arbitrary_checkpoint_interval_is_rejected(self):
        import json

        config = json.loads(
            (ROOT / "configs/so101/clean60_training_base3000.json").read_text(
                encoding="utf-8"))
        config["checkpointing"]["save_interval"] = 1234
        self.assertIn("checkpointing.save_interval is not 3000 or 5000",
                      contracts.training_config_problems(config))


if __name__ == "__main__":
    unittest.main()
