"""Contract tests for batch-size-dependent training schedules."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "libero"))

import bowl_plate_49  # noqa: E402


class BowlPlateScheduleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol_path = ROOT / "configs" / "libero" / "bowl_plate_49_lora.json"
        cls.protocol = bowl_plate_49.load_protocol(
            cls.protocol_path,
            bowl_plate_49.sha256_file(cls.protocol_path),
        )

    def test_base_schedule_matches_requested_config(self) -> None:
        schedule = bowl_plate_49.effective_schedule(self.protocol, 32)
        self.assertEqual(schedule["optimizer_updates"], 1500)
        self.assertEqual(schedule["warmup_steps"], 100)
        self.assertEqual(schedule["cosine_decay_steps"], 1500)
        self.assertEqual(schedule["log_interval"], 20)
        self.assertEqual(schedule["checkpoint_steps"], [500, 1000, 1500])

    def test_every_reduction_preserves_sample_budget_and_ratios(self) -> None:
        for batch_size, scale in ((16, 2), (8, 4), (4, 8)):
            with self.subTest(batch_size=batch_size):
                schedule = bowl_plate_49.effective_schedule(
                    self.protocol, batch_size)
                self.assertEqual(schedule["batch_scale"], scale)
                self.assertEqual(
                    schedule["optimizer_updates"] * batch_size, 48_000)
                self.assertEqual(
                    schedule["warmup_steps"], 100 * scale)
                self.assertEqual(
                    schedule["cosine_decay_steps"],
                    schedule["optimizer_updates"],
                )
                self.assertEqual(
                    schedule["log_interval"], 20 * scale)
                self.assertEqual(
                    schedule["save_interval"], 500 * scale)
                self.assertEqual(
                    schedule["keep_period"], 500 * scale)
                self.assertEqual(
                    schedule["checkpoint_steps"],
                    [500 * scale, 1000 * scale, 1500 * scale],
                )

    def test_unregistered_batch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "not registered"):
            bowl_plate_49.effective_schedule(self.protocol, 12)


if __name__ == "__main__":
    unittest.main()
