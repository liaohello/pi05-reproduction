"""Unit tests for the per-sampled-copy, episode-equal open-loop metric re-derivation."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "so101"))

import e4t_percopy_metrics as metrics  # noqa: E402

THRESHOLD = {"open_percent": 25.0, "closed_percent": 10.0}


class EpisodeEqualAggregationTest(unittest.TestCase):
    def test_episode_equal_mean_is_not_a_window_mean(self):
        # episode 0 has one window, episode 1 has three; a window mean would be 2.5.
        rows = [{"episode": 0, "value": [1.0]},
                {"episode": 1, "value": [3.0]},
                {"episode": 1, "value": [3.0]},
                {"episode": 1, "value": [3.0]}]
        self.assertAlmostEqual(metrics._episode_macro(rows, lambda row: row["value"]), 2.0)

    def test_pooled_p95_is_taken_over_sampled_errors(self):
        rows = [{"pool": [0.0] * 96 + [10.0] * 4}]
        self.assertAlmostEqual(metrics._pooled_p95(rows, "pool"), 0.0)
        rows = [{"pool": [0.0] * 90 + [10.0] * 10}]
        self.assertAlmostEqual(metrics._pooled_p95(rows, "pool"), 10.0)

    def test_per_copy_values_are_kept_separate(self):
        # two copies with opposite errors: their mean trajectory would be perfect, each copy is not.
        rows = [{"episode": 0, "copies": 2, "value": [1.0, 1.0]}]
        self.assertAlmostEqual(metrics._episode_macro(rows, lambda row: row["value"]), 1.0)


class HorizonAggregationTest(unittest.TestCase):
    def test_horizon_metrics_use_episode_equal_means_and_keep_recovery_separate(self):
        import numpy as np

        records = [
            {"episode": 0, "category": "clean", "phase": "active"},
            {"episode": 1, "category": "clean", "phase": "active"},
            {"episode": 2, "category": "A", "disturbance_phase": "post_event",
             "phase": "active"},
        ]
        truth = np.zeros((3, 10, 12), dtype=float)
        predictions = np.zeros((3, 2, 10, 12), dtype=float)
        predictions[0] += 1.0
        predictions[1] += 3.0
        predictions[2] += 5.0

        result = metrics.horizon_metrics_by_group(records, truth, predictions)

        self.assertEqual(result["clean_active"]["windows"], 2)
        self.assertEqual(result["clean_active"]["episodes"], 2)
        self.assertAlmostEqual(result["clean_active"]["joint_macro_mae_deg"]["0"], 2.0)
        self.assertAlmostEqual(result["clean_active"]["joint_p95_deg"]["0"], 3.0)
        self.assertEqual(result["clean_active"]["joint_mae_deg_by_joint"]["0"], [2.0] * 10)
        self.assertAlmostEqual(result["clean_active"]["gripper_macro_mae_pct"]["0"], 2.0)
        self.assertAlmostEqual(result["clean_active"]["gripper_p95_pct"]["0"], 3.0)
        self.assertEqual(result["clean_active"]["gripper_mae_pct_by_joint"]["0"], [2.0, 2.0])
        self.assertAlmostEqual(result["recovery_post_active"]["joint_macro_mae_deg"]["9"], 5.0)
        self.assertAlmostEqual(result["recovery_post_active"]["joint_p95_deg"]["9"], 5.0)
        self.assertAlmostEqual(result["recovery_post_active"]["gripper_macro_mae_pct"]["9"], 5.0)


class EventTimingTest(unittest.TestCase):
    def test_timing_is_reported_per_copy_and_worst_copy_is_the_gate_value(self):
        pairs = [
            {"kind": "close", "copy": 0, "episode": 0, "start": 10, "arm": "left", "error_frames": 0},
            {"kind": "close", "copy": 1, "episode": 0, "start": 10, "arm": "left", "error_frames": 4},
        ]
        misses = [{"kind": "close", "copy": 0, "episode": 0, "start": 20, "arm": "right"}]
        block = metrics.timing_block(pairs, misses, "close", 2)
        self.assertEqual(block["max_abs_frames"], 4)
        self.assertEqual(block["within_3_frames"], 1)
        self.assertEqual(block["unpaired"], 1)
        self.assertEqual([item["max_abs_frames"] for item in block["per_copy"]], [0, 4])
        self.assertEqual(block["per_copy"][0]["unpaired"], 1)
        self.assertEqual(block["per_copy"][1]["unpaired"], 0)

    def test_a_two_frame_ramp_is_not_a_pre_registered_single_frame_transition(self):
        import numpy as np
        ramp = np.asarray([[30.0] * 12, [20.0] * 12, [5.0] * 12], dtype=float)
        self.assertEqual(metrics.detect_gripper_events(ramp, THRESHOLD), [])
        jump = np.asarray([[30.0] * 12, [5.0] * 12], dtype=float)
        events = metrics.detect_gripper_events(jump, THRESHOLD)
        self.assertEqual([(item["arm"], item["event"], item["frame"]) for item in events],
                         [("left", "close", 1), ("right", "close", 1)])


if __name__ == "__main__":
    unittest.main()
