"""Budget evidence must include the train curve and isolate supplementary window sets."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "so101"))
import e4t_budget  # noqa: E402


class EvaluationBudgetTest(unittest.TestCase):
    def test_evaluation_evidence_counts_train_validation_and_supplements(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "evaluation").mkdir()
            for split, rows in {
                "train": [
                    {"label": "train_init_step000001", "success": True,
                     "started": "2026-01-01T00:00:00+00:00"},
                    {"label": "train_step005001", "success": False,
                     "started": "2026-01-01T00:01:00+00:00"},
                    {"label": "train_step005001", "success": True,
                     "started": "2026-01-01T00:02:00+00:00"},
                ],
                "validation": [
                    {"label": "step005001", "success": True,
                     "started": "2026-01-01T00:03:00+00:00"},
                ],
            }.items():
                (run / f"evaluation_{split}_driver.json").write_text(
                    json.dumps({"evaluations": rows}), encoding="utf-8")

            reports = {
                "train_step005001": {"split": "train", "label": "train_step005001",
                                     "window_set": "frozen_plan",
                                     "created": "2026-01-01T00:02:30+00:00"},
                "step005001": {"split": "validation", "label": "step005001",
                               "window_set": "frozen_plan",
                               "created": "2026-01-01T00:03:30+00:00"},
                "step005001_anchored": {"split": "validation", "label": "step005001",
                                        "window_set": "event_anchored_supplementary",
                                        "created": "2026-01-01T00:04:00+00:00"},
            }
            for label, report in reports.items():
                folder = run / "evaluation" / label
                folder.mkdir()
                report_path = folder / "report.json"
                report_path.write_text(json.dumps(report), encoding="utf-8")
            dense_folder = run / "evaluation" / "step030000_dense10"
            dense_folder.mkdir()
            dense_report = dense_folder / "report.json"
            dense_report.write_text(json.dumps({"split": "validation", "label": "step030000",
                                               "window_set": "frozen_plan_plus_dense_sweep"}),
                                    encoding="utf-8")
            timestamp = datetime.fromisoformat("2026-01-01T00:06:00+00:00").timestamp()
            os.utime(dense_report, (timestamp, timestamp))

            (run / "resources.json").write_text(json.dumps({"entries": [{
                "exact_name": "so101-e4t-eval-test", "created": "2026-01-01T00:05:00+00:00",
                "finished_at": "2026-01-01T00:07:00+00:00"
            }]}), encoding="utf-8")

            result = e4t_budget.evaluation_evidence(run)
            self.assertEqual(result["driver_attempts_by_split"], {"train": 3, "validation": 1})
            self.assertEqual(result["successful_driver_checkpoints_by_split"],
                             {"train": 2, "validation": 1})
            self.assertEqual(result["frozen_plan_reports_by_split"], {"train": 1, "validation": 1})
            self.assertEqual(result["supplementary_reports_by_window_set"],
                             {"event_anchored_supplementary": 1,
                              "frozen_plan_plus_dense_sweep": 1})
            self.assertEqual(result["report_creation_span_s"], 90)
            self.assertEqual(result["registered_evaluation_launches"], 1)
            self.assertEqual(result["evaluation_wall_upper_bound_s"], 420)


if __name__ == "__main__":
    unittest.main()
