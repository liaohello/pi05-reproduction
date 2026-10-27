"""SO101 ledger records must never unlock a LIBERO gate."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("track_status", ROOT / "scripts" / "track" / "status.py")
status = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(status)

CFG = {"ssh": {"host": "example.invalid", "port": 22, "user": "example", "key": "~/.ssh/id_ed25519"},
       "remote": {"label": "remote-gpu-server", "root": "/tmp/remote", "runs_dir": "/tmp/remote/runs",
                  "artifact_dir": "/tmp/remote/artifacts", "log_dir": "/tmp/remote/logs"},
       "assets": {"checkpoint": "/tmp/checkpoint", "norm_stats_sha256": "0" * 64},
       "images": {}, "sync": {"excludes": []}, "local": {"min_free_gb": 1}}

SO101_TRAIN = {"run_id": "1-so101-E2-train", "ts": "2026-10-25T00:00:00+00:00",
               "scope": "so101", "phase": "E2", "kind": "train", "status": "success",
               "episodes": 0, "successes": 0, "metrics": {}, "suite": ""}
LIBERO_TRAIN = {"run_id": "2-libero-D5-train", "ts": "2026-10-25T00:01:00+00:00",
                "scope": "libero", "phase": "D5", "kind": "train", "status": "success",
                "episodes": 0, "successes": 0, "metrics": {}, "suite": ""}
LEGACY_EVAL = {"run_id": "3-D4-eval-libero_10", "ts": "2026-10-25T00:02:00+00:00",
               "phase": "D4", "kind": "eval", "status": "success",
               "episodes": 10, "successes": 4, "metrics": {"sr": 0.4}, "suite": "libero_10"}


class ScopeIsolationTest(unittest.TestCase):
    def test_so101_train_cannot_unlock_libero_train_gate(self):
        ctx = status.Ctx(runs=[SO101_TRAIN], cfg=CFG)
        state, _ = status.evaluate(ctx, "train_run")
        self.assertEqual(state, status.PENDING)

        ctx = status.Ctx(runs=[SO101_TRAIN, LIBERO_TRAIN], cfg=CFG)
        state, _ = status.evaluate(ctx, "train_run")
        self.assertEqual(state, status.PASS)

    def test_legacy_records_without_scope_stay_libero(self):
        ctx = status.Ctx(runs=[LEGACY_EVAL], cfg=CFG)
        self.assertEqual(len(ctx.runs_of("eval")), 1)
        self.assertEqual(ctx.runs_of_scope("libero"), [LEGACY_EVAL])

    def test_so101_records_are_reported_separately(self):
        ctx = status.Ctx(runs=[SO101_TRAIN, LIBERO_TRAIN], cfg=CFG)
        report = status.build(ctx)
        self.assertEqual(report["metrics"]["train"], 1)
        self.assertEqual(report["so101"]["runs"], 1)
        self.assertEqual(report["so101"]["phases"]["E2"]["runs"], 1)
        self.assertEqual(report["so101"]["phases"]["E2"]["successes"], 1)
        self.assertNotIn(SO101_TRAIN["run_id"], [run["run_id"] for run in report["runs"]])


if __name__ == "__main__":
    unittest.main()
