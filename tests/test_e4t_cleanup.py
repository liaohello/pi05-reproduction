"""Host-only cleanup tests for the E4-T launcher; no Docker or GPU is used."""
from __future__ import annotations

import subprocess
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "so101"))
import e4t_run_training as launcher  # noqa: E402


class FakeProcess:
    def __init__(self, waits=(), *, terminate_error=None, kill_error=None):
        self.waits = list(waits)
        self.terminate_error = terminate_error
        self.kill_error = kill_error
        self.wait_calls = []
        self.terminated = 0
        self.killed = 0

    def wait(self, timeout):
        self.wait_calls.append(timeout)
        outcome = self.waits.pop(0) if self.waits else "done"
        if outcome == "timeout":
            raise subprocess.TimeoutExpired("fake", timeout)
        if isinstance(outcome, Exception):
            raise outcome
        return 0

    def terminate(self):
        self.terminated += 1
        if self.terminate_error:
            raise self.terminate_error

    def kill(self):
        self.killed += 1
        if self.kill_error:
            raise self.kill_error


class E4TCleanupTests(unittest.TestCase):
    def setUp(self):
        self.registry = Path("mock-registry.json")
        self.stop = mock.patch.object(launcher, "registry_call", return_value={})
        self.registry_call = self.stop.start()
        self.addCleanup(self.stop.stop)

    @contextlib.contextmanager
    def mocked_main(self, directory, *, wait_error=None, stream_start_error=None):
        root = Path(directory)
        run_dir = root / "runs" / "run-1"
        train_view = root / "train-view"
        base = root / "base"
        tokenizer = root / "tokenizer.model"
        run_dir.mkdir(parents=True)
        train_view.mkdir()
        base.mkdir()
        (base / "params").mkdir()
        (base / "params" / "placeholder").write_text("base placeholder")
        tokenizer.write_text("tokenizer placeholder")
        cache_token = run_dir / "openpi_cache/big_vision/paligemma_tokenizer.model"
        cache_token.parent.mkdir(parents=True)
        cache_token.write_text("cached tokenizer")

        calls = []
        def registry_call(_registry, argv, timeout=86400):
            calls.append(argv)
            if argv[0] == "allocate":
                return {"registry_entry_id": "entry-main", "container_id": "container-main"}
            if argv[0] == "wait" and wait_error is not None:
                raise wait_error
            return {"exit_code": 0}

        monitor = FakeProcess()
        popen_calls = []
        def popen(*args, **kwargs):
            popen_calls.append(args[0])
            if len(popen_calls) == 1:
                return monitor
            if stream_start_error is not None:
                raise stream_start_error
            return FakeProcess()

        argv = [
            "launcher", "--mode", "smoke", "--run-dir", str(run_dir),
            "--train-view", str(train_view), "--base-checkpoint", str(base),
            "--tokenizer", str(tokenizer), "--exp-name", "test-run",
            "--gpu-indices", "0", "--required-memory-mib", "100",
            "--batch-size", "1", "--steps", "1", "--decay-steps", "1",
            "--timeout-seconds", "1",
        ]
        with (mock.patch.object(launcher, "REPO", root),
              mock.patch.object(launcher, "gpu_state", return_value={
                  "index": 0, "uuid": "GPU-test", "memory_total_mib": 24000,
                  "memory_used_mib": 100, "memory_free_mib": 20000}),
              mock.patch.object(launcher, "run", return_value="GPU-test"),
              mock.patch.object(launcher, "registry_call", side_effect=registry_call),
              mock.patch.object(launcher.subprocess, "Popen", side_effect=popen),
              mock.patch.object(launcher.subprocess, "run", side_effect=AssertionError(
                  "unexpected external process")),
              mock.patch.object(sys, "argv", argv),
              contextlib.redirect_stdout(io.StringIO()),
              contextlib.redirect_stderr(io.StringIO())):
            yield calls, popen_calls, monitor

    def test_main_wait_error_stops_exact_entry_and_preserves_original_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            original = RuntimeError("synthetic registry wait failure")
            with self.mocked_main(directory, wait_error=original) as (calls, popen_calls, _):
                with self.assertRaises(RuntimeError) as caught:
                    launcher.main()
            self.assertIs(caught.exception, original)
            self.assertEqual(len(popen_calls), 2)
            self.assertEqual(calls[-1], ["stop", "--entry-id", "entry-main", "--confirm",
                                         "STOP-REGISTERED-EXACT-ENTRIES"])

    def test_main_log_stream_start_failure_reaps_monitor_and_stops_container(self):
        with tempfile.TemporaryDirectory() as directory:
            original = OSError("synthetic log stream startup failure")
            with self.mocked_main(directory, stream_start_error=original) as (calls, popen_calls, monitor):
                with self.assertRaises(OSError) as caught:
                    launcher.main()
            self.assertIs(caught.exception, original)
            self.assertEqual(len(popen_calls), 2)
            self.assertEqual(monitor.wait_calls, [5])
            self.assertEqual(calls[-1], ["stop", "--entry-id", "entry-main", "--confirm",
                                         "STOP-REGISTERED-EXACT-ENTRIES"])

    def test_normal_completion_reaps_children_and_stops_registered_entry(self):
        stream, monitor = FakeProcess(), FakeProcess()
        errors = launcher._cleanup_owned_run(self.registry, "entry-1", monitor, stream)
        self.assertEqual(errors, [])
        self.registry_call.assert_called_once_with(
            self.registry, ["stop", "--entry-id", "entry-1", "--confirm",
                            "STOP-REGISTERED-EXACT-ENTRIES"])

    def test_wait_timeout_terminates_child_and_still_stops_entry(self):
        stream = FakeProcess(["timeout", "done"])
        errors = launcher._cleanup_owned_run(self.registry, "entry-2", None, stream)
        self.assertEqual(errors, [])
        self.assertEqual(stream.terminated, 1)
        self.assertEqual(stream.killed, 0)
        self.registry_call.assert_called_once()

    def test_persistent_stream_timeout_and_stop_error_are_reported(self):
        stream = FakeProcess(["timeout"] * 3, terminate_error=OSError("term failed"),
                             kill_error=OSError("kill failed"))
        self.registry_call.side_effect = RuntimeError("stop failed")
        errors = launcher._cleanup_owned_run(self.registry, "entry-3", None, stream)
        self.assertEqual(stream.terminated, 1)
        self.assertEqual(stream.killed, 1)
        self.assertTrue(any("still running after kill" in error for error in errors))
        self.assertTrue(any("controlled container cleanup failed" in error for error in errors))
        self.registry_call.assert_called_once()


if __name__ == "__main__":
    unittest.main()
