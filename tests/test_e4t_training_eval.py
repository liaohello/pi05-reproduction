"""E4-T training/evaluation guard tests that run on the host without JAX."""
from __future__ import annotations

import json
import gzip
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "so101"))

import summarize_e4t_openloop as summarize  # noqa: E402
import export_e4t_view as export  # noqa: E402
import e4t_eval_checkpoints as driver  # noqa: E402
import plot_e4t_curves as plot_curves  # noqa: E402
import e4t_training_evidence as evidence  # noqa: E402
import e4t_run_training as training_launcher  # noqa: E402

GATES = json.loads((ROOT / "configs/so101/e4t_training.json").read_text())["validation_gate"]
NOISE = "shared-draws"


def _event_block(kind: str, frames: int, unpaired: int = 0, ground: int = 12) -> dict:
    return {"ground_truth_events": ground, "paired": ground - unpaired, "unpaired": unpaired,
            "max_abs_frames": frames if ground - unpaired else None,
            "mean_abs_frames": float(frames), "measurable": bool(ground - unpaired)}


def _report(step: int, clean: float, recovery: float, gripper: float = 1.0,
            timing: int = 1, hold: float = 6.0, mean: float = 5.0, unpaired: int = 0,
            windows: int = 640, windows_expected: int | None = 640, noise: str | None = NOISE,
            success: bool = True) -> dict:
    def group(mae):
        return {"windows": 320, "joint_macro_mae_deg": mae, "joint_p95_deg": mae * 2.0,
                "gripper_macro_mae_pct": gripper, "hold_joint_mae_deg": hold,
                "train_mean_joint_mae_deg": mean, "pred_delta_rms_deg_per_frame": 0.4,
                "gt_delta_rms_deg_per_frame": 0.5, "pred_reversals_per_window": 1.0}
    groups = {"clean_active": group(clean), "recovery_post_active": group(recovery)}
    report = {"label": f"step{step}", "split": "validation", "checkpoint_train_state_step": step,
              "windows": windows, "windows_expected": windows_expected, "success": success,
              "noise_set_sha256": noise, "run_config_sha256": "aa",
              "horizon_mae_joint_deg": [mae for mae in ([clean] * 10)],
              "event_timing": {"pairing_limit_frames": 10,
                               "close": _event_block("close", timing, unpaired),
                               "release": _event_block("release", timing, unpaired),
                               "unpaired_examples": []},
              "aggregates": groups}
    # The summariser requires the corrected per-sampled-copy, episode-equal derivation and the
    # event-anchored timing pass.  These fixture blocks share the same dict objects as the legacy
    # aggregates so a test that empties a group also empties it here.
    report["per_copy"] = {"aggregates": groups, "event_timing": report["event_timing"]}
    report["anchored"] = {"per_copy": {"event_timing": report["event_timing"]}}
    return report


def _eval(report: dict, initial: dict | None = None) -> dict:
    return summarize.evaluate(report, GATES, initial, NOISE)


class NoiseSetHashTest(unittest.TestCase):
    def test_record_noise_hash_matches_evaluator_serialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {"episode": 4, "start": 8, "noise_hashes": ["z", "a"]},
                {"episode": 1, "start": 3, "noise_hashes": ["m", "b"]},
            ]
            with gzip.open(root / "records.jsonl.gz", "wt", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            canonical = {f"{row['episode']}:{row['start']}": sorted(row["noise_hashes"])
                         for row in rows}
            expected = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
            actual, source = summarize.noise_set_digest({
                "_path": str(root / "report.json"), "noise_set_sha256": "wrong-format-hash"})
            self.assertEqual(actual, expected)
            self.assertEqual(source, "records.jsonl.gz")


class OpenLoopGateTest(unittest.TestCase):
    def test_failed_gate_diagnostic_uses_error_best_not_latest_step(self):
        rows = [
            {"step": 5001, "primary_objective_deg": 1.1},
            {"step": 10001, "primary_objective_deg": 0.8},
            {"step": 15001, "primary_objective_deg": 0.9},
        ]
        selected = summarize.diagnostic_checkpoint(rows)
        self.assertEqual(selected["step"], 10001)

    def test_evaluation_launches_get_isolated_resource_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = training_launcher.run_artifact_paths(root, "eval", "run-a")
            second = training_launcher.run_artifact_paths(root, "eval", "run-b")
            self.assertNotEqual(first["gpu_log"], second["gpu_log"])
            self.assertNotEqual(first["stdout_log"], second["stdout_log"])
            self.assertNotEqual(first["run_payload"], second["run_payload"])
            initial = training_launcher.run_artifact_paths(root, "initial", "run-c")
            self.assertEqual(initial["stdout_log"], root / "train_initial.log")

    def test_train_split_labels_cannot_overwrite_validation_reports(self):
        self.assertEqual(driver.evaluation_label(5001), "step005001")
        self.assertEqual(driver.evaluation_label(5001, "train_"), "train_step005001")

    def test_train_curve_reader_excludes_validation_and_supplementary_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label, split, window_set in (
                    ("train_step005001", "train", "frozen_plan"),
                    ("step005001", "validation", "frozen_plan"),
                    ("train_dense_step005001", "train", "frozen_plan_plus_dense_sweep")):
                folder = root / label
                folder.mkdir()
                (folder / "report.json").write_text(json.dumps({
                    "label": label, "split": split, "window_set": window_set}))
            rows = plot_curves.reports(root, "train")
            self.assertEqual([report["label"] for report, _ in rows], ["train_step005001"])

    def test_public_resource_summary_omits_machine_identity(self):
        raw = {"mode": "eval", "exit_code": 0, "wall_time_s": 12.0,
               "container_id": "private-id", "exact_name": "private-name", "owner": "private-owner",
               "gpu": {"selected_uuid": "private-uuid", "requested_index": 2,
                       "memory_total_mib": 24564, "peak_mib_sampled": 12000,
                       "jax_memory_fraction": 0.5,
                       "utilisation_is_not_an_admission_gate": True}}
        public = evidence.public_container_record(raw)
        self.assertNotIn("container_id", public)
        self.assertNotIn("exact_name", public)
        self.assertNotIn("owner", public)
        self.assertNotIn("selected_uuid", public["gpu"])
        self.assertNotIn("requested_index", public["gpu"])
        self.assertNotIn("memory_total_mib", public["gpu"])
        self.assertEqual(public["gpu"]["peak_mib_sampled"], 12000)

    def test_registered_resource_payloads_are_matched_by_run_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run_eval_run-a.json").write_text(json.dumps({
                "mode": "eval", "run_id": "run-a", "exit_code": 0,
                "gpu": {"peak_mib_sampled": 12345},
            }), encoding="utf-8")
            registry = [{"run_id": "run-a", "exact_name": "so101-e4t-eval-run-a",
                         "matched_all_registration_fields": True,
                         "command": ["-u", "evaluate_e4t_openloop.py", "--label", "train_step015001"]},
                        {"run_id": "run-b", "exact_name": "so101-e4t-eval-run-b",
                         "matched_all_registration_fields": True,
                         "command": ["-u", "evaluate_e4t_openloop.py", "--label", "train_step020001"]}]
            rows = evidence.registered_container_records(root, registry)
            self.assertEqual([key for key, _ in rows], ["run-a"])
            public = evidence.public_container_record(rows[0][1])
            self.assertEqual(public["evaluation_label"], "train_step015001")
            self.assertEqual(public["owner_verified_at_registration"], True)
            self.assertNotIn("run_id", public)

    def test_training_split_curve_uses_only_frozen_train_reports(self):
        base = {"run_config_sha256": "config", "windows": 640, "windows_expected": 640,
                "noise_set_sha256": "fixed-noise", "aggregates": {
                    "clean_active": {"joint_macro_mae_deg": 1.0, "true12_mse": 0.1},
                    "recovery_post_active": {"joint_macro_mae_deg": 1.2, "true12_mse": 0.2}}}
        reports = {
            "train_step005001@frozen_plan": {**base, "split": "train", "window_set": "frozen_plan",
                                               "checkpoint_train_state_step": 5001},
            "train_dense_step005001@frozen_plan_plus_dense_sweep": {
                **base, "split": "train", "window_set": "frozen_plan_plus_dense_sweep",
                "checkpoint_train_state_step": 5001},
            "step005001@frozen_plan": {**base, "split": "validation", "window_set": "frozen_plan",
                                         "checkpoint_train_state_step": 5001},
        }
        curve = evidence.split_curve(reports, "train")
        self.assertEqual([row["label"] for row in curve], ["train_step005001@frozen_plan"])
        self.assertEqual(curve[0]["clean_active"]["true12_mse"], 0.1)

    def test_passing_checkpoint_beats_every_baseline(self):
        trained = _eval(_report(5000, 1.8, 1.9), _report(0, 9.0, 11.0))
        self.assertTrue(trained["gate_passed"], trained["gate_failed_checks"])
        self.assertEqual(trained["first_filter_passed"], True)

    def test_unmeasured_active_group_is_not_a_pass(self):
        initial = _report(0, 9.0, 11.0)
        report = _report(5000, 1.5, 1.5)
        report["aggregates"]["recovery_post_active"] = {}
        verdict = _eval(report, initial)
        self.assertFalse(verdict["gate_passed"])
        self.assertIn("recovery_post_active_joint_macro_mae", verdict["gate_failed_checks"])
        self.assertEqual(verdict["checks"]["recovery_post_active_joint_macro_mae"], "unmeasured")

    def test_static_only_success_is_rejected_by_the_activity_split(self):
        report = _report(5000, 0.1, 0.1)
        report["aggregates"]["clean_active"] = {}
        report["aggregates"]["clean"] = {"joint_macro_mae_deg": 0.1}
        verdict = _eval(report, _report(0, 9.0, 11.0))
        self.assertFalse(verdict["gate_passed"])

    def test_must_beat_hold_and_train_mean_baselines(self):
        initial = _report(0, 9.0, 11.0)
        worse = _report(5000, 7.0, 7.5, hold=6.0, mean=5.0)
        verdict = _eval(worse, initial)
        self.assertFalse(verdict["gate_passed"])
        self.assertEqual(verdict["checks"]["beat_hold_clean_active"], "fail")
        self.assertEqual(verdict["checks"]["beat_train_mean_clean_active"], "fail")

    def test_event_timing_failure_is_excluded_from_best_selection(self):
        rows = [_eval(report, _report(0, 9.0, 11.0))
                for report in (_report(5000, 1.0, 1.0, timing=8), _report(10000, 1.6, 1.7))]
        candidates = [row for row in rows if row["first_filter_passed"]]
        self.assertEqual([row["step"] for row in candidates], [10000])

    def test_an_unpaired_ground_truth_event_fails_the_timing_gate(self):
        """A missed event is a timing error of unknown size, not a smaller sample of errors."""
        verdict = _eval(_report(5000, 1.0, 1.0, unpaired=3), _report(0, 9.0, 11.0))
        self.assertEqual(verdict["checks"]["key_event_timing_frames"], "fail")
        self.assertEqual(verdict["event_timing_verdict"], {"close": "fail", "release": "fail"})
        self.assertFalse(verdict["first_filter_passed"])

    def test_timing_without_any_ground_truth_event_is_unmeasured(self):
        report = _report(5000, 1.0, 1.0)
        for kind in ("close", "release"):
            report["event_timing"][kind] = _event_block(kind, 1, ground=0)
        verdict = _eval(report, _report(0, 9.0, 11.0))
        self.assertEqual(verdict["checks"]["key_event_timing_frames"], "unmeasured")
        self.assertFalse(verdict["gate_passed"])

    def test_a_short_window_set_fails_instead_of_silently_scoring_fewer_windows(self):
        verdict = _eval(_report(5000, 1.0, 1.0, windows=613), _report(0, 9.0, 11.0))
        self.assertEqual(verdict["checks"]["evaluation_uses_frozen_window_plan"], "fail")
        self.assertFalse(verdict["gate_passed"])

    def test_a_report_without_a_window_plan_is_unmeasured(self):
        verdict = _eval(_report(5000, 1.0, 1.0, windows_expected=None), _report(0, 9.0, 11.0))
        self.assertEqual(verdict["checks"]["evaluation_uses_frozen_window_plan"], "unmeasured")

    def test_a_different_noise_set_between_checkpoints_fails_the_gate(self):
        verdict = _eval(_report(5000, 1.0, 1.0, noise="other-draws"), _report(0, 9.0, 11.0))
        self.assertEqual(verdict["checks"]["shared_noise_set"], "fail")
        self.assertFalse(verdict["gate_passed"])

    def test_a_missing_initialisation_reference_fails_the_gate(self):
        """must_beat_initialization cannot be satisfied by simply not having a reference."""
        verdict = _eval(_report(5000, 1.0, 1.2), initial=None)
        self.assertEqual(verdict["checks"]["initialization_reference_available"], "fail")
        self.assertFalse(verdict["gate_passed"])

    def test_a_gate_that_only_needs_a_passing_run_still_needs_active_windows(self):
        report = _report(5000, 1.0, 1.2)
        report["aggregates"]["clean_active"]["windows"] = 0
        verdict = _eval(report, _report(0, 6.0, 6.5))
        self.assertEqual(verdict["checks"]["active_groups_measured"], "fail")
        self.assertFalse(verdict["gate_passed"])

    def test_the_frame_and_millisecond_tolerances_have_to_agree(self):
        initial = _report(0, 9.0, 11.0)
        verdict = _eval(_report(5000, 1.0, 1.2), initial)
        self.assertEqual(verdict["checks"]["timing_tolerance_units_consistent"], "pass")
        broken = dict(GATES, key_event_timing_tolerance_ms=100.0)
        self.assertEqual(summarize.evaluate(_report(5000, 1.0, 1.2), broken, initial, NOISE)
                         ["checks"]["timing_tolerance_units_consistent"], "fail")

    def test_gripper_gate_is_a_first_filter_before_error_ranking(self):
        rows = [_eval(report, _report(0, 9.0, 11.0))
                for report in (_report(5000, 0.5, 0.5, gripper=4.5), _report(10000, 1.6, 1.7))]
        candidates = [row for row in rows if row["first_filter_passed"]]
        self.assertEqual([row["step"] for row in candidates], [10000])
        self.assertFalse(rows[0]["gate_passed"])

    def test_initialisation_must_be_beaten_on_active_groups(self):
        initial = _report(0, 1.4, 1.5)
        later = _eval(_report(5000, 1.6, 1.7), initial)
        self.assertEqual(later["checks"]["beat_initialization_clean_active"], "fail")
        self.assertFalse(later["gate_passed"])


class CheckpointConfigBindingTest(unittest.TestCase):
    """A curve may only be built from checkpoints that share the run's config."""

    def _write(self, root: Path, label: str, step: int, sha: str) -> None:
        directory = root / label
        directory.mkdir(parents=True)
        report = _report(step, 1.0, 1.5)
        report["run_config_sha256"] = sha
        (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")

    def _run(self, root: Path, argv_extra=()):
        import contextlib
        import io
        output = root / "gate.json"
        argv = ["summarize", "--reports", str(root), "--output", str(output), *argv_extra]
        with contextlib.redirect_stdout(io.StringIO()):
            with _patched_argv(argv):
                return summarize.main()

    def test_mixed_run_configs_are_refused(self):
        import sys
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, "a", 5000, "aa")
            self._write(root, "b", 10000, "bb")
            with self.assertRaises(SystemExit) as caught:
                self._run(root)
            self.assertIn("run config", str(caught.exception))

    def test_expected_config_hash_is_enforced(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, "a", 5000, "aa")
            with self.assertRaises(SystemExit) as caught:
                self._run(root, ("--expected-config-sha256", "zz"))
            self.assertIn("mismatch", str(caught.exception))

    def test_matching_reports_produce_a_bound_summary(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, "a", 5000, "aa")
            self._write(root, "b", 10000, "aa")
            self.assertEqual(self._run(root, ("--expected-config-sha256", "aa")), 0)
            summary = json.loads((root / "gate.json").read_text())
            self.assertEqual(summary["run_config_sha256"], "aa")
            self.assertEqual([row["step"] for row in summary["curve"]], [5000, 10000])
            self.assertIn("velocity_shape", summary["curve"][0])

    def _write_labelled(self, root: Path, directory: str, label: str, step: int) -> None:
        path = root / directory
        path.mkdir(parents=True)
        report = _report(step, 1.0, 1.5)
        report["label"] = label
        report["run_config_sha256"] = "aa"
        (path / "report.json").write_text(json.dumps(report), encoding="utf-8")

    def test_a_re_scored_report_does_not_reuse_the_superseded_label(self):
        """A second report carrying the initialisation label must not silently pick one of them."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_labelled(root, "init_step1", "init_step1", 1)
            self._write_labelled(root, "init_step1_copy", "init_step1", 1)
            with self.assertRaises(SystemExit) as caught:
                self._run(root, ("--initial-label", "init_step1"))
            self.assertIn("share the label", str(caught.exception))

    def test_the_initial_reference_may_carry_its_own_run_config(self):
        """The reference state comes from the probe run; it must not break the checkpoint binding."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = _report(1, 6.0, 6.5)
            reference["label"] = "init_step1"
            reference["run_config_sha256"] = "probe-config"
            (root / "init_step1").mkdir(parents=True)
            (root / "init_step1" / "report.json").write_text(json.dumps(reference), encoding="utf-8")
            self._write(root, "a", 5000, "aa")
            self.assertEqual(self._run(root, ("--initial-label", "init_step1")), 0)
            summary = json.loads((root / "gate.json").read_text())
            self.assertEqual(summary["run_config_sha256"], "aa")
            self.assertEqual(summary["initial_reference_run_config_sha256"], "probe-config")
            self.assertTrue(summary["curve"][1]["gate_passed"],
                            summary["curve"][1]["gate_failed_checks"])

    def test_a_dense_or_anchored_report_does_not_join_the_frozen_plan_curve(self):
        """A superset sweep of the same checkpoint is its own curve, never a second point here."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_labelled(root, "step015001", "step015001", 15001)
            dense = _report(15001, 1.0, 1.2)
            dense["label"] = "step015001"
            dense["window_set"] = "frozen_plan_plus_dense_sweep"
            directory = root / "step015001_dense200"
            directory.mkdir(parents=True)
            (directory / "report.json").write_text(json.dumps(dense), encoding="utf-8")
            self.assertEqual(self._run(root), 0)
            summary = json.loads((root / "gate.json").read_text())
            self.assertEqual([row["step"] for row in summary["curve"]], [15001])
            self.assertEqual(summary["window_set"], "frozen_plan")

    def test_superseded_directories_are_ignored_and_recorded(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, "step005000", 5000, "aa")
            self._write(root, "step005000_superseded_admission_problem", 5000, "aa")
            self.assertEqual(self._run(root), 0)
            summary = json.loads((root / "gate.json").read_text())
            self.assertEqual(len(summary["curve"]), 1)
            self.assertEqual(summary["superseded_report_directories_ignored"],
                             ["step005000_superseded_admission_problem"])


class InitialisationReferenceTest(unittest.TestCase):
    """The named initialisation reference must actually be beaten, not merely recorded."""

    def _write(self, root: Path, label: str, step: int, clean: float, recovery: float) -> None:
        directory = root / label
        directory.mkdir(parents=True)
        report = _report(step, clean, recovery)
        report["label"] = label
        report["run_config_sha256"] = "aa"
        (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")

    def _summary(self, root: Path) -> dict:
        import contextlib
        import io
        argv = ["summarize", "--reports", str(root), "--initial-label", "init_step1",
                "--output", str(root / "gate.json")]
        with contextlib.redirect_stdout(io.StringIO()):
            with _patched_argv(argv):
                summarize.main()
        return json.loads((root / "gate.json").read_text())

    def test_beating_the_reference_is_required(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, "init_step1", 1, 0.5, 0.6)
            self._write(root, "step005000", 5000, 1.0, 1.2)
            summary = self._summary(root)
            self.assertEqual(summary["initial_reference_label"], "init_step1")
            row = summary["curve"][1]
            self.assertEqual(row["checks"]["beat_initialization_clean_active"], "fail")
            self.assertFalse(row["gate_passed"])

    def test_a_reference_that_is_beaten_keeps_the_gate_open(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, "init_step1", 1, 20.0, 21.0)
            self._write(root, "step005000", 5000, 1.0, 1.2)
            summary = self._summary(root)
            self.assertTrue(summary["curve"][1]["checks"]["beat_initialization_clean_active"] == "pass")
            self.assertTrue(summary["curve"][1]["gate_passed"])
            self.assertEqual(summary["decision"], "eligible_for_closed_loop")

    def test_an_unknown_reference_label_is_refused(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, "init_step1", 1, 20.0, 21.0)
            with self.assertRaises(SystemExit):
                import contextlib
                import io
                with contextlib.redirect_stdout(io.StringIO()):
                    with _patched_argv(["summarize", "--reports", str(root),
                                        "--initial-label", "nope", "--output", str(root / "g.json")]):
                        summarize.main()


def _sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StatisticsBindingTest(unittest.TestCase):
    """The curve must be bound to one frozen view and one run config, checked from both sides."""

    def _setup(self, root: Path, stats_in_config: str, stats_in_meta: str) -> tuple[Path, Path]:
        view = root / "view"
        view.mkdir(parents=True)
        meta = {"view_sha256": "view-hash", "norm_stats_sha256": stats_in_meta,
                "norm_stats_canonical_sha256": "canonical-hash"}
        (view / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        run_config = root / "config_initial.json"
        run_config.write_text(json.dumps({"norm_stats_sha256": stats_in_config,
                                          "train_view_meta_sha256": _sha(view / "meta.json")}),
                              encoding="utf-8")
        for name, step in (("init_step1", 1), ("step005000", 5000)):
            directory = root / "evaluation" / name
            directory.mkdir(parents=True)
            report = _report(step, 6.0 if step == 1 else 1.0, 6.5 if step == 1 else 1.2)
            report["label"] = name
            report["view_sha256"] = meta["view_sha256"]
            report["run_config_sha256"] = _sha(run_config)
            (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
        return view, run_config

    def _run(self, root: Path, view: Path, run_config: Path) -> dict:
        import contextlib
        import io
        output = root / "gate.json"
        argv = ["summarize", "--reports", str(root / "evaluation"), "--initial-label", "init_step1",
                "--view", str(view), "--run-config", str(run_config),
                "--expected-config-sha256", _sha(run_config), "--output", str(output)]
        with contextlib.redirect_stdout(io.StringIO()):
            with _patched_argv(argv):
                self.assertEqual(summarize.main(), 0)
        return json.loads(output.read_text())

    def test_agreeing_statistics_are_published(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view, run_config = self._setup(root, "file-stats", "file-stats")
            summary = self._run(root, view, run_config)
            self.assertEqual(summary["statistics_and_view_bindings"]["norm_stats_file_sha256"],
                             "file-stats")
            self.assertEqual(summary["statistics_and_view_bindings"]["norm_stats_canonical_sha256"],
                             "canonical-hash")
            self.assertTrue(summary["curve"][1]["gate_passed"],
                            summary["curve"][1]["gate_failed_checks"])

    def test_a_run_config_naming_other_statistics_is_refused(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view, run_config = self._setup(root, "file-stats", "some-other-stats")
            with self.assertRaises(SystemExit) as caught:
                self._run(root, view, run_config)
            self.assertIn("statistics", str(caught.exception))

    def test_a_report_scored_on_another_view_is_refused(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view, run_config = self._setup(root, "file-stats", "file-stats")
            path = root / "evaluation" / "step005000" / "report.json"
            report = json.loads(path.read_text())
            report["view_sha256"] = "another-view"
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaises(SystemExit) as caught:
                self._run(root, view, run_config)
            self.assertIn("frozen view", str(caught.exception))


class _patched_argv:
    def __init__(self, argv):
        self.argv = argv
        self.previous = None

    def __enter__(self):
        import sys
        self.previous = sys.argv
        sys.argv = self.argv
        return self

    def __exit__(self, *exc):
        import sys
        sys.argv = self.previous
        return False


class CheckpointDiscoveryTest(unittest.TestCase):
    """A checkpoint may only be evaluated when its Orbax save is committed."""

    def setUp(self):
        import tempfile
        self.root = Path(tempfile.mkdtemp())
        self.exp = self.root / "checkpoints" / "pi05_so101_e2" / "e4t_30k_state_on_b8"
        self.exp.mkdir(parents=True)

    def _make(self, directory: str, *files: str) -> Path:
        path = self.exp / directory
        for name in files:
            target = path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("x", encoding="utf-8")
        return path

    def test_only_committed_saves_are_found(self):
        complete = self._make("4999", "params/_METADATA", "train_state/_METADATA",
                              "_CHECKPOINT_METADATA", "assets/so101/norm_stats.json")
        self._make("9999", "params/_METADATA")  # half-written
        found = dict(driver.completed_checkpoints(self.root / "checkpoints"))
        self.assertEqual(found, {4999: complete})

    def test_committed_checkpoints_are_sorted_by_numeric_step(self):
        for index in (5000, 10000, 15000, 20000, 25000):
            self._make(str(index), "params/_METADATA", "train_state/_METADATA",
                       "_CHECKPOINT_METADATA", "assets/so101/norm_stats.json")
        found = driver.completed_checkpoints(self.root / "checkpoints")
        self.assertEqual([index for index, _ in found], [5000, 10000, 15000, 20000, 25000])

    def test_directory_index_maps_to_the_authoritative_step(self):
        # measured: the batch-8 probe saved directory 0 with TrainState.step 1,
        # and the E2 run saved directory 99 with TrainState.step 100.
        self.assertEqual(driver.expected_step(0), 1)
        self.assertEqual(driver.expected_step(99), 100)
        self.assertEqual(driver.expected_step(4999), 5000)

    def test_log_steps_are_read_case_insensitively(self):
        run = self.root / "run"
        run.mkdir()
        (run / "train_initial.log").write_text(
            "Step 41: loss=0.4\n04:53:56 Progress on: 441it/30.0kit rate:1.2s/it\n", encoding="utf-8")
        self.assertIn(41, driver.logged_steps(run))
        self.assertIn(441, driver.logged_steps(run))


class WindowPlanTest(unittest.TestCase):
    def test_event_crossing_windows_leave_only_legal_starts(self):
        starts = export._eligible_starts(25, 15, 10)
        self.assertEqual(starts, [0, 1, 2, 3, 4, 5])
        self.assertNotIn(6, starts)                # 6..15 ends on the event frame
        self.assertNotIn(15, starts)               # 15..24 starts on the event frame
        self.assertEqual(len(starts), 6)

    def test_clean_episode_keeps_every_full_horizon_window(self):
        self.assertEqual(export._eligible_starts(12, None, 10), [0, 1, 2])

    def test_episode_shorter_than_horizon_has_no_window(self):
        self.assertEqual(export._eligible_starts(9, None, 10), [])


if __name__ == "__main__":
    unittest.main()
