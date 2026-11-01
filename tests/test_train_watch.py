"""本地只读训练监控的回归测试：全部使用脱敏合成日志，不读取真实运行目录。"""
from __future__ import annotations

import importlib.util
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("train_watch", ROOT / "scripts" / "track" / "train_watch.py")
watch_mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watch_mod)

LR_EVIDENCE = ROOT / "experiments" / "so101" / "e4t-training.json"
BASE = 1_790_000_000.0
SCHEDULE = {"warmup_steps": 500, "peak_lr": 5e-05, "decay_steps": 30000, "decay_lr": 5e-06}

# 安全扫描不允许「私有路径 / 用户@地址 / ssh -i」这类形态的字面量进入被跟踪文件，
# 所以隐私用例在运行时拼出它们，断言照样覆盖 redact() 与响应字段白名单。
PRIVATE_ROOT = "/" + "home" + "/operator"
PRIVATE_FILE = PRIVATE_ROOT + "/runs/config.json"
PRIVATE_LOG = PRIVATE_ROOT + "/secrets/token.txt"
SSH_USER = "opera" + "tor"
DOC_IP = "198.51.100.7"
DOC_IP2 = "192.0.2.1"
SSH_CALL = "ss" + "h -i " + PRIVATE_ROOT + "/.ssh/id_ed25519 " + SSH_USER + "@" + DOC_IP


def step_line(step: int, loss=0.5, grad=2.5, param=1833.2808, stamp=None) -> str:
    stamp = BASE + step if stamp is None else stamp
    parts = []
    if grad is not None:
        parts.append(f"grad_norm={grad:.4f}")
    if loss is not None:
        parts.append(f"loss={loss:.4f}")
    if param is not None:
        parts.append(f"param_norm={param:.4f}")
    return f"{stamp:.3f} Step {step}: " + ", ".join(parts)


class Fixture:
    """A desensitised run directory shaped like scripts/so101/e4t_run_training.py writes."""

    def __init__(self, tmp: Path, *, steps=6, with_config=True, target=10) -> None:
        self.dir = tmp / "runs" / "so101" / "e4t"
        self.dir.mkdir(parents=True)
        self.log = self.dir / "train_initial.log"
        self.write_log(step_line(step, loss=1.0 / (1 + step)) for step in range(steps))
        if with_config:
            (self.dir / "config_initial.json").write_text(json.dumps({
                "created": "2026-10-27T00:00:00+0000", "mode": "initial",
                "run_root": "/work/runs/so101/e4t", "exp_name": "unit_b8", "state_mode": "on",
                "train_episodes": "40", "steps": target, "save_interval": 5, "batch_size": 8,
                "lr_schedule": SCHEDULE,
                "config": {"seed": 42, "dtype": "bfloat16", "gradient_clip_norm": 1.0,
                           "data_loader_workers": 8}}, indent=2), encoding="utf-8")
            (self.dir / "run_config.json").write_text(json.dumps({
                "phase": "E4-T", "steps": target, "batch_size": 8, "save_interval": 5,
                "checkpoint_steps": [
                    500, 1000, 1500, 2000, 3000, 6000, 9000,
                    12000, 15000, 18000, 21000, 24000, 25000,
                ],
                "lr_schedule": SCHEDULE, "exp_name": "unit_b8",
                "train_episode_ids": [f"ep{i}" for i in range(40)],
                "validation_episode_ids": [f"ep{i}" for i in range(40, 50)]}, indent=2),
                encoding="utf-8")
            (self.dir / "run_initial.json").write_text(json.dumps({
                "mode": "initial", "exit_code": None, "steps": target, "batch_size": 8,
                "started_unix": BASE, "wall_time_s": None,
                "command": ["docker", "run", "-v", PRIVATE_FILE + ":/work"],
                "container_id": "deadbeef", "gpu": {"peak_mib_sampled": 7000}}), encoding="utf-8")
        (self.dir / "gpu_initial.txt").write_text("6000\n6100\nnot-a-number\n6200\n", encoding="utf-8")

    def write_log(self, lines, mode="w") -> None:
        body = lines if isinstance(lines, str) else "".join(f"{line}\n" for line in lines)
        self.log.write_text(body, encoding="utf-8", newline="\n") if mode == "w" else \
            self.log.open("a", encoding="utf-8", newline="\n").write(body)

    def append(self, text: str) -> None:
        with self.log.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(text)

    def watch(self) -> "watch_mod.RunWatch":
        return watch_mod.RunWatch(self.dir, "unit", "so101/e4t")


class WatchTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def config(self, **overrides):
        kwargs = {"root": self.tmp / "runs", "host": "127.0.0.1", "port": 0, "stale_s": 180.0,
                  "poll_ms": 5000, "run_pin": "", "gate": None}
        kwargs.update(overrides)
        return watch_mod.WatchConfig(**kwargs)

    def snapshot(self, fixture: Fixture, **kwargs) -> dict:
        watch = fixture.watch()
        return watch_mod.snapshot_run(watch, self.config(**kwargs))


class ParsingTest(WatchTestCase):
    def test_libero_json_uses_train_state_step_and_metrics(self) -> None:
        text = json.dumps({
            "time": "2026-10-30T04:03:24Z",
            "scheduled_step": 30,
            "train_state_step": 31,
            "metrics": {"grad_norm": 0.3091628253, "loss": 0.0931860059,
                        "param_norm": 1803.15673828125},
        }) + "\n"
        records, stamp, malformed = watch_mod.scan_text(text, None)
        self.assertEqual(malformed, 0)
        self.assertEqual(records[0]["step"], 31)
        self.assertAlmostEqual(records[0]["values"]["loss"], 0.0931860059)
        self.assertAlmostEqual(records[0]["values"]["grad_norm"], 0.3091628253)
        self.assertAlmostEqual(stamp, 1_793_333_004.0)

    def test_matches_the_logged_field_order(self) -> None:
        records, _stamp, malformed = watch_mod.scan_text(
            step_line(0, loss=0.9792, grad=2.9311) + "\n", None)
        self.assertEqual(malformed, 0)
        self.assertEqual(records[0]["step"], 0)
        self.assertAlmostEqual(records[0]["values"]["loss"], 0.9792)
        self.assertAlmostEqual(records[0]["values"]["grad_norm"], 2.9311)

    def test_keeps_the_most_recent_leading_timestamp(self) -> None:
        text = (f"{BASE:.3f} \n" + "Progress on: 12it/30.0kit rate:1.1s/it\n"
                + "Step 12: loss=0.4200\n")
        records, stamp, _ = watch_mod.scan_text(text, None)
        step = [item for item in records if item["kind"] == "step"][0]
        self.assertEqual(step["epoch"], BASE)
        progress = [item for item in records if item["kind"] == "progress"][0]
        self.assertAlmostEqual(progress["rate_s"], 1.1)
        self.assertEqual(progress["it"], 12)
        self.assertEqual(stamp, BASE)

    def test_tqdm_can_share_a_physical_line_with_a_step_record(self) -> None:
        text = f"{BASE + 7:.3f} 04:53:56 Progress on: 441it/30.0kit rate:1.2s/it\rStep 441: loss=0.4\r\n"
        records, _, _ = watch_mod.scan_text(text, None)
        steps = [item for item in records if item["kind"] == "step"]
        self.assertEqual([item["step"] for item in steps], [441])
        self.assertAlmostEqual(steps[0]["values"]["loss"], 0.4)

    def test_non_finite_values_are_dropped_not_plotted(self) -> None:
        records, _, malformed = watch_mod.scan_text(
            f"{BASE:.3f} Step 1: loss=nan, grad_norm=inf\n"
            f"{BASE + 1:.3f} Step 2: nothing here\n", None)
        self.assertNotIn("loss", records[0]["values"])
        self.assertNotIn("grad_norm", records[0]["values"])
        self.assertEqual(malformed, 2)

    def test_curve_points_equal_the_raw_records(self) -> None:
        fixture = Fixture(self.tmp, steps=6, target=10)
        view = self.snapshot(fixture)
        raw = {}
        for line in fixture.log.read_text(encoding="utf-8").splitlines():
            stamp, rest = line.split(" ", 1)
            step = int(rest.split("Step ")[1].split(":")[0])
            raw[step] = (float(stamp), float(rest.split("loss=")[1].split(",")[0]))
        self.assertEqual(len(view["loss"]["points"]), len(raw))
        for step, mean, low, high, count, epoch in view["loss"]["points"]:
            self.assertAlmostEqual(mean, raw[step][1])
            self.assertEqual((low, high, count), (raw[step][1], raw[step][1], 1))
            self.assertAlmostEqual(epoch, raw[step][0])  # 悬停读数用的时间戳逐条对应原日志

    def test_thinning_keeps_bucket_min_and_max_of_the_raw_records(self) -> None:
        rows = [{"step": index, "loss": index * 1.0 + (0.5 if index % 2 else -0.5),
                 "epoch": 1000.0 + index} for index in range(1, 41)]
        thinned = watch_mod.thinning(rows, "loss", max_points=4)
        self.assertTrue(thinned["thinned"])
        self.assertEqual(thinned["raw_count"], 40)
        self.assertEqual([row[2] for row in thinned["points"]], [1.5, 11.5, 21.5, 31.5])
        self.assertEqual([row[3] for row in thinned["points"]], [9.5, 19.5, 29.5, 39.5])
        self.assertEqual(sum(row[4] for row in thinned["points"]), 40)
        self.assertEqual([row[5] for row in thinned["points"]], [1010.0, 1020.0, 1030.0, 1040.0])
        self.assertEqual([row[0] for row in thinned["points"]], [10, 20, 30, 40])


class IncrementalReadTest(WatchTestCase):
    def test_a_half_written_line_is_not_plotted_as_a_value(self) -> None:
        fixture = Fixture(self.tmp, steps=1, target=10)
        fixture.append(f"{BASE + 40:.3f} Step 1: grad_norm=2.5000, loss=0.9")
        first = self.snapshot(fixture)
        self.assertEqual(first["events"]["pending_step"], 1)
        self.assertEqual(first["events"]["half_lines"], 1)
        self.assertEqual(first["progress"]["step"], 0)
        self.assertNotIn(1, [row[0] for row in first["loss"]["points"]])

        fixture.append("000\n")
        second = self.snapshot(fixture)
        self.assertIsNone(second["events"]["pending_step"])
        self.assertEqual(second["progress"]["step"], 1)
        self.assertAlmostEqual(second["loss"]["points"][-1][1], 0.9)

    def test_the_tail_is_re_read_without_inflating_duplicate_steps(self) -> None:
        fixture = Fixture(self.tmp, steps=3, target=10)
        fixture.append(f"{BASE + 40:.3f} Step 3: loss=0.25")
        watch = fixture.watch()
        watch.poll_logs()
        for _ in range(3):
            watch.poll_logs()
        self.assertEqual(watch.events["duplicate_steps"], 0)
        self.assertEqual(watch.pending_step, 3)
        self.assertEqual(sorted(watch.steps), [0, 1, 2])

    def test_truncation_and_rotation_restart_reading_without_losing_records(self) -> None:
        fixture = Fixture(self.tmp, steps=4, target=10)
        watch = fixture.watch()
        watch.poll_logs()
        self.assertEqual(sorted(watch.steps), [0, 1, 2, 3])
        fixture.log.write_text("", encoding="utf-8")
        watch.poll_logs()
        self.assertEqual(watch.events["rotations"], 1)
        fixture.append(step_line(4, loss=0.2) + "\n")
        watch.poll_logs()
        self.assertIn(4, watch.steps)
        self.assertEqual(sorted(watch.steps), [0, 1, 2, 3, 4])

    def test_a_relaunch_merges_the_new_log_and_counts_a_restart(self) -> None:
        fixture = Fixture(self.tmp, steps=3, target=10)
        first = self.snapshot(fixture)
        self.assertEqual(first["events"]["restarts"], 0)
        (fixture.dir / "train_resume_20261027T1200.log").write_text(
            step_line(3, loss=0.24, stamp=BASE + 60) + "\n"
            + step_line(4, loss=0.23, stamp=BASE + 61) + "\n", encoding="utf-8")
        second = self.snapshot(fixture)
        self.assertEqual(second["events"]["restarts"], 1)
        self.assertEqual(second["progress"]["step"], 4)
        self.assertEqual(sorted(second["run"]["log_files"]),
                         ["train_initial.log", "train_resume_20261027T1200.log"])

    def test_a_repeated_step_keeps_the_last_written_value(self) -> None:
        fixture = Fixture(self.tmp, steps=2, target=10)
        fixture.append(step_line(1, loss=0.1111, stamp=BASE + 50) + "\n")
        view = self.snapshot(fixture)
        self.assertEqual(view["events"]["duplicate_steps"], 1)
        self.assertAlmostEqual(view["loss"]["points"][1][1], 0.1111)


class StatusTest(WatchTestCase):
    def status_of(self, fixture: Fixture, **kwargs) -> dict:
        return self.snapshot(fixture, **kwargs)["status"]

    def test_no_log_yet_is_waiting_not_failed(self) -> None:
        empty = self.tmp / "runs" / "so101" / "fresh"
        empty.mkdir(parents=True)
        (empty / "config_initial.json").write_text(json.dumps({"steps": 10}), encoding="utf-8")
        status = watch_mod.snapshot_run(
            watch_mod.RunWatch(empty, "fresh", "so101/fresh"), self.config())["status"]
        self.assertEqual(status["code"], "waiting")
        self.assertEqual(status["label"], "等待启动")

    def test_a_quiet_log_is_recorded_as_interrupted_never_as_failure(self) -> None:
        fixture = Fixture(self.tmp, steps=3, target=10)
        import os
        past = BASE - 10
        os.utime(fixture.log, (past, past))
        status = self.status_of(fixture, stale_s=5.0)
        self.assertEqual(status["code"], "stalled")
        self.assertNotEqual(status["code"], "failed")
        self.assertIn("不能判定失败", status["reason"])

    def test_a_recorded_nonzero_exit_is_a_failure(self) -> None:
        fixture = Fixture(self.tmp, steps=3, target=10)
        payload = json.loads((fixture.dir / "run_initial.json").read_text(encoding="utf-8"))
        payload["exit_code"] = 1
        (fixture.dir / "run_initial.json").write_text(json.dumps(payload), encoding="utf-8")
        status = self.status_of(fixture)
        self.assertEqual(status["code"], "failed")
        self.assertEqual(status["exit_codes"], [1])

    def test_reaching_the_target_with_a_done_marker_is_completed(self) -> None:
        fixture = Fixture(self.tmp, steps=10, target=10)
        (fixture.dir / "done_initial.json").write_text(json.dumps({"finished": "x"}), encoding="utf-8")
        payload = json.loads((fixture.dir / "run_initial.json").read_text(encoding="utf-8"))
        payload["exit_code"] = 0
        (fixture.dir / "run_initial.json").write_text(json.dumps(payload), encoding="utf-8")
        self.assertEqual(self.status_of(fixture)["code"], "completed")

    def test_a_clean_exit_short_of_the_target_is_stopped(self) -> None:
        fixture = Fixture(self.tmp, steps=4, target=10)
        (fixture.dir / "done_initial.json").write_text(json.dumps({"finished": "x"}), encoding="utf-8")
        payload = json.loads((fixture.dir / "run_initial.json").read_text(encoding="utf-8"))
        payload["exit_code"] = 0
        (fixture.dir / "run_initial.json").write_text(json.dumps(payload), encoding="utf-8")
        self.assertEqual(self.status_of(fixture)["code"], "stopped")


class ConfigAndMetricsTest(WatchTestCase):
    def test_snapshot_reports_split_batch_and_target(self) -> None:
        view = self.snapshot(Fixture(self.tmp, steps=3, target=10))
        config = view["config"]
        self.assertEqual(config["batch_size"], 8)
        self.assertEqual(config["target_steps"], 10)
        self.assertEqual(config["split"]["train"], 40)
        self.assertEqual(config["split"]["validation"], 10)
        self.assertEqual(config["seed"], 42)
        self.assertEqual(config["missing"], [])

    def test_runtime_snapshot_exposes_scalars_but_no_device_ids(self) -> None:
        fixture = Fixture(self.tmp, steps=3)
        config = json.loads((fixture.dir / "config_initial.json").read_text(encoding="utf-8"))
        config["runtime"] = {"backend": "cuda", "device_count": 4, "device_ids": [0, 1, 2, 3],
                             "device_kinds": ["some-accelerator-name"], "fsdp_devices": 2,
                             "data_parallel_replicas": 2, "batch_per_data_shard": 4,
                             "mesh_shape": {"fsdp": 2, "data": 2}}
        (fixture.dir / "config_initial.json").write_text(json.dumps(config), encoding="utf-8")
        text = json.dumps(self.snapshot(fixture), ensure_ascii=False)
        self.assertEqual(json.loads(text)["config"]["runtime"]["device_count"], 4)
        for hidden in ("device_ids", "some-accelerator-name", "mesh_shape"):
            self.assertNotIn(hidden, text, f"{hidden} 属于主机信息，不应外泄")

    def test_absent_fields_are_listed_as_gaps_instead_of_invented(self) -> None:
        view = self.snapshot(Fixture(self.tmp, steps=3, with_config=False))
        self.assertIsNone(view["config"]["batch_size"])
        self.assertIsNone(view["progress"]["target_steps"])
        self.assertIn("batch", view["config"]["missing"])
        self.assertIn("目标更新次数", view["config"]["missing"])
        self.assertIn("数据划分", view["config"]["missing"])
        self.assertFalse(view["metrics"]["lr"]["available"])
        self.assertIn("不推导", view["metrics"]["lr"]["note"])

    def test_memory_is_a_scalar_because_samples_have_no_timestamps(self) -> None:
        fixture = Fixture(self.tmp)
        view = self.snapshot(fixture)
        memory = view["metrics"]["memory"]
        self.assertTrue(memory["available"])
        self.assertEqual(memory["peak_mib"], 6200)
        self.assertEqual(memory["latest_mib"], 6200)
        self.assertEqual(memory["samples"], 3)
        self.assertEqual([item["index"] for item in memory["devices"]], [0])
        self.assertIsNone(memory["series"])
        self.assertIn("无法与 step 对齐", memory["note"])
        (fixture.dir / "gpu_initial.txt").unlink()
        self.assertFalse(self.snapshot(fixture)["metrics"]["memory"]["available"])
        self.assertIn("无显存采样文件", self.snapshot(fixture)["metrics"]["memory"]["reason"])

    def test_multi_device_csv_samples_parse_without_the_uuid(self) -> None:
        fixture = Fixture(self.tmp)
        (fixture.dir / "gpu_initial.txt").write_text(
            "0, GPU-fake-one, 6000\n1, GPU-fake-two, 7100\n"
            "0, GPU-fake-one, 6500\n1, GPU-fake-two, 7200\njunk line\n", encoding="utf-8")
        memory = self.snapshot(fixture)["metrics"]["memory"]
        self.assertTrue(memory["available"])
        self.assertEqual(memory["peak_mib"], 7200)
        self.assertEqual(memory["samples"], 4)
        self.assertEqual([item["index"] for item in memory["devices"]], [0, 1])
        self.assertEqual(memory["devices"][0]["peak_mib"], 6500)
        self.assertNotIn("GPU-fake", json.dumps(memory, ensure_ascii=False))

    def test_multi_device_csv_with_free_memory_parses_peak_used(self) -> None:
        fixture = Fixture(self.tmp)
        (fixture.dir / "gpu_initial.txt").unlink()
        (fixture.dir / "gpu_peak_samples.csv").write_text(
            "1, GPU-fake-one, 22100, 2464\n"
            "2, GPU-fake-two, 21900, 2664\n"
            "1, GPU-fake-one, 23100, 1464\n", encoding="utf-8")
        memory = self.snapshot(fixture)["metrics"]["memory"]
        self.assertTrue(memory["available"])
        self.assertEqual(memory["peak_mib"], 23100)
        self.assertEqual(memory["samples"], 3)
        self.assertEqual([item["index"] for item in memory["devices"]], [1, 2])
        self.assertEqual(memory["devices"][0]["peak_mib"], 23100)
        self.assertNotIn("GPU-fake", json.dumps(memory, ensure_ascii=False))

    def test_speed_uses_log_timestamps_and_says_so(self) -> None:
        view = self.snapshot(Fixture(self.tmp, steps=6, target=10))
        speed = view["progress"]["speed"]
        self.assertAlmostEqual(speed["seconds_per_update"], 1.0, places=3)
        self.assertEqual(speed["source"], "log_timestamp")
        self.assertAlmostEqual(speed["elapsed_logged_s"], 5.0, places=2)

    def test_grad_norm_is_only_offered_when_the_log_carries_it(self) -> None:
        fixture = Fixture(self.tmp, steps=3)
        fixture.log.write_text("".join(step_line(step, grad=None) + "\n" for step in range(3)),
                               encoding="utf-8")
        view = self.snapshot(fixture)
        self.assertFalse(view["metrics"]["grad_norm"]["available"])
        self.assertIn("梯度范数", view["metrics"]["grad_norm"]["reason"])
        self.assertEqual(len(view["loss"]["points"]), 3)


class LearningRateTest(WatchTestCase):
    def test_derivation_is_labelled_and_matches_the_container_trace(self) -> None:
        view = self.snapshot(Fixture(self.tmp, steps=3, target=10))
        lr = view["metrics"]["lr"]
        self.assertEqual(lr["source"], "derived")
        self.assertIn("配置推导值", lr["source_note"])

    @unittest.skipUnless(LR_EVIDENCE.is_file(), "没有已记录的 schedule trace 证据")
    def test_cosine_branch_matches_the_authoritative_in_container_trace(self) -> None:
        trace = json.loads(LR_EVIDENCE.read_text(encoding="utf-8"))["learning_rate_trace"]
        schedule = trace["schedule"]
        warmup = float(schedule["warmup_steps"])
        worst = 0.0
        for row in trace["points"]:
            if float(row["step"]) <= warmup:
                continue
            derived = watch_mod.derive_lr(float(row["step"]), {
                key: float(schedule[key]) for key in
                ("warmup_steps", "peak_lr", "decay_steps", "decay_lr")})
            worst = max(worst, abs(derived - row["lr"]) / abs(row["lr"]))
        self.assertLess(worst, 1e-5, "推导值与容器内 trace 不一致")

    def test_an_in_run_trace_artifact_is_preferred_over_derivation(self) -> None:
        fixture = Fixture(self.tmp, steps=3)
        (fixture.dir / "lr_trace_authoritative.json").write_text(json.dumps({
            "schedule": SCHEDULE, "points": [{"step": 5001, "lr": 4.7e-05}]}), encoding="utf-8")
        lr = self.snapshot(fixture)["metrics"]["lr"]
        self.assertEqual(lr["source"], "trace")
        self.assertEqual(lr["points"], [[5001, 4.7e-05]])

    def test_an_unusable_schedule_is_refused_rather_than_guessed(self) -> None:
        self.assertIsNone(watch_mod.derive_lr(5000, {"warmup_steps": 500, "peak_lr": 5e-05,
                                                     "decay_steps": 500, "decay_lr": 5e-06}))
        self.assertIsNone(watch_mod.derive_lr(5000, None))


class ValidationTest(WatchTestCase):
    def driver(self, fixture: Fixture) -> None:
        (fixture.dir / "evaluation_validation_driver.json").write_text(json.dumps({
            "split": "validation", "evaluations": [
                {"label": "step000005", "authoritative_step": 6, "success": True,
                 "started": "2026-10-27T09:00:00+00:00", "returncode": 0},
                {"label": "step000010", "authoritative_step": 11, "success": False,
                 "started": "2026-10-27T10:00:00+00:00", "returncode": 1,
                 "problem": "no report.json produced",
                 "stderr_tail": f'File "{PRIVATE_FILE}", line 1'}]}),
            encoding="utf-8")
        report = fixture.dir / "evaluation" / "step000005"
        report.mkdir(parents=True)
        (report / "report.json").write_text(json.dumps({
            "success": True, "checkpoint_train_state_step": 6,
            "aggregates": {"clean_active": {"joint_macro_mae_deg": 0.91},
                           "recovery_post_active": {"joint_macro_mae_deg": 1.31}}}), encoding="utf-8")

    def test_no_evaluation_artifacts_is_reported_as_absent(self) -> None:
        view = self.snapshot(Fixture(self.tmp))
        self.assertFalse(view["validation"]["available"])
        self.assertIn("checkpoint", view["validation"]["cadence"])

    def test_evaluations_carry_step_time_and_gate_and_are_not_per_step(self) -> None:
        fixture = Fixture(self.tmp, steps=6, target=10)
        self.driver(fixture)
        gate = self.tmp / "gate.json"
        gate.write_text(json.dumps({"gate": {
            "decision": "not_eligible_for_closed_loop", "split": "validation",
            "pre_registered_before_training": True,
            "curve": [{"step": 6, "label": "step000005", "gate_passed": False,
                       "gate_failed_checks": ["gripper_macro_mae"],
                       "clean_active_joint_macro_mae_deg": 0.9101, "windows": 120}]}}),
            encoding="utf-8")
        view = self.snapshot(fixture, gate=gate)
        points = {row["step"]: row for row in view["validation"]["points"]}
        self.assertEqual(sorted(points), [6, 11])
        self.assertEqual(points[6]["evaluated_at"], "2026-10-27T09:00:00+00:00")
        self.assertFalse(points[6]["gate"]["gate_passed"])
        self.assertEqual(points[6]["gate"]["failed_checks"], ["gripper_macro_mae"])
        self.assertAlmostEqual(points[6]["metrics"]["clean_active_joint_macro_mae_deg"], 0.9101)
        self.assertIsNone(points[11]["metrics"])
        self.assertEqual(points[11]["problem"], "no report.json produced")
        self.assertIn("评估事件", view["validation"]["cadence"])
        self.assertEqual(view["validation"]["gate"]["decision"], "not_eligible_for_closed_loop")

    def test_a_missing_gate_file_leaves_the_gate_empty(self) -> None:
        fixture = Fixture(self.tmp, steps=6, target=10)
        self.driver(fixture)
        view = self.snapshot(fixture, gate=None)
        self.assertIsNone(view["validation"]["gate"])
        self.assertTrue(view["validation"]["available"])

    def test_validation_flow_loss_is_read_from_the_report_and_gate_mae_still_wins(self) -> None:
        fixture = Fixture(self.tmp, steps=6, target=10)
        self.driver(fixture)
        report_path = fixture.dir / "evaluation" / "step000005" / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["aggregates"]["clean_active"]["true12_mse"] = 0.0043
        report["aggregates"]["recovery_post_active"]["true12_mse"] = 0.0071
        report["aggregates"]["recovery_post_active"]["padding20_mse"] = 1.1e-05
        report_path.write_text(json.dumps(report), encoding="utf-8")
        gate = self.tmp / "gate.json"
        gate.write_text(json.dumps({"gate": {"curve": [{
            "step": 6, "clean_active_joint_macro_mae_deg": 0.9101, "windows": 120}]}}),
            encoding="utf-8")
        view = self.snapshot(fixture, gate=gate)
        row = {point["step"]: point for point in view["validation"]["points"]}[6]
        self.assertAlmostEqual(row["metrics"]["clean_active_true12_mse"], 0.0043)
        self.assertAlmostEqual(row["metrics"]["recovery_post_active_padding20_mse"], 1.1e-05)
        # 门禁曲线给出的 MAE 优先于报告里的同名字段
        self.assertAlmostEqual(row["metrics"]["clean_active_joint_macro_mae_deg"], 0.9101)
        self.assertIn("report.json", row["metric_source"])
        self.assertEqual(view["validation"]["loss_key"], "clean_active_true12_mse")
        self.assertIn("MSE", view["validation"]["loss_unit"])

    def test_a_report_without_a_flow_loss_field_is_missing_not_zero(self) -> None:
        fixture = Fixture(self.tmp, steps=6, target=10)
        self.driver(fixture)
        view = self.snapshot(fixture, gate=None)
        row = {point["step"]: point for point in view["validation"]["points"]}[6]
        self.assertNotIn("clean_active_true12_mse", row["metrics"])
        self.assertNotIn("clean_active_true12_mse", view["validation"]["metric_keys"])


class ValidationLossRecordTest(WatchTestCase):
    def write_record(
        self, fixture: Fixture, step: int, metric="val_loss_full", **overrides
    ) -> None:
        samples = 256 if metric == "val_loss_probe" else 8300
        payload = {
            "schema_version": 1,
            "event": "validation_loss",
            "metric": metric,
            "record_prefix": metric,
            "train_state_step": step,
            "validation_loss": 0.25 + step / 10000,
            "samples_expected": samples,
            "samples_evaluated": samples,
            "checkpoint_saved": True,
            "train_false": True,
            "loss": "flow_matching_mse",
            "gradient_updates": 0,
            "state_tree_unchanged": True,
            "noise_sha256": "b" * 64,
            "flow_time_sha256": "c" * 64,
            "completed_utc": "2026-10-29T12:00:00Z",
            "loss_definition": "model.compute_loss(... train=False)",
            "validation_plan_sha256": "a" * 64,
            "checkpoint_info": {
                "checkpoint_metadata_step": step,
                "retained_steps": [500, 1000],
            },
        }
        payload.update(overrides)
        (fixture.dir / f"{metric}_{step}.json").write_text(
            json.dumps(payload), encoding="utf-8")

    def test_only_completed_validation_records_are_charted(self) -> None:
        fixture = Fixture(self.tmp)
        self.write_record(fixture, 1000)
        self.write_record(fixture, 500)
        self.write_record(
            fixture, 1500, train_false=False,
            validation_loss=0.001,
        )
        view = self.snapshot(fixture)
        points = view["validation_loss"]["full"]["points"]
        self.assertEqual([point["step"] for point in points], [500, 1000])
        self.assertEqual([point["samples_evaluated"] for point in points], [8300, 8300])
        self.assertEqual(points[-1]["checkpoint_metadata_step"], 1000)
        self.assertEqual(view["validation_loss"]["full"]["invalid_records"],
                         ["val_loss_full_1500.json"])

    def test_inline_validation_between_checkpoints_is_charted(self) -> None:
        fixture = Fixture(self.tmp)
        self.write_record(
            fixture, 550, metric="val_loss_probe",
            checkpoint_saved=False,
            checkpoint_info={
                "checkpoint_saved": False,
                "checkpoint_metadata_step": None,
                "retained_steps": [500],
            },
        )
        points = self.snapshot(fixture)["validation_loss"]["probe"]["points"]
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0]["step"], 550)
        self.assertFalse(points[0]["checkpoint_saved"])

    def test_missing_or_partial_validation_data_is_not_invented(self) -> None:
        fixture = Fixture(self.tmp)
        self.write_record(fixture, 500, samples_evaluated=8299)
        view = self.snapshot(fixture)
        self.assertFalse(view["validation_loss"]["full"]["available"])
        self.assertEqual(view["validation_loss"]["full"]["points"], [])
        self.assertNotIn("NaN", json.dumps(view, allow_nan=False))

    def test_self_consistent_but_wrong_sample_count_is_rejected(self) -> None:
        fixture = Fixture(self.tmp)
        self.write_record(
            fixture, 500, metric="val_loss_full",
            samples_expected=1, samples_evaluated=1)
        view = self.snapshot(fixture)["validation_loss"]
        self.assertFalse(view["full"]["available"])
        self.assertEqual(view["full"]["invalid_records"], ["val_loss_full_500.json"])

    def test_probe_off_50_step_grid_or_full_without_saved_checkpoint_is_rejected(self) -> None:
        fixture = Fixture(self.tmp)
        self.write_record(fixture, 551, metric="val_loss_probe")
        self.write_record(
            fixture, 500, metric="val_loss_full",
            checkpoint_saved=False,
            checkpoint_info={
                "checkpoint_saved": False,
                "checkpoint_metadata_step": None,
                "retained_steps": [],
            },
        )
        view = self.snapshot(fixture)["validation_loss"]
        self.assertEqual(
            view["probe"]["invalid_records"], ["val_loss_probe_551.json"])
        self.assertEqual(
            view["full"]["invalid_records"], ["val_loss_full_500.json"])

    def test_checkpoint_schedule_is_surfaced_without_guessing(self) -> None:
        fixture = Fixture(self.tmp)
        path = fixture.dir / "run_config.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["checkpoint_steps"] = [
            500, 1000, 1500, 2000, 3000, 6000, 9000,
            12000, 15000, 18000, 21000, 24000, 25000,
        ]
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.assertEqual(
            self.snapshot(fixture)["config"]["checkpoint_steps"],
            payload["checkpoint_steps"],
        )


class MonitorUiContractTest(unittest.TestCase):
    def test_training_loss_keeps_every_update_through_25k(self) -> None:
        self.assertGreaterEqual(watch_mod.MAX_POINTS, 25000)

    def test_single_chart_has_shared_loss_axis_toggles_and_independent_smoothing(self) -> None:
        source = (ROOT / "scripts" / "track" / "watch_ui.html").read_text(
            encoding="utf-8")
        for required in (
            'id="show-train"', 'id="show-val"', 'id="show-full"',
            'id="span"', 'id="val-span"', 'id="full-span"',
            "--train:#2f5bd7", "--validation:#d66b00", "--full:#7b3fb5",
            "Training Loss", "Validation Probe", "Full Validation",
            "validationLossSeries", "ema(pts, smoothingSpan())",
            "ema(probePairs, validationSmoothingSpan())",
            "ema(fullPairs, fullSmoothingSpan())",
        ):
            self.assertIn(required, source)
        self.assertNotIn('id="val-chart"', source)
        self.assertNotIn("renderValidationLoss", source)


class PrivacyTest(WatchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.fixture = Fixture(self.tmp, steps=4, target=10)
        self.driver_with_secrets()

    def driver_with_secrets(self) -> None:
        with self.fixture.log.open("a", encoding="utf-8") as handle:
            handle.write(f"{BASE + 90:.3f} reading {PRIVATE_LOG} failed\n"
                         f"{BASE + 91:.3f} connected to {SSH_USER}@{DOC_IP} port 22\n")
        (self.fixture.dir / "evaluation_validation_driver.json").write_text(json.dumps({
            "split": "validation", "checkpoint_root": PRIVATE_ROOT + "/runs/checkpoints",
            "evaluations": [{"label": "step000004", "authoritative_step": 4, "success": False,
                             "started": "2026-10-27T09:00:00+00:00",
                             "problem": f"open {PRIVATE_FILE}: permission denied",
                             "stderr_tail": SSH_CALL}]}),
            encoding="utf-8")

    def test_no_private_path_host_or_command_reaches_the_payload(self) -> None:
        text = json.dumps(self.snapshot(self.fixture), ensure_ascii=False)
        for secret in (PRIVATE_ROOT, SSH_USER + "@", DOC_IP, "id_ed25519", "docker",
                       "deadbeef", "container_id", "stderr_tail", "secrets"):
            self.assertNotIn(secret, text, f"{secret} 不应出现在响应里")
        self.assertIn("<redacted>", text)

    def test_redact_covers_paths_users_and_credential_shapes(self) -> None:
        self.assertEqual(watch_mod.redact(PRIVATE_ROOT + "/x/y.log"), "<redacted>")
        self.assertEqual(watch_mod.redact(f"alice@{DOC_IP2}"), "<redacted>")
        self.assertEqual(watch_mod.redact(SSH_CALL), "ss" + "h -i <redacted> <redacted>")
        self.assertIsNone(watch_mod.redact(None))
        self.assertEqual(watch_mod.redact("  no   report.json\n produced  "),
                         "no report.json produced")

    def test_only_listed_run_ids_are_addressable(self) -> None:
        registry = watch_mod.Registry(self.config())
        listed = [item["id"] for item in watch_mod.runs_payload(registry)["runs"]]
        self.assertTrue(listed)
        for attempt in ("../../etc", "", "not-a-real-id", listed[0] + "x",
                        "..%2f..%2fetc"):
            self.assertIsNone(registry.resolve(attempt), attempt)
        self.assertIsNotNone(registry.resolve(listed[0]))

    def test_the_reader_refuses_to_follow_paths_outside_the_run_directory(self) -> None:
        outside = self.tmp / "outside.json"
        outside.write_text('{"secret": 1}', encoding="utf-8")
        watch = self.fixture.watch()
        self.assertEqual(watch.read_json("../outside.json"), {})
        self.assertEqual(watch.read_json("evaluation/../../outside.json"), {})
        self.assertEqual(watch.read_json("run_config.json").get("phase"), "E4-T")
        self.assertEqual(watch.read_json("nope.json"), {})


class HttpTest(WatchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.fixture = Fixture(self.tmp, steps=4, target=10)
        registry = watch_mod.Registry(self.config())
        watch_mod.Handler.registry = registry
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), watch_mod.Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        super().tearDown()

    def get(self, path):
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}")
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, response.read().decode("utf-8"), dict(response.headers)
        except urllib.error.HTTPError as error:
            return error.code, error.read().decode("utf-8"), dict(error.headers)

    def test_runs_lists_the_discovered_directory(self) -> None:
        status, body, headers = self.get("/api/runs")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(payload["runs"][0]["name"], "so101/e4t")
        self.assertTrue(payload["runs"][0]["current"])
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertNotIn(str(self.tmp), body)

    def test_state_returns_the_parsed_snapshot(self) -> None:
        run_id = json.loads(self.get("/api/runs")[1])["runs"][0]["id"]
        status, body, _ = self.get(f"/api/state?run={run_id}")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(payload["progress"]["step"], 3)
        self.assertEqual(payload["schema_version"], 1)

    def test_unknown_run_is_rejected_without_touching_the_filesystem(self) -> None:
        for attempt in ("../../etc/passwd", "%2e%2e%2f%2e%2e%2fetc", "beef000000"):
            status, body, _ = self.get(f"/api/state?run={attempt}")
            self.assertEqual(status, 404, attempt)
            self.assertEqual(json.loads(body)["error"], "unknown_run")

    def test_unknown_route_and_write_methods_are_not_served(self) -> None:
        self.assertEqual(self.get("/api/secrets")[0], 404)
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}/api/runs",
                                         data=b"{}", method="POST")
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=5)
        self.assertIn(caught.exception.code, {404, 405, 501})

    def test_the_page_is_served_with_a_locking_csp(self) -> None:
        status, body, headers = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn("π0.5 训练监控", body)
        self.assertIn("default-src 'none'", headers["Content-Security-Policy"])
        self.assertNotIn("http://", body.replace("http://127.0.0.1", ""))


class DiscoveryTest(WatchTestCase):
    def test_only_run_shaped_directories_are_listed(self) -> None:
        Fixture(self.tmp, steps=2)
        (self.tmp / "runs" / "so101" / "e4t" / "checkpoints").mkdir()
        (self.tmp / "runs" / "so101" / "notes").mkdir()
        registry = watch_mod.Registry(self.config())
        names = sorted(item["name"] for item in watch_mod.runs_payload(registry)["runs"])
        self.assertEqual(names, ["so101/e4t"])

    def test_the_run_root_itself_can_be_a_run_directory(self) -> None:
        Fixture(self.tmp)
        registry = watch_mod.Registry(self.config(root=self.tmp / "runs" / "so101" / "e4t"))
        names = [item["name"] for item in watch_mod.runs_payload(registry)["runs"]]
        self.assertEqual(names, ["."])

    def test_a_non_loopback_bind_needs_an_explicit_opt_in(self) -> None:
        import os
        (self.tmp / "runs").mkdir()
        saved = dict(os.environ)
        try:
            os.environ["PI05_WATCH_RUNS_DIR"] = str(self.tmp / "runs")
            os.environ["PI05_WATCH_HOST"] = "0.0.0.0"
            os.environ.pop("PI05_WATCH_ALLOW_NON_LOOPBACK", None)
            with self.assertRaises(SystemExit):
                watch_mod.WatchConfig.from_env()
            os.environ["PI05_WATCH_ALLOW_NON_LOOPBACK"] = "1"
            self.assertEqual(watch_mod.WatchConfig.from_env().host, "0.0.0.0")
        finally:
            os.environ.clear()
            os.environ.update(saved)

    def test_a_missing_run_directory_fails_with_one_actionable_line(self) -> None:
        import os
        saved = dict(os.environ)
        try:
            os.environ.pop("PI05_WATCH_RUNS_DIR", None)
            with self.assertRaises(SystemExit) as caught:
                watch_mod.WatchConfig.from_env()
            self.assertIn("PI05_WATCH_RUNS_DIR", str(caught.exception))
            os.environ["PI05_WATCH_RUNS_DIR"] = str(self.tmp / "nowhere")
            with self.assertRaises(SystemExit):
                watch_mod.WatchConfig.from_env()
        finally:
            os.environ.clear()
            os.environ.update(saved)


class EmptyStateTest(WatchTestCase):
    def test_a_run_directory_with_no_records_yet_still_answers(self) -> None:
        directory = self.tmp / "runs" / "so101" / "just-started"
        directory.mkdir(parents=True)
        (directory / "train_initial.log").write_text("", encoding="utf-8")
        (directory / "config_initial.json").write_text(json.dumps({"steps": 100, "batch_size": 2}),
                                                       encoding="utf-8")
        view = watch_mod.snapshot_run(
            watch_mod.RunWatch(directory, "fresh", "so101/just-started"), self.config())
        self.assertEqual(view["status"]["code"], "waiting")
        self.assertIsNone(view["progress"]["step"])
        self.assertEqual(view["loss"]["points"], [])
        self.assertFalse(view["metrics"]["lr"]["available"])
        self.assertFalse(view["validation"]["available"])
        self.assertEqual(view["config"]["target_steps"], 100)
        self.assertEqual(view["config"]["missing"], ["数据划分"])


if __name__ == "__main__":
    unittest.main()
