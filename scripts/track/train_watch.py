#!/usr/bin/env python3
"""本地只读训练监控后端：增量读取运行目录里的真实日志、配置快照与评估产物。

用法：
    PI05_WATCH_RUNS_DIR=runs/so101/e4t ./scripts/run.sh watch
    ./scripts/run.sh watch --stop

约定：只读；默认只监听 127.0.0.1；响应不含命令、绝对路径、主机信息或凭据；
缺失指标输出 null 并写明原因，不补造数值；由 schedule 求值的学习率标为 derived。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
UI_FILE = Path(__file__).with_name("watch_ui.html")
PID_FILE = ROOT / "experiments" / ".state" / "watch.pid"
DEFAULT_GATE = ROOT / "experiments" / "so101" / "e4t-openloop.json"

# 行格式与 scripts/so101/summarize_training.py、plot_e4t_loss.py 保持一致：启动器给容器输出
# 加了单调时间前缀，tqdm 又会用 \r 把 Step 文本挤到同一物理行里，所以时间戳按“最近一次出现”
# 携带，Step 记录在片段任意位置匹配。
STEP_RE = re.compile(r"Step\s+(\d+):\s*([^\r\n]*)")
VALUE_RE = re.compile(r"(loss|grad_norm|param_norm)=([-+0-9.eE]+)")
STAMP_RE = re.compile(r"^\s*(\d{9,14}(?:\.\d+)?)\b")
PROGRESS_RE = re.compile(r"Progress on:\s*(\d+)it")
RATE_RE = re.compile(r"rate:\s*([0-9.]+)\s*([ms])/it")
REDACT_RE = re.compile(
    r"(?:/(?:home|Users|mnt|Volumes|data)/\S+)|(?:\b[A-Za-z0-9._-]+@(?:\d{1,3}\.){3}\d{1,3}\b)")

MAX_READ_BYTES = 4 << 20
MAX_POINTS = 30000
DISCOVERY_DEPTH = 3
SKIP_DIRS = {"checkpoints", "evaluation", "home", "jax_cache", "openpi_cache", "__pycache__", "wandb"}
MEMORY_NOTE = "显存按秒采样且无时间戳，无法与 step 对齐；只给出峰值与最新值，不绘制曲线。"
CADENCE_NOTE = "验证误差只在 checkpoint 评估完成后更新，是评估事件而非逐步指标；训练 Probe/Full Loss 另算。"
MAE_UNIT = "°关节宏平均 MAE（开环预测误差）"
LOSS_UNIT = "闭环评估报告中的 flow-matching MSE；与 pi05 训练 compute_loss 的32维口径分开"
LOSS_KEY = "clean_active_true12_mse"

STATUS_LABELS = {"waiting": "等待启动", "running": "运行中", "completed": "已完成",
                 "failed": "已失败", "stopped": "已停止", "stalled": "记录中断"}


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def number(text: object) -> float | None:
    try:
        value = float(text)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def as_number(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, str):
        return number(value)
    return None


def redact(value: object, limit: int = 160) -> str | None:
    """Never let a raw path, host or credential-shaped string reach the browser."""
    if not isinstance(value, str):
        return None
    cleaned = REDACT_RE.sub("<redacted>", " ".join(value.split()))
    return cleaned[:limit] or None


def within(root: Path, candidate: Path) -> bool:
    try:
        Path(candidate).resolve().relative_to(root)
    except (ValueError, OSError):
        return False
    return True


def newest_mtime(paths) -> float | None:
    newest = None
    for path in paths:
        try:
            value = path.stat().st_mtime
        except OSError:
            continue
        newest = value if newest is None else max(newest, value)
    return newest


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit(f"{name} 需要是整数，当前值无法解析") from None
    if not minimum <= value <= maximum:
        raise SystemExit(f"{name} 需要在 {minimum}..{maximum} 之间")
    return value


def _env_float(name: str, default: float, *, minimum: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise SystemExit(f"{name} 需要是数字，当前值无法解析") from None
    if value < minimum:
        raise SystemExit(f"{name} 不能小于 {minimum}")
    return value


class WatchConfig:
    def __init__(self, root: Path, host: str, port: int, stale_s: float, poll_ms: int,
                 run_pin: str, gate: Path | None) -> None:
        self.root = root
        self.host = host
        self.port = port
        self.stale_s = stale_s
        self.poll_ms = poll_ms
        self.run_pin = run_pin
        self.gate = gate

    @classmethod
    def from_env(cls) -> "WatchConfig":
        raw_root = os.environ.get("PI05_WATCH_RUNS_DIR", "").strip()
        if not raw_root:
            raise SystemExit("未设置 PI05_WATCH_RUNS_DIR：请指向运行目录，见 configs/monitor/example.env")
        root = Path(raw_root).expanduser()
        root = (root if root.is_absolute() else ROOT / root).resolve()
        if not root.is_dir():
            raise SystemExit(f"PI05_WATCH_RUNS_DIR 指向的不是目录：{raw_root}")

        host = os.environ.get("PI05_WATCH_HOST", "").strip() or "127.0.0.1"
        if host not in {"localhost", "127.0.0.1", "::1"} and not _env_flag("PI05_WATCH_ALLOW_NON_LOOPBACK"):
            raise SystemExit(
                f"监控默认只允许本机地址，当前 PI05_WATCH_HOST={host}；"
                "确需监听其他地址请显式设置 PI05_WATCH_ALLOW_NON_LOOPBACK=1")
        gate_raw = os.environ.get("PI05_WATCH_GATE_FILE", "").strip()
        if gate_raw:
            gate = Path(gate_raw).expanduser().resolve()
            if not gate.is_file():
                raise SystemExit(f"PI05_WATCH_GATE_FILE 指向的文件不存在：{gate_raw}")
        else:
            gate = DEFAULT_GATE if DEFAULT_GATE.is_file() else None
        return cls(
            root, host,
            _env_int("PI05_WATCH_PORT", 8765, minimum=1, maximum=65535),
            _env_float("PI05_WATCH_STALE_S", 180.0, minimum=5.0),
            _env_int("PI05_WATCH_POLL_MS", 5000, minimum=1000, maximum=600000),
            os.environ.get("PI05_WATCH_RUN", "").strip().strip("/"),
            gate)


# --------------------------------------------------------------------------- log parsing


def scan_text(text: str, stamp: float | None) -> tuple[list[dict], float | None, int]:
    """Split a log chunk into records; one tqdm write may share a physical line with a Step."""
    records: list[dict] = []
    malformed = 0
    for fragment in re.split(r"[\r\n]+", text):
        fragment = fragment.strip()
        if not fragment:
            continue
        if fragment.startswith("{"):
            try:
                row = json.loads(fragment)
            except (TypeError, ValueError):
                row = None
            if isinstance(row, dict) and isinstance(row.get("train_state_step"), int):
                metrics = row.get("metrics")
                values = ({
                    key: as_number(metrics.get(key)) for key in RECORD_KEYS
                    if isinstance(metrics, dict) and as_number(metrics.get(key)) is not None
                })
                timestamp = row.get("time")
                if isinstance(timestamp, str):
                    try:
                        stamp = datetime.fromisoformat(
                            timestamp.replace("Z", "+00:00")).timestamp()
                    except ValueError:
                        pass
                if not values:
                    malformed += 1
                records.append({
                    "kind": "step",
                    "step": int(row["train_state_step"]),
                    "epoch": stamp,
                    "values": values,
                })
                continue
        lead = STAMP_RE.match(fragment)
        if lead:
            candidate = number(lead.group(1))
            if candidate is not None and 1e8 < candidate < 1e11:
                stamp = candidate
        step_match = STEP_RE.search(fragment)
        if step_match:
            values = {key: as_number(raw) for key, raw in VALUE_RE.findall(step_match.group(2))}
            if not any(value is not None for value in values.values()):
                malformed += 1
            records.append({"kind": "step", "step": int(step_match.group(1)),
                            "epoch": stamp, "values": values})
            continue
        progress = PROGRESS_RE.search(fragment)
        if progress:
            rate = RATE_RE.search(fragment)
            seconds = as_number(rate.group(1)) if rate else None
            if seconds is not None and rate is not None and rate.group(2) == "m":
                seconds *= 60.0
            records.append({"kind": "progress", "it": int(progress.group(1)),
                            "epoch": stamp, "rate_s": seconds})
    return records, stamp, malformed


RECORD_KEYS = ("loss", "grad_norm", "param_norm")


def same_record(old: dict, new: dict) -> bool:
    """A re-read of an unterminated tail must not be counted as a duplicated step."""
    return (old.get("epoch") == new.get("epoch") and old.get("source") == new.get("source")
            and all(old.get(key) == new.get(key) for key in RECORD_KEYS))


class Tail:
    """Incremental reader for one log file; holds a byte offset and never opens for writing."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.name = path.name
        self.offset = 0
        self.inode: int | None = None
        self.stamp: float | None = None
        self.tail_open = False
        self.pending_bytes = 0
        self.last_step: int | None = None
        self.backward_jumps = 0

    def poll(self) -> tuple[list[dict], dict]:
        events = {"rotations": 0, "malformed": 0, "half_lines": 0}
        try:
            stat = self.path.stat()
        except OSError:
            return [], events
        if self.inode is None:
            self.inode = stat.st_ino
        elif stat.st_ino != self.inode or stat.st_size < self.offset:
            # 轮转或截断：同一路径现在指向另一条流，从头重读并由 step 去重收敛
            self.inode = stat.st_ino
            self.offset = 0
            events["rotations"] += 1
        if stat.st_size == self.offset:
            return [], events

        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            data = handle.read(MAX_READ_BYTES)
        if not data:
            return [], events

        cut = max(data.rfind(b"\n"), data.rfind(b"\r"))
        # 只有到最后一个终止符为止的内容算写完；之后的未终止尾部是半行，不推进偏移量
        head = data[: cut + 1] if cut >= 0 else b""
        tail = data[cut + 1:] if cut >= 0 else data
        advance = len(head)
        self.tail_open = tail != b""
        if self.tail_open:
            if len(data) != self.pending_bytes:
                events["half_lines"] += 1
            self.pending_bytes = len(data)
        else:
            self.pending_bytes = 0
        records, self.stamp, malformed = scan_text(head.decode("utf-8", errors="replace"), self.stamp)
        if tail:
            provisional, self.stamp, extra = scan_text(
                tail.decode("utf-8", errors="replace"), self.stamp)
            for record in provisional:
                record["provisional"] = True
            records += provisional
            malformed += extra
        events["malformed"] += malformed
        for record in records:
            if record["kind"] == "step":
                step = record["step"]
                if self.last_step is not None and step < self.last_step:
                    self.backward_jumps += 1
                self.last_step = step if self.last_step is None else max(self.last_step, step)
        self.offset += advance
        return records, events


# --------------------------------------------------------------------------- run state


class RunWatch:
    """Merged view of one run directory: step records, config snapshot, evaluation artifacts."""

    def __init__(self, run_dir: Path, run_id: str, name: str) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self.name = name
        self.tails: dict[str, Tail] = {}
        self.steps: dict[int, dict] = {}
        self.progress: list[dict] = []
        self.pending_step: int | None = None
        self.relaunches = 0
        self.events = {"rotations": 0, "duplicate_steps": 0, "half_lines": 0,
                       "malformed": 0, "backward_jumps": 0}
        self.lock = threading.Lock()

    def log_files(self) -> list[Path]:
        return sorted((path for path in self.run_dir.glob("train_*.log")
                       if path.name.endswith(".log") and path.is_file()),
                      key=lambda path: (path.stat().st_mtime, path.name))

    def poll_logs(self) -> None:
        for path in self.log_files():
            tail = self.tails.get(path.name)
            if tail is None:
                tail = self.tails[path.name] = Tail(path)
                if self.steps:
                    self.relaunches += 1
            records, events = tail.poll()
            for key, value in events.items():
                self.events[key] += value
            for record in records:
                if record["kind"] == "progress":
                    if record.get("provisional"):
                        continue  # 半行的进度速率同样可能是被截断的值
                    if self.progress and (self.progress[-1].get("it") == record["it"]
                                          and self.progress[-1].get("epoch") == record["epoch"]):
                        continue  # 半行重读，不是新的进度
                    self.progress = (self.progress + [record])[-40:]
                    continue
                step = record["step"]
                if record.get("provisional"):
                    # 未结束的半行只报 step：数值可能被截断，写入曲线等于造数
                    self.pending_step = step
                    continue
                if self.pending_step == step:
                    self.pending_step = None
                merged = {"step": step, "epoch": record["epoch"], "source": tail.name,
                          **record["values"]}
                existing = self.steps.get(step)
                if existing is not None and not same_record(existing, merged):
                    self.events["duplicate_steps"] += 1
                self.steps[step] = merged

    @property
    def restarts(self) -> int:
        return self.relaunches + self.events["backward_jumps"]

    def ordered_steps(self) -> list[dict]:
        return [self.steps[key] for key in sorted(self.steps)]

    def read_json(self, relative: str) -> dict:
        path = self.run_dir / relative
        if not within(self.run_dir, path) or not path.is_file():
            return {}
        return load_json(path)

    def glob_json(self, pattern: str) -> list[tuple[str, dict]]:
        out = []
        for path in sorted(self.run_dir.glob(pattern)):
            data = load_json(path)
            if data:
                out.append((path.name, data))
        return out

    def newest_json(self, pattern: str) -> list[dict]:
        """Snapshot documents ordered newest first, so a resumed run's record wins."""
        found = []
        for path in self.run_dir.glob(pattern):
            data = load_json(path)
            if not data:
                continue
            try:
                stamp = path.stat().st_mtime
            except OSError:
                stamp = 0.0
            found.append((stamp, path.name, data))
        return [item[2] for item in sorted(found, key=lambda row: (row[0], row[1]), reverse=True)]

    def config_snapshot(self) -> tuple[dict, dict]:
        """Merge whitelisted scalar fields across snapshots; paths, commands and ids stay local."""
        ordered = [self.read_json("run_config.json")] + self.newest_json("config_*.json")
        ordered = [item for item in ordered if item]
        launches = self.newest_json("run_*.json")
        launch = launches[0] if launches else {}

        def field(key):
            return first_not_none(*[item.get(key) for item in ordered])

        inner: dict = {}
        for item in reversed(ordered + launches):
            block = item.get("config")
            if isinstance(block, dict):
                inner.update(block)
        # runtime 块由 train_so101.py 写入；只取非标识性标量，device_ids/device_kinds 不读
        runtime = next((item.get("runtime") for item in ordered + launches
                        if isinstance(item.get("runtime"), dict)), None)

        batch = as_number(first_not_none(field("batch_size"),
                                         *(item.get("batch_size") for item in launches)))
        target = as_number(first_not_none(field("steps"),
                                          *(item.get("steps") for item in launches)))
        train_ids = field("train_episode_ids")
        val_ids = field("validation_episode_ids")
        split = {
            "train": len(train_ids) if isinstance(train_ids, list) else None,
            "validation": len(val_ids) if isinstance(val_ids, list) else None,
            "declared": redact(field("train_episodes"), 60),
        }
        missing = []
        if split["train"] is None and split["validation"] is None:
            missing.append("数据划分")
        if batch is None:
            missing.append("batch")
        if target is None:
            missing.append("目标更新次数")
        snapshot = {
            "batch_size": batch,
            "target_steps": target,
            "save_interval": as_number(field("save_interval")),
            "checkpoint_steps": (
                [int(value) for value in field("checkpoint_steps")]
                if isinstance(field("checkpoint_steps"), list)
                and all(as_number(value) is not None for value in field("checkpoint_steps"))
                else None
            ),
            "seed": as_number(inner.get("seed")),
            "dtype": redact(inner.get("dtype"), 24),
            "state_mode": redact(first_not_none(field("state_mode"), launch.get("mode")), 16),
            "exp_name": redact(field("exp_name"), 64),
            "mode": redact(first_not_none(field("mode"), launch.get("mode")), 16),
            "phase": redact(field("phase"), 16),
            "split": split,
            "lr_schedule": clean_schedule(field("lr_schedule")),
            "created": redact(field("created"), 40),
            "data_loader_workers": as_number(first_not_none(
                inner.get("data_loader_workers"), field("data_loader_workers"))),
            "gradient_clip_norm": as_number(inner.get("gradient_clip_norm")),
            "runtime": None if runtime is None else {
                "backend": redact(runtime.get("backend"), 12),
                "device_count": as_number(runtime.get("device_count")),
                "fsdp_devices": as_number(runtime.get("fsdp_devices")),
                "data_parallel_replicas": as_number(runtime.get("data_parallel_replicas")),
                "batch_per_data_shard": as_number(runtime.get("batch_per_data_shard")),
            },
            "missing": missing,
        }
        return snapshot, launch

    def memory(self) -> dict:
        files = sorted((path for path in self.run_dir.glob("gpu_*") if path.is_file()),
                       key=lambda path: path.stat().st_mtime)
        if not files:
            return {"available": False, "reason": "无显存采样文件", "note": MEMORY_NOTE}
        # 单卡写裸整数（scripts/so101/run_training.sh），多卡写 index,uuid,memory.used
        # （scripts/so101/e4t_run_training.py）；uuid 属于主机信息，不读也不返回。
        per_device: dict[str, list[int]] = {}
        for line in files[-1].read_text(encoding="utf-8", errors="replace").splitlines():
            parts = [value.strip() for value in line.split(",")]
            if len(parts) == 4 and parts[0].isdigit() and parts[2].isdigit():
                per_device.setdefault(parts[0], []).append(int(parts[2]))
            elif len(parts) == 3 and parts[0].isdigit() and parts[2].isdigit():
                per_device.setdefault(parts[0], []).append(int(parts[2]))
            elif line.strip().isdigit():
                per_device.setdefault("0", []).append(int(line.strip()))
        if not per_device:
            return {"available": False, "reason": "显存采样为空或格式不认识", "note": MEMORY_NOTE}
        devices = [{"index": int(index), "peak_mib": max(values), "latest_mib": values[-1],
                    "samples": len(values)}
                   for index, values in sorted(per_device.items(), key=lambda item: int(item[0]))]
        return {"available": True,
                "peak_mib": max(item["peak_mib"] for item in devices),
                "latest_mib": max(item["latest_mib"] for item in devices),
                "samples": sum(item["samples"] for item in devices),
                "devices": devices, "series": None, "note": MEMORY_NOTE}

    def evaluations(self, gate: Path | None) -> dict:
        rows: list[dict] = []
        gate_document: dict | None = None
        gate_rows: dict[int, dict] = {}
        if gate is not None and gate.is_file():
            document = load_json(gate).get("gate") or {}
            if document:
                gate_document = {
                    "decision": redact(document.get("decision"), 60),
                    "split": redact(document.get("split"), 24),
                    "best_checkpoint": redact(document.get("best_checkpoint"), 40),
                    "best_is_gate_passing": document.get("best_is_gate_passing"),
                    "pre_registered_before_training": document.get("pre_registered_before_training"),
                }
            for curve in document.get("curve", []) or []:
                if isinstance(curve, dict) and as_number(curve.get("step")) is not None:
                    gate_rows[int(curve["step"])] = curve

        for driver, manifest in self.glob_json("evaluation*driver*.json"):
            split = redact(manifest.get("split"), 24) or "unknown"
            for row in manifest.get("evaluations", []) or []:
                if not isinstance(row, dict):
                    continue
                step = as_number(first_not_none(row.get("authoritative_step"), row.get("recorded_step")))
                if step is None:
                    continue
                label = redact(row.get("label"), 40)
                entry = {
                    "step": int(step),
                    "split": split,
                    "label": label,
                    "evaluated_at": redact(row.get("started"), 40),
                    "success": row.get("success") if isinstance(row.get("success"), bool) else None,
                    "problem": redact(row.get("problem"), 120),
                    "returncode": as_number(row.get("returncode")),
                    "source": "driver",
                    "driver": driver,
                    "report": {},
                }
                if label:
                    entry["report"] = report_metrics(self.read_json(f"evaluation/{label}/report.json"))
                gate_row = gate_rows.get(int(step))
                if gate_row:
                    entry["gate"] = {
                        "gate_passed": gate_row.get("gate_passed"),
                        "first_filter_passed": gate_row.get("first_filter_passed"),
                        "failed_checks": [redact(item, 60)
                                          for item in (gate_row.get("gate_failed_checks") or [])][:12],
                    }
                    # 门禁曲线只带门禁阈值那一组数；报告里还有验证 flow loss，
                    # 所以按「门禁优先、缺失项回退报告」合并，而不是整块替换。
                    recorded = {key: value for key, value in gate_metrics(gate_row).items()
                                if value is not None}
                    fallback = entry["report"] or {}
                    entry["metrics"] = {**fallback, **recorded} or None
                    entry["metric_source"] = (
                        "门禁汇总曲线" if not fallback or set(fallback) <= set(recorded)
                        else "门禁汇总曲线 + report.json（门禁未记录的项取报告）")
                else:
                    entry["metrics"] = entry["report"] or None
                    entry["metric_source"] = "report.json" if entry["report"] else None
                rows.append(entry)

        for step, curve in sorted(gate_rows.items()):
            if any(row["step"] == step and row.get("source") == "gate" for row in rows):
                continue
            if not any(row["step"] == step for row in rows):
                rows.append({
                    "step": step,
                    "split": redact(curve.get("split"), 24) or "validation",
                    "label": redact(curve.get("label"), 40),
                    "evaluated_at": None,
                    "success": None,
                    "problem": None,
                    "returncode": None,
                    "source": "gate",
                    "driver": None,
                    "gate": {"gate_passed": curve.get("gate_passed"),
                             "first_filter_passed": curve.get("first_filter_passed"),
                             "failed_checks": [redact(item, 60)
                                               for item in (curve.get("gate_failed_checks") or [])][:12]},
                    "metrics": gate_metrics(curve),
                    "metric_source": "门禁汇总曲线",
                })

        rows.sort(key=lambda row: (row["step"], row["split"] or ""))
        if not rows:
            return {"available": False, "reason": "尚无已完成的 checkpoint 评估产物",
                    "points": [], "cadence": CADENCE_NOTE, "gate": gate_document,
                    "metric_keys": [], "unit": MAE_UNIT, "loss_unit": LOSS_UNIT,
                    "loss_key": LOSS_KEY}
        keys = sorted({key for row in rows if isinstance(row.get("metrics"), dict)
                       for key, value in row["metrics"].items() if value is not None})
        return {"available": True, "points": rows, "cadence": CADENCE_NOTE,
                "gate": gate_document, "metric_keys": keys,
                "unit": MAE_UNIT, "loss_unit": LOSS_UNIT, "loss_key": LOSS_KEY}

    def validation_losses(self) -> dict:
        """Read 50-step probe and checkpoint-only full series without mixing them."""
        config_snapshot, _ = self.config_snapshot()
        full_steps = set(config_snapshot.get("checkpoint_steps") or [])

        def read_series(metric: str, requires_checkpoint: bool) -> dict:
            required_samples = 256 if metric == "val_loss_probe" else 8300
            points = []
            invalid = []
            seen: set[int] = set()
            for filename, row in self.glob_json(f"{metric}_*.json"):
                step = as_number(row.get("train_state_step"))
                loss = as_number(row.get("validation_loss"))
                expected = as_number(row.get("samples_expected"))
                evaluated = as_number(row.get("samples_evaluated"))
                checkpoint = row.get("checkpoint_info")
                checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
                checkpoint_saved = row.get("checkpoint_saved")
                metadata_step = as_number(
                    checkpoint.get("checkpoint_metadata_step"))
                checkpoint_ok = (
                    checkpoint_saved is True and metadata_step == step
                ) or (
                    checkpoint_saved is False
                    and not requires_checkpoint
                    and metadata_step is None
                )
                schedule_ok = (
                    step is not None
                    and int(step) % 50 == 0
                    and (metric == "val_loss_probe" or int(step) in full_steps)
                )
                valid = (
                    row.get("metric") == metric
                    and step is not None
                    and loss is not None
                and expected == required_samples
                and evaluated == required_samples
                    and checkpoint_ok
                    and schedule_ok
                    and row.get("train_false") is True
                    and row.get("loss") == "flow_matching_mse"
                    and as_number(row.get("gradient_updates")) == 0
                    and row.get("state_tree_unchanged") is True
                    and len(str(row.get("noise_sha256") or "")) == 64
                    and len(str(row.get("flow_time_sha256") or "")) == 64
                )
                if not valid:
                    invalid.append(filename)
                    continue
                point_step = int(step)
                if point_step in seen:
                    invalid.append(filename)
                    continue
                seen.add(point_step)
                points.append({
                    "step": point_step,
                    "loss": loss,
                    "samples_evaluated": int(evaluated),
                    "checkpoint_saved": bool(checkpoint_saved),
                    "checkpoint_metadata_step": metadata_step,
                    "checkpoint": redact(filename, 120),
                    "completed_utc": redact(row.get("completed_utc"), 40),
                    "wall_time_s": as_number(row.get("wall_time_s")),
                    "loss_definition": redact(row.get("loss_definition"), 240),
                    "validation_plan_sha256": redact(
                        row.get("validation_plan_sha256"), 64),
                })
            points.sort(key=lambda item: item["step"])
            if not points:
                reason = f"尚无完成的 {metric} 记录"
                if invalid:
                    reason += f"；{len(invalid)} 个记录不完整，已忽略"
                return {
                    "available": False,
                    "metric": metric,
                    "reason": reason,
                    "points": [],
                    "invalid_records": invalid,
                }
            return {
                "available": True,
                "metric": metric,
                "points": points,
                "invalid_records": invalid,
                "unit": "flow-matching MSE",
                "definition": points[-1].get("loss_definition"),
            }

        return {
            "probe": read_series("val_loss_probe", requires_checkpoint=False),
            "full": read_series("val_loss_full", requires_checkpoint=True),
            "unit": "flow-matching MSE",
            "separate_metrics": True,
        }


def load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def clean_schedule(value) -> dict | None:
    if not isinstance(value, dict):
        return None
    out = {key: as_number(value.get(key)) for key in
           ("warmup_steps", "peak_lr", "decay_steps", "decay_lr")}
    return out if any(item is not None for item in out.values()) else None


def gate_metrics(curve: dict) -> dict:
    return {
        "clean_active_joint_macro_mae_deg": as_number(curve.get("clean_active_joint_macro_mae_deg")),
        "recovery_post_active_joint_macro_mae_deg": as_number(
            curve.get("recovery_post_active_joint_macro_mae_deg")),
        "clean_active_joint_p95_deg": as_number(curve.get("clean_active_joint_p95_deg")),
        "gripper_macro_mae_pct_clean": as_number(curve.get("gripper_macro_mae_pct_clean")),
        "windows": as_number(curve.get("windows")),
    }


def report_metrics(report: dict) -> dict:
    """Per-group numbers from one evaluation report: action error and validation flow loss.

    ``true12_mse``/``padding20_mse`` are the flow-matching squared error computed by
    scripts/so101/evaluate_e4t_openloop.py on the 12 real action dims and on the 20 padded
    dims; they live in the same ``aggregates`` block as the gate's degrees, and are kept
    apart by suffix because the units are not interchangeable.
    """
    aggregates = report.get("aggregates") if isinstance(report.get("aggregates"), dict) else {}
    out: dict[str, float | None] = {}
    for group in ("clean_active", "recovery_post_active"):
        row = aggregates.get(group)
        if isinstance(row, dict):
            for source_key in ("joint_macro_mae_deg", "true12_mse", "padding20_mse"):
                value = as_number(row.get(source_key))
                if value is not None:
                    out[f"{group}_{source_key}"] = value
    return out


# --------------------------------------------------------------------------- learning rate


def derive_lr(step: float, schedule: dict | None) -> float | None:
    """Cosine branch matches the in-container trace; the warmup ramp is a labelled approximation."""
    if not schedule:
        return None
    warmup, peak, decay, end = (schedule.get(key) for key in
                                ("warmup_steps", "peak_lr", "decay_steps", "decay_lr"))
    if None in (warmup, peak, decay, end) or decay <= warmup or warmup <= 0:
        return None
    if step <= warmup:
        return peak * (step / warmup)
    span = decay - warmup
    position = min(step - warmup, span)
    return end + (peak - end) * 0.5 * (1.0 + math.cos(math.pi * position / span))


def lr_series(rows: list[dict], schedule: dict | None, trace: dict | None) -> dict:
    if trace and isinstance(trace.get("points"), list):
        points = [[as_number(item.get("step")), as_number(item.get("lr"))]
                  for item in trace["points"] if isinstance(item, dict)]
        points = [item for item in points if item[0] is not None and item[1] is not None]
        if points:
            return {"available": True, "source": "trace",
                    "source_note": "运行目录内的 schedule trace 产物（容器内 schedule 求值）",
                    "points": points, "note": None}
    if not schedule:
        return {"available": False, "source": None, "points": [],
                "reason": "配置快照没有 lr_schedule，训练器也不打印学习率",
                "note": "学习率不可用：无 schedule 即不推导"}
    return {"available": True, "source": "derived",
            "source_note": "配置推导值：由 run 配置里的 schedule 计算，OpenPI 训练器不打印学习率",
            "points": [[row["step"], derive_lr(row["step"], schedule)] for row in rows],
            "note": "衰减段与容器内 trace 一致（相对误差 <1e-6），warmup 段为线性近似"}


# --------------------------------------------------------------------------- series


def thinning(rows: list[dict], key: str, max_points: int = MAX_POINTS) -> dict:
    """Bucket raw records into [step, mean, min, max, count, last_epoch]; never invents a value."""
    pairs = [(row["step"], row[key], row.get("epoch")) for row in rows
             if as_number(row.get(key)) is not None]
    missing = sum(1 for row in rows if as_number(row.get(key)) is None)
    base = {"raw_count": len(pairs), "records_missing_value": missing}
    if not pairs:
        return {**base, "points": [], "thinned": False,
                "reason": f"日志中没有 {key} 记录" if rows else "日志中还没有记录"}
    if len(pairs) <= max_points:
        return {**base, "points": [[step, value, value, value, 1, epoch]
                                   for step, value, epoch in pairs],
                "thinned": False, "reason": None}
    size = math.ceil(len(pairs) / max_points)
    points = []
    for start in range(0, len(pairs), size):
        chunk = pairs[start:start + size]
        values = [value for _, value, _ in chunk]
        points.append([chunk[-1][0], sum(values) / len(values), min(values), max(values),
                       len(values), chunk[-1][2]])
    return {**base, "points": points, "thinned": True,
            "reason": f"原始记录 {len(pairs)} 条超过 {max_points} 个绘制点，按桶聚合（保留桶内最小/最大值）"}


def metric_block(rows: list[dict], key: str, label: str) -> dict:
    if not any(as_number(row.get(key)) is not None for row in rows):
        return {"available": False, "reason": f"日志中没有 {label} 记录", "points": []}
    return {"available": True, "source": "log", **thinning(rows, key)}


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def speed(rows: list[dict], target: float | None, progress: list[dict]) -> dict:
    timed = [row for row in rows if row.get("epoch") is not None]
    per_update = []
    for previous, current in zip(timed, timed[1:]):
        gap = current["epoch"] - previous["epoch"]
        span = current["step"] - previous["step"]
        if gap > 0 and span > 0:
            per_update.append(gap / span)
    seconds = median(per_update[-120:]) if per_update else None
    source = "log_timestamp"
    if seconds is None:
        rates = [row["rate_s"] for row in progress[-10:] if row.get("rate_s")]
        seconds = median(rates) if rates else None
        source = "progress_bar"
    if seconds is None:
        source = None
    elapsed = (timed[-1]["epoch"] - timed[0]["epoch"]) if len(timed) > 1 else None
    # 剩余量按已记录的更新次数计，日志 step 从 0 开始所以不能直接拿最后一条当计数
    remaining = max(0.0, target - len(rows)) if target and rows else None
    return {
        "seconds_per_update": seconds,
        "updates_per_second": (1.0 / seconds) if seconds else None,
        "source": source,
        "samples": len(per_update),
        "elapsed_logged_s": elapsed,
        "estimated_remaining_s": (remaining * seconds) if (seconds and remaining is not None) else None,
        "note": None if seconds else "日志时间戳不足，无法计算训练速度",
    }


def build_status(watch: RunWatch, target: float | None, launch: dict, stale_s: float) -> dict:
    """A terminal state needs an explicit record; a log that merely stops growing is never failed.

    日志 step 从 0 计数（30k 更新的实际运行最后一条是 Step 29999），所以完成判定用记录条数，
    与 scripts/so101/summarize_training.py 的 updates_completed 口径一致。
    """
    rows = watch.ordered_steps()
    last = rows[-1] if rows else None
    updates = len(rows)
    mtime = newest_mtime(watch.log_files())
    age = (time.time() - mtime) if mtime is not None else None
    exits = [as_number(data.get("exit_code")) for _, data in watch.glob_json("run_*.json")]
    exits = [item for item in exits if item is not None]
    markers = sorted(path.name for path in watch.run_dir.glob("done_*.json"))
    reached = bool(target) and updates >= target

    if not rows and mtime is None and not launch:
        code, reason = "waiting", "运行目录里还没有训练日志"
    elif any(item != 0 for item in exits):
        code, reason = "failed", f"启动器记录了非零退出码 {int(max(exits))}"
    elif last is None:
        code, reason = "waiting", "日志尚无 Step 记录（可能仍在加载权重或编译 XLA）"
    elif markers and reached:
        code, reason = "completed", f"完成标记存在，已记录 {updates} 次更新"
    elif 0 in exits and markers:
        code, reason = "stopped", f"容器正常退出但只记录了 {updates} 次更新"
    elif age is not None and age > stale_s:
        code, reason = "stalled", (f"日志已 {int(age)} 秒未增长且没有退出记录；"
                                   "仅凭不增长不能判定失败")
    else:
        code, reason = "running", "日志在增量增长且没有终止记录"
    return {"code": code, "label": STATUS_LABELS[code], "reason": reason, "updates": updates,
            "stale_seconds": age, "stale_limit_seconds": stale_s, "has_log": mtime is not None,
            "completion_markers": len(markers), "exit_codes": [int(item) for item in exits]}


def snapshot_run(watch: RunWatch, config: WatchConfig) -> dict:
    with watch.lock:
        watch.poll_logs()
        snapshot, launch = watch.config_snapshot()
        rows = watch.ordered_steps()
        last = rows[-1] if rows else None
        trace = None
        for _, data in watch.glob_json("lr_trace*.json"):
            if isinstance(data.get("points"), list):
                trace = data
                break
        target = snapshot.get("target_steps")
        latest_epoch = last.get("epoch") if last else None
        return {
            "schema_version": 1,
            "generated_at": now_iso(),
            "run": {"id": watch.run_id, "name": watch.name, "log_files": sorted(watch.tails)},
            "status": build_status(watch, target, launch, config.stale_s),
            "progress": {
                "step": last["step"] if last else None,
                "updates": len(rows),
                "records": len(rows),
                "target_steps": target,
                "percent": (min(100.0, len(rows) / target * 100.0) if target and rows else None),
                "last_record_at": latest_epoch,
                "last_record_age_s": (time.time() - latest_epoch) if latest_epoch else None,
                "first_record_at": rows[0].get("epoch") if rows else None,
                "started_at": as_number(launch.get("started_unix")),
                "wall_time_s": as_number(launch.get("wall_time_s")),
                "speed": speed(rows, target, watch.progress),
            },
            "config": snapshot,
            "loss": thinning(rows, "loss"),
            "metrics": {
                "grad_norm": metric_block(rows, "grad_norm", "梯度范数"),
                "param_norm": metric_block(rows, "param_norm", "参数范数"),
                "lr": lr_series(rows, snapshot.get("lr_schedule"), trace),
                "memory": watch.memory(),
            },
            "validation": watch.evaluations(config.gate),
            "validation_loss": watch.validation_losses(),
            "events": {**watch.events, "restarts": watch.restarts,
                       "pending_step": watch.pending_step,
                       "tail_open": any(tail.tail_open for tail in watch.tails.values())},
            "history": ledger_history(),
        }


def ledger_history(limit: int = 12) -> dict:
    """The ledger records finished runs; it stays separate from the live run-directory view."""
    path = ROOT / "experiments" / "ledger.jsonl"
    if not path.is_file():
        return {"available": False, "reason": "没有 experiments/ledger.jsonl", "runs": []}
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict) or record.get("kind") != "train":
            continue
        metrics = record.get("metrics") or {}
        rows.append({
            "run_id": redact(record.get("run_id"), 80),
            "ts": redact(record.get("ts"), 40),
            "scope": redact(record.get("scope") or "libero", 16),
            "phase": redact(record.get("phase"), 12),
            "substage": redact(record.get("substage"), 24),
            "config": redact(record.get("config"), 64),
            "status": redact(record.get("status"), 16),
            "updates": as_number(metrics.get("updates")),
            "batch": as_number(metrics.get("batch")),
            "wall_s": as_number(metrics.get("wall_s")),
        })
    return {"available": True, "runs": list(reversed(rows[-limit:])),
            "note": "台账记录为历史运行，与当前运行目录分开展示"}


# --------------------------------------------------------------------------- http


def runs_payload(registry: "Registry") -> dict:
    watches = registry.refresh()
    ordered = sorted(watches, key=lambda item: (newest_mtime(item.log_files()) or 0.0, item.name),
                     reverse=True)
    items = []
    for watch in ordered:
        newest = newest_mtime(watch.log_files())
        items.append({
            "id": watch.run_id,
            "name": watch.name,
            "updated_at": (datetime.fromtimestamp(newest, timezone.utc).astimezone()
                           .isoformat(timespec="seconds")) if newest else None,
            "has_log": newest is not None,
            "current": False,
        })
    pinned = registry.config.run_pin
    if pinned:
        for item in items:
            item["current"] = item["name"] == pinned
        if not any(item["current"] for item in items):
            items.insert(0, {"id": None, "name": pinned, "updated_at": None,
                             "has_log": False, "current": True, "unknown": True})
    elif items:
        items[0]["current"] = True
    return {"runs": items, "polling_ms": registry.config.poll_ms,
            "stale_limit_s": registry.config.stale_s, "run_count": len(items),
            "history": ledger_history()}


class Registry:
    """Run discovery plus per-run readers; a client can only address an id already listed."""

    def __init__(self, config: WatchConfig) -> None:
        self.config = config
        self.runs: dict[str, RunWatch] = {}
        self.scan_lock = threading.Lock()
        self.refresh()

    @staticmethod
    def run_id(relative: str) -> str:
        return hashlib.sha256(relative.encode("utf-8")).hexdigest()[:10]

    def refresh(self) -> list[RunWatch]:
        with self.scan_lock:  # 多个标签页会并发轮询
            return self._scan()

    def _scan(self) -> list[RunWatch]:
        root = self.config.root
        found: list[Path] = []
        candidates = [root]
        for depth in range(1, DISCOVERY_DEPTH + 1):
            for candidate in root.glob("/".join(["*"] * depth)):
                relative = candidate.relative_to(root)
                if any(part in SKIP_DIRS or part.startswith(".") for part in relative.parts[:-1]):
                    continue
                candidates.append(candidate)
        for path in candidates:
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if not resolved.is_dir() or not within(root, resolved):
                continue
            if path.is_symlink() and path is not root:
                continue
            if not looks_like_run(resolved):
                continue
            found.append(resolved)
        watches = []
        for path in sorted(set(found)):
            relative = path.relative_to(root).as_posix() or "."
            run_id = self.run_id(relative)
            watch = self.runs.get(run_id) or RunWatch(path, run_id, relative)
            self.runs[run_id] = watch
            watches.append(watch)
        return watches

    def resolve(self, run_id: str) -> RunWatch | None:
        watch = self.runs.get(run_id or "")
        if watch is None or not within(self.config.root, watch.run_dir):
            return None
        return watch


def looks_like_run(path: Path) -> bool:
    if path.name in SKIP_DIRS:
        return False
    return bool(next(path.glob("train_*.log"), None) or next(path.glob("config_*.json"), None)
                or next(path.glob("run_*.json"), None))


class Handler(BaseHTTPRequestHandler):
    registry: Registry
    server_version = "pi05-watch/1"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._serve_ui()
        elif parsed.path == "/api/runs":
            self._json(runs_payload(self.registry))
        elif parsed.path == "/api/state":
            run_id = (parse_qs(parsed.query).get("run") or [""])[0]
            watch = self.registry.resolve(run_id)
            if watch is None:
                self._json({"error": "unknown_run",
                            "hint": "只能访问 /api/runs 返回的运行 id"}, status=404)
            else:
                self._json(snapshot_run(watch, self.registry.config))
        else:
            self._json({"error": "not_found"}, status=404)

    do_HEAD = do_GET

    def _serve_ui(self) -> None:
        try:
            body = UI_FILE.read_bytes()
        except OSError:
            self._json({"error": "ui_missing", "hint": str(UI_FILE.name)}, status=500)
            return
        self._respond(200, "text/html; charset=utf-8", body,
                      {"Content-Security-Policy":
                       "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
                       "connect-src 'self'; img-src 'none'; form-action 'none'; base-uri 'none'"})

    def _json(self, payload: dict, status: int = 200) -> None:
        try:
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError):
            body = json.dumps({"error": "encode_failed"}).encode("utf-8")
            status = 500
        self._respond(status, "application/json; charset=utf-8", body, {})

    def _respond(self, status: int, content_type: str, body: bytes, extra: dict) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for key, value in extra.items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def log_message(self, *_args) -> None:
        return


# --------------------------------------------------------------------------- process


def write_pid() -> None:
    path = pid_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{os.getpid()}\n", encoding="utf-8")
    except OSError:
        pass


def pid_file() -> Path:
    raw = os.environ.get("PI05_WATCH_PID_FILE", "").strip()
    if not raw:
        return PID_FILE
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def watch_cmdline(pid: int) -> bool:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
    return "train_watch.py" in raw.decode("utf-8", errors="replace").replace("\x00", " ")


def stop_process() -> int:
    path = pid_file()
    if not path.is_file():
        print("没有 pid 文件，监控可能未在本机启动过")
        return 1
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        print("pid 文件无法解析")
        return 1
    if not pid_alive(pid):
        path.unlink(missing_ok=True)
        print(f"进程 {pid} 已不存在，清理 pid 文件")
        return 0
    if not watch_cmdline(pid):
        print(f"pid {pid} 的进程不是训练监控，拒绝终止")
        return 1
    os.kill(pid, signal.SIGTERM)
    for _ in range(50):
        if not pid_alive(pid):
            break
        time.sleep(0.1)
    path.unlink(missing_ok=True)
    print(f"已停止监控进程 {pid}")
    return 0


def print_state() -> None:
    running = False
    path = pid_file()
    if path.is_file():
        try:
            running = pid_alive(int(path.read_text(encoding="utf-8").strip()))
        except (OSError, ValueError):
            running = False
    print("运行中" if running else "未运行")
    raise SystemExit(0 if running else 1)


def print_once(config: WatchConfig) -> int:
    registry = Registry(config)
    items = runs_payload(registry)["runs"]
    if not items:
        print(json.dumps({"error": "no_run_directory", "hint": "PI05_WATCH_RUNS_DIR 下没有运行目录"},
                         ensure_ascii=False))
        return 1
    chosen = next((item for item in items if item["current"]), items[0])
    watch = registry.resolve(chosen["id"]) if chosen["id"] else None
    if watch is None:
        print(json.dumps({"error": "run_not_found", "name": chosen["name"]}, ensure_ascii=False))
        return 1
    print(json.dumps(snapshot_run(watch, config), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def serve(config: WatchConfig) -> int:
    registry = Registry(config)
    Handler.registry = registry
    address = ("127.0.0.1" if config.host == "localhost" else config.host)
    try:
        server = ThreadingHTTPServer((address, config.port), Handler)
    except OSError as exc:
        raise SystemExit(f"无法监听 {config.host}:{config.port}：{exc.strerror or exc}") from None
    server.daemon_threads = True
    write_pid()
    print(f"只读监控已启动：http://{config.host}:{config.port}/ "
          f"（运行目录含 {len(registry.runs)} 个候选）", flush=True)
    print("页面按轮询增量读取日志；Ctrl-C 或 ./scripts/run.sh watch --stop 停止", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        pid_file().unlink(missing_ok=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="本地只读训练实时监控")
    parser.add_argument("--stop", action="store_true", help="停止后台监控进程")
    parser.add_argument("--status", action="store_true", help="显示监控进程状态")
    parser.add_argument("--once", action="store_true", help="打印一次 JSON 快照后退出")
    args = parser.parse_args(argv)

    if args.stop:
        return stop_process()
    if args.status:
        print_state()
    config = WatchConfig.from_env()
    if args.once:
        return print_once(config)
    return serve(config)


if __name__ == "__main__":
    sys.exit(main())
