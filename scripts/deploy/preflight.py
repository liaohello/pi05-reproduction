#!/usr/bin/env python3
"""部署体检：本地契约 + 远程环境，逐项给 pass/fail。

用法：
    python3 deploy/preflight.py              # 只检查本地
    python3 deploy/preflight.py --remote     # 本地 + 服务器
    python3 deploy/preflight.py --json       # 只输出 JSON 报告

结果同时写入 deploy/preflight-report.json，供 track/status.py 消费。
任一 FAIL 时退出码为 1。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROOT = common.ROOT
REPORT = common.state_path("preflight-report.json")


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Checker:
    def __init__(self) -> None:
        self.checks: list[dict] = []

    def add(self, cid: str, name: str, status: str, detail: str) -> None:
        self.checks.append({"id": cid, "name": name, "status": status, "detail": detail})

    def has_fail(self) -> bool:
        return any(c["status"] == "FAIL" for c in self.checks)

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for check in self.checks:
            counts[check["status"]] = counts.get(check["status"], 0) + 1
        return counts


def check_local(cfg: dict, checker: Checker) -> bool:
    """返回本地门禁是否全通过。"""
    key = common.key_path(cfg)
    if not key.is_file():
        checker.add("L0-01", "SSH 私钥", "FAIL",
                    f"缺少 {key}（服务器公钥认证必需；从 Windows 复制该私钥到此处即可）")
        key_ok = False
    else:
        mode = stat.S_IMODE(key.stat().st_mode)
        perm_note = "" if mode & 0o077 == 0 else f"，权限 {oct(mode)} 建议改为 600"
        body = key.read_text(encoding="utf-8", errors="replace")
        if "PRIVATE KEY" not in body:
            checker.add("L0-01", "SSH 私钥", "FAIL", f"{key} 不是私钥文件")
            key_ok = False
        else:
            checker.add("L0-01", "SSH 私钥", "PASS" if not perm_note else "WARN",
                        f"{key}{perm_note}")
            key_ok = True

    missing = [tool for tool in ("ssh", "rsync") if shutil.which(tool) is None]
    checker.add("L0-02", "工具链 ssh/rsync", "PASS" if not missing else "FAIL",
                "均可用" if not missing else f"缺少 {', '.join(missing)}")

    manifest = ROOT / cfg["local"]["source_manifest"]
    source_tree = Path(os.environ.get("PI05_OPENPI_SRC", ROOT / "third_party" / "openpi"))
    if not manifest.is_file() or not source_tree.is_dir():
        checker.add("L0-03", "源码基线", "SKIP",
                    "第三方源码未就位（可选）：设置 PI05_OPENPI_SRC 与 PI05_SOURCE_MANIFEST 后再校验")
        source_ok = True
    else:
        result = common.run([sys.executable, str(ROOT / "scripts" / "verify" / "source_lock.py")], timeout=900)
        tail = [line for line in result.stdout.splitlines() if line.strip()][-4:]
        passed = result.returncode == 0 and "SOURCE LOCK PASSED" in result.stdout
        checker.add("L0-03", "源码基线", "PASS" if passed else "FAIL", " | ".join(tail))
        source_ok = passed

    norm = ROOT / "configs" / "libero" / "norm_stats.json"
    expected = cfg["assets"]["norm_stats_sha256"]
    if norm.is_file():
        got = sha256(norm)
        checker.add("L0-04", "归一化统计副本", "PASS" if got == expected else "WARN",
                    f"sha256={got[:16]}…" + ("" if got == expected else f" 期望 {expected[:16]}…"))
    else:
        checker.add("L0-04", "归一化统计副本", "WARN", "本地副本不存在（不阻塞部署）")

    commit = common.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"])
    if commit.returncode == 0:
        dirty = common.run(["git", "-C", str(ROOT), "status", "--porcelain"]).stdout.strip()
        count = len([line for line in dirty.splitlines() if line.strip()])
        checker.add("L0-05", "Git 版本锚点", "PASS" if count == 0 else "WARN",
                    f"commit={commit.stdout.strip()}" + (f"，未提交改动 {count} 处" if count else "，工作区干净"))
    else:
        checker.add("L0-05", "Git 版本锚点", "FAIL", "仓库未初始化（无法把实验对应到代码版本）")

    free_gb = shutil.disk_usage(ROOT).free / (1024 ** 3)
    need = cfg["local"]["min_free_gb"]
    checker.add("L0-06", "本地磁盘余量", "PASS" if free_gb >= need else "FAIL",
                f"{free_gb:.1f} GB 可用（需要 ≥{need} GB）")
    return key_ok and source_ok and not missing


def check_remote(cfg: dict, checker: Checker, remote_ok: bool) -> None:
    if not remote_ok:
        for cid, name in [
            ("R1-01", "SSH 连通"), ("R1-02", "远程目录可写"), ("R1-03", "Docker 镜像"),
            ("R1-04", "模型权重"), ("R1-05", "GPU"), ("R1-06", "数据资产"), ("R1-07", "远程磁盘"),
        ]:
            checker.add(cid, name, "SKIP", "本地 SSH 前置未就绪，未执行")
        return

    probe, error = common.remote_probe(cfg)
    if not probe:
        checker.add("R1-01", "SSH 连通", "FAIL", f"连接失败：{error[:300]}")
        for cid, name in [("R1-02", "远程目录可写"), ("R1-03", "Docker 镜像"), ("R1-04", "模型权重"),
                          ("R1-05", "GPU"), ("R1-06", "数据资产"), ("R1-07", "远程磁盘")]:
            checker.add(cid, name, "SKIP", "连接失败")
        return

    checker.add("R1-01", "SSH 连通", "PASS",
                f"kernel={probe.get('kernel')} cpu={probe.get('nproc')}核 "
                f"内存={probe.get('mem_total_gb')}GB")

    writable = probe.get("root_writable") == "yes"
    checker.add("R1-02", "远程目录可写", "PASS" if writable else "FAIL",
                "远程工作目录可写" if writable else "远程工作目录不可写")

    images_raw = probe.get("docker_images", "")
    remote_images = {}
    for chunk in images_raw.split(";"):
        if "=" in chunk:
            tag, _, image_id = chunk.partition("=")
            remote_images[tag.strip()] = image_id.strip()
    image_lines = []
    images_ok = True
    for tag, expected_id in cfg["images"].items():
        got = remote_images.get(tag, "")
        want = expected_id.removeprefix("sha256:")[:12]
        hit = got.startswith(want) if got else False
        images_ok = images_ok and hit
        image_lines.append(f"{tag}={'OK' if hit else 'MISSING/不符'}")
    if probe.get("docker_ok") != "yes":
        checker.add("R1-03", "Docker 镜像", "FAIL", "远程 docker 不可用（无权限或未运行）")
    else:
        checker.add("R1-03", "Docker 镜像", "PASS" if images_ok else "FAIL", "，".join(image_lines))

    ckpt_ok = probe.get("ckpt_exists") == "yes"
    ckpt_gb = int(probe.get("ckpt_bytes") or 0) / (1024 ** 3)
    checker.add("R1-04", "模型权重", "PASS" if ckpt_ok and ckpt_gb > 1 else "FAIL",
                f"{probe.get('ckpt_files')} 个文件，{ckpt_gb:.2f} GB")

    norm_got = probe.get("norm_stats_sha256", "")
    norm_ok = norm_got == cfg["assets"]["norm_stats_sha256"]
    checker.add("R1-04b", "归一化统计", "PASS" if norm_ok else "FAIL",
                f"sha256={norm_got[:16]}…" if norm_got else "缺失")

    gpu_count = int(probe.get("gpu_count") or 0)
    checker.add("R1-05", "GPU", "PASS" if gpu_count > 0 else "FAIL",
                common.short_gpu_summary(cfg, probe))

    lerobot = probe.get("lerobot_exists") == "yes"
    data_gb = int(probe.get("lerobot_bytes") or 0) / (1024 ** 3)
    checker.add("R1-06", "训练数据资产", "INFO" if lerobot else "WARN",
                f"LeRobot {data_gb:.1f} GB" if lerobot else "LeRobot 数据缺失（仅影响训练，不影响推理）")

    free = int(probe.get("root_free_gb") or 0)
    checker.add("R1-07", "远程磁盘", "PASS" if free >= 20 else "WARN",
                f"{cfg['remote']['root']} 所在分区剩余 {free} GB")


def render(checks: list[dict]) -> None:
    width_id = max(len(c["id"]) for c in checks)
    width_name = max(len(c["name"]) for c in checks)
    marks = {"PASS": "PASS", "FAIL": "FAIL", "WARN": "WARN", "SKIP": "SKIP", "INFO": "INFO"}
    for check in checks:
        print(f"{check['id']:<{width_id}}  {check['name']:<{width_name}}  "
              f"{marks.get(check['status'], check['status']):<5}  {check['detail']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", action="store_true", help="同时检查服务器")
    parser.add_argument("--json", action="store_true", help="只打印 JSON")
    args = parser.parse_args()

    cfg = common.load_config()
    checker = Checker()
    local_ok = check_local(cfg, checker)
    if args.remote:
        check_remote(cfg, checker, local_ok)

    report = {
        "ts": now(),
        "mode": "remote" if args.remote else "local",
        "target": common.label(cfg),
        "root": cfg["remote"]["root"],
        "checks": checker.checks,
        "summary": checker.summary(),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"部署体检 · {report['target']} · 模式={report['mode']} · {report['ts']}")
        print("-" * 88)
        render(checker.checks)
        print("-" * 88)
        print("汇总：" + "  ".join(f"{k}={v}" for k, v in sorted(checker.summary().items())))
        print(f"报告写入 {REPORT.relative_to(ROOT)}")
    return 1 if checker.has_fail() else 0


if __name__ == "__main__":
    sys.exit(main())
