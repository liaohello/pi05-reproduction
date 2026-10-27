#!/usr/bin/env python3
"""阶段门禁与运行台账汇总：控制台表格 + track/STATUS.md + track/dashboard.html。

门禁是自动判定的，不靠手填：
  D0 取 Git 提交与源码基线校验
  D1 取 deploy/last-sync.json 与体检报告的 SSH 结果
  D2 取 deploy/preflight-report.json 的镜像/权重/GPU 结果
  D3/D4/D5 取 track/ledger.jsonl 的运行记录

用法：
    python3 track/status.py            # 打印 + 写 STATUS.md + dashboard.html
    python3 track/status.py --quiet    # 只写文件
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deploy"))

import common  # noqa: E402

ROOT = common.ROOT
LEGACY = ROOT / "experiments" / "ledger.jsonl"
PHASES = ROOT / "experiments" / "phases.json"
STATUS_MD = ROOT / "reports" / "status.md"
DASHBOARD = ROOT / "reports" / "dashboard.html"
PREFLIGHT = common.STATE_DIR / "preflight-report.json"
LAST_SYNC = common.STATE_DIR / "last-sync.json"

PASS, FAIL, PENDING = "pass", "fail", "pending"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Ctx:
    def __init__(self, *, runs: list[dict] | None = None, cfg: dict | None = None) -> None:
        self.cfg = common.load_config_or_exit() if cfg is None else cfg
        self.phase_def = json.loads(PHASES.read_text(encoding="utf-8"))
        self.runs = self._load_runs() if runs is None else list(runs)
        self.preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8")) if PREFLIGHT.is_file() else {}
        self.last_sync = json.loads(LAST_SYNC.read_text(encoding="utf-8")) if LAST_SYNC.is_file() else {}

    @staticmethod
    def _load_runs() -> list[dict]:
        if not LEGACY.is_file():
            return []
        return [json.loads(line) for line in LEGACY.read_text(encoding="utf-8").splitlines() if line.strip()]

    def probe(self, cid: str) -> dict:
        for check in self.preflight.get("checks", []):
            if check["id"] == cid:
                return check
        return {}

    def runs_of(self, kind: str) -> list[dict]:
        """LIBERO gate input. Legacy records without scope count as LIBERO."""
        return [r for r in self.runs
                if r.get("kind") == kind and r.get("scope", "libero") == "libero"]

    def runs_of_scope(self, scope: str) -> list[dict]:
        return [r for r in self.runs if r.get("scope", "libero") == scope]


def evaluate(ctx: Ctx, kind: str) -> tuple[str, str]:
    cfg = ctx.cfg

    if kind == "git_commit":
        result = common.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"])
        if result.returncode != 0:
            return FAIL, "仓库未初始化"
        commit = result.stdout.strip()
        dirty = [line for line in common.run(
            ["git", "-C", str(ROOT), "status", "--porcelain"]).stdout.splitlines() if line.strip()]
        return PASS, f"commit={commit}" + (f"（{len(dirty)} 处未提交）" if dirty else "（干净）")

    if kind == "source_baseline":
        result = common.run([sys.executable, str(ROOT / "scripts" / "verify" / "source_lock.py")], timeout=900)
        lines = [line.strip() for line in result.stdout.splitlines() if "mismatches=" in line]
        if result.returncode == 2:
            return PENDING, "第三方源码未就位（可选），设置 PI05_OPENPI_SRC 后校验"
        ok = result.returncode == 0 and "SOURCE LOCK PASSED" in result.stdout
        return (PASS if ok else FAIL), " | ".join(lines) or "校验未通过"

    if kind == "norm_stats_local":
        path = ROOT / "configs" / "libero" / "norm_stats.json"
        if not path.is_file():
            return PENDING, "本地副本缺失"
        got = sha256(path)
        expected = cfg["assets"]["norm_stats_sha256"]
        return (PASS if got == expected else FAIL), f"sha256={got[:16]}…"

    if kind == "ssh_reachable":
        check = ctx.probe("R1-01")
        if not check:
            return PENDING, "未执行远程体检（deploy/preflight.py --remote）"
        return (PASS if check["status"] == "PASS" else FAIL), check["detail"][:70]

    if kind == "synced":
        data = ctx.last_sync
        if not data:
            return PENDING, "尚无同步记录"
        ok = data.get("returncode") == 0 and data.get("mode") in {"sync", "dry-run", "check"}
        return (PASS if ok else FAIL), \
            f"{data.get('ts', '')} 模式={data.get('mode')} git={data.get('git', '')}"

    if kind == "verified":
        data = ctx.last_sync
        if not data:
            return PENDING, "尚无内容级核验记录"
        ok = data.get("mode") == "check" and bool(data.get("ok"))
        return (PASS if ok else FAIL), ("远程与本地逐字节一致" if ok
                                       else f"差异 {data.get('stats', {}).get('changed', '?')} 处")

    if kind == "images":
        checks = [ctx.probe(cid) for cid in ("R1-03",)]
        if not checks[0]:
            return PENDING, "未执行远程体检"
        return (PASS if checks[0]["status"] == "PASS" else FAIL), checks[0]["detail"][:70]

    if kind == "checkpoint":
        pairs = [ctx.probe("R1-04"), ctx.probe("R1-04b")]
        if not pairs[0]:
            return PENDING, "未执行远程体检"
        ok = all(c["status"] == "PASS" for c in pairs)
        return (PASS if ok else FAIL), "；".join(c["detail"][:40] for c in pairs)

    if kind == "gpu":
        check = ctx.probe("R1-05")
        if not check:
            return PENDING, "未执行远程体检"
        return (PASS if check["status"] == "PASS" else FAIL), check["detail"][:70]

    if kind == "smoke_suites":
        needed = ctx.phase_def["suite_required"]
        done = []
        for suite in needed:
            hit = [r for r in ctx.runs_of("smoke")
                   if r.get("suite") == suite and r.get("status") == "success" and r.get("successes", 0) >= 1]
            if hit:
                done.append(suite)
        return (PASS if len(done) == len(needed) else (PENDING if not done else FAIL)), \
               f"{len(done)}/{len(needed)} 套件成功" + (f"：{'、'.join(done)}" if done else "")

    if kind == "smoke_evidence":
        smoke = [r for r in ctx.runs_of("smoke") if r.get("status") == "success"]
        good = [r for r in smoke if r.get("artifacts") and r.get("command")]
        if not smoke:
            return PENDING, "无成功冒烟记录"
        return (PASS if len(good) == len(smoke) else FAIL), f"{len(good)}/{len(smoke)} 条带命令与产物"

    if kind == "eval_coverage":
        need = ctx.phase_def["episodes_per_suite"]
        parts, ok = [], True
        for suite in ctx.phase_def["suite_required"]:
            total = sum(r.get("episodes", 0) for r in ctx.runs_of("eval")
                        if r.get("suite") == suite and r.get("status") == "success")
            ok = ok and total >= need
            parts.append(f"{suite}={total}/{need}")
        return (PASS if ok else PENDING), "，".join(parts)

    if kind == "eval_metrics":
        good = [r for r in ctx.runs_of("eval") if "sr" in r.get("metrics", {})]
        return (PASS if good else PENDING), f"{len(good)} 条评估含 SR 指标"

    if kind == "base_weights":
        return PENDING, "未探测；需 pi05_base 约 12.442 GB（当前材料缺口）"

    if kind == "train_run":
        good = [r for r in ctx.runs_of("train") if r.get("status") == "success"]
        return (PASS if good else PENDING), f"{len(good)} 条训练记录"

    return PENDING, "未知检查类型"


def build(ctx: Ctx) -> dict:
    key_blocked = ctx.probe("L0-01").get("status") == "FAIL"
    phases = []
    for phase in ctx.phase_def["phases"]:
        checks = []
        for check in phase["checks"]:
            state, detail = evaluate(ctx, check["type"])
            checks.append({"id": check["id"], "label": check["label"], "state": state, "detail": detail})
        passed = sum(1 for c in checks if c["state"] == PASS)
        failed = sum(1 for c in checks if c["state"] == FAIL)
        if passed == len(checks):
            status = "DONE"
        elif failed:
            status = "BLOCKED" if passed == 0 else "IN_PROGRESS"
        else:
            status = "IN_PROGRESS" if passed else "PENDING"
        if key_blocked and status != "DONE" and phase["id"] in {"D1", "D2"}:
            status = "BLOCKED"
        phases.append({**phase, "checks": checks, "passed": passed,
                       "total": len(checks), "status": status})

    done = sum(1 for p in phases if p["status"] == "DONE")
    libero_runs = [r for r in ctx.runs if r.get("scope", "libero") == "libero"]
    metrics = {
        "runs": len(libero_runs),
        "smoke": len(ctx.runs_of("smoke")),
        "eval": len(ctx.runs_of("eval")),
        "train": len(ctx.runs_of("train")),
        "episodes": sum(r.get("episodes", 0) for r in libero_runs),
        "successes": sum(r.get("successes", 0) for r in libero_runs),
    }
    so101_runs = ctx.runs_of_scope("so101")
    so101 = {
        "runs": len(so101_runs),
        "phases": {
            phase: {
                "runs": sum(1 for r in so101_runs if r.get("phase") == phase),
                "successes": sum(1 for r in so101_runs
                                 if r.get("phase") == phase and r.get("status") == "success"),
            }
            for phase in ("E0", "E1", "E2", "E3")
        },
        "last": sorted(so101_runs, key=lambda r: r["ts"], reverse=True)[:5],
    }
    return {"ts": now(), "phases": phases, "metrics": metrics, "so101": so101,
            "done": done, "total": len(phases),
            "runs": sorted(libero_runs, key=lambda r: r["ts"], reverse=True)}


def render_console(report: dict) -> None:
    marks = {"DONE": "DONE  ", "IN_PROGRESS": "进行中", "BLOCKED": "阻塞  ", "PENDING": "未开始"}
    print(f"π0.5 + LIBERO 复现追踪 · {report['ts']}")
    print(f"服务器：{common.label(common.load_config())}")
    print("-" * 78)
    print(f"{'阶段':<5}{'名称':<28}{'状态':<8}{'门禁':<8}")
    for phase in report["phases"]:
        name = phase["name"] if len(phase["name"]) <= 12 else phase["name"][:11] + "…"
        print(f"{phase['id']:<6}{name:<26}{marks.get(phase['status'], phase['status']):<9}"
              f"{phase['passed']}/{phase['total']}")
    print("-" * 78)
    for phase in report["phases"]:
        if phase["status"] == "DONE":
            continue
        for check in phase["checks"]:
            if check["state"] != PASS:
                print(f"  [{phase['id']}] {check['label']}：{check['detail'][:90]}")
    m = report["metrics"]
    print("-" * 78)
    if any(p["status"] == "BLOCKED" for p in report["phases"]):
        print("  阻塞：缺少 SSH 私钥，设置 PI05_SSH_KEY_PATH 指向你的私钥")
    print(f"run 记录 {m['runs']} 条（冒烟 {m['smoke']} / 评估 {m['eval']} / 训练 {m['train']}），"
          f"累计 episode {m['episodes']}，成功 {m['successes']}")
    print(f"阶段完成 {report['done']}/{report['total']}")
    so101 = report.get("so101", {})
    phases_text = "  ".join(f"{phase} run={value['runs']} 成功记录={value['successes']}"
                            for phase, value in so101.get("phases", {}).items())
    print(f"SO101 工程分支（独立 scope，不计入上述门禁）：run {so101.get('runs', 0)} 条；{phases_text}")


def render_md(report: dict) -> str:
    lines = [
        "# π0.5 + LIBERO 复现追踪",
        "",
        f"- 生成时间：{report['ts']}",
        f"- 服务器：`{common.label(common.load_config())}`",
        f"- 阶段完成：{report['done']}/{report['total']}",
        "",
        "| 阶段 | 名称 | 状态 | 门禁 |",
        "| --- | --- | --- | --- |",
    ]
    for phase in report["phases"]:
        lines.append(f"| {phase['id']} | {phase['name']} | {phase['status']} | "
                     f"{phase['passed']}/{phase['total']} |")
    lines += ["", "## 门禁明细", ""]
    for phase in report["phases"]:
        lines += [f"### {phase['id']} {phase['name']}", "", f"> {phase['goal']}", ""]
        for check in phase["checks"]:
            mark = {"pass": "x", "fail": " ", "pending": " "}[check["state"]]
            lines.append(f"- [{mark}] {check['id']} {check['label']} — {check['detail']}")
        lines.append("")
    if report["runs"]:
        lines += ["## 最近运行", "", "| 时间 | 类型 | 套件 | episodes | SR | 状态 | run_id |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for run in report["runs"][:15]:
            sr = run.get("metrics", {}).get("sr")
            lines.append(f"| {run['ts'][:19]} | {run['kind']} | {run.get('suite') or '-'} | "
                         f"{run.get('episodes', 0)} | {sr if sr is not None else '-'} | "
                         f"{run['status']} | `{run['run_id']}` |")
        lines.append("")
    so101 = report.get("so101", {})
    lines += ["## SO101 工程分支（独立 scope）", "",
              "以下记录不参与上面的 LIBERO 门禁，也不会改变 D0–D5 状态。", "",
              "| 阶段 | run 记录 | 成功记录 |", "| --- | --- | --- |"]
    for phase, value in so101.get("phases", {}).items():
        lines.append(f"| {phase} | {value['runs']} | {value['successes']} |")
    if so101.get("last"):
        lines += ["", "| 时间 | 阶段 | 类型 | episodes | 成功 | 状态 | run_id |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for run in so101["last"]:
            lines.append(f"| {run['ts'][:19]} | {run.get('phase')} | {run['kind']} | "
                         f"{run.get('episodes', 0)} | {run.get('successes', 0)} | "
                         f"{run['status']} | `{run['run_id']}` |")
    lines.append("")
    return "\n".join(lines)


CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#14161a;--muted:#5d6470;--line:#e3e6ec;
--ok:#1a7f4b;--okbg:#e7f5ee;--bad:#b3261e;--badbg:#fdecea;--wait:#8a6d00;--waitbg:#fdf3d7;--accent:#2f5bd7}
@media (prefers-color-scheme: dark){
:root{--bg:#12141a;--card:#1b1f27;--ink:#eef1f6;--muted:#9aa3b2;--line:#2b3140;
--ok:#5fd39a;--okbg:#16311f;--bad:#ff8a80;--badbg:#3a1c1a;--wait:#e5c05a;--waitbg:#3a3116;--accent:#7aa2ff}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
header{padding:28px 32px 18px;border-bottom:1px solid var(--line);background:var(--card)}
h1{margin:0 0 6px;font-size:20px;letter-spacing:.2px}
.sub{color:var(--muted);font-size:13px}
main{padding:24px 32px 40px;max-width:1180px}
.bar{height:8px;border-radius:99px;background:var(--line);overflow:hidden;margin:14px 0 6px}
.bar>i{display:block;height:100%;background:var(--accent)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:20px 0 26px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.kpi b{display:block;font-size:22px;font-weight:650}
.kpi span{color:var(--muted);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px}
.phase{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 18px}
.phase h2{margin:0;font-size:15px;display:flex;align-items:center;gap:8px}
.pid{font:600 12px/1 ui-monospace,Menlo,monospace;color:var(--muted);border:1px solid var(--line);border-radius:6px;padding:3px 6px}
.pill{margin-left:auto;font-size:11px;padding:3px 9px;border-radius:99px;font-weight:600}
.DONE{color:var(--ok);background:var(--okbg)}
.IN_PROGRESS{color:var(--wait);background:var(--waitbg)}
.PENDING{color:var(--muted);background:var(--bg)}
.BLOCKED{color:var(--bad);background:var(--badbg)}
.goal{color:var(--muted);font-size:12.5px;margin:6px 0 10px}
ul{list-style:none;margin:0;padding:0}
li{display:flex;gap:8px;padding:5px 0;border-top:1px dashed var(--line);font-size:13px}
li:first-child{border-top:0}
.dot{width:7px;height:7px;border-radius:99px;margin-top:8px;flex:none}
.dot.pass{background:var(--ok)}.dot.fail{background:var(--bad)}.dot.pending{background:var(--muted)}
.detail{color:var(--muted);font-size:12px;display:block}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden;font-size:13px}
th,td{padding:9px 12px;text-align:left;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:600;font-size:12px}
tr:last-child td{border-bottom:0}
code{font:12px/1.4 ui-monospace,Menlo,monospace;color:var(--accent)}
h3{margin:30px 0 10px;font-size:15px}
footer{color:var(--muted);font-size:12px;padding:0 32px 32px}
"""


def render_html(report: dict) -> str:
    cfg = common.load_config()
    pct = int(report["done"] / report["total"] * 100)
    kpis = [
        ("阶段完成", f"{report['done']}/{report['total']}"),
        ("run 记录", report["metrics"]["runs"]),
        ("累计 episode", report["metrics"]["episodes"]),
        ("冒烟成功记录", report["metrics"]["smoke"]),
    ]
    for phase in report["phases"]:
        for check in phase["checks"]:
            check["_esc_label"] = html.escape(check["label"])
            check["_esc_detail"] = html.escape(check["detail"])

    cards = []
    for phase in report["phases"]:
        items = "".join(
            f'<li><span class="dot {c["state"]}"></span><div>{c["_esc_label"]}'
            f'<span class="detail">{c["_esc_detail"]}</span></div></li>'
            for c in phase["checks"])
        cards.append(
            f'<section class="phase"><h2><span class="pid">{phase["id"]}</span>'
            f'{html.escape(phase["name"])}<span class="pill {phase["status"]}">'
            f'{phase["status"]} {phase["passed"]}/{phase["total"]}</span></h2>'
            f'<p class="goal">{html.escape(phase["goal"])}</p><ul>{items}</ul></section>')

    rows = []
    for run in report["runs"][:15]:
        sr = run.get("metrics", {}).get("sr")
        rows.append(
            f"<tr><td>{html.escape(run['ts'][:19])}</td><td>{html.escape(run['kind'])}</td>"
            f"<td>{html.escape(run.get('suite') or '-')}</td><td>{run.get('episodes', 0)}</td>"
            f"<td>{'-' if sr is None else f'{sr:.3f}'}</td><td>{html.escape(run['status'])}</td>"
            f"<td><code>{html.escape(run['run_id'])}</code></td></tr>")
    table = ("<table><thead><tr><th>时间</th><th>类型</th><th>套件</th><th>episodes</th>"
             "<th>SR</th><th>状态</th><th>run_id</th></tr></thead><tbody>"
             + ("".join(rows) if rows else
                '<tr><td colspan="7" style="color:var(--muted)">尚无运行记录</td></tr>')
             + "</tbody></table>")

    kpi_html = "".join(f'<div class="kpi"><b>{v}</b><span>{k}</span></div>' for k, v in kpis)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>pi05 + LIBERO 复现追踪</title><style>{CSS}</style></head>
<body><header>
<h1>π0.5 + LIBERO 复现追踪</h1>
<div class="sub">服务器 {html.escape(common.label(cfg))} · 生成于 {html.escape(report['ts'])}</div>
<div class="bar"><i style="width:{pct}%"></i></div>
<div class="sub">阶段完成度 {pct}%</div>
</header><main>
<div class="kpis">{kpi_html}</div>
<div class="grid">{''.join(cards)}</div>
<h3>最近运行</h3>{table}
</main>
<footer>门禁自动取自 Git、源码基线、deploy/preflight-report.json、deploy/last-sync.json 与 track/ledger.jsonl。</footer>
</body></html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    ctx = Ctx()
    report = build(ctx)
    STATUS_MD.write_text(render_md(report), encoding="utf-8")
    DASHBOARD.write_text(render_html(report), encoding="utf-8")
    if not args.quiet:
        render_console(report)
        print("已写入 reports/status.md 与 reports/dashboard.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
