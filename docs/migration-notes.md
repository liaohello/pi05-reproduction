# 迁移记录

## 来源

| 项 | 值 |
| --- | --- |
| 源仓库 | 内部实验仓库 `pi05Libero`（未公开） |
| 源版本 | `5d85d13`（5 个提交，全部为部署与追踪脚本迭代） |
| 迁移方式 | 按白名单复制文件，重建干净历史；不合并、不 cherry-pick、不复制 `.git` |
| 迁移时间 | 2026-10-24 |

源仓库的历史里包含早期硬编码的服务器地址与私有路径，因此不迁移历史是安全要求，不是偏好。

## 文件去向

| 源 | 目标 | 处理 |
| --- | --- | --- |
| `deploy/`（preflight / sync / smoke / release / 探针） | `scripts/deploy/` | 配置改为环境变量解析；输出中的主机名、用户名、私有路径全部移除 |
| `deploy/config.json` | `scripts/deploy/config.example.json` | 拆成变量映射模板；真实配置改为 `config.local.json`，不入库 |
| `track/ledger.jsonl` | `experiments/ledger.jsonl` | 删除 `target`、`remote_root`；checkpoint 改为 id；产物路径改占位符；git 引用标注为 `internal-*` |
| `track/phases.json` | `experiments/phases.json` | 门禁标签去掉端口号 |
| `track/record.py`、`status.py` | `scripts/track/` | 路径改到 `experiments/` 与 `reports/`；输出改用机器别名 |
| `tools/check_sources.py` + `tools/inspect_runtime.py` | `scripts/verify/source_lock.py` | 两个工具合并：本地校验 + 直接从镜像取哈希；不再写死主机与端口 |
| `deployment/previous/scripts/evaluate_openpi_libero.py` | `evaluation/simulation/libero_episode_runner.py` | 保留评估逻辑，文件头补派生说明与许可 |
| `deployment/previous/scripts/probe_libero.py` | `evaluation/simulation/probe_env.py` | 同上 |
| `deployment/previous/*.md`、`deployment.json` | `docs/` 与本文件 | 只保留可公开的参数与结论，删除地址、端口、实例标签 |
| `deployment/previous/scripts/run-libero.sh` | 不迁移 | 被 `scripts/deploy/` 取代；旧脚本绑定私有 workspace 与实例标签，保留会造成两套部署路径 |
| `resources/pi05_libero/norm_stats.json` | `configs/libero/norm_stats.json` | 2 KB 统计量，保留以便复算动作归一化 |
| `runtime-source-manifest.json`（262 KB） | `third_party/PROVENANCE.md` + 本地 `third_party/lock/` | 只公开聚合哈希与获取方式；逐文件清单可由镜像重新生成 |
| `docs/materials-and-sync.md` | `docs/phase1-libero.md` | 重写：去掉 Windows/服务器绝对路径与端口 |
| `docs/reading-guide.md` | `docs/reading-guide.md` | 原样迁移（不含机器信息） |
| `artifacts/smoke/pi05_libero_smoke_grid.mp4` | `reports/assets/` | 人工复核：仅仿真画面与任务标签 |
| 4 段原始 episode 录像、`evaluation.json` | 不迁移 | 留在实验机；仓库只放合成后的公开证据 |
| `openpi/`（429 MB / 1,256 文件） | 不迁移 | 见 `third_party/PROVENANCE.md` |

## 脱敏规则（已执行的替换）

| 类别 | 处理 |
| --- | --- |
| 公网 IP、SSH 端口 | 从所有文件移除；只保留 `PI05_REMOTE_HOST` / `PI05_REMOTE_PORT` 变量名 |
| 用户名、主机名 | 从代码与报告中移除；输出改用 `remote-gpu-server` 之类的别名 |
| 私钥路径 | 从代码与文档移除；改为 `PI05_SSH_KEY_PATH`（默认 `~/.ssh/id_ed25519`） |
| 私有绝对路径（工作目录、资产目录、数据集路径） | 改为 `<remote-workdir>` / `<assets>` 占位符或环境变量 |
| 镜像仓库地址 | 不公开；只保留内容 digest |
| 台账中的命令 | 去掉 `cd`、绝对路径与主机前缀，保留可复现的参数 |

## 证据口径调整

- 四次成功改写为 `4/4 smoke-test episodes succeeded; not a benchmark success-rate estimate.`
- 阶段命名从内部 `D0–D5` 调整为对外的 `P1.0–P1.3`；`experiments/phases.json` 保留门禁定义，
  但 README 与报告只使用 P 编号。
- M0 记为 `in progress`，因为训练复现与 2,000 episode 评估都还没有。

## 迁移后的验证

| 检查 | 结果 |
| --- | --- |
| `scripts/security/scan.sh` | PASS，0 命中（gitleaks 未安装，仅模式扫描生效） |
| `./scripts/run.sh preflight` | PASS 5 / WARN 1；源码锁 140 + 1,116 文件、聚合哈希与迁移前一致 |
| `./scripts/run.sh preflight-remote` | PASS 12 / INFO 1；镜像 digest、checkpoint、norm_stats、GPU 全部匹配 |
| `./scripts/run.sh deploy` + `verify` | `VERIFY PASSED`，远端工作目录与本地逐字节一致 |
| `./scripts/run.sh smoke` | 迁移后重跑 `libero_spatial` task 0：1/1 成功，台账新增 1 条脱敏记录 |
| 干净克隆 | 38 个受控文件，无机器配置时输出单行提示并以 2 退出 |
| 许可证 | 见 `THIRD_PARTY_LICENSES.md` |

## 没有迁入但需要注意

- 源仓库的 5 个提交仍停留在本机旧目录；若旧仓库曾经公开过任何凭据，需要轮换，而不是删除历史。
- 私有镜像不会公开。第三方若要复现，必须按 `third_party/PROVENANCE.md` 自行构建等价镜像。
