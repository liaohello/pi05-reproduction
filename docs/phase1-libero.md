# Phase 1 · LIBERO 使用说明

本文说明如何在自有 GPU 机器上复现 Phase 1 的推理与评估。所有与机器相关的值都从环境变量读取，
仓库内不保存任何地址、端口、用户名或私钥路径。

## 1. 准备

需要：

1. 一台带 NVIDIA GPU 的 Linux 机器，Docker 可用且当前用户在 `docker` 组；
2. 两个镜像：`openpi-server`（推理）与 `openpi-client`（LIBERO 仿真），内容要求见
   `third_party/PROVENANCE.md`；
3. `pi05_libero` checkpoint（约 11.58 GB，16 个文件）与 PaliGemma tokenizer；
4. 本机 Python 3.10+（只跑编排脚本，不需要 JAX/CUDA）。

```bash
cp configs/hosts/example.env .env.local
$EDITOR .env.local          # 填 host / port / user / key / 远端目录 / 资产路径
set -a && source .env.local && set +a
```

`scripts/deploy/config.example.json` 里写的是变量名映射，不是值；真实值只存在于 `.env.local`
或 shell 环境。若你更愿意用配置文件，可创建被 Git 忽略的
`scripts/deploy/config.local.json`，字段与 `config.example.json` 相同，但直接写值。

## 2. 部署

```bash
./scripts/run.sh preflight          # 本地：配置、源码锁、git、磁盘
./scripts/run.sh preflight-remote   # 远端：SSH、目录、镜像 ID、checkpoint、GPU
./scripts/run.sh deploy             # 同步到远端工作目录
./scripts/run.sh verify             # 内容级核验，必须输出 VERIFY PASSED
```

`deploy` 使用 rsync 加 `--delete`，排除项包含 `.git`、`.venv`、`experiments/.state`、
`reports/private`、`cache/`、`home/`、`runs/`。远端 `cache/` 与 `home/` 是容器运行时目录，
不能被同步删除。

## 3. 跑一次 episode

```bash
./scripts/run.sh smoke --suite libero_spatial --tasks 0 --episodes 1
```

`smoke.py` 的行为：

1. 在远端挑空闲显存最大的 GPU；低于阈值（默认 16000 MiB，`PI05_MIN_FREE_MB` 可调）直接拒绝启动；
2. 建内部网络 `pi05-libero`，启动策略服务，等到日志出现 `server listening`；
3. 在客户端容器里运行 `evaluation/simulation/libero_episode_runner.py`，录像与 JSON 写入
   `<远端工作目录>/runs/<时间戳>-smoke-<套件>/`；
4. 停容器、删容器，把结果解析成台账记录；
5. 失败时保留服务日志，退出码非零。

## 4. 协议参数

固定值在 `configs/libero/protocol.json`：`seed=7`、`replan_steps=10`、`settle=10`、
渲染 256、策略输入 224、相机旋转 180°、每套件步数上限 220/280/300/520。
任何一次运行如果改了其中一项，必须在台账的 `notes` 里写清楚。

## 5. 记录与报告

```bash
./scripts/run.sh record --phase P1 --kind eval --suite libero_10 \
  --tasks 10 --episodes 500 --successes 481 --metric sr=0.962 --metric spl=0.91 \
  --command "./scripts/run.sh ..." --artifact <远端产物目录>
./scripts/run.sh status
```

台账字段固定：`run_id`、时间、阶段、类型、套件、任务数、episode 数、成功数、指标、
`policy_config`、`checkpoint` id、两套镜像 digest、`norm_stats_sha256`、git 引用、命令、
产物（私有前缀替换为占位符）、状态与备注。

`status` 会把阶段门禁渲染到 `reports/status.md` 与 `reports/dashboard.html`（两者都是生成物，
不进 Git）。

## 6. 已完成的证据

P1.0 的四次 rollout 见 `reports/phase1-inference-smoke.md`，台账在 `experiments/ledger.jsonl`。
口径固定为：**4/4 冒烟 episode 成功，不代表基准成功率**。

## 7. 常见问题

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `Permission denied (publickey)` | 私钥不存在或路径不对 | 检查 `PI05_SSH_KEY_PATH` 与权限 600 |
| `VERIFY FAILED` | 远端有手工改动或被旧版本覆盖 | 先看差异文件，确认后再 `deploy` |
| `GPU ... below threshold` | 显存被别人占满 | 换卡或等；不要为了让脚本通过而把阈值调到危险值 |
| `server listening` 等不到 | checkpoint 不完整或端口被占 | 看 `runs/<id>/logs/policy-server.log` |
| 客户端渲染全黑 | 缺 `MUJOCO_GL`/`PYOPENGL_PLATFORM` | 检查 `remote_smoke.sh` 的环境变量是否被改动 |
