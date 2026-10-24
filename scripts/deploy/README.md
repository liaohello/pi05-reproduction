# 部署脚本

本目录只做三件事：体检、同步、跑一次 episode。所有机器相关的值都来自环境变量或被 Git 忽略的
`config.local.json`，源码里没有地址、端口、用户名与私钥路径。

## 配置

```bash
cp ../../configs/hosts/example.env ../../.env.local
set -a && source ../../.env.local && set +a
```

`config.example.json` 描述字段到环境变量的映射；`common.py` 按顺序查找
`config.local.json` → `config.json` → `config.example.json`，遇到 `{"env": "NAME"}` 就从环境读取，
缺失时给出明确报错而不是用默认值猜。

## 脚本

| 文件 | 作用 | 通过标准 |
| --- | --- | --- |
| `preflight.py` | 本地 + 远端体检，结果写 `experiments/.state/preflight-report.json` | 无 FAIL |
| `sync.py` | rsync 同步；`--check` 用 `-c` 逐字节比对；`--dry-run` 预览 | `VERIFY PASSED` |
| `smoke.py` | 远端跑一次 episode，解析 `evaluation.json` 并写台账 | 输出 `episode 完成 N，成功 M` |
| `release.py` | 删除本项目容器与网络，报告 GPU 是否归还 | 输出「本项目容器 无 / 网络 无」 |
| `remote_probe.sh` | 远端只读探针（不拉镜像、不加载模型、不占 GPU） | 通过 `ssh bash -s` 执行 |
| `remote_smoke.sh` | 远端执行体：起服务 → 跑 episode → 停服务 | 输出 `RESULT_JSON` 与 `SUMMARY` |
| `_env.py` | 打印解析后的变量，便于手敲命令 | `./scripts/run.sh env` |

## 约定

- 远端只有一个工作目录，代码永远由 `sync.py` 覆盖，不在远端手改。
- 权重、数据集、镜像留在资源池，用环境变量引用，不复制进仓库。
- `--delete` 生效，但排除项受保护：`.git`、`.venv`、`cache/`、`home/`、`runs/`、`experiments/.state`。
- 每次同步把 git 引用与差异统计写入 `experiments/.state/last-sync.json`，作为阶段证据。

## 失败处理

| 现象 | 处理 |
| --- | --- |
| `missing environment variable PI05_*` | 没有 source `.env.local` |
| `Permission denied (publickey)` | 私钥路径错或权限不是 600 |
| `VERIFY FAILED` | 远端有手工改动，先看差异文件 |
| `GPU ... below threshold` | 显存不足，换卡或等待；不要为通过而调低阈值 |
| `server listening` 超时 | 看 `runs/<id>/logs/policy-server.log` |
