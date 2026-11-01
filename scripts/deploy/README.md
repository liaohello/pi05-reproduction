# 部署脚本

脚本负责配置体检、同步和远程 smoke。机器值只从环境或忽略的 `config.local.json` 读取；使用 [`example.env`](../../configs/hosts/example.env)。字段映射见 `config.example.json`；`common.py` 按 local → config → example 读取，缺变量时报错。

## 命令

- `preflight.py`：本地/远端检查，报告写入忽略的 `experiments/.state/`；要求无 FAIL。
- `sync.py`：rsync；`--check` 校验内容，`--dry-run` 预览；要求 `VERIFY PASSED`。
- `smoke.py`：远端跑 episode，解析 JSON 并记账。
- `release.py`：清理本项目容器/网络并检查 GPU 是否归还。
- `remote_probe.sh`：只读探针，不拉镜像、不加载模型、不占 GPU。
- `remote_smoke.sh`：起服务、跑 episode、停服务，输出结果摘要。
- `_env.py`：显示解析值；入口见 `scripts/run.sh env`。

远端工作目录由同步覆盖，禁止远端手改。权重、数据和镜像不进仓库。同步保护 `.git`、`.venv`、缓存、home、runs 和运行状态；记录 git 引用与差异统计。故障排查：环境缺失时 source 本地 env；同步失败先看差异；GPU不足应等待或换资源，不能降低验收阈值。运行日志只在本地受控目录查看，发布前清理路径与连接信息。
