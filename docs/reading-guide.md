# LIBERO代码阅读路线

上游OpenPI/LIBERO源码不在本仓库；本文是定位路线，不证明当前已安装版本一致。完整链路为：任务/初态 → 图像与8维state → WebSocket策略 → π0.5采样7维末端增量 → 环境step与成功谓词。

建议顺序：
1. 本仓库入口与协议：[Phase 1手册](phase1-libero.md)、`configs/libero/protocol.json`。
2. 上游评估循环 `examples/libero/main.py`：任务、初态、相机预处理、replan和成功检查。
3. `policies/libero_policy.py`、`transforms.py`：8维state映射、动作裁剪到7维、归一化。动作不是关节角；LIBERO与SO101契约不同。
4. `serving/websocket_policy_server.py`、client：进程边界和推理接口。
5. `training/config.py` 的 `pi05_libero`、`data_loader.py`、`scripts/train.py`：配置、数据和优化更新。训练依赖外部OpenPI运行时。

π0.5的action horizon、每次执行的replan步数、flow采样步数是不同概念。P1完整门禁见[路线图](roadmap.md)；本仓库当前代码导航见[SO101 review](so101-code-review.md)。
