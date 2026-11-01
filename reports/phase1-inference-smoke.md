# LIBERO P1.0 推理冒烟

状态：通过。**4/4 smoke-test episodes succeeded; not a benchmark success-rate estimate.** 日期：2026-10-23。

在公开 `pi05_libero` checkpoint 上，每套件 task 0 执行一次闭环：指令→policy server→模拟器→成功判定。4 条原始记录及 1 条迁移回归见 [`ledger.jsonl`](../experiments/ledger.jsonl)（`kind=smoke, phase=D3`）；视频见 [`assets/pi05_libero_smoke_grid.mp4`](assets/pi05_libero_smoke_grid.mp4)。复现：`./scripts/run.sh smoke --suite libero_spatial --tasks 0 --episodes 1`。

该测试只证明推理链路可运行，不估计成功率、吞吐或训练效果。首个推理约 20 秒含 JAX 编译；共享 GPU 下其余调用不代表稳定延迟。未完成 P1.1 的 2,000 episode 正式评估、P1.2 训练复现或 ACT 对照。M0 / Phase 1 仍为 **in progress**。
