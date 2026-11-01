# SO101 E3.1：故障归因与学习诊断

E3.1 保留原 E3 **0/20**。专家回放是**一条示范内 4/4 个物体归位**，并非 4 回合；另检查状态输入、离线动作误差和单轨迹配对训练。

配对训练场景中两支策略都未放置物体，且以超时结束；没有发生接触，也没有接触安全停止。因此 E3.1 仍未学会放置。历史 E3 的失败曾触发接触安全停止，两者原因不同。E4 闭环入口门禁为 false，未启动 E4-R。验证器 112 项完整性检查通过只说明证据/契约一致，不代表策略门禁通过。

详见 [`oracle`](../experiments/so101/e31-oracle.json)、[`状态输入`](../experiments/so101/e31-state.json)、[`离线误差`](../experiments/so101/e31-offline.json)、[`配对训练`](../experiments/so101/e31-paired.json)、[`验证`](../experiments/so101/e31-validation.json)。操作步骤见 [`SO101 runbook`](../docs/so101-engineering.md)。
