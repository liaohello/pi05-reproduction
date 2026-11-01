# SO101 E2：100 次更新与恢复检查

日期：2026-10-25。E2 是训练链工程检查，不是正式训练，也不证明任务质量。基于 E1 的 16 集训练数据，以 `pi05_libero` 初始化 π0.5 LoRA；batch 1、seed 42、100 次更新。训练视图和 train-only 统计哈希均锁定。

合成数据的一步 smoke 仅验证流水线，不作为 E2 指标。正式运行保存 checkpoint 后执行恢复检查，核对权重及 `TrainState.step`；恢复后的继续更新仅验证可恢复性。详尽配置、loss、参数变化及资源记录见 [`e2-training.json`](../experiments/so101/e2-training.json)。

限制：100 次更新仅为工程验证；没有据此宣称策略成功或泛化。E1 数据、E0 参数加载检查和 E2 记录属于 SO101 scope，不影响 LIBERO P1.2 门禁。
