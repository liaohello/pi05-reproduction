# SO101 E4-T：30,000 更新与开放环门禁

30,000 次 state-on 更新完成，batch 8、seed 42，`TrainState.step=30000`，退出码 0。loader worker 调整后前 60 步逐项一致；训练墙钟上限因实测吞吐由 8 小时修订为 9 小时并留有记录。详细训练、采样、checkpoint 与资源证据见 [`e4t-training.json`](../experiments/so101/e4t-training.json)。

开放环连续动作误差门禁通过；**key-event timing 门禁失败**，因此预注册规则下没有 gate-passed checkpoint，也没有“最佳 checkpoint”。step 30,000 仅作诊断状态。E4-R 未启动。初始 checkpoint 的 key-event 计时也失败，早期评估器漏掉事件窗口的问题已在最终重评分中修正。

关键限制：训练 loss 下降不等于闭环任务成功。E4-T 数据阶段 B 类为零，样本类别不完整；E4-T 的单场景 gate-waived 闭环诊断是另一个明确标注的诊断事件，不可覆盖本开放环失败门禁。指标曲线及判定见 [`e4t-openloop.json`](../experiments/so101/e4t-openloop.json)，数据偏差见 [`e4t-data.json`](../experiments/so101/e4t-data.json)。
