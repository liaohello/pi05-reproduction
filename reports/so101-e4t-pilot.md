# SO101 E4-T：扰动采集试验索引

本轮经历 v1–v14 迭代。v13 时门禁失败，不能启动正式数据集与训练；后续 v14 的门禁通过后，采集按冻结规则继续。最终数据偏差为 A15/B0/C10，B 类没有合格的物理有效且 replay-exact 机制，因此数据不满足原计划类别配额。

阶段事件须区分：早期 E4-T data gate 曾失败；v14 后续通过并允许采集/训练。数据门禁通过不等于自主恢复通过。30,000 更新训练完成，但开放环 key-event timing 检查失败，故没有符合规则的 checkpoint，未启动 E4-R。详见 [`e4t-pilot.json`](../experiments/so101/e4t-pilot.json)、[`e4t-data.json`](../experiments/so101/e4t-data.json) 与 [`训练及门禁报告`](so101-e4t-training.md)。

各次试验、排除和预算调整保留于机器记录；不要把单场景 waived 诊断误写成一般 gate pass。该报告是导航摘要，不取代证据 JSON。
