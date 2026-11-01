# SO101 实验记录索引

SO101 E0–E4-T 脱敏记录索引；原始数据不入 Git。记录不代表所有门禁通过。

| 记录 | 内容 |
|---|---|
| [E0 基线](e0-baseline.json) | 源码/权重指纹与 scope 隔离 |
| [E1 示范](e1-manifest.json)、[审核](e1-verification.json)、[训练视图](e1-train-view.json) | 20 集示范、审核、拆分及训练视图 |
| [E2 训练](e2-training.json)、[E3 诊断](e3-diagnostics.json) | 100 次更新/恢复；E3 20 场景 0 成功 |
| [E3.1 专家回放](e31-oracle.json)、[状态](e31-state.json)、[离线](e31-offline.json)、[配对](e31-paired.json)、[验证](e31-validation.json) | 回放、状态、离线/配对训练及完整性检查 |
| [E4-T 试验](e4t-pilot.json)、[数据](e4t-data.json)、[训练](e4t-training.json)、[开放环](e4t-openloop.json)、[预算](e4t-budget.json) | 扰动、数据、训练与门禁；详见 `e4t-*.json` |

E4-T 类别为 A15/B0/C10；B 类被排除。30,000 更新完成，但 key-event timing 失败；无 best checkpoint，E4-R 未开始。单场景 gate-waived 诊断独立于正式门禁。详见 [`数据报告`](../../reports/so101-e4t-data.md)、[`训练报告`](../../reports/so101-e4t-training.md) 和 [`E3.1 摘要`](../../reports/so101-e31-diagnostics.md)。

台账：[`../ledger.jsonl`](../ledger.jsonl)；SO101 scope 不影响 LIBERO gates。LIBERO P1.0 为冒烟；M0/Phase 1 仍 in progress。
