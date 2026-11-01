# SO101工程手册索引

SO101接口为三路RGB、12维绝对关节目标（左6+右6；角度degree、夹爪百分比）；LIBERO契约不可套用。机器值来自环境或忽略配置。完整接口见[SO101契约](../inference/SO101_CONTRACT.md)。

- [运行与复现](so101-runbook.md)：部署、配置和ledger边界。
- [E0–E3.1](so101-e0-e31.md)：专家数据、训练恢复及早期诊断事实。
- [E4-T结果](so101-e4t.md)：50条数据、30k训练与未通过的开环门。
- [训练代码导航](so101-code-review.md)：适配器和入口。
- [实验任务原则](agent-execution-prompts.md)：预算、安全和证据。

阶段编号与LIBERO门禁分离。正式数字以[ledger](../experiments/ledger.jsonl)和对应报告为准；不要从摘要推算新指标。E4-R必须在夹爪事件时序门通过后另行启动。

Clean60轮实测：[布局pilot](../reports/so101-clean60-layout-pilot.md)、
[训练范围](../reports/so101-clean60-training-scope.md)、
[多卡probe](../reports/so101-clean60-multicard-probe.md)、
[冻结与短更新](../reports/so101-clean60-freeze-verify.md)。
