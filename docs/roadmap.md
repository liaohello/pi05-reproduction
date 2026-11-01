# 路线图

当前以 SO101 π0.5 工程链为主；LIBERO、RoboTwin、真机各自验收，不能互相代替。未启动阶段不建空实现。

- **P1 LIBERO**：P1.0仅推理冒烟，规范结论为“4/4 smoke-test episodes succeeded; not a benchmark success-rate estimate.” P1.1的2,000回合协议评估、P1.2正式训练复现、P1.3 ACT对照未完成；M0仍 in progress。见[Phase 1](phase1-libero.md)和[阅读路线](reading-guide.md)。
- **SO101**：E0–E3.1、E4-T已执行；E4-T开环连续误差低，但夹爪事件时序门未过，E4-R不具备资格。结果以[交接摘要](session-handoff.md)、[实验手册](so101-engineering.md)及对应[报告](../reports/so101-e4t-training.md)为准。
- **RoboTwin**：多模型同数据/种子对照仍是独立研究；源码、权重和数据按[路线计划](roadmap-future.md)准备后再启动。
- **真机**：待仿真闭环、接管、限位及急停验证后另行定义。

SO101任务编号与LIBERO阶段门禁独立。记录必须关联ledger；视频、截图和评估JSON提交前须人工审阅。第三方源码、权重和原始数据不入Git。迁移决策见[迁移记录](migration-notes.md)。
