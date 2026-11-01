# 会话交接

唯一主仓库为 pi05-reproduction；旧内部Git历史不迁入。外置OpenPI、模拟器、模型权重和数据不入库，真实部署值只在忽略配置或环境中。规则见[AGENTS](../AGENTS.md)及[迁移记录](migration-notes.md)。

## 当前事实

- LIBERO P1.0为推理冒烟；完整评估和正式训练未完成，M0仍 in progress。
- SO101训练适配代码在本仓库，实际优化器复用外置OpenPI/JAX。E4-T有50条成功示范（40/10）、30,000步训练；连续动作误差低，但夹爪事件时序门未过，无checkpoint通过开环准入。后有一次获准的单场景豁免诊断：0/4归位并超时，不属E4-R。不得说成任务成功率。
- E4-R须先处理事件时序与恢复数据覆盖。无真实follower运行。
- Scene1：单块抓取测试通过（单场景闭环，非成功率估计，未入台账，不可引用为指标）。KI联合训练（PI-FAST+Flow Matching+Knowledge Insulation，语言侧与动作专家均LoRA、非LoRA冻结）目前只有冻结草案`configs/so101/scene1_pifast_ki_training_v2.json`与b8两卡FSDP2梯度矩阵诊断；四卡复算、端到端训练与评估均未验证。
- Clean60轮（独立`runs/so101/clean60_25k`、协议`clean60_spread_25k_v1`）：布局pilot否证草案0.12 m间距（520次认证采样仅1次达到），已按实测登记0.08 m；训练范围实测为语言骨干全冻结+LoRA可训练、动作专家427.9M与SigLIP 414.8M全量可训练，合计872,771,344可训练参数。见[布局pilot](../reports/so101-clean60-layout-pilot.md)与[训练范围](../reports/so101-clean60-training-scope.md)。60条采集、归一化、多卡probe与25k训练尚未完成。

具体数字以[实验台账](../experiments/ledger.jsonl)、[E4-T训练报告](../reports/so101-e4t-training.md)和[数据报告](../reports/so101-e4t-data.md)为准；历史阶段和复跑入口见[SO101手册](so101-engineering.md)。训练源码导航：[code review](so101-code-review.md)。
