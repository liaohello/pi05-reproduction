# SO101 单方块离线验收状态

离线策略评估未执行：本线没有训练 checkpoint，且按用户指示不占用 GPU。本报告只记录数据与前置审计状态，不给出 action MAE、成功率或任何策略结论。

已完成：64 集成功数据的 view/SHA、train-only norm、loader 与运行时观测/动作管线一致性、base 装载、参数树/冻结范围、三相机渲染同步与视频时长对齐。详见[数据报告](so101-scene1-data.md)与[训练前置](so101-scene1-training.md)；norm 审计需在正式训练前按最终 config 重跑绑定。

未执行：20k 训练、checkpoint 保存/恢复、开环动作评估、按红蓝/位置/托盘分组的模型失败分布、闭环仿真 rollout。评估种子 16000–16099 仍保留；闭环范围未确认。

后续启动条件：4 卡各至少 16.4 GB 真实空闲且不停止他人；先跑 FSDP4/global batch 8 的短更新与完整保存恢复 probe，再跑 20k；离线验收门限在首个 checkpoint 后冻结。当前协议与训练配置已把这些项标为 pending。
