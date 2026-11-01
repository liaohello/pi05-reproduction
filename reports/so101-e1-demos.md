# SO101 E1：专家示范数据

日期：2026-10-25。直接分拣任务共采集 20 次尝试、20 次脚本专家成功；这是**专家采集比例**，不是学习策略成功率。数据为 LeRobot v3、50 Hz、三相机 640×480、12-D 状态/动作；16 集训练，4 集验证。20 个评估场景在训练前冻结。

审核通过：20 集、54,141 帧；帧、时间戳、动作块、图像及抽样状态/动作对齐均无问题；20/20 物理回放成功。训练视图与统计只用训练集，状态/动作对照误差为 0。原始录制留在仓库外。

机器证据：[`manifest`](../experiments/so101/e1-manifest.json)、[`audit`](../experiments/so101/e1-verification.json)、[`train view`](../experiments/so101/e1-train-view.json)。复跑说明见 [`实验索引`](../experiments/so101/README.md) 与 [`runbook`](../docs/so101-engineering.md)。
