# SO101运行要点

依赖外置双SO101 MuJoCo项目、固定OpenPI镜像、tokenizer、checkpoint与数据视图；私有位置由环境变量/忽略配置解析。部署前检查同步差异并执行verify，禁止盲目删除远端文件。详见[provenance](../third_party/PROVENANCE.md)。

核心步骤：核外部源码内容指纹和checkpoint形状；采集后审计/物理回放；仅train拆分计算统计；启动训练时记录配置快照、数据hash、checkpoint、实际step和精确容器owner；开环评估不得把动作送入模拟器。SO101运行与LIBERO D5统计隔离。

恢复报告只能声称实际验证过的参数、optimizer与step恢复；未保存RNG/loader游标时不称逐样本精确续训。每个实验记录失败、超时和停止原因。复跑脚本位于 `scripts/so101/`；证据门槛见[AGENTS](../AGENTS.md)，当前阶段索引见[手册](so101-engineering.md)。
