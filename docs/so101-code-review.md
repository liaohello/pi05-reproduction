# SO101训练代码导航

本仓库实现SO101训练适配、配置、数据导出和评估；优化循环调用外置OpenPI/JAX训练器。运行需外部代码、镜像、checkpoint、tokenizer和数据，均不随仓库分发。

| 用途 | 文件 |
|---|---|
| 参数与启动 | [`e4t_training.json`](../configs/so101/e4t_training.json)、[`e4t_launch_training.py`](../scripts/so101/e4t_launch_training.py)、[`train_so101.py`](../scripts/so101/train_so101.py) |
| 模型/loader | [`openpi_so101.py`](../scripts/so101/openpi_so101.py) |
| 数据视图 | [`export_e4t_view.py`](../scripts/so101/export_e4t_view.py) |
| 开环评估/门禁 | [`evaluate_e4t_openloop.py`](../scripts/so101/evaluate_e4t_openloop.py)、[`summarize_e4t_openloop.py`](../scripts/so101/summarize_e4t_openloop.py) |
| 动作语义 | [`SO101_CONTRACT.md`](../inference/SO101_CONTRACT.md) |

复跑资产与实验边界见[工程手册](so101-engineering.md)，结果见[报告](../reports/so101-e4t-training.md)。查看测试依赖 `requirements-review.txt`；安全扫描为 `bash scripts/security/scan.sh`。公开证据提交须人工审阅，许可/来源见[清单](../THIRD_PARTY_LICENSES.md)。
