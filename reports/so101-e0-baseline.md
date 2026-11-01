# SO101 E0：基线与记录隔离

日期：2026-10-25。E0 验证外部源码绑定、π0.5 参数装载和 LIBERO/SO101 记录隔离；不代表训练或任务成功。

使用外部 SO101 源码快照（109 个 Python 文件，`source_sha256=201898ff…`）和固定 OpenPI image（源码聚合哈希 `d8ee60bc…`）。π0.5 LoRA 模型的 51 个 checkpoint 张量形状匹配；缺少项仅 LoRA adapter。LeRobot v3 数据由训练视图适配至 image 内旧 reader API，未篡改版本标签。

`record.py` 加 scope，`status.py` 只用 LIBERO 记录计算 D0–D5；SO101 train 记录不会使 P1.2 变绿。来源指纹及参数检查见 [`e0-baseline.json`](../experiments/so101/e0-baseline.json)。E0 仅建立可追溯边界；不证明训练策略质量、LIBERO 完成或真实机器人能力。
