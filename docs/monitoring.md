# 本地训练监控（只读）

在浏览器里看 π0.5 LoRA 训练进度：只读取运行目录（唯一写入是自己的 pid 文件 `experiments/.state/watch.pid`，已被忽略），默认只监听 127.0.0.1。

## 启动与停止

```bash
cp configs/monitor/example.env configs/monitor/local.env   # 填 PI05_WATCH_RUNS_DIR
./scripts/run.sh watch          # 前台，Ctrl-C 停止
./scripts/run.sh watch --once   # 打印一次 JSON 快照
./scripts/run.sh watch --stop   # 停止后台进程
```

## 显示内容

运行状态、真实更新数、batch、划分、速度与资源来自只读记录。Training Loss 每产生一条 optimizer-update 记录即保留并绘制；25k 曲线不截断。曲线可切原始/平滑，鼠标悬停显示 step、loss、checkpoint 与验证信息。

Scene1 主图绘制 Training Loss（蓝色、每次 update）、50 步 Probe（橙色）和 Checkpoint Full Validation（紫色标记）。三者共用 Global Step 与 Flow Matching MSE 轴，可独立隐藏和平滑；缺少结果时绝不插值。

## 状态口径

等待启动、运行中、已完成、已失败、已停止、记录中断、监控连接中断。日志暂时不增长只判「记录中断」，失败要启动器记下退出码。

## 验证

`python3 -m unittest tests.test_train_watch`：脱敏合成日志覆盖半行、轮转、重启、重复 step、缺失字段与路径穿越。规则见[AGENTS](../AGENTS.md)，索引见[文档目录](index.md)。
