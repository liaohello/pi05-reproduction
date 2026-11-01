# 复现

先跑体检，再读文档。命令是定义，散文不是。

| 等级 | 含义 | 门槛 |
|---|---|---|
| R1 结构 | 协议、配置、证据齐全，pin 可机读 | 无需任何私有配置 |
| R2 环境 | 镜像与源码树在位且与记录一致 | 需 `PI05_REGISTRY` |
| R3 数值 | 记录的 run 可重跑并比对台账 | 需 `PI05_CKPT` |

```bash
./scripts/run.sh repro doctor --compile   # 新克隆的第一条命令
./scripts/run.sh repro plan               # 到下一级要做什么
./scripts/run.sh repro fetch-images       # 拉镜像并校验 image id
./scripts/run.sh repro verify-source      # 源码树比对内容锁
```

当前：R1 可达。R2 被四个未填字段挡住 —— OpenPI 与 LIBERO 归档各缺 URL 和 SHA-256，列在 `configs/repro/manifest.json` 的 `status: blocked` 项里。权重与数据集不可再分发，R3 只在持有 `PI05_CKPT` 的机器上成立。

规则见 [policy/reproducibility.md](../policy/reproducibility.md)。
