# π0.5 Reproduction

π0.5 推理复现与双 SO101 微调研究。仓库保存适配代码、协议和脱敏证据；上游源码、权重、数据集与原始录像在仓库外，但**获取路径与校验和入库**。

## 当前状态

- LIBERO：P1.0 推理冒烟完成；完整评估与训练复现未完成，M0 仍为 in progress。
- `4/4 smoke-test episodes succeeded; not a benchmark success-rate estimate.`
- SO101：训练、数据和评估代码已入库；E4-T 开环门未通过。一次获准绕过门禁的单场景诊断不代表 E4-R 验收。
- Scene1：单块抓取测试通过（单场景闭环，非成功率估计）；KI 联合训练（PI-FAST + Flow Matching + Knowledge Insulation，语言侧与动作专家均为 LoRA）仅有冻结草案与梯度矩阵诊断，端到端训练与评估未验证。
- RoboTwin、RTC、真机仍按独立阶段推进。

## 复现

新克隆先跑一条命令，它给出 R1/R2/R3 等级并列出阻塞项：

```bash
./scripts/run.sh repro doctor --compile
```

R1（结构可复现）是本仓库的最低状态；R2 需镜像访问，R3 需权重。`repro plan` 打印到下一级的命令。详见 [复现](docs/reproduction.md)。

## 阅读入口

- [完整文档目录](docs/index.md)
- [代码与测试](docs/so101-code-review.md)
- [研究路线](docs/roadmap.md) · [阅读指南](docs/reading-guide.md)
- [LIBERO 操作](docs/phase1-libero.md) · [SO101 操作](docs/so101-engineering.md)
- [实验台账](experiments/ledger.jsonl) · [SO101 证据](experiments/so101/README.md)
- [仓库规则](AGENTS.md) · [许可](THIRD_PARTY_LICENSES.md) · [来源](third_party/PROVENANCE.md)

环境参数使用模板与环境变量；提交前运行 `bash scripts/security/scan.sh`。实验证据须人工审阅后提交。
