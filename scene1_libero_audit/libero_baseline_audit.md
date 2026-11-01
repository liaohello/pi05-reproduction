# LIBERO 交叉验证审计（P4）

**结论：正在跑的 LIBERO 不是"官方 pi05_libero 微调"，而是从 `pi05_base` 从头微调的
`libero_goal` 任务 8。它可以检验共享训练基础设施，但不能充当 L0 基线，也不能为 Scene1 的
状态路径背书。L0 `BLOCKED`，L1 部分可用，L2 未触发。**

---

## 1. 实验身份（已识别，非假设）

| 问题 | 答案 | 来源 |
|---|---|---|
| 初始化权重 | `pi05_base`，tree sha `9909d717…` | `config_initial.json:4-8, 64-66` |
| 是否 `pi05_libero` 微调 | **否** | 同上；`resume_from_probe = false`，`start_train_state_step = 0` |
| 是否基于上游官方配置 | 否，用本仓库 `configs/libero/s2_training.json`（sha `23fcec4b…`） | `config_initial.json:102` |
| 微调范围 | action expert 全量 + 语言 LoRA(r16) + 投影全量，视觉冻结 | `result.json:68-73` |
| 数据 | `libero_goal`，task 8 "put the bowl on the plate"，39 训练 / 10 验证 | `config_initial.json:15-34` |
| `discrete_state_input` | **false** | `:43` |
| 动作 | 7 维 delta EEF，horizon 25，内部 32 维 | `:16-24` |
| EMA | 0.999（开启） | `:84` |
| Norm Stats | sha `dba8704a…`，`validation_in_norm_stats: false` | `:25, 33` |

一个值得肯定的细节：`result.json:53-62` 记录了两次失败启动（argparse 拒绝 `--python-arg`；
镜像 digest 校验拒绝镜像名），并注明"未创建 training checkpoint"。**这是正确的溯源纪律**——
失败尝试没有被伪装成一次训练。

## 2. 与 Scene1 的共享面 / 差异面

共享（LIBERO 正常 = 这些正常）：

- 同一个 base checkpoint tree（sha 完全相同）
- 同一个容器镜像 `405af520…`、同一个 JAX 内存计划 0.7119
- 同一套 AdamW、warmup→cosine、EMA 0.999、seed 42、batch 8、clip 1.0
- 同一份 `train.py` / `checkpoints.py` / `compute_loss` / `transforms.py`

不共享（LIBERO 正常 **不能** 推出这些正常）：

- `discrete_state_input`: **false vs true** —— 状态走连续投影 vs 走 prompt 字符串，是两条完全不同的路径
- 动作语义：7 维 delta EEF vs 12 维绝对关节
- horizon 25 vs 10；相机 2 vs 3；fps 10 vs 50
- Norm Stats（不同文件、不同分布）

因此：**LIBERO 能验证"训练循环/优化器/checkpoint/图像编码"这些公共件；它一个字也证明不了
Scene1 的状态编码、动作尺度、任务条件。**

## 3. L0 官方 pi05_libero 推理基线

**`BLOCKED`。**

- 磁盘上没有任何 `pi05_libero` 权重工件；现有唯一 base 树是 `pi05_base`。
- 取得官方权重需要下载数 GB 资产，任务明确要求"不在未经确认的情况下重复下载大型模型资产"。
- 更硬的理由：4 张 4090 当前被 LIBERO 全量占用（jax fraction 0.71、需求 17.1 GB），
  任何 Scene1/基线 GPU 推理都会与正在运行的训练争抢显存。任务要求先做低成本静态检查，
  这一点已经遵守。

L0 的目的本来是"验证现有推理链路是否可信"。在拿不到官方权重前，替代做法是：用 LIBERO 本次
训练产出的第一个 checkpoint（step 3000）跑一次闭环，检验"本仓库的推理链路能否把一个自己
训练出来的权重跑通"。那要到约 2.5 小时后才存在，且仍需 GPU。

## 4. L1 当前 LIBERO 训练（进行中，只读）

| 量 | 值 | 判读 |
|---|---|---|
| 状态 | RUNNING，194 updates @ 3.0 s/update | 正常推进 |
| loss | 0.0918 → 0.0788 | 32 维口径，7 真 + 25 padding，**不可与 Scene1 比较** |
| grad_norm | 0.263 → 0.269 | **远低于 clip 1.0，裁剪未触发** |
| param_norm | 1803.157 | 稳定（变化 < 1e-3） |
| checkpoint | 无（首个在 3000） | 无法评估 EMA vs current |
| 验证 | 配置了 10 集，尚无记录 | — |

**唯一可用的跨任务推断**：`grad_norm ≈ 0.26 ≪ 1.0`，说明在这套优化器/模型规模下，全局裁剪
很可能长期不触发。这削弱了 `padding_gradient_audit.md §4` 里"padding 通过加重裁剪间接伤害
学习"的假设——但**不能**据此判定 Scene1，因为任务、维度和 horizon 都不同。保持 `LIKELY 削弱`，
不升级为结论。

## 5. L2 官方源码隔离对照

**未触发。** 触发条件是"L0 无法通过，或发现现有源码与官方实现存在重要差异"。L0 尚未执行
（不是失败），且已核对的关键文件（`transforms.py` 的 Normalize/Pad/Tokenize、`pi0.py` 的
compute_loss/embed_prefix）与上游 openpi 语义一致，未见需要隔离对照的差异。**不下载官方源码。**

一个必须记录的溯源缺口：Scene1 与 LIBERO 都从 `/opt/openpi` 加载训练入口
（`train_so101.py:31-42`），而该目录**不在仓库中**，也没有像 LIBERO 的
`openpi_src_sha256` 那样为 Scene1 记录源码树哈希。Scene1 侧只有
`inference_source_sha256 = 6def3aad…`（口径为推理侧源码），与 LIBERO 的
`openpi_src = 6bb8d9f5…` **口径不同，不可直接比较**。这是可复现性缺口，不是本次故障的原因。

## 6. 对 Scene1 的证据贡献

| 问题 | LIBERO 能否回答 |
|---|---|
| 训练循环是否工作 | 能，且目前正常 |
| 优化器/裁剪/EMA 是否被正确接线 | 部分能（grad_norm、param_norm 稳定） |
| Scene1 的状态 token 路径是否有问题 | **不能**（discrete=false） |
| Scene1 的 12 维动作尺度是否合理 | **不能** |
| Scene1 为何总是走向红盘 | **不能** |

这也正是本轮的关键结论之一：**用户启动了正确的对照，但这个对照的覆盖面比预期窄。**
若要让它覆盖 Scene1 的状态路径，需要另一个 `discrete_state_input=true` 的运行，或者直接对
Scene1 做 prompt 交换探针。
