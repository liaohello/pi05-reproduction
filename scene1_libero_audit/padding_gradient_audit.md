# Padding Loss 与梯度审计（P3）

**结论：机制分析可用，数值全部 `BLOCKED`。上一轮那份"逐维 loss"实现未被采用——它是对
`compute_loss` 的手工复制，本身就违反本轮"不能信任未执行的复制实现"的要求。这里给出一个
不需要复制前向的做法。**

---

## 1. 为什么不能再用上一轮的 `_per_dim_loss`

上一轮 `scene1_root_cause_audit/tools/scene1_audit.py` 里的 `_per_dim_loss` 把
`compute_loss` 的 25 行（split rng、beta 采样、x_t/u_t、embed_prefix/suffix、attn mask、
llm 调用、action_out_proj）抄了一遍，只把最后的 `mean(axis=-1)` 换成返回张量。

它从未运行过。任何一行抄错（例如 `train=False` 与 `train=True` 的预处理差异、adarms_cond
的传参位置），都会得到一个"看起来合理"的错误梯度。**这正是本轮要避免的。**

## 2. 替代方案：包一层 `action_out_proj`，其余全部走官方前向

`compute_loss` 的最后一跳是 `v_t = self.action_out_proj(suffix_out[:, -H:])`，然后
`mean((v_t − u_t)², axis=-1)`。所以只要在调用**官方** `compute_loss` 之前，把
`model.action_out_proj` 换成一个"记录输出再原样返回"的包装器，就能在不复制任何前向逻辑的
前提下拿到 `v_t`；`u_t` 与 `x_t` 由同一 rng 在外部按 `pi0.py:196-200` 三行重建。
随后 `L_real / L_pad / L_total` 只是一次 `square` 加三种 `mean` 的算术。

这样"模型前向"只有一份实现（官方那份），可出错面从 25 行降到 3 行，而且那 3 行可以用
`L_total` 与官方返回值的数值一致性来交叉验证（要求 `|Δ| < 1e-6`）。

## 3. 要测的量（未测）

| 量 | 含义 | 状态 |
|---|---|---|
| `L_real` / `L_pad` / `L_total` | 12 维、20 维、32 维均值 | `BLOCKED` |
| `L32 = (12·L_real + 20·L_pad)/32` | 恒等式校验 | `BLOCKED` |
| 逐维 loss（32 个） | 找异常维 | `BLOCKED` |
| 分阶段 × 分 horizon 的 loss | 任务要求 | `BLOCKED` |
| `‖g_real‖, ‖g_pad‖, ‖g_total‖` 按参数组 | Action Expert / 语言 LoRA / 投影 / 时间投影 | `BLOCKED` |
| `cos(g_real, g_pad)` | 是否同向竞争 | `BLOCKED` |
| 各组贡献占比 | — | `BLOCKED` |
| 是否触发全局裁剪（`clip_gradient_norm=1.0`） | **关键** | `BLOCKED` |
| AdamW 更新方向与范数对比（32 维 vs 12 维 mask） | 任务要求 | `BLOCKED` |

## 4. 一个有具体机理的假设，取代"62.5 % 浪费"

上一轮已经排除两条流行论证（padding 目标可零损失学会；AdamW 对 loss 常数缩放不敏感）。
剩下唯一能让 padding **真实伤害** 12 维学习的通道是**全局梯度裁剪**：

```
g_total = (12·g_real + 20·g_pad) / 32
if ‖g_total‖ > 1.0:  g_total ← g_total · 1.0 / ‖g_total‖
```

只有当 `‖g_total‖` **确实超过 1.0** 时，`g_pad` 才会通过撑大范数、加重裁剪，间接缩小
`g_real` 的有效步长。若范数长期远小于 1.0，裁剪不触发，这条通道就是关闭的——padding 对
优化动力学没有影响，只剩下"指标口径"这一个真实危害。

这是一个**可判定**的假设，而且判据只有两个数：`‖g_total‖` 的分布，以及去掉 20 维后
`‖g_real‖` 是否仍触发裁剪。它在 LIBERO 上已经有旁证但不可直接搬用：LIBERO 的
`grad_norm` 在 step 1–194 期间稳定在 **0.26–0.27**，远低于 1.0；但那是 7 维 delta EEF、
horizon 25、`discrete_state_input=false` 的另一个任务，**不能据此推断 Scene1**。

## 5. mask 对照（仅隔离诊断，不改生产）

若第 4 节的判据显示裁剪确实被触发，才做 12 维 mask 与 32 维的 A/B。判据固定为：
在相同 batch/rng 下比较**前 200 步的 12 维 loss 轨迹**，而不是比较 32 维 loss——后者在两种
目标下不可比。这属于训练方法消融，必须先获得授权，本审计不执行。

## 6. 交付物状态

| 文件 | 状态 |
|---|---|
| `per_dimension_loss.json` | **`BLOCKED`**（schema only） |
| `gradient_contribution.json` | **`BLOCKED`**（schema only） |
| `padding_gradient_audit.md` | 本文件 |

补齐命令：

```bash
python scene1_libero_audit/tools/padding_grad.py \
  --checkpoint .../formal/checkpoints --step 2000 \
  --train-view .../data_view_v3 \
  --batch 8 --seed 42 --capture-identity-check \
  --out scene1_libero_audit/gradient_contribution.measured.json
```
