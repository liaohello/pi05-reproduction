# Norm Stats 逐维审计

**结论：Norm Stats 本身没有发现足以解释失败的异常（`VERIFIED` 无泄漏、无量纲退化、无 runaway）；
但它的下游——离散状态分词——存在一个可复现的编码缺陷（`VERIFIED` 机制，影响 `INCONCLUSIVE`）。**

数据来源：`norm_audit_v4/action_norm_audit.json` 的 `statistics.recomputed_train_only`
（训练集 36,623 帧、52 集）。逐维数值见 `norm_stats_per_dim.json`。以下所有"归一化极值"
都是由声明过的 min/q01/q99/max 直接算出的算术结果，**不是直方图统计**。

---

## 1. 通过的检查（这些是真的通过了）

| 检查 | 结果 | 依据 |
|---|---|---|
| 是否 train-only | 是 | `recomputed_from_train_only_matches_view = true`；`recorded_train_only` 与 `recomputed_train_only` 的 mean/q01/q99/std 完全一致 |
| 验证集是否泄漏 | 否 | `validation_in_norm_stats = false`；同文件记录了"若把验证集算进去会偏移多少"：mean 1.28°、q01 3.19°、q99 4.84°，而实际统计里没有这个偏移 |
| 是否分位数归一化 | 是 | `normalisation_uses_quantiles = true`；`Normalize(..., use_quantiles=True)` |
| 是否有近常量维 | 无 | `near_constant_dims_std_lt_1e-6.actions/state = []` |
| 是否有 NaN/Inf | 无 | `all_values_finite = true` |
| state/action 维度顺序 | 一致 | 两者 units 均为 5×degree + percent + 5×degree + percent |
| 归一化后再反归一化 | 往返误差 ≤ 2.1e-14 | 45,031 样本全部往返 |
| 推理端顺序 | `Unnormalize → SO101Outputs` | 已对照 pinned source 校验 |

**没有发现 "分位数跨度极小的关节"**：12 维 action 的 q99−q01 落在 40.0–113.5 之间，
最小的是两个夹爪（40 percentage points，这是定义域本身），没有一维塌缩。

**没有发现"归一化后大量超出正常范围"**：

| 区间 | 训练集是否出现 |
|---|---|
| \|x_norm\| > 1 | 是，尾部约 1 %/维（分位数归一化的定义使然） |
| \|x_norm\| > 2 | **否**（由 min/max 反算，最大仅 1.32） |
| \|x_norm\| > 5 / 10 | **否** |

这一条很重要：分位数归一化把整份训练数据压进 `[-1.32, +1.24]`，所以 **Norm Stats 不可能
是"数值爆掉"型故障的原因**。若这里有病，症状会是训练 loss 的尖峰，而不是"总是往红盘走"。

## 2. 两个真实的尺度发现

**(a) 左右臂第 4/10 关节跨度和均值不对称（`VERIFIED`）**

| 关节对 | q01–q99 跨度 | 均值 | 判定 |
|---|---|---|---|
| 左 4 | 113.5° | +17.9° | — |
| 右 10 | 93.4° | +9.3° | 左比右宽 22 % |
| 其余 5 对 | — | — | 差异 < 5 % |

这不是 bug，但它是一个**可被模型利用的先验**：左臂在该关节上的运动范围更大、均值更高。
一个没有真正绑定颜色条件的模型，在"猜"的时候更容易输出左臂量级的动作。这与闭环观察
（蓝色任务仍走左臂）方向一致，属于 `LIKELY` 的次要放大因素，不是根因。

**(b) 夹爪的 state 与 action 不在同一把尺子上（`VERIFIED`，影响可忽略）**

| | 最小 | 最大 | 0 % 映射到 |
|---|---|---|---|
| action 夹爪 | 0.0 | 40.0 | −1.000 |
| state 夹爪（左） | 1.189 | 39.998 | −1.013 |
| state 夹爪（右） | 0.880 | 39.998 | −1.006 |

同一个物理开合度，在状态 token 和动作目标里相差约 0.02 个归一化单位。量级可忽略，
但它解释了为什么 state 的夹爪维会**恰好越过 −1**（见下）。

## 3. 真正的发现：离散状态分词的 −1 缺陷（`VERIFIED` 机制）

Scene1 用 `discrete_state_input=True`，状态不是张量而是**字符串**进 prompt：

```python
# openpi/models/tokenizer.py:26-28
discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
state_str = " ".join(map(str, discretized_state))
prefix = f"Task: {cleaned_text}, State: {state_str};\n"
```

`np.digitize` 对 `x < bins[0] = −1` 返回 **0**，减 1 之后得到 **−1**。于是：

- `x ≥ −1` → 合法 bin `0..255`
- `x < −1` → 词元 **`-1`**，一个两位字符的字符串，不是 bin 编号
- `x ≥ 0.9922` → 饱和到 `255`（高侧饱和是合法 bin）

而分位数归一化**恰好把 q01 映射到 −1.0**，所以每一维都有约 1 % 的训练值落在 q01 以下，
也就是落进 `-1`。夹爪维因为 state 的 q01 是 1.189 而不是 0，其最小值 0.697 同样落到 −1.013。

影响评估：

| 项 | 值 | 判定 |
|---|---|---|
| 单维落入 `-1` 的比例 | ≈1 %（q01 定义） | `VERIFIED`（算术） |
| 单帧至少一维落入 `-1` 的比例 | 2 %–10 %（取决于关节相关性） | **`BLOCKED`**（需跑数据） |
| 闭环时该比例是否升高 | 会升高（越出训练分布） | `LIKELY` |
| 是否改变 token 长度 | 是：`-1` 比 `123` 短、比 `7` 长 | `VERIFIED` |
| 是否导致截断 | 否：prompt 远小于 `max_token_len=200` | `VERIFIED`（量级判断） |
| 是否是失败主因 | 未知 | **`INCONCLUSIVE`** |

这是 upstream openpi 的既有行为，不是本仓库引入的 bug，而且它在所有 pi05 + 分位数归一化的
数据集上都会出现。把它当根因是不成立的；把它当"完全无害"同样没有证据。

**但它有一个确定的推论，直接影响本轮交叉验证的设计：**

> `discrete_state_input=False` 的 LIBERO 训练**根本不走这条路径**。
> 所以即使 LIBERO 训练/推理完全正常，也不能证明 Scene1 的状态编码路径是好的。

## 4. 仍然缺失的（`BLOCKED`）

| 任务要求 | 状态 | 原因 |
|---|---|---|
| 逐维直方图 / 归一化后分布 | `BLOCKED` | 需 45,031 帧数据 + Python 进程 |
| 按红/蓝、左/右臂、活动/非活动臂分组 | `BLOCKED` | 审计文件不含颜色/手臂分区；需回读 episode |
| 训练 vs 验证分布检验（KS/分位对比） | `BLOCKED` | 同上 |
| 闭环状态分布 | `BLOCKED` | 六集闭环未保存逐帧状态 |
| 夹爪百分比定义与归一化范围是否一致 | **已核**：35 % open / 0 % closed，归一化范围 0–40，一致 | `VERIFIED` |

补齐命令（本机无法执行）：

```bash
python scene1_libero_audit/tools/norm_probe.py \
  --train-view runs/so101/scene1-pi05base-val50-20261029T184107Z-92c99192/data_view_v3 \
  --out scene1_libero_audit/norm_train_vs_inference.measured.json
```
