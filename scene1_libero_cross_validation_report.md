# Scene1 π0.5 故障排查与 LIBERO 交叉验证报告

Run `scene1-pi05base-val50-20261029T184107Z-92c99192` · ckpt step 2000 · 2026-10-30
Commit `b0d869c19170cecfcc8ad5739a77d6885220f07b`（读自 `.git/refs/heads/main`；工作区是否干净 `BLOCKED`）

> **执行环境**：本机第三轮仍然无法 `fork(2)`（`fork: EAGAIN`，MemFree 0.71 GB，
> Committed_AS 69.7 GB vs CommitLimit 20.4 GB）。因此**所有需要起进程的测量都标 `BLOCKED`**，
> 没有一项被写成"通过"。四个 GPU 由正在运行的 LIBERO 全量占用（jax fraction 0.7119，
> 需求 17.1 GB），这也使任何 Scene1 GPU 诊断在物理上不可行——与任务"先做低成本静态检查"
> 的要求一致。**LIBERO 未被停止、重启、修改或连接。**

---

## 0. 一句话结论

Norm Stats、输入管线顺序、动作时间对齐、图像路径**都没有发现可复现异常**；Padding 与 EMA 的
数值判定仍 `BLOCKED`。唯一被证实的新缺陷是**离散状态分词把 `< −1` 的归一化值编码成字面量
`-1`**（`VERIFIED` 机制，影响 `INCONCLUSIVE`）。而这一问题恰好落在
`discrete_state_input=true` 的 Scene1 独有路径上——**正在跑的 LIBERO 因为
`discrete_state_input=false`，完全覆盖不到它**。

## 1. 判定汇总

| 编号 | 检查 | 判定 | 一句话 |
|---|---|---|---|
| P0-a | 本机/容器/GPU/进程 | `BLOCKED` | fork 失败，无法枚举进程；GPU 由 LIBERO 占用 |
| P0-b | Scene1 是否被重新拉起 | `BLOCKED`（维持 `AUTO_RESTART_UNRESOLVED`） | 工件显示曾多次 resume（`-resume-2000b`、2002→2050 快照），无 `done_resume.json` |
| P0-c | LIBERO 身份 | `VERIFIED` | 从 `pi05_base` 从头微调，**不是** `pi05_libero` |
| P1 | Norm Stats | `VERIFIED` 正常；一处 `INCONCLUSIVE` | 无泄漏、无退化维、无 runaway；分词器 −1 缺陷在下游 |
| P2 | 输入管线 | `VERIFIED` 顺序/对齐/图像；`BLOCKED` token 级 | 并**撤回**上一轮"20 个零状态 token"的错误结论 |
| P3 | Padding loss/梯度 | `BLOCKED` | 给出可用官方前向的替代测法 + 唯一存活机理（裁剪） |
| P4-L0 | 官方 pi05_libero 基线 | `BLOCKED` | 无官方权重，且无空闲 GPU；不建议为此下载 |
| P4-L1 | 当前 LIBERO 训练 | 正常（194 步 `VERIFIED`） | 只证公共基础设施，不证 Scene1 |
| P4-L2 | 官方源码隔离对照 | 未触发 | 已核关键文件与上游语义一致 |
| P5 | EMA | `BLOCKED`（"EMA 损坏"已 `RULED_OUT`） | 树健康，能跑出 1 m 定向运动；差异大小未测 |

## 2. P0 源码身份

| 项 | 值 | 来源 |
|---|---|---|
| 容器镜像 | `sha256:405af52051682d5e13169d1de1523fd0adc116eceae4fc314cadfacd45b1ad52` | LIBERO `config_initial.json:40`；与 Scene1 state-read 探针同镜像 |
| Scene1 base 树 | `9909d717ec51c078a3d5fd3464cb7612ae7b6a0a713121190e9455420b2350cd` | `norm_audit_v4/action_norm_audit.json:3` |
| LIBERO base 树 | 同上（完全相同） | `config_initial.json:7` |
| Scene1 训练入口 | `/opt/openpi/scripts/train.py`，**未入仓** | `train_so101.py:31-42` |
| Scene1 推理源码哈希 | `6def3aad58f449a8472e62addb7d1dfc67e5db7961b512b65a258a02d81dccd6` | 审计文件 `transform.inference_source_sha256` |
| LIBERO openpi 源码树 | `6bb8d9f5babb8ab3b655a5a320ea75bb7380851418d8e0cb22f4da610343d7ad`（56 文件） | `config_initial.json:68-72` |

**溯源缺口（`VERIFIED`，非本次故障原因）**：Scene1 没有像 LIBERO 那样记录
`openpi_src`/`openpi_scripts` 的树哈希，只有推理侧源码哈希；两者哈希口径不同，**不可直接比较**。
`gemma_300m_lora` vs `gemma_300m` 这个具体风险点，本轮结论是**没有混用**：运行时配置为
`action_expert_variant = "gemma_300m"`，冻结规则显式排除 `.*lora.*` 与 `.*llm.*_1.*`
（`config_resume.json:74-75`）。历史快照里的 `make_model_config` 只是上游工厂函数同时支持两者，
不代表实际选了 lora 版。

## 3. P1 Norm Stats —— 唯一新增的实质发现

详见 `scene1_libero_audit/norm_stats_per_dim.json` 与 `norm_distribution.md`。

通过的项：train-only（`recomputed_from_train_only_matches_view = true`）、无验证泄漏
（`validation_in_norm_stats = false`，并记录了"若含验证集会偏 3.2°–4.8°"）、无近常量维、
无 NaN、往返误差 ≤ 2.1e-14、推理端顺序 `Unnormalize → SO101Outputs` 已核。

**没有 runaway**：由声明过的 min/max 反算，训练集归一化值最大只到 **1.32**，
`|x|>2/5/10` 的比例为 **0**。所以"Norm 数值爆掉导致训练崩"这一支可以划掉。

两处尺度问题（都不足以解释 180° 方向错误）：

1. **左 4 关节 q01–q99 跨度 113.5°，右 10 关节 93.4°，相差 22 %**；其余 5 对差异 < 5 %。
   左臂运动范围更大、均值更高，是一个可被"不读颜色"的策略利用的先验。
2. **夹爪 state 与 action 不同尺**：action 0 % → −1.000，state 0 % → −1.013 / −1.006。

**新发现（`VERIFIED` 机制）**：`tokenizer.py:26`

```python
discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
```

`x < −1` 时 `digitize` 返回 0，减 1 得到 **−1**，于是 prompt 里出现字面量 `-1` 而不是 bin 编号。
分位数归一化恰好把 q01 映到 −1.0，因此**每一维约 1 % 的训练值落进 `-1`**；夹爪 state 最小值
0.697 也落在 −1。闭环越出分布时该比例只会更高。

影响边界：不导致截断（prompt ≪ `max_token_len=200`）；每帧至少一维命中的比例 2 %–10 %
（`BLOCKED`，需数据）。**它是上游 openpi 的既有行为，不是本仓库引入的 bug**；
把它当根因不成立，把它当"完全无害"同样没有证据。

## 4. P2 输入管线 —— 并撤回上一轮一处错误

顺序由运行自身记录核对：`SO101Inputs → Normalize → InjectDefaultPrompt → ResizeImages →
TokenizePrompt → PadStatesAndActions`，与 `data_loader.py:186-189` 一致。Normalize 在 Pad 之前。

**更正**：上一轮报告称"20 个 padding 零会变成 20 个状态 token"，**这是错的**。
`TokenizePrompt` 在 Pad 之前执行，只看到 12 维状态；而 `pi0.py:106-137` 的 `embed_prefix`
只读图像 + `tokenized_prompt`，`pi0.py:151` 的 `state_proj` 仅在 `if not self.pi05` 时执行。
Scene1 是 `pi05=true`，所以**32 维连续状态在整个前向里从未被读取**。state 的 padding 是死代码。

其余 `VERIFIED`：前 12 维与标签一致（`pad_first_12_max_abs_error = 0.0`）、尾 20 维恒零
（`padded_tail_max_abs = 0.0`）、窗口 `[t, t+10)` 与 `state[t]` 同源、图像
`raw_jpg_equals_loader_frame = true`、三路 mask 全 True、闭环首帧 MAE ≤ 0.99 px。

`BLOCKED`：逐帧 token id、解码文本、红蓝序列差异位置、resize 前后对照图。

一个必须承认的任务事实：指令**只用颜色暗示手臂**，没有任何"用左/右臂"的字面表述。
一旦颜色绑定没学会，模型没有第二条线索——这让"条件没学会"与"任务对模型过难"在当前数据下
不可区分。

## 5. P3 Padding —— 机理收窄到一个可判定量

不采用上一轮那份手抄 `compute_loss` 的实现（本轮明确要求不信任未执行的复制）。替代方案：
包一层 `model.action_out_proj` 捕获 `v_t`，**其余全走官方前向**，再用同一 rng 重建
`u_t = noise − actions`，最后只做 `square` 与三种 `mean`。可出错面从 25 行降到 3 行，
且能用"重算 `L32` 与官方返回值差 < 1e-6"自校验。

在排除了"padding 目标是噪声"和"AdamW 在乎 loss 常数缩放"之后，唯一还能让 padding 真正伤害
12 维学习的通道是**全局梯度裁剪**：`g_total = (12·g_real + 20·g_pad)/32`，只有当
`‖g_total‖ > 1.0` 时，`g_pad` 才会通过撑大范数间接压缩 `g_real` 的有效步长。
判据只有两个数：`‖g_total‖` 分布、去掉 20 维后是否仍触发裁剪。

旁证（不可搬用）：LIBERO 的 `grad_norm` 稳定在 **0.26–0.27 ≪ 1.0**，裁剪未触发。
但那是 7 维 delta EEF、horizon 25 的另一个任务，只能算 `LIKELY 削弱`，不能判定 Scene1。

## 6. P4 LIBERO

身份见 `libero_experiment_identity.json`，审计见 `libero_baseline_audit.md`。

- **不是官方 `pi05_libero` 基线**：从 `pi05_base`（同一 base 树哈希）从头微调
  `libero_goal` task 8，`start_train_state_step = 0`。
- **`discrete_state_input = false`**（Scene1 为 true）——这是最关键的一条：LIBERO 走连续
  `state_proj`，Scene1 走 prompt 字符串。**LIBERO 正常不能为 Scene1 的状态路径背书。**
- 动作 7 维 delta EEF / horizon 25 / 2 相机 / 10 fps，与 Scene1 完全不同，loss 不可比。
- L1 只读状态：RUNNING，194 步，3.0 s/步，loss 0.0918→0.0788，`param_norm` 1803.157 稳定。
  首个 checkpoint 在 step 3000（约 2.5 小时后），此前无法评估 EMA/闭环/验证。
- L0 `BLOCKED`：无官方权重，且 4 GPU 已满。**不建议为此下载官方权重**——它验证的是"推理
  链路"，而当前更便宜、更对症的替代是 Scene1 的 prompt 交换探针。
- L2 未触发，**不需要**重新下载 OpenPI 官方源码。

一个正向细节：`result.json:53-62` 记录了两次失败启动（argparse、镜像 digest 校验）并注明
"未创建 checkpoint"——失败尝试没有被伪装成训练。

## 7. P5 EMA（最小验证）

`BLOCKED`，未测。已知边界不变：validation 绑 `TrainState.params`
（`scene1_validation.py:166`，记录自述 `weights_source`），推理加载 `params/` = EMA
（`checkpoints.py:145-152` + `policy_config.py:57`）。**"EMA 损坏/零初始化"已 `RULED_OUT`**：
蓝色回合里部署策略仍跑出 1 m 定向运动并撞上红盘，坏权重做不到。差异大小必须在同一 batch、
同一 rng 下比 12 维 loss，未做即不升级。

## 8. 证据矩阵（沿用任务的判定表）

| 检查结果 | 本轮是否成立 | 优先方向 |
|---|---|---|
| Norm 出现异常尺度/极端分布 | **否**（max \|x_norm\| = 1.32，无 >2） | 排除 |
| 输入 Token/Mask/Action 时间对齐错误 | **否**（顺序、窗口、首帧均核）；但存在 −1 词元缺陷 | 记录，暂不作为根因 |
| Padding 梯度明显冲突并改变有效更新 | `BLOCKED` | 需一个 GPU 小时判定 |
| 官方 LIBERO 基线在本环境成功 | **未测**（无官方权重） | 不建议为此下载 |
| 官方基线失败、隔离环境成功 | 未触发 | 不需要官方源码对照 |
| **LIBERO 训练正常而 Scene1 失败** | **当前一致成立**（194 步正常） | **优先查 SO101 适配与任务条件学习** |
| EMA 与 Current 显著差异 | `BLOCKED` | 顺带在探针里测 |
| 以上均无异常 | 无法排除 | 训练不足 / 专家覆盖 / 闭环分布偏移 |

## 9. 建议的下一步（按成本排序，均需先解封）

| # | 动作 | 成本 | 判据 |
|---|---|---|---|
| 1 | `restart_audit.sh` ×2，确认 Scene1 真停、step 2000 不被覆写 | 分钟 | `AUTO_RESTART_RESOLVED` 两次 |
| 2 | **prompt 交换探针**：固定观测/噪声，只换红/蓝指令，两棵树各跑 8 场景×3 种子 | ~1 GPU·h | 手臂选择是否随指令翻转。翻转⇒条件已学，问题在执行链；不翻转⇒条件未学 |
| 3 | 12 维 / 20 维 loss 拆分 + 裁剪判据（`‖g_total‖` vs 1.0） | ~1 GPU·h | 裁剪未触发⇒padding 只剩"指标口径"问题 |
| 4 | 统计 `-1` 词元出现率（训练集/闭环） | ~0.2 GPU·h | 若闭环显著高于训练，需重新考虑状态编码 |
| 5 | LIBERO step 3000 落地后，只读跑一次闭环，验证推理链路 | ~0.5 GPU·h | 通过⇒公共链路可信，火力集中在 SO101 适配 |

**不建议**：恢复 Scene1 正式训练（条件 1 未满足，且条件未学的证据已经存在）；下载官方
`pi05_libero` 权重；现在改 padding mask、改分词器或改 norm stats。

## 10. 明确回答三个是非题

1. **是否需要修复生产代码？**
   不需要为"训练正确性"改任何东西。`-1` 词元是上游行为，若要改属于**输入表示变更**，必须走
   受控实验（改前先测出现率），不能静默打补丁。
2. **是否有充分证据恢复 Scene1 训练？**
   **没有。** 三条独立理由：训练是否真停仍未证（`AUTO_RESTART_UNRESOLVED`）；颜色→手臂
   条件在 step 2000 已被证未学会（两集蓝色闭环）；padding/EMA 的定量判定未做。
3. **是否真的需要重新下载 OpenPI 官方源码？**
   **不需要。** L0 未执行（不是失败），L2 的触发条件不成立；已核对的
   `transforms.py`/`pi0.py` 关键路径与上游语义一致。

---

### 交付清单

| 文件 | 状态 |
|---|---|
| `scene1_libero_cross_validation_report.md` | 本文件 |
| `scene1_libero_audit/norm_stats_per_dim.json` | 部分（数值 VERIFIED，分布字段 BLOCKED） |
| `scene1_libero_audit/norm_distribution.md` | 完成 |
| `scene1_libero_audit/input_pipeline_audit.md` | 完成（含一处撤回） |
| `scene1_libero_audit/padding_gradient_audit.md` | 完成（数值 BLOCKED） |
| `scene1_libero_audit/libero_experiment_identity.json` | 完成 |
| `scene1_libero_audit/libero_baseline_audit.md` | 完成 |
| `scene1_libero_audit/model_input_trace.json` | **BLOCKED**（schema only） |
| `scene1_libero_audit/per_dimension_loss.json` | **BLOCKED**（schema only） |
| `scene1_libero_audit/gradient_contribution.json` | **BLOCKED**（schema only） |

### 局限

- 无一项模型前向被执行；所有"逐维/逐阶段/梯度/EMA"数字缺失。
- Norm 的分布统计只有分位数轮廓，没有直方图，也没有红/蓝、活动/非活动臂分区。
- 六集闭环仍不是同质实验（`rollout_six` vs `rollout_six_v2`，chunk 复用 9.9 vs 1.67 帧/请求），
  不可汇总为成功率。
- `/opt/openpi` 未入仓，Scene1 的训练循环本体无法逐行审计。

报告到此为止，未进入修复或训练阶段。
