# 训练输入端到端审计（P2）

**结论：管线顺序、维度、时间对齐、图像路径均 `VERIFIED` 无误；但本轮必须更正上一轮报告
中的一处错误结论。真正的输入侧异常只有一处——离散状态分词（见 `norm_distribution.md §3`），
其影响仍是 `INCONCLUSIVE`。**

---

## 1. 顺序：用运行自己的记录核对，而不是用代码猜

运行产出的 `action_norm_audit.json → transform.train_input_order` 记录为：

```
["SO101Inputs", "Normalize", "InjectDefaultPrompt", "ResizeImages", "TokenizePrompt", "PadStatesAndActions"]
```

与 `openpi/training/data_loader.py:186-189` 的构造顺序（repack → data_transforms → Normalize →
model_transforms）逐项一致。**Normalize 在 Pad 之前**，这是后面一切结论的前提。

## 2. State：上一轮的一个结论是错的，这里撤回

上一轮报告说"20 个 padding 零会变成 20 个状态 token"。**这是错的。** 证据链如下：

1. `TokenizePrompt` 在 `PadStatesAndActions` **之前**执行（见上），所以它拿到的是 **12 维**状态。
2. `tokenizer.py:26` 把 12 维状态离散成 12 个整数，拼进 prompt 字符串。
3. `pi0.py:106-137 embed_prefix` 只拼接 **图像 token + `obs.tokenized_prompt`**，完全没有读 `obs.state`。
4. `pi0.py:151` `if not self.pi05:` 才做 `state_proj(obs.state)`。Scene1 是 `pi05=True`，所以
   **连续状态张量在整个前向里从未被读取**。

因此：

| 项 | 实际值 | 判定 |
|---|---|---|
| token 化时的状态维度 | 12 | `VERIFIED` |
| prompt 里的状态条目数 | 12 个整数 | `VERIFIED`（代码路径） |
| 32 维 padded state 是否进入模型 | **否，完全惰性** | `VERIFIED` |
| padding 对 state 路径的影响 | 0 | `VERIFIED` |

这条更正同时也说明：**"padding 32 维" 这件事只作用于 actions 和一段死代码里的 state**。
把 state padding 当作干扰源是站不住的。

## 3. Actions：前 12 维严格对应，后 20 维恒零（`VERIFIED`）

| 检查 | 结果 | 依据 |
|---|---|---|
| 原始动作 | 10×12 绝对关节目标 + 夹爪百分比 | `action_semantics` 未在本运行声明；单位表为 5×deg + percent ×2 |
| Normalize 后 | 12 维，分位数归一化 | 同上 |
| Pad 后 | 10×32，尾部零 | `padded_tail_max_abs = 0.0` |
| 前 12 维是否与标签一致 | 是，`pad_first_12_max_abs_error = 0.0` | 同上 |
| 是否偏移/重复/越界 | 未见；`action_sequence_keys = ("action",)` | `openpi_so101.py:111` |
| 时间关系 | `action[t]` 是控制帧 t 施加的目标，`state` 是控制前观测，窗口 `[t, t+10)` | 审计文件注记 |

`action_sequence_keys=("action",)` 且 `SO101EpisodeDataset.__getitem__` 返回
`arrays["action"][frame:frame+horizon]`（`openpi_so101.py:329`），窗口起点与 `state[frame]`
共用同一个 `frame`。**没有 t 与 t+1 错位**。

## 4. 图像：`VERIFIED`

| 检查 | 结果 |
|---|---|
| 原始内容 vs loader 读取 | `raw_jpg_equals_loader_frame = true`，`image_max_abs_error = 0.0` |
| 通道序 | `PIL.Image.convert("RGB")`，HWC，uint8（`openpi_so101.py:320-325`） |
| resize | `ResizeImages(224, 224)` |
| mask | 三路相机 `image_mask` 全 True（`openpi_so101.py:73-77`） |
| 闭环首帧一致性 | 六集闭环首帧审计通过：MAE ≈ 0.83–0.99 px，p95 ≤ 4 px |

## 5. 语言：路径正确，但每帧的实际 token 序列 `BLOCKED`

- 每集 prompt 来自 `episode.npz["task"]`，`task_per_episode_matches_scene = true`。
- `tokenizer.py:23` 会 `lower().strip().replace("_"," ")`，最终格式为
  `"Task: put the red block into the red tray., State: <12 ints>;\nAction: "`。
- 红/蓝两条指令在**字符层**只差 `red`/`blue` 两处。

**没有做到**：取出某一帧的 token id 列表、解码回文本、比对红蓝序列的差异位置。这需要
tokenizer 与数据，`BLOCKED`。

一个必须承认的设计事实：指令**只通过颜色暗示手臂**，没有任何"用左臂/右臂"的字面表述。
所以一旦颜色绑定没学会，模型没有任何别的线索可用——这与闭环观察一致，但也意味着
"条件学习失败"和"任务定义本身对模型过难"在当前数据下不可区分。

## 6. 交付物状态

| 文件 | 状态 |
|---|---|
| `model_input_trace.json` | **`BLOCKED`**（只有 schema，无数据） |
| `input_pipeline_audit.md` | 本文件 |
| `sample_visual_comparison/` | **`BLOCKED`**（无法渲染 resize 前后对照） |

补齐命令：

```bash
python scene1_libero_audit/tools/input_trace.py \
  --train-view runs/so101/scene1-pi05base-val50-20261029T184107Z-92c99192/data_view_v3 \
  --episodes 62,13,11,2 --frames 0,200,400 --colour-pair \
  --out scene1_libero_audit/model_input_trace.measured.json
```
