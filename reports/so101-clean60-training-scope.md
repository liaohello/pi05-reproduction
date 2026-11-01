# Clean60 参数范围状态

旧版固定镜像参数树记录了 61 个张量、872,771,344 个候选可训练参数；这是显式冻结规则
修改前的审计，不能直接当作本轮最终数量。旧报告的 40 步参数变化审计未通过，详见
[冻结验证状态](so101-clean60-freeze-verify.md)。

本轮冻结范围已明确为：Gemma 2B 原始语言权重冻结、注意力/MLP LoRA 可训练；Gemma 300M
动作专家非 LoRA 全量训练；SigLIP 全量训练；四个 action/time 投影层 kernel 与 bias 全量
训练。新参数树、LoRA rank/alpha、各组数量和 dtype，须在固定镜像中重跑后发布。
