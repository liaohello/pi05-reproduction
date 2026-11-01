# SO101 单方块数据

64 成功：红 32、蓝 32；物理尝试 78（红 32/32，蓝 32/46），14 次物理失败全为蓝臂，2 次蓝场景布局拒绝。红成功 yaw 4.9–69.7°（中位 36.0），蓝成功 yaw 0.1–61.2°（中位 22.9）：蓝高 yaw 被系统性拒绝，数据分布不声称覆盖 [0,72) 全域。

失败分布：place_release 9 次（yaw 60.3–69.0，夹爪-托盘壁约 0.0005 m 边缘接触）、place_withdraw IK 2 次、carry IK 3 次（yaw 1.4/1.4/11.6，block x 0.27–0.28）。布局拒绝 2 次（400 draw 内 place_checks/tray_keepout 耗尽）。未放宽接触守卫或限位。

质量：45,031 帧，单集 657–775；Δt=0.020 s，substeps 全 10，incomplete=0；dataset 与 trace 的 state/action 最大误差 0；三相机同帧渲染通过，视频时长对齐误差 5.7e-14；每集任务与 SceneSpec 一致；contact monitor/soft contact 0。

Split：52/12，红蓝各 26/6，family 交叉 0；norm 仅用 52 集 36,623 帧，无常数维；view SHA `3c4911b5…`，norm SHA `c44567cd…`。无扰动、无视觉随机化。原始记录见 `attempts.jsonl`、`collection_manifest.json`。
