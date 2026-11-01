# Clean60 容量实测（4 卡 FSDP）

`--mode probe`（XLA 预分配关闭）、4×RTX 4090、`fsdp_devices=4`、全局批 8、
state-on、1 次真实更新 + checkpoint 落盘，退出码 0。
原始数据：`runs/so101/clean60_runtime/run_probe_20261028T083051Z-clean60-4card-capacity.json`。

| 卡 | 他人常驻基线 | 峰值 | 本轮真实需求 |
|---|---|---|---|
| #0 | 6,642 MiB | 22,951 | **16,309 MiB** |
| #1 | 6,213 | 22,523 | **16,310** |
| #2 | 17 | 16,329 | **16,312** |
| #3 | 6,214 | 22,523 | **16,309** |

峰值由 init_train_state 的瞬时占用主导（底模先整份落到每卡再分片），
不是稳态分片量；异步 checkpoint 落盘耗时 52.6 s。

2 卡方案已被独立 probe 否证：同阶段每卡落地 7,418 MiB、池上限 18,494 MiB、
空闲卡物理峰值 17,050 MiB，forward_backward 还要 4,251,348,328 B（4.25 GiB）即
RESOURCE_EXHAUSTED。瓶颈同样是那个初始化瞬时量。

作废旧估计：10.5 GiB 只算了可训练部分；状态体量实为 14.43 GiB
（params 7.93 + Adam 双矩 6.50），另有冻结骨干 bf16 驻留 4.69 GiB。
此前 40 步 smoke 的 peak_sampled 是预分配预留量，不等于需求，已改以本 probe 为准。

旧准入数 16,312 + 1,500 MiB 已作废。新余量为 512 MiB，阈值至少 16,824 MiB/卡；
但真实参数更新和 checkpoint 缓冲尚未覆盖，必须重跑 probe 后才可准入正式训练。
