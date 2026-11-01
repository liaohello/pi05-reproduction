# SO101 单方块 pilot

## 布局/IK pilot

200 个种子（15050–15249），全部由认证快照与逐场景检查生成：接受 199，拒绝 1；首 draw 接受 83/200（41.7%），总 draw 916；拒绝原因：tray_keepout 417、place_checks 222、pick_region 72、hover_ik 6。接受 yaw 0.009–71.073°，30–60° 76 个；托盘位移最大 0.039691 m，>0.01 m 178 个、>0.02 m 109 个、>0.04 m 0 个；全部 scene SHA 唯一；未执行专家、未保存 episode。原始记录：[layout_pilot](../runs/so101/scene1_layout_pilot_final_20261029/layout_pilot.json)。

## 物理 pilot

8 次尝试（目标 8），8 成功；红 4、蓝 4；yaw 16.07–66.67°，30–60° 3 次；托盘位移 >0.01 m 6 次；严格接触监控错误 0、soft contact 0；总 wall 174.4 s。原始记录：[attempts](../runs/so101/scene1_physical_pilot_final_20261029/attempts.jsonl)。

结论：场景与专家链路可用；正式采集上限冻结为 128 次物理尝试。pilot 不进入数据集。修正项见[协议](so101-scene1-protocol.md)。

工程注记：首次物理 pilot 因本机 EGL 设备号配置错误（设备 2 不可见）未执行任何 episode，记录保留在 `runs/so101/scene1_physical_pilot_20261029`，未计入上述 8 次。
