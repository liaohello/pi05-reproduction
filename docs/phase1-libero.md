# Phase 1：LIBERO

本页是本地推理复跑入口。主机/镜像/checkpoint从环境或忽略的 `scripts/deploy/config.local.json` 提供，不能写进Git。资产来源见[PROVENANCE](../third_party/PROVENANCE.md)。

```sh
./scripts/run.sh preflight
./scripts/run.sh preflight-remote
./scripts/run.sh deploy-dry   # 审阅同步项
./scripts/run.sh deploy
./scripts/run.sh verify
./scripts/run.sh smoke --suite libero_spatial --tasks 0 --episodes 1
```

协议在 `configs/libero/protocol.json`；修改seed、replan、settle、图像或步数上限时记入run。结果通过 `scripts/track/record.py` 关联ledger，再看 `./scripts/run.sh status`。生成的status/dashboard不是手改证据。

P1.0规范结论：`4/4 smoke-test episodes succeeded; not a benchmark success-rate estimate.` P1.1完整2,000回合评估及P1.2训练复现未完成，M0仍 in progress。训练阅读路线见[reading guide](reading-guide.md)。公开视频、截图、评估JSON须先人工审阅；详见[AGENTS](../AGENTS.md)。
