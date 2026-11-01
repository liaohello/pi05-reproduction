# SO101 scene1 运行命令

`SO=${SO101_SOURCE_ROOT}`、`PY=${SO101_PYTHON}`、`CKPT=${PI05_CKPT}`、`TOK=${PI05_TOKENIZER}`；仓库根运行。

物理 pilot：`$PY scripts/so101/scene1_collect.py --root $SO --output runs/so101/scene1_physical_pilot_final_20261029 --pilot --pilot-attempts 12 --target 8`

采集（run-id `20261029T102500Z-scene1-single-random-v3`）：`$PY scripts/so101/scene1_collect.py --root $SO --output runs/so101/scene1_single_random_v1/<run>/collection --max-attempts 128`

视图：`$PY scripts/so101/scene1_export_view.py --collection <run>/collection --output runs/so101/scene1_view_v1 --camera-sync <camera> --workers 8`；norm 在 `openpi-server` 跑 `scene1_norm_audit.py`。

前置：`scene1_base_load_audit.py`、`parameter_tree.py`、`scene1_freeze_audit.py`、`tests/test_scene1_contracts.py`、`scan_content.py`。

训练未执行：`scene1_run_training.py --mode initial --steps 20000 --batch-size 8 --fsdp-devices 4 ...`（不占 GPU）。补丁：`patches/so101_scene1_single_random.patch`。
