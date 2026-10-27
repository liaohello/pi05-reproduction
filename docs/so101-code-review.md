# SO101 training code review

The SO101 implementation is available for source review. Its training entry point builds a
project-specific OpenPI configuration and data loader, then calls the external OpenPI trainer.
OpenPI and the dual-SO101 simulator remain external dependencies; their source, weights and
datasets are not included in this repository.

## Start here

| Responsibility | Source |
| --- | --- |
| Registered training parameters | [`configs/so101/e4t_training.json`](../configs/so101/e4t_training.json) |
| Configuration-driven launch | [`e4t_launch_training.py`](../scripts/so101/e4t_launch_training.py) |
| Container launch and GPU admission | [`e4t_run_training.py`](../scripts/so101/e4t_run_training.py) |
| Training entry, admission checks and configuration snapshot | [`train_so101.py`](../scripts/so101/train_so101.py) |
| Model, LoRA freeze filter, optimizer, schedule and data loader | [`openpi_so101.py`](../scripts/so101/openpi_so101.py) |
| Dataset export and train-only statistics | [`export_e4t_view.py`](../scripts/so101/export_e4t_view.py) |
| Open-loop prediction | [`evaluate_e4t_openloop.py`](../scripts/so101/evaluate_e4t_openloop.py) |
| Checkpoint selection and validation gate | [`summarize_e4t_openloop.py`](../scripts/so101/summarize_e4t_openloop.py) |
| Exact container ownership | [`e4t_resources.py`](../scripts/so101/e4t_resources.py) |
| Observation and action contract | [`SO101_CONTRACT.md`](../inference/SO101_CONTRACT.md) |

The data-collection, replay, diagnostic and plotting tools are under `scripts/so101/`.
SO101 experiment tracking is isolated from the LIBERO phase gates. The shared `scripts/run.sh`
still exposes the LIBERO deployment and tracking commands; SO101 uses the entries above.

## Host-side checks

These unit tests use synthetic fixtures and require NumPy and Pillow. They do not start training,
connect to a remote host, apply simulator actions or require JAX, model weights or a GPU.

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements-review.txt
.venv/bin/python -m unittest discover -s tests -v
bash scripts/security/scan.sh
```

Running training requires the external OpenPI runtime, a prepared dataset view, checkpoint and
tokenizer assets. Deployment asset locations are resolved from environment variables or ignored
local configuration. See [`PROVENANCE.md`](../third_party/PROVENANCE.md) and
[`THIRD_PARTY_LICENSES.md`](../THIRD_PARTY_LICENSES.md).

## Evidence review boundary

This source-review commit excludes the new experiment ledger entries, evaluation JSON, reports
and generated figures. Those materials remain local for human review before a separate evidence
commit. Their absence from this snapshot does not describe the local experiment history.
This commit does not complete the LIBERO training-reproduction or benchmark gates.
