#!/bin/bash
# Run one G1 oracle group on the test host.
#
# Paths come from the private environment; the reference source is reached only
# through PYTHONPATH so nothing is installed into the read-only tree.
set -u
: "${REFERENCE_ROOT:?}" "${ASSET_ROOT:?}" "${FINEART_JAX_ROOT:?}" "${REF_PYTHON:?}"
GROUP=${1:?group}
OUT=${2:?out dir}

export PYTHONPATH="$REFERENCE_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1

exec "$REF_PYTHON" "$FINEART_JAX_ROOT/tests/migration/oracle/export_g1_data.py" \
  --group "$GROUP" \
  --out "$OUT" \
  --text-tokenizer "$TEXT_TOKENIZER_DIR" \
  --fast-tokenizer "$ASSET_ROOT/tokenizers/fast" \
  --dataset-root "$FIT_DATASET_ROOT"
