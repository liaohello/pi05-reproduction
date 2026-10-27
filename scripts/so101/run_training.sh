#!/usr/bin/env bash
# E2 launcher: run one OpenPI training process in the pinned image, offline.
#
# Required environment:
#   SO101_REPO                repository root (auto-detected from this script)
#   SO101_RUN_DIR             E2 run directory, inside the repository
#   SO101_TRAIN_VIEW          exported E1 training view, inside the repository
#   SO101_PI05_CHECKPOINT     host path of the pi05_libero checkpoint directory
#   SO101_TOKENIZER           host path of paligemma_tokenizer.model
# Optional:
#   SO101_MODE=initial|resume (default initial)
#   SO101_GPU_INDEX=2
#   SO101_EXP_NAME=e2_lora_b1_s42
#   SO101_IMAGE=openpi-server:latest
set -euo pipefail

repo="${SO101_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
: "${SO101_RUN_DIR:?set SO101_RUN_DIR}"
: "${SO101_TRAIN_VIEW:?set SO101_TRAIN_VIEW}"
: "${SO101_PI05_CHECKPOINT:?set SO101_PI05_CHECKPOINT}"
: "${SO101_TOKENIZER:?set SO101_TOKENIZER}"
mode="${SO101_MODE:-initial}"
gpu="${SO101_GPU_INDEX:-2}"
exp_name="${SO101_EXP_NAME:-e2_lora_b1_s42}"
image="${SO101_IMAGE:-openpi-server:latest}"
state_mode="${SO101_STATE_MODE:-off}"
train_episodes="${SO101_TRAIN_EPISODES:-}"
extra=()
[[ -n "${SO101_TRAIN_STEPS:-}" ]] && extra+=(--steps "$SO101_TRAIN_STEPS")
[[ -n "${SO101_SAVE_INTERVAL:-}" ]] && extra+=(--save-interval "$SO101_SAVE_INTERVAL")

case "$mode" in initial|resume|smoke) ;; *) echo "SO101_MODE must be initial, resume or smoke" >&2; exit 2;; esac
mkdir -p "$SO101_RUN_DIR" "$SO101_RUN_DIR/home" "$SO101_RUN_DIR/jax_cache" \
         "$SO101_RUN_DIR/openpi_cache/big_vision"
if [[ ! -s "$SO101_RUN_DIR/openpi_cache/big_vision/paligemma_tokenizer.model" ]]; then
  cp "$SO101_TOKENIZER" "$SO101_RUN_DIR/openpi_cache/big_vision/paligemma_tokenizer.model"
fi
tokenizer_sha=$(sha256sum "$SO101_RUN_DIR/openpi_cache/big_vision/paligemma_tokenizer.model" | cut -d' ' -f1)

rel_run="${SO101_RUN_DIR#"$repo"/}"
rel_view="${SO101_TRAIN_VIEW#"$repo"/}"
container_run="/work/$rel_run"
container_view="/work/$rel_view"
# JAX preallocation is computed from the memory actually free on the chosen card, with a margin,
# capped below the whole card; it is never a hard-coded fraction of the total.
margin_mib="${SO101_MEMORY_MARGIN_MIB:-1500}"
read -r free_mib total_mib <<<"$(nvidia-smi --query-gpu=memory.free,memory.total \
  --index "$gpu" --format=csv,noheader,nounits | tr ',' ' ')"
fraction=$(python3 -c "
import sys
free, total, margin, index = (float(v) for v in sys.argv[1:])
if free <= margin:
    raise SystemExit(f'GPU {int(index)} has only {free:.0f} MiB free, which is not more than the '
                     f'{margin:.0f} MiB margin')
print(f'{max(0.10, min(0.80, (free - margin) / total)):.6f}')" \
  "$free_mib" "$total_mib" "$margin_mib" "$gpu")
echo "gpu $gpu free=$free_mib/$total_mib MiB -> XLA_PYTHON_CLIENT_MEM_FRACTION=$fraction (margin ${margin_mib} MiB)"
if [[ "$rel_run" == "$SO101_RUN_DIR" || "$rel_view" == "$SO101_TRAIN_VIEW" ]]; then
  echo "SO101_RUN_DIR and SO101_TRAIN_VIEW must be inside SO101_REPO" >&2
  exit 2
fi

log="$SO101_RUN_DIR/train_${mode}.log"
gpu_log="$SO101_RUN_DIR/gpu_${mode}.txt"
: > "$log"; : > "$gpu_log"
(
  while true; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null \
      | tr -d ' ' >> "$gpu_log" || true
    sleep 1
  done
) &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT

started=$(date +%s.%N)
set +e
docker run --rm \
  --gpus "device=$gpu" \
  --network none \
  --entrypoint python3 \
  -e HOME=/work/"$rel_run"/home \
  -e OPENPI_DATA_HOME=/work/"$rel_run"/openpi_cache \
  -e SO101_TRAIN_VIEW="$container_view" \
  -e PYTHONPATH=/work/scripts/so101 \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -e XLA_PYTHON_CLIENT_MEM_FRACTION="$fraction" \
  -e JAX_PLATFORMS=cuda \
  -v "$repo:/work" \
  -v "$SO101_PI05_CHECKPOINT:/base:ro" \
  "$image" \
  -u /work/scripts/so101/train_so101.py \
    --mode "$mode" \
    --run-root "$container_run" \
    --train-view "$container_view" \
    --base-checkpoint /base \
    --exp-name "$exp_name" \
    --state-mode "$state_mode" \
    --train-episodes "$train_episodes" \
    "${extra[@]}" \
  2>&1 | python3 -u -c 'import sys,time
for line in sys.stdin:
    sys.stdout.write(f"{time.time():.3f} {line}"); sys.stdout.flush()' \
  | tee "$log"
status=${PIPESTATUS[0]}
set -e
kill "$monitor" 2>/dev/null || true
trap - EXIT
docker run --rm --network none --entrypoint bash -v "$repo:/work" "$image" \
  -c "chmod -R a+rwX /work/$rel_run" >/dev/null 2>&1 || true
finished=$(date +%s.%N)
peak=$(sort -n "$gpu_log" 2>/dev/null | tail -1 || true)
python3 - "$SO101_RUN_DIR" "$mode" "$status" "$started" "$finished" "$peak" "$gpu" "$tokenizer_sha" <<'PY'
import json, sys
from pathlib import Path
run, mode, status, started, finished, peak, gpu, tokenizer_sha = sys.argv[1:]
payload = {"mode": mode, "exit_code": int(status), "started_unix": float(started),
           "finished_unix": float(finished), "wall_time_s": float(finished) - float(started),
           "gpu_index": int(gpu), "gpu_peak_mib_sampled": int(peak) if peak else None,
           "tokenizer_sha256": tokenizer_sha,
           "log": f"train_{mode}.log", "gpu_log": f"gpu_{mode}.txt"}
Path(run, f"run_{mode}.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
exit "$status"
