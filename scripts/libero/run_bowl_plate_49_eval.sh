#!/usr/bin/env bash
# Run LIBERO Goal task 8 closed-loop episodes against an all-49 LoRA checkpoint.
set -euo pipefail

ROOT="${PI05_REMOTE_ROOT:?set PI05_REMOTE_ROOT}"
TRAIN_RUN="${PI05_TRAIN_RUN:?set PI05_TRAIN_RUN}"
VIEW="${PI05_VIEW_DIR:?set PI05_VIEW_DIR}"
TOKENIZER="${PI05_TOKENIZER:?set PI05_TOKENIZER}"
STEP="${PI05_STEP:-6000}"
EPISODES="${PI05_EVAL_EPISODES:-10}"
SEED="${PI05_EVAL_SEED:-7}"
REPLAN="${PI05_EVAL_REPLAN:-10}"
SERVER_GPU="${PI05_SERVER_GPU:?set PI05_SERVER_GPU}"
CLIENT_GPU="${PI05_CLIENT_GPU:?set PI05_CLIENT_GPU}"
SERVER_IMAGE="${PI05_SERVER_IMAGE:-openpi-server:latest}"
CLIENT_IMAGE="${PI05_CLIENT_IMAGE:-openpi-client:latest}"
PROTOCOL_SHA256="${PI05_PROTOCOL_SHA256:?set PI05_PROTOCOL_SHA256}"
SERVER_MEM_FRACTION="${PI05_SERVER_MEM_FRACTION:-0.45}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
NET="libero-bowl-plate-eval-${STAMP,,}"
SERVER="libero-bowl-plate-server-${STAMP,,}"
OUT="${TRAIN_RUN}/evaluation_${STEP}"
mkdir -p "$OUT/logs" "$OUT/home" "$OUT/cache/numba" "$OUT/cache/big_vision"
cp "$TOKENIZER" "$OUT/cache/big_vision/paligemma_tokenizer.model"
docker image inspect --format '{{.Id}}' "$SERVER_IMAGE" "$CLIENT_IMAGE" \
  >"$OUT/logs/image-ids.txt"
docker image inspect --format '{{index .RepoDigests 0}}' "$SERVER_IMAGE" "$CLIENT_IMAGE" \
  >>"$OUT/logs/image-ids.txt"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader >"$OUT/logs/gpu-before.csv"

cleanup() {
  docker rm -f "$SERVER" >/dev/null 2>&1 || true
  docker network rm "$NET" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker network create --internal --label "pi05.repro=1" "$NET" >/dev/null
CHECKPOINT="/run/train/checkpoints/pi05_libero_bowl_plate_49_lora/bowl_plate_49_lora/${STEP}"
OUT_CONTAINER="/run/evaluation_${STEP}"
docker run -d --name "$SERVER" --label "pi05.repro=1" \
  --network "$NET" --gpus "\"device=${SERVER_GPU}\"" --shm-size=16g \
  -v "${ROOT}:/work:ro" -v "${TRAIN_RUN}:/run" -v "${VIEW}:/view:ro" \
  -v "${OUT}/cache:/tokenizer_cache" \
  -e OPENPI_DATA_HOME=/tokenizer_cache -e JAX_PLATFORMS=cuda \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -e XLA_PYTHON_CLIENT_MEM_FRACTION="$SERVER_MEM_FRACTION" -e PYTHONUNBUFFERED=1 \
  -e PI05_SERVER_LOG_DIR="${OUT_CONTAINER}/logs" \
  --entrypoint bash "$SERVER_IMAGE" \
  /run/setup/run_policy_server.sh python3 /run/setup/serve_bowl_plate_49.py \
  --protocol /run/setup/bowl_plate_49_lora.json \
  --expected-protocol-sha256 "$PROTOCOL_SHA256" \
  --view /view --checkpoint "$CHECKPOINT" --expected-step "$STEP" >/dev/null

ready=0
for i in $(seq 1 240); do
  if grep -q "POLICY_SERVER_READY" "$OUT/logs/server.stdout.log" 2>/dev/null \
    || docker logs "$SERVER" 2>&1 | grep -q "POLICY_SERVER_READY"; then
    ready=1
    break
  fi
  status="$(docker inspect -f '{{.State.Status}}' "$SERVER" 2>/dev/null || echo gone)"
  if [[ "$status" == "exited" || "$status" == "dead" || "$status" == "gone" ]]; then
    break
  fi
  # A loaded host can spend a minute in "created" before the GPU runtime starts it.
  if [[ "$status" == "created" && $i -gt 90 ]]; then
    break
  fi
  sleep 2
done
docker inspect "$SERVER" >"$OUT/logs/server-inspect.json" 2>/dev/null || true
docker logs "$SERVER" >"$OUT/logs/policy-server.log" 2>&1 || true
if [[ "$ready" != "1" ]]; then
  echo "policy server failed to become ready" >&2
  tail -40 "$OUT/logs/policy-server.log" >&2
  exit 1
fi
grep -m1 "POLICY_SERVER_READY" "$OUT/logs/server.stdout.log" \
  >"$OUT/logs/readiness.log" 2>/dev/null \
  || docker logs "$SERVER" 2>&1 | grep -m1 "POLICY_SERVER_READY" \
  >"$OUT/logs/readiness.log" || true
python3 "$(cd "$(dirname "$0")" && pwd)/check_server_metadata.py" \
  --readiness "$OUT/logs/readiness.log" \
  --protocol "$(cd "$(dirname "$0")" && pwd)/bowl_plate_49_lora.json" \
  --expected-step "$STEP" | tee "$OUT/logs/server-metadata-check.log"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader >"$OUT/logs/gpu-server.csv"

set +e
docker run --rm --name "libero-bowl-plate-client-${STAMP,,}" \
  --label "pi05.repro=1" --network "$NET" --gpus "\"device=${CLIENT_GPU}\"" \
  --shm-size=4g -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl -e PYNPUT_BACKEND=dummy \
  -e HOME=/workspace/home -e NUMBA_CACHE_DIR=/workspace/cache/numba \
  -v "${TRAIN_RUN}:/workspace" --entrypoint python "$CLIENT_IMAGE" \
  /workspace/eval_code/libero_episode_runner.py \
  --host "$SERVER" --port 8000 --suite libero_goal --task-ids 8 \
  --episodes "$EPISODES" --seed "$SEED" --replan-steps "$REPLAN" \
  --output "/workspace/evaluation_${STEP}"
client_rc=$?
set -e

docker logs "$SERVER" >"$OUT/logs/policy-server-final.log" 2>&1 || true
docker rm -f "$SERVER" >/dev/null 2>&1 || true
docker network rm "$NET" >/dev/null 2>&1 || true
trap - EXIT

if [[ $client_rc -ne 0 ]]; then
  echo "LIBERO client failed with exit code $client_rc" >&2
  exit "$client_rc"
fi
test -f "$OUT/evaluation.json"
echo "EVALUATION_JSON=$OUT/evaluation.json"
