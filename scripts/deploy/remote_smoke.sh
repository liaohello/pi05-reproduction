#!/usr/bin/env bash
# 在服务器上跑一次 episode：起策略服务 → 跑 LIBERO 客户端 → 收结果 → 停服务。
# 只读资源池权重，不改官方源码；容器名带固定前缀和标签，重复运行只清理自己上次的容器。
# 输出：RESULT_JSON=<路径> / VIDEO_DIR=<路径> / SUMMARY=<episodes>/<successes>
set -euo pipefail

SUITE="${PI05_SMOKE_SUITE:-libero_spatial}"
TASK_IDS="${PI05_SMOKE_TASKS:-0}"
EPISODES="${PI05_SMOKE_EPISODES:-1}"
REPLAN="${PI05_SMOKE_REPLAN:-10}"
MIN_FREE="${PI05_MIN_FREE_MB:-16000}"
GPU="${PI05_GPU:-auto}"
ROOT="${PI05_REMOTE_ROOT:?需要 PI05_REMOTE_ROOT}"
CKPT="${PI05_CKPT:?需要 PI05_CKPT}"
TOKENIZER="${PI05_TOKENIZER:?需要 PI05_TOKENIZER}"
SERVER_IMAGE="${PI05_SERVER_IMAGE:-openpi-server:latest}"
CLIENT_IMAGE="${PI05_CLIENT_IMAGE:-openpi-client:latest}"
NET="pi05-libero"
SERVER="pi05-libero-policy"
LABEL="pi05.repro=1"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$ROOT/runs/${STAMP}-smoke-${SUITE}"

log() { printf '[smoke] %s\n' "$*" >&2; }

cleanup_stale() {
  for name in "$SERVER" "pi05-libero-eval"; do
    if docker container inspect "$name" >/dev/null 2>&1; then
      owner="$(docker inspect --format '{{index .Config.Labels "pi05.repro"}}' "$name" 2>/dev/null || true)"
      if [[ "$owner" == "1" ]]; then
        docker rm -f "$name" >/dev/null
        log "已清理上次遗留容器 $name"
      else
        echo "容器 $name 已存在且不属于本项目，拒绝覆盖" >&2
        exit 3
      fi
    fi
  done
}

pick_gpu() {
  if [[ "$GPU" != "auto" ]]; then
    echo "$GPU"
    return
  fi
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' '
}

mkdir -p "$OUT/logs" "$ROOT/cache/big_vision" "$ROOT/home" "$ROOT/cache/numba"
if [[ ! -f "$ROOT/cache/big_vision/paligemma_tokenizer.model" ]]; then
  cp "$TOKENIZER" "$ROOT/cache/big_vision/"
  log "已放置 tokenizer"
fi

cleanup_stale
GPU="$(pick_gpu)"
free_mb="$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
log "GPU=$GPU 空闲 ${free_mb}MiB"
if (( free_mb < MIN_FREE )); then
  echo "GPU $GPU 空闲 $free_mb MiB 低于阈值 $MIN_FREE MiB，拒绝启动（避免 OOM 影响他人任务）" >&2
  exit 4
fi

if ! docker network inspect "$NET" >/dev/null 2>&1; then
  docker network create --internal --label "$LABEL" "$NET" >/dev/null
  log "已创建内部网络 $NET"
fi

log "启动策略服务（加载权重约 1-2 分钟）"
docker run -d --name "$SERVER" --label "$LABEL" \
  --network "$NET" --gpus "device=$GPU" --shm-size=2g \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -e OPENPI_DATA_HOME=/workspace/cache -e HOME=/workspace/home \
  -v "$CKPT:/checkpoint:ro" -v "$ROOT:/workspace" \
  -w /opt/openpi --entrypoint /opt/openpi/.venv/bin/python "$SERVER_IMAGE" \
  scripts/serve_policy.py --env LIBERO --port 8000 \
  policy:checkpoint --policy.config pi05_libero --policy.dir /checkpoint >/dev/null

ready=0
for _ in $(seq 1 120); do
  if docker logs "$SERVER" 2>&1 | grep -q "server listening"; then ready=1; break; fi
  if [[ "$(docker inspect --format '{{.State.Running}}' "$SERVER")" != "true" ]]; then break; fi
  sleep 5
done
docker logs "$SERVER" >"$OUT/logs/policy-server.log" 2>&1 || true
if [[ "$ready" != "1" ]]; then
  echo "策略服务未在 600 秒内就绪，日志：$OUT/logs/policy-server.log" >&2
  tail -30 "$OUT/logs/policy-server.log" >&2 || true
  docker rm -f "$SERVER" >/dev/null || true
  exit 5
fi
log "策略服务就绪"

set +e
docker run --rm --name "pi05-libero-eval" --label "$LABEL" \
  --network "$NET" --gpus "device=$GPU" --shm-size=2g \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl -e PYNPUT_BACKEND=dummy \
  -e HOME=/workspace/home -e NUMBA_CACHE_DIR=/workspace/cache/numba \
  -v "$ROOT:/workspace" --entrypoint python "$CLIENT_IMAGE" \
  /workspace/evaluation/simulation/libero_episode_runner.py \
  --host "$SERVER" --port 8000 --suite "$SUITE" --task-ids "$TASK_IDS" \
  --episodes "$EPISODES" --replan-steps "$REPLAN" \
  --output "/workspace/runs/$(basename "$OUT")"
client_rc=$?
set -e

docker logs "$SERVER" >"$OUT/logs/policy-server-final.log" 2>&1 || true
docker rm -f "$SERVER" >/dev/null 2>&1 || true

if [[ $client_rc -ne 0 ]]; then
  echo "客户端退出码 $client_rc，产物目录：$OUT" >&2
  exit 6
fi

RESULT="$OUT/evaluation.json"
if [[ ! -f "$RESULT" ]]; then
  echo "缺少 $RESULT" >&2
  exit 7
fi
read -r episodes successes < <(python3 -c "
import json,sys
d=json.load(open('$RESULT'))
print(d.get('completed_episodes',0), d.get('successes',0))
")
echo "RESULT_JSON=$RESULT"
echo "VIDEO_DIR=$OUT"
echo "SUMMARY=${episodes}/${successes}"
