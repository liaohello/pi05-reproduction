#!/usr/bin/env bash
# E2 sequence: wait for the real training view, run 100 updates, inspect,
# resume in a new process for one update, inspect again.
#
# Required environment:
#   SO101_TRAIN_VIEW, SO101_RUN_DIR, SO101_PI05_CHECKPOINT, SO101_TOKENIZER,
#   SO101_SOURCE_ROOT, SO101_COLLECTION (for the export-failure marker)
# Optional:
#   SO101_REPO, SO101_GPU_INDEX=2, SO101_EXP_NAME=e2_lora_b1_s42
set -euo pipefail
repo="${SO101_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
: "${SO101_TRAIN_VIEW:?}"
: "${SO101_RUN_DIR:?}"
: "${SO101_PI05_CHECKPOINT:?}"
: "${SO101_TOKENIZER:?}"
: "${SO101_SOURCE_ROOT:?}"
: "${SO101_COLLECTION:?}"
gpu="${SO101_GPU_INDEX:-2}"
exp_name="${SO101_EXP_NAME:-e2_lora_b1_s42}"
cd "$repo"
mkdir -p "$SO101_RUN_DIR"
log="$SO101_RUN_DIR/sequence.log"
exec > >(tee -a "$log") 2>&1

echo "[sequence] waiting for $SO101_TRAIN_VIEW/meta.json"
while [[ ! -f "$SO101_TRAIN_VIEW/meta.json" ]]; do
  if [[ -f "$SO101_COLLECTION/post/export.exit" ]]; then
    status=$(cat "$SO101_COLLECTION/post/export.exit")
    if [[ "$status" != "0" ]]; then
      echo "[sequence] export failed with exit $status" >&2
      exit 1
    fi
  fi
  sleep 30
done

rel_run="${SO101_RUN_DIR#"$repo"/}"
rel_view="${SO101_TRAIN_VIEW#"$repo"/}"
container_run="/work/$rel_run"
container_view="/work/$rel_view"
ckpt_root="$SO101_RUN_DIR/checkpoints/pi05_so101_e2/$exp_name"
container_ckpt="$container_run/checkpoints/pi05_so101_e2/$exp_name"

run_container_cpu() {
  docker run --rm --network none --entrypoint python3 -e JAX_PLATFORMS=cpu \
    -e PYTHONPATH=/work/scripts/so101 -v "$repo:/work" \
    -v "$SO101_PI05_CHECKPOINT:/base:ro" openpi-server:latest "$@"
}

echo "[sequence] initial 100 updates"
SO101_REPO="$repo" SO101_RUN_DIR="$SO101_RUN_DIR" SO101_TRAIN_VIEW="$SO101_TRAIN_VIEW" \
SO101_PI05_CHECKPOINT="$SO101_PI05_CHECKPOINT" SO101_TOKENIZER="$SO101_TOKENIZER" \
SO101_MODE=initial SO101_GPU_INDEX="$gpu" SO101_EXP_NAME="$exp_name" \
  bash scripts/so101/run_training.sh
initial_status=$?
if [[ "$initial_status" != "0" ]]; then
  echo "[sequence] initial run failed: $initial_status" >&2
  exit "$initial_status"
fi

latest=$(find "$ckpt_root" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | grep -E '^[0-9]+$' | sort -n | tail -1)
if [[ -z "$latest" ]]; then
  echo "[sequence] no checkpoint produced" >&2
  exit 1
fi
echo "[sequence] initial latest Orbax step: $latest"
run_container_cpu /work/scripts/so101/read_train_state.py \
  --checkpoint-root "$container_ckpt" --step "$latest" --train-view "$container_view" \
  --base-checkpoint /base --output "$container_run/train_state_$latest.json"
state_step=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['train_state_step'])" "$SO101_RUN_DIR/train_state_$latest.json")
echo "[sequence] TrainState.step after initial: $state_step"
run_container_cpu /work/scripts/so101/inspect_checkpoint.py --base /base \
  --trained "$container_ckpt/$latest" --checkpoint-root "$container_ckpt" \
  --train-state-step "$state_step" --output "$container_run/inspect_$latest.json"

echo "[sequence] resume in a new process for one update"
SO101_REPO="$repo" SO101_RUN_DIR="$SO101_RUN_DIR" SO101_TRAIN_VIEW="$SO101_TRAIN_VIEW" \
SO101_PI05_CHECKPOINT="$SO101_PI05_CHECKPOINT" SO101_TOKENIZER="$SO101_TOKENIZER" \
SO101_MODE=resume SO101_GPU_INDEX="$gpu" SO101_EXP_NAME="$exp_name" \
  bash scripts/so101/run_training.sh
resume_status=$?
if [[ "$resume_status" != "0" ]]; then
  echo "[sequence] resume run failed: $resume_status" >&2
  exit "$resume_status"
fi

latest2=$(find "$ckpt_root" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | grep -E '^[0-9]+$' | sort -n | tail -1)
echo "[sequence] resume latest Orbax step: $latest2"
run_container_cpu /work/scripts/so101/read_train_state.py \
  --checkpoint-root "$container_ckpt" --step "$latest2" --train-view "$container_view" \
  --base-checkpoint /base --output "$container_run/train_state_$latest2.json"
state_step2=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['train_state_step'])" "$SO101_RUN_DIR/train_state_$latest2.json")
echo "[sequence] TrainState.step after resume: $state_step2"
run_container_cpu /work/scripts/so101/inspect_checkpoint.py --base /base \
  --trained "$container_ckpt/$latest2" --previous "$container_ckpt/$latest" \
  --checkpoint-root "$container_ckpt" --train-state-step "$state_step2" \
  --output "$container_run/inspect_$latest2.json"

python3 scripts/so101/summarize_training.py --run-dir "$SO101_RUN_DIR" --mode initial
python3 scripts/so101/summarize_training.py --run-dir "$SO101_RUN_DIR" --mode resume
python3 - "$SO101_RUN_DIR" "$latest" "$state_step" "$latest2" "$state_step2" <<'PY'
import json, sys
from pathlib import Path
run, latest, step1, latest2, step2 = sys.argv[1:]
payload = {"initial_orbax_step": int(latest), "initial_train_state_step": int(step1),
           "resume_orbax_step": int(latest2), "resume_train_state_step": int(step2),
           "initial_inspect": f"inspect_{latest}.json",
           "resume_inspect": f"inspect_{latest2}.json",
           "initial_summary": "summary_initial.json", "resume_summary": "summary_resume.json"}
Path(run, "e2_sequence.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
docker run --rm --network none --entrypoint bash -v "$repo:/work" openpi-server:latest \
  -c "chmod -R a+rwX /work/$rel_run" >/dev/null 2>&1 || true
echo "[sequence] complete"
