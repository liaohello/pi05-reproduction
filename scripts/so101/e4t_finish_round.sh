#!/usr/bin/env bash
# E4-T end-of-round pipeline. Runs detached on the GPU host and writes one status document.
#
# It waits for the registered training container to report, waits for the evaluation driver to have
# scored every committed checkpoint, then adds the supplementary passes (event-anchored timing and the
# dense curve on the selected checkpoint), regenerates the delivery (gate summary, figures, evidence
# document), reads the final TrainState back authoritatively, and rebuilds the training and budget
# records from the artifacts.
#
# Usage: e4t_finish_round.sh <run-dir> <view-dir> <python>
# Requires PI05_CKPT and PI05_TOKENIZER in the environment for the container steps.
# Nothing here starts training, applies a learned action to the simulator, or touches another
# project's containers: every launch goes through the registered, exact-owner path in
# e4t_run_training.py.
set -uo pipefail

RUN="${1:?usage: e4t_finish_round.sh <run-dir> <view-dir> <python>}"
VIEW="${2:?missing frozen view directory}"
PY="${3:?missing python executable}"
: "${SO101_IMAGE:=openpi-server:latest}"
REPO="$(pwd)"
STEP_LOG="$RUN/finish_round.log"
STATUS="$RUN/finish_round.json"
declare -a STEPS=()

log() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$STEP_LOG"; }
step() { STEPS+=("$1=$2"); log "step $1 -> $2"; }

wait_for() {  # wait_for <description> <max-iterations> <command...>
  local what="$1" limit="$2"; shift 2
  for _ in $(seq 1 "$limit"); do
    if "$@" >/dev/null 2>&1; then log "$what ready"; return 0; fi
    sleep 60
  done
  log "$what NOT ready after $limit minutes"; return 1
}

log "waiting for the training run to report"
if wait_for "run_initial.json" 480 test -f "$RUN/run_initial.json"; then
  step training_reported ok
else
  step training_reported timeout
fi

log "waiting for the evaluation driver to score every committed checkpoint"
missing_checkpoint() {
  "$PY" - "$RUN" <<'PY' 2>/dev/null
import sys
from pathlib import Path
sys.path.insert(0, "scripts/so101")
import e4t_eval_checkpoints as driver
import so101_common as common

run = Path(sys.argv[1])
missing = [str(path.name) for index, path in driver.completed_checkpoints(run / "checkpoints")
           if not (run / "evaluation" / f"step{driver.expected_step(index):06d}" / "report.json").is_file()]
sys.exit(1 if missing else 0)
PY
}
if wait_for "checkpoint evaluations" 180 missing_checkpoint; then
  step evaluations_complete ok
else
  step evaluations_complete timeout
fi

log "gate determination over the frozen plan, to pick the supplementary checkpoint"
PER_COPY="$RUN/per_copy"
mkdir -p "$PER_COPY"
for report in "$RUN"/evaluation/*/report.json; do
  [ -e "$report" ] || continue
  label="$(basename "$(dirname "$report")")"
  case "$label" in *superseded*) continue;; esac
  if [ ! -f "$PER_COPY/$label.json" ] || [ "$report" -nt "$PER_COPY/$label.json" ]; then
    log "per-copy re-derivation for $label"
    "$PY" scripts/so101/e4t_percopy_metrics.py --evaluation "$(dirname "$report")" \
      --output "$PER_COPY/$label.json" >>"$STEP_LOG" 2>&1 || log "per-copy failed for $label"
  fi
done
INITIAL_LABEL="${INITIAL_LABEL:-init_step1}"
RUN_CONFIG_SHA="$("$PY" -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" \
  "$RUN/config_initial.json")"
"$PY" scripts/so101/summarize_e4t_openloop.py --reports "$RUN/evaluation" \
  --initial-label "$INITIAL_LABEL" --run-config "$RUN/config_initial.json" --view "$VIEW" \
  --per-copy-dir "$PER_COPY" \
  --expected-config-sha256 "$RUN_CONFIG_SHA" --output "$RUN/gate_summary.json" >>"$STEP_LOG" 2>&1
step gate_summary "$?"

BEST_STEP="$("$PY" - "$RUN/gate_summary.json" <<'PY' 2>/dev/null
import json, sys
summary = json.loads(open(sys.argv[1]).read())
best = summary.get("best_checkpoint")
diagnostic = summary.get("diagnostic_checkpoint")
rows = [row for row in summary["curve"] if row["step"] > 0]
if best:
    print(best["step"])
elif diagnostic:
    # No candidate cleared the first gripper/timing filters. Keep the pre-registered
    # error-best checkpoint for diagnostics; do not silently substitute the latest step.
    print(diagnostic["step"])
elif rows:
    # Compatibility fallback for summaries created before diagnostic_checkpoint existed.
    eligible = [row for row in rows if row.get("primary_objective_deg") is not None]
    print(min(eligible, key=lambda row: (row["primary_objective_deg"], row["step"]))["step"]
          if eligible else max(row["step"] for row in rows))
else:
    print(0)
PY
)"
BEST_DIR="$("$PY" - "$RUN" "$BEST_STEP" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "scripts/so101")
import e4t_eval_checkpoints as driver
run = Path(sys.argv[1]); step = int(sys.argv[2])
print(next((str(path.name) for index, path in driver.completed_checkpoints(run / "checkpoints")
            if driver.expected_step(index) == step), ""))
PY
)"
log "supplementary checkpoint: step=$BEST_STEP orbax directory=$BEST_DIR"
step supplementary_target "$([ -n "$BEST_DIR" ] && echo ok || echo missing)"

if [ -n "$BEST_DIR" ]; then
  log "event-anchored timing pass on the selected checkpoint"
  bash "$RUN/anchor_eval.sh" "$BEST_DIR" "$BEST_STEP" >>"$STEP_LOG" 2>&1
  step anchored_pass "$?"
  log "dense curve (stride 10) on the selected checkpoint"
  bash "$RUN/dense_eval.sh" "$BEST_DIR" "$BEST_STEP" 10 >>"$STEP_LOG" 2>&1
  step dense_pass "$?"
fi

log "final delivery: gate summary, figures and the evidence document"
if [ -d "$RUN/openloop_figures" ]; then
  # the delivery refuses to reuse a figure directory, so the previous one is archived, not deleted
  mv "$RUN/openloop_figures" "$RUN/openloop_figures_prev_$(date -u +%H%M%S)"
fi
SO101_PY="$PY" LR_TRACE="${LR_TRACE:-$RUN/lr_trace_authoritative.json}" \
  bash scripts/so101/e4t_deliver_openloop.sh "$RUN" "$VIEW" "$RUN/e4t-openloop-final.json" \
  >>"$STEP_LOG" 2>&1
step delivery "$?"

FINAL_DIR="$("$PY" - "$RUN" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "scripts/so101")
import e4t_eval_checkpoints as driver
run = Path(sys.argv[1])
done = driver.completed_checkpoints(run / "checkpoints")
print(str(done[-1][1].name) if done else "")
PY
)"
if [ -n "$FINAL_DIR" ] && [ -n "${PI05_CKPT:-}" ]; then
  log "authoritative TrainState read-back for directory $FINAL_DIR"
  docker run --rm --entrypoint python3 \
    -e HOME="/work/$RUN/home" -e OPENPI_DATA_HOME="/work/$RUN/openpi_cache" \
    -e PYTHONPATH=/work/scripts/so101 -e JAX_PLATFORMS=cpu \
    -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
    -v "$REPO:/work" -v "$PI05_CKPT:/base:ro" -w /work "$SO101_IMAGE" \
    -u /work/scripts/so101/read_train_state.py \
    --checkpoint-root "/work/$RUN/checkpoints/pi05_so101_e2/e4t_30k_state_on_b8" \
    --step "$FINAL_DIR" --train-view "/work/$VIEW" --base-checkpoint /base --state-mode on \
    --output "/work/$RUN/train_state_final.json" >>"$STEP_LOG" 2>&1
  step train_state_readback "$?"
fi

log "rebuilding the training and budget records"
"$PY" scripts/so101/e4t_training_evidence.py --run-dir "$RUN" --view "$VIEW" \
  --narrative experiments/so101/e4t-training.json --output "$RUN/e4t-training-final.json" \
  >>"$STEP_LOG" 2>&1
step training_evidence "$?"
"$PY" scripts/so101/e4t_budget.py --output "$RUN/e4t-budget-final.json" >>"$STEP_LOG" 2>&1
step budget "$?"

{
  printf '{\n  "finished_at": "%s",\n  "supplementary_step": %s,\n  "supplementary_directory": "%s",\n  "steps": {\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${BEST_STEP:-0}" "${BEST_DIR:-}"
  for entry in "${STEPS[@]}"; do printf '    "%s": "%s",\n' "${entry%%=*}" "${entry#*=}"; done | sed '$ s/,$//'
  printf '  }\n}\n'
} > "$STATUS"
log "wrote $STATUS"
