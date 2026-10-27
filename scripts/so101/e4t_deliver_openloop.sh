#!/usr/bin/env bash
# E4-T T2/T3 delivery: aggregate every evaluated checkpoint into the gate determination and the
# PNG/PDF figures, then write one sanitised evidence document.
#
# Usage: e4t_deliver_openloop.sh <run-dir> <view-dir> <evidence-json>
# Read-only with respect to the evaluation outputs; it refuses to overwrite an existing figure dir.
set -euo pipefail

RUN="${1:?usage: e4t_deliver_openloop.sh <run-dir> <view-dir> <evidence-json>}"
VIEW="${2:?missing training view directory}"
EVIDENCE="${3:?missing evidence output path}"
: "${SO101_PY:?set SO101_PY to the external python}"

EVAL="$RUN/evaluation"
OUT="$RUN/openloop_figures"
if [ ! -d "$EVAL" ]; then
  echo "no evaluation outputs under $EVAL yet" >&2
  exit 1
fi
if [ -e "$OUT" ]; then
  echo "refusing to reuse an existing figure directory: $OUT" >&2
  exit 1
fi
mkdir -p "$OUT"

echo "[0/4] per-sampled-copy, episode-equal re-derivation for every checkpoint"
# The first pass averaged the noise copies into a trajectory the sampler never emits.  The corrected
# metrics are written outside the root-owned evaluation directories and consumed below.
PER_COPY="$RUN/per_copy"
mkdir -p "$PER_COPY"
for report in "$EVAL"/*/report.json; do
  [ -e "$report" ] || continue
  label="$(basename "$(dirname "$report")")"
  case "$label" in *superseded*) continue;; esac
  if [ ! -f "$PER_COPY/$label.json" ] || [ "$report" -nt "$PER_COPY/$label.json" ]; then
    echo "  per-copy $label"
    "$SO101_PY" scripts/so101/e4t_percopy_metrics.py --evaluation "$(dirname "$report")" \
      --output "$PER_COPY/$label.json"
  fi
done

echo "[1/4] gate determination over every evaluated checkpoint"
# the initialisation reference is not a fine-tune checkpoint, so it is named explicitly; without it
# the "must beat initialisation" checks would be silently dropped
: "${INITIAL_LABEL:=init_step1}"
RUN_CONFIG_SHA="$(python3 -c "
import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" \
  "$RUN/config_initial.json")"
"$SO101_PY" scripts/so101/summarize_e4t_openloop.py --reports "$EVAL" \
  --initial-label "$INITIAL_LABEL" --run-config "$RUN/config_initial.json" --view "$VIEW" \
  --per-copy-dir "$PER_COPY" \
  --expected-config-sha256 "$RUN_CONFIG_SHA" --output "$RUN/gate_summary.json"

echo "[2/4] figures (PNG + PDF)"
# the overlays also mark the expert's own detection/correction/regrasp moments, which live in the
# episode diagnostics referenced by the frozen view
"$SO101_PY" scripts/so101/plot_e4t_curves.py --eval-root "$EVAL" --output "$OUT" \
  --summary "$RUN/gate_summary.json" --view "$VIEW" --per-copy-dir "$PER_COPY" \
  --lr-trace "${LR_TRACE:-$RUN/lr_trace.json}" --run-config-sha256 "$RUN_CONFIG_SHA" \
  --collection runs/so101/e4t/main_20261026_050822 \
  --collection runs/so101/e4t/topupA_20261026_091910 \
  --collection runs/so101/e4t/topupC_20261026_091910

echo "[3/4] evidence document"
"$SO101_PY" - "$RUN" "$VIEW" "$EVIDENCE" <<'PY'
import hashlib, json, pathlib, sys
from datetime import datetime, timezone

run = pathlib.Path(sys.argv[1]); view = pathlib.Path(sys.argv[2]); out = pathlib.Path(sys.argv[3])
summary = json.loads((run / "gate_summary.json").read_text())
driver = json.loads((run / "evaluation_validation_driver.json").read_text())
figures = sorted(str(p.relative_to(run)) for p in (run / "openloop_figures").glob("*") if p.is_file())

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

prediction_artifacts = []
for report_path in sorted((run / "evaluation").glob("*/report.json")):
    if "superseded" in report_path.parent.name:
        continue
    source = json.loads(report_path.read_text())
    window_set = source.get("window_set") or "frozen_plan"
    if source.get("split") != "validation" or window_set not in {
            "frozen_plan", "frozen_plan_plus_dense_sweep", "event_anchored_supplementary"}:
        continue
    arrays = report_path.parent / "arrays.npz"
    records = report_path.parent / "records.jsonl.gz"
    if not arrays.is_file() or not records.is_file():
        raise SystemExit(f"missing prediction arrays or records for {report_path.parent.name}")
    prediction_artifacts.append({
        "label": source.get("label", report_path.parent.name),
        "checkpoint_train_state_step": source.get("checkpoint_train_state_step"),
        "split": source.get("split"), "window_set": window_set,
        "windows": source.get("windows"),
        "noise_copies": source.get("noise_copies"),
        "noise_set_sha256": source.get("noise_set_sha256"),
        "report_sha256": sha256(report_path),
        "arrays_sha256": sha256(arrays),
        "records_sha256": sha256(records),
    })
diagnostic = summary.get("diagnostic_checkpoint") or {}
diagnostic_label = diagnostic.get("label")
diagnostic_per_copy_path = run / "per_copy" / f"{diagnostic_label}.json" if diagnostic_label else None
diagnostic_breakdown = None
if diagnostic_per_copy_path and diagnostic_per_copy_path.is_file():
    per_copy = json.loads(diagnostic_per_copy_path.read_text())
    groups = per_copy.get("aggregates") or {}
    wanted = ("clean_active", "clean_static", "recovery_pre_active", "recovery_post_active",
              "recovery_post_static", "recovery_A_post_active", "recovery_B_post_active",
              "recovery_C_post_active")
    diagnostic_breakdown = {
        "label": diagnostic_label,
        "aggregation": per_copy.get("aggregation"),
        "noise_copies": per_copy.get("noise_copies"),
        "noise_set_sha256": per_copy.get("noise_set_sha256"),
        "groups": {
            key: {field: (groups.get(key) or {}).get(field) for field in
                  ("episodes", "windows", "joint_macro_mae_deg", "joint_p95_deg",
                   "gripper_macro_mae_pct", "true12_mse", "padding20_mse")}
            for key in wanted if key in groups
        },
        "groups_without_windows": [key for key in wanted
                                   if not int((groups.get(key) or {}).get("windows") or 0)],
        "horizon_metrics_by_group": {
            key: (per_copy.get("horizon_metrics_by_group") or {}).get(key)
            for key in ("clean_active", "recovery_post_active")
            if key in (per_copy.get("horizon_metrics_by_group") or {})
        },
        "source_sha256": sha256(diagnostic_per_copy_path),
    }
figure_hashes = [{"file": str(path.relative_to(run)), "sha256": sha256(path)}
                 for path in sorted((run / "openloop_figures").glob("*")) if path.is_file()]
report = {
    "schema_version": 1, "branch": "so101", "phase": "E3", "substage": "E4-T-T2-T3-openloop",
    "created": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    "run_artifact_root": "ignored E4-T run artifacts", "view": view.name,
    "view_sha256": json.loads((view / "meta.json").read_text())["view_sha256"],
    "checkpoints_evaluated": [row["label"] for row in driver["evaluations"]],
    "evaluation_failures": [
        {key: row.get(key) for key in ("label", "authoritative_step", "success", "returncode", "problem")}
        for row in driver["evaluations"] if not row.get("success")],
    "driver_sha256": hashlib.sha256((run / "evaluation_validation_driver.json").read_bytes()).hexdigest(),
    "figures": figures,
    "figure_hashes": figure_hashes,
    "prediction_artifacts": prediction_artifacts,
    "selected_checkpoint_dense_sweep": {
        "diagnostic_step": diagnostic.get("step"),
        "completed": any(
            item.get("checkpoint_train_state_step") == diagnostic.get("step")
            and item.get("window_set") == "frozen_plan_plus_dense_sweep"
            for item in prediction_artifacts),
        "labels": [
            item["label"] for item in prediction_artifacts
            if item.get("checkpoint_train_state_step") == diagnostic.get("step")
            and item.get("window_set") == "frozen_plan_plus_dense_sweep"],
    },
    "diagnostic_breakdown": diagnostic_breakdown,
    "gate": summary,
    "no_learned_action_executed_in_simulation": True,
    "claim": "open-loop action prediction on the frozen validation split; not a success rate",
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"evidence": str(out), "figures": len(figures),
                  "decision": summary.get("decision"), "best": summary.get("best_checkpoint")}, indent=2))
PY

echo "figures: $OUT"
