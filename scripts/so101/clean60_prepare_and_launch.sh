#!/usr/bin/env bash
# Clean60 data gate chain. Formal launch is opt-in and requires a READY manifest.
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

VIEW=""
PYTHON_BIN="${CLEAN60_PYTHON:-python3}"
TRAIN_CONFIG="${CLEAN60_TRAINING_CONFIG:-configs/so101/clean60_training.json}"
RUN_ID="${CLEAN60_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-clean60-prepare-$(python3 -c 'import secrets; print(secrets.token_hex(4))')}"
RUN_DIR=""
READY_MANIFEST=""
LAUNCH=0

while (($#)); do
  case "$1" in
    --view) VIEW="${2:?--view needs a path}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?--python needs a path}"; shift 2 ;;
    --training-config) TRAIN_CONFIG="${2:?--training-config needs a path}"; shift 2 ;;
    --run-id) RUN_ID="${2:?--run-id needs a value}"; shift 2 ;;
    --run-dir) RUN_DIR="${2:?--run-dir needs a path}"; shift 2 ;;
    --ready-manifest) READY_MANIFEST="${2:?--ready-manifest needs a path}"; shift 2 ;;
    --launch) LAUNCH=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

: "${VIEW:?provide --view for the new versioned clean60 view}"
if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="runs/so101/clean60_25k/$RUN_ID"
fi
if [[ "$LAUNCH" == 1 ]]; then
  : "${GPU_INDICES:?set after the final per-GPU ownership and memory check}"
  : "${FSDP_DEVICES:?set to the measured single-process FSDP shard count}"
  : "${READY_MANIFEST:?formal launch requires --ready-manifest}"
fi

DEMAND_MIB="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["resource_policy"]["measured_per_card_demand_mib"])' "$TRAIN_CONFIG")"
MARGIN_MIB="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["resource_policy"]["memory_margin_mib"])' "$TRAIN_CONFIG")"
IMAGE="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["runtime_binding"]["image"])' "$TRAIN_CONFIG")"

echo "[1/4] checking explicit clean provenance and frozen view bindings"
"$PYTHON_BIN" - "$VIEW" "$TRAIN_CONFIG" <<'PY'
import json, pathlib, sys
sys.path.insert(0, "scripts/so101")
import clean60_contracts as c
view, config_path = map(pathlib.Path, sys.argv[1:])
config = json.loads(config_path.read_text())
meta = json.loads((view / "meta.json").read_text())
binding = json.loads((view / "protocol_binding.json").read_text())
split = json.loads((view / "splits.json").read_text())
admission = json.loads((view / "e4t_admission.json").read_text())
scene_audit = json.loads((view / "scene_audit.json").read_text())
data_audit = json.loads((view / "data_audit.json").read_text())
stats_report = json.loads((view / "stats_recompute.json").read_text())
problems = c.training_config_problems(config)
problems += c.view_contract_problems(
    meta, protocol_binding=binding,
    expected_counts=(config["data"]["view_episodes"], config["data"]["train"],
                     config["data"]["validation"]), require_clean=True)
problems += c.admission_contract_problems(
    admission, meta=meta, split=split,
    train_stats_sha256=c.sha256_file(view / "train_stats.json"))
problems += c.scene_audit_problems(scene_audit, view_sha256=meta["view_sha256"])
problems += c.data_audit_problems(data_audit, view_sha256=meta["view_sha256"])
problems += c.stats_recompute_problems(
    stats_report, view_sha256=meta["view_sha256"],
    norm_stats_sha256=c.sha256_file(view / "assets/so101/norm_stats.json"))
if problems:
    raise SystemExit("clean60 view contract failed: " + "; ".join(problems))
print(json.dumps({"view": view.name, "episodes": len(meta["episodes"]),
                  "train": config["data"]["train"], "validation": config["data"]["validation"],
                  "clean_source": meta["clean_profile"]["source"],
                  "view_sha256": meta["view_sha256"]}, indent=2))
PY

echo "[2/4] freezing the clean-only sampling plan"
"$PYTHON_BIN" scripts/so101/build_e4t_sampling_plan.py \
  --training-config "$TRAIN_CONFIG" --view "$VIEW" \
  --output "$VIEW/e4t_sampling_plan.json"

echo "[3/4] action normalization audit inside the pinned image"
: "${PI05_CKPT:?export PI05_CKPT before auditing}"
: "${PI05_TOKENIZER:?export PI05_TOKENIZER before auditing}"
CACHE="$RUN_DIR/openpi_cache"
CONTAINER_VIEW=/work/clean60_view
mkdir -p "$CACHE/big_vision"
python3 scripts/so101/clean60_container_exec.py --run-dir "$RUN_DIR" --label normaudit \
  --image "$IMAGE" \
  --mount "$PI05_CKPT":/base:ro \
  --mount "$VIEW":"$CONTAINER_VIEW" \
  --mount "$PI05_TOKENIZER":/work/"$CACHE"/big_vision/paligemma_tokenizer.model:ro \
  --env "HOME=/work/$RUN_DIR/home" --env "OPENPI_DATA_HOME=/work/$CACHE" \
  --env "SO101_TRAIN_VIEW=$CONTAINER_VIEW" --timeout 2400 \
  --container-python-file /work/scripts/so101/clean60_norm_audit.py \
  --python-arg=--view --python-arg="$CONTAINER_VIEW" \
  --python-arg=--run-config --python-arg=/work/"$TRAIN_CONFIG" \
  --python-arg=--base-checkpoint --python-arg=/base \
  --python-arg=--fsdp-devices --python-arg=1
"$PYTHON_BIN" - "$VIEW" "$TRAIN_CONFIG" "$PI05_CKPT" <<'PY'
import json, pathlib, sys
sys.path.insert(0, "scripts/so101")
import clean60_contracts as c
view, config_path, base = map(pathlib.Path, sys.argv[1:])
meta = json.loads((view / "meta.json").read_text())
audit = json.loads((view / "action_norm_audit.json").read_text())
problems = c.norm_audit_problems(
    audit, view_sha256=meta["view_sha256"],
    norm_stats_sha256=c.sha256_file(view / "assets/so101/norm_stats.json"),
    base_checkpoint_sha256=c.sha256_tree(base),
    training_config_sha256=c.sha256_file(config_path))
if problems:
    raise SystemExit("action normalization gate failed: " + "; ".join(problems))
print(json.dumps({"norm_audit": "passed",
                  "worst_recovery_error": audit["transform"]["worst_recovery_error"],
                  "internal_dim": audit["transform"]["internal_action_dim"],
                  "loader_episodes_covered": audit["loader_consistency"]["episodes_covered"],
                  "dims_exceeding_unit": audit["transform"]["dims_exceeding_unit_interval_after_normalisation"],
                  "norm_stats_sha256": audit["norm_stats_sha256"]}, indent=2))
PY

if [[ "$LAUNCH" == 0 ]]; then
  echo "prepare-only complete; no formal training was started"
  exit 0
fi

echo "[4/4] launching the reviewed formal run"
mkdir -p "$RUN_DIR"
setsid nohup "$PYTHON_BIN" scripts/so101/e4t_launch_training.py \
  --view "$VIEW" --python "$PYTHON_BIN" \
  --gpu-indices "$GPU_INDICES" --fsdp-devices "$FSDP_DEVICES" \
  --required-memory-mib "$DEMAND_MIB" --memory-margin-mib "$MARGIN_MIB" \
  --training-config "$TRAIN_CONFIG" --run-directory "$RUN_DIR" --run-id "$RUN_ID" \
  --ready-manifest "$READY_MANIFEST" \
  > "$RUN_DIR/launch_$RUN_ID.log" 2>&1 &
echo "launch_pid=$! run_id=$RUN_ID log=$RUN_DIR/launch_$RUN_ID.log"
