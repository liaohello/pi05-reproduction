#!/usr/bin/env bash
# E4-T finalize: audit the new 30 episodes, re-replay the 20 reused E1 episodes, then export the
# frozen 50-episode training view with train-only statistics and the admission manifest.
#
# Usage: SO101_ROOT=<patched snapshot> SO101_PY=<python> bash scripts/so101/e4t_finalize.sh <main-collection-dir>
#
# Nothing here overwrites an existing artifact: every output directory must not exist yet, and the
# E1 collection and the historical E2 view are read-only inputs.
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: e4t_finalize.sh <collection-dir> [<more collection dirs>...]" >&2
  exit 1
fi
COLLECTIONS=("$@")
: "${SO101_ROOT:?set SO101_ROOT to the patched external snapshot}"
: "${SO101_PY:?set SO101_PY to the external python}"

TS="$(date +%Y%m%d_%H%M%S)"
AUDIT_ROOT="runs/so101/e4t/audit_${TS}"
OLD="runs/so101/e4t/old_replay_${TS}"
VIEW="runs/so101/e4t/train_view_50_${TS}"
E1="runs/so101/e1/collection_20261025"
OLD_VIEW="runs/so101/e2/train_view"

for candidate in "$AUDIT_ROOT" "$OLD" "$VIEW"; do
  if [ -e "$candidate" ]; then
    echo "refusing to reuse an existing output directory: $candidate" >&2
    exit 1
  fi
done
for required in "$E1/e1_manifest.json" "$OLD_VIEW/meta.json"; do
  if [ ! -f "$required" ]; then
    echo "missing input: $required" >&2
    exit 1
  fi
done

mkdir -p "$AUDIT_ROOT"
EXPORT_ARGS=()
for collection in "${COLLECTIONS[@]}"; do
  for required in "$collection/attempts.jsonl" "$collection/e4t_manifest.json"; do
    if [ ! -f "$required" ]; then
      echo "missing input: $required" >&2
      exit 1
    fi
  done
done

echo "[1/4] auditing every collection shard (schemas, disturbance events, physical replay)"
for collection in "${COLLECTIONS[@]}"; do
  name="$(basename "$collection")"
  echo "  - $collection"
  shard_flag=()
  if [ "$#" -gt 0 ] && [ "${#COLLECTIONS[@]}" -gt 1 ]; then shard_flag=(--partial); fi
  if ! "$SO101_PY" scripts/so101/verify_e4t.py --root "$SO101_ROOT" --collection "$collection" \
      --output "$AUDIT_ROOT/${name}.json" --public-output "$AUDIT_ROOT/${name}.public.json" "${shard_flag[@]}"; then
    echo "  ! $name reports problems; the export step decides whether they only concern excluded episodes" >&2
  fi
  EXPORT_ARGS+=(--new-collection "$collection" --new-audit "$AUDIT_ROOT/${name}.json")
done

echo "[2/4] replaying the 20 reused E1 episodes under this round's runtime"
mkdir -p "$OLD"
"$SO101_PY" scripts/so101/verify_demos.py --root "$SO101_ROOT" --collection "$E1" \
  --public-output "$OLD/public_verification.json"

echo "[3/4] exporting the frozen 50-episode view with train-only statistics"
"$SO101_PY" scripts/so101/export_e4t_view.py --old-view "$OLD_VIEW" --old-manifest "$E1/e1_manifest.json" \
  --old-replay "$OLD/public_verification.json" \
  "${EXPORT_ARGS[@]}" --output "$VIEW" \
  --public-manifest "$VIEW/public_view.json"

echo "[4/4] freezing the sampling plan against the exported view"
"$SO101_PY" scripts/so101/build_e4t_sampling_plan.py --view "$VIEW" \
  --output "$VIEW/e4t_sampling_plan.json"

echo "next: launch the state-on run with"
echo "  --train-view $VIEW --data-admission $VIEW/e4t_admission.json \\"
echo "  --sampling-plan $VIEW/e4t_sampling_plan.json"

echo "admission summary:"
"$SO101_PY" - "$VIEW" "$AUDIT_ROOT" "$OLD" <<'PY'
import json, sys, pathlib
view = pathlib.Path(sys.argv[1])
admission = json.loads((view / "e4t_admission.json").read_text())
print(json.dumps({
    "view": str(view),
    "view_sha256": admission.get("view_sha256"),
    "data_gate_passed": (admission.get("data_gate") or {}).get("passed"),
    "totals": admission.get("totals"),
    "counts": admission.get("counts"),
    "view_sha256": admission.get("view_sha256"),
    "train_stats_sha256": admission.get("train_stats_sha256"),
    "norm_stats_canonical_sha256": admission.get("norm_stats_canonical_sha256"),
    "audit": sys.argv[2], "old_replay": sys.argv[3],
}, indent=2, default=str))
PY
