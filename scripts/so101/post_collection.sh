#!/usr/bin/env bash
# E1 post-processing: wait for collection, then audit (replay) and export in parallel.
#
# Required environment:
#   SO101_COLLECTION       E1 collection directory
#   SO101_COLLECTION_PID   PID of the running collection process
#   SO101_TRAIN_VIEW      new output directory for the training view
#   SO101_SOURCE_ROOT      external SO101 snapshot
#   SO101_PYTHON           external Python interpreter
# Optional:
#   SO101_REPO             repository root (auto-detected)
#   SO101_EGL_DEVICE=2
set -euo pipefail
repo="${SO101_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
: "${SO101_COLLECTION:?}"
: "${SO101_COLLECTION_PID:?}"
: "${SO101_TRAIN_VIEW:?}"
: "${SO101_SOURCE_ROOT:?}"
: "${SO101_PYTHON:?}"
ego="${SO101_EGL_DEVICE:-2}"
mkdir -p "$SO101_COLLECTION/post"
cd "$repo"
while kill -0 "$SO101_COLLECTION_PID" 2>/dev/null; do sleep 20; done
if [[ ! -f "$SO101_COLLECTION/summary.json" || ! -f "$SO101_COLLECTION/e1_manifest.json" ]]; then
  echo "collection did not finish cleanly: missing summary.json or e1_manifest.json" >&2
  exit 1
fi
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SO101_SOURCE_ROOT SO101_PYTHON BIMANUAL_EGL_DEVICE_ID="$ego" MUJOCO_GL=egl
(
  set +e
  "$SO101_PYTHON" scripts/so101/export_training_view.py \
    --collection "$SO101_COLLECTION" --output "$SO101_TRAIN_VIEW" --workers 4 \
    --public-manifest "$SO101_COLLECTION/public_train_view.json" \
    > "$SO101_COLLECTION/post/export.log" 2>&1
  echo $? > "$SO101_COLLECTION/post/export.exit"
) &
export_pid=$!
(
  set +e
  "$SO101_PYTHON" scripts/so101/verify_demos.py --root "$SO101_SOURCE_ROOT" \
    --collection "$SO101_COLLECTION" \
    --public-output "$SO101_COLLECTION/public_verification.json" \
    > "$SO101_COLLECTION/post/verify.log" 2>&1
  echo $? > "$SO101_COLLECTION/post/verify.exit"
) &
verify_pid=$!
wait "$export_pid" || true
wait "$verify_pid" || true
export_status=$(cat "$SO101_COLLECTION/post/export.exit")
verify_status=$(cat "$SO101_COLLECTION/post/verify.exit")
python3 - "$SO101_COLLECTION" "$export_status" "$verify_status" <<'PY'
import json, sys
from pathlib import Path
collection, export_status, verify_status = sys.argv[1:]
payload = {"export_exit": int(export_status), "verify_exit": int(verify_status),
           "train_view_ready": Path(collection, "post", "export.exit").read_text().strip() == "0"}
Path(collection, "post", "post_status.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
test "$export_status" = 0
