#!/usr/bin/env bash
# Scan the working tree, index, and all commits reachable from HEAD.
# Reports contain file/line/type only; matched text is never copied to the report.
set -euo pipefail
cd "$(dirname "$0")/../.."
exec python3 scripts/security/scan_content.py "$@"
