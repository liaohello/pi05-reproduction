#!/usr/bin/env bash
# Run a server command with stdout and stderr teed to separate files.
set -euo pipefail
LOG_DIR="${PI05_SERVER_LOG_DIR:?set PI05_SERVER_LOG_DIR}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/server.stdout.log") 2> >(tee -a "$LOG_DIR/server.stderr.log" >&2)
exec "$@"
