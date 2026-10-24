#!/usr/bin/env bash
# Single entry point. Only requires python3, ssh and rsync.
#   ./scripts/run.sh help
set -euo pipefail
cd "$(dirname "$0")/.."

usage() {
  cat <<'EOF'
pi05-reproduction · local deploy and experiment tracking

  ./scripts/run.sh preflight         local checks: config, source lock, git, disk
  ./scripts/run.sh preflight-remote  local + remote: ssh, workdir, images, checkpoint, GPU
  ./scripts/run.sh deploy-dry        preview what would be synced to the remote workdir
  ./scripts/run.sh deploy            sync source to the remote workdir
  ./scripts/run.sh verify            content-level check: remote must equal local
  ./scripts/run.sh smoke --suite ... run one LIBERO episode on the server, record to ledger
  ./scripts/run.sh release           remove this project's containers and network
  ./scripts/run.sh status            refresh phase gates, write reports/status.md
  ./scripts/run.sh baseline          verify third-party source against the lock
  ./scripts/run.sh record ...        append one run record (see scripts/track/record.py -h)
  ./scripts/run.sh env               print resolved deployment variables
  ./scripts/run.sh scan              public-repository security scan
  ./scripts/run.sh ssh [cmd...]      login, or run one command remotely
EOF
}

cmd="${1:-help}"
[ $# -gt 0 ] && shift || true

case "$cmd" in
  preflight)        exec python3 scripts/deploy/preflight.py "$@" ;;
  preflight-remote) exec python3 scripts/deploy/preflight.py --remote "$@" ;;
  deploy-dry)       exec python3 scripts/deploy/sync.py --dry-run "$@" ;;
  deploy)           exec python3 scripts/deploy/sync.py "$@" ;;
  verify)           exec python3 scripts/deploy/sync.py --check "$@" ;;
  smoke)            exec python3 scripts/deploy/smoke.py "$@" ;;
  release)          exec python3 scripts/deploy/release.py "$@" ;;
  status)           exec python3 scripts/track/status.py "$@" ;;
  baseline)         exec python3 scripts/verify/source_lock.py "$@" ;;
  record)           exec python3 scripts/track/record.py "$@" ;;
  env)              exec python3 scripts/deploy/_env.py ;;
  scan)             exec bash scripts/security/scan.sh "$@" ;;
  ssh)
    # shellcheck disable=SC1091
    eval "$(python3 scripts/deploy/_env.py)"
    exec ssh -i "$PI05_SSH_KEY_PATH" -p "$PI05_REMOTE_PORT" "$PI05_REMOTE_USER@$PI05_REMOTE_HOST" "$@"
    ;;
  help|-h|--help)   usage ;;
  *)                echo "unknown subcommand: $cmd" >&2; usage; exit 2 ;;
esac
