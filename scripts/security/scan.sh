#!/usr/bin/env bash
# Public-repository security gate.
# Fails when any real deployment detail or credential leaks into tracked files.
#
#   bash scripts/security/scan.sh          # scan tracked + staged + working tree
#   bash scripts/security/scan.sh --all    # include ignored-but-present files
#
# Reports are written to security-reports/ (git-ignored) and never outside.
set -uo pipefail
cd "$(dirname "$0")/../.."

mkdir -p security-reports
report="security-reports/scan-$(date -u +%Y%m%dT%H%M%SZ).txt"
findings=0

say() { printf '%s\n' "$*" | tee -a "$report"; }

# 1. Hard-coded deployment facts. Patterns are intentionally narrow so that
#    documentation placeholders (${PI05_REMOTE_HOST}) never match.
PATTERNS=(
  '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'
  'BEGIN [A-Z ]*PRIVATE KEY'
  '(ssh|scp)[[:space:]]+-i[[:space:]]+/'
  'id_ed25519_cec_sim|remoteServer'
  '/(home|Users)/[A-Za-z0-9._-]+'
  '[A-Za-z0-9._-]+@[0-9]{1,3}(\.[0-9]{1,3}){3}'
  'ssh[[:space:]]+-p[[:space:]]+[0-9]{2,5}'
  '/data/openpi'
)
# Benign addresses that may legitimately appear in docs and code.
ALLOW='127\.0\.0\.1|0\.0\.0\.0|255\.255\.255\.255|10\.0\.2\.2'

say "== public repository security scan · $(date -u +%FT%TZ) =="

files_list() {
  if git rev-parse --git-dir >/dev/null 2>&1; then
    if [ "${1:-}" = "--all" ]; then
      git ls-files -z --cached --others
    else
      # tracked + untracked-but-not-ignored + staged: everything a commit could carry
      { git ls-files -z; git ls-files --others --exclude-standard -z; \
        git diff --cached --name-only -z; } | sort -zu
    fi
  else
    find . -type f -not -path './.git/*' -print0
  fi
}

while IFS= read -r -d '' file; do
  case "$file" in
    *.mp4|*.png|*.jpg|*.jpeg|*.gif|*.pdf|*.zip) continue ;;
    scripts/security/scan.sh) continue ;;   # the scanner contains the patterns themselves
  esac
  [ -f "$file" ] || continue
  for pattern in "${PATTERNS[@]}"; do
    hits="$(grep -nEI -- "$pattern" "$file" 2>/dev/null | grep -Ev "$ALLOW" | head -3)"
    if [ -n "$hits" ]; then
      say "FAIL [$pattern] $file"
      printf '%s\n' "$hits" | sed 's/^/     /' | tee -a "$report"
      findings=$((findings + 1))
    fi
  done
done < <(files_list "${1:-}")

# 2. Optional gitleaks pass (secret material, entropy).
if command -v gitleaks >/dev/null 2>&1; then
  say "-- gitleaks --"
  if gitleaks dir --redact --no-banner --exit-code 7 . >>"$report" 2>&1; then
    say "PASS gitleaks"
  else
    rc=$?
    if [ "$rc" = "7" ]; then
      say "FAIL gitleaks reported findings (see $report)"
      findings=$((findings + 1))
    else
      say "SKIP gitleaks rc=$rc"
    fi
  fi
else
  say "-- gitleaks not installed, skipped (pattern scan above still enforced) --"
fi

if [ "$findings" -eq 0 ]; then
  say "RESULT: PASS (0 findings)"
  exit 0
fi
say "RESULT: FAIL ($findings groups) — fix before committing or pushing"
say "report: $report"
exit 1
