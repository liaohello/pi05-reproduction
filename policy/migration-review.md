# Migration and evidence review

Binding rules from AGENTS.md.

## Clean migration

Repository history starts fresh. Do not merge, cherry-pick or re-import the internal repository's history, and do not copy its `.git` directory. Migrate only files on the reviewed whitelist. Keep a written record of the source revision in [`docs/migration-notes.md`](../docs/migration-notes.md).

## Security gate

Run `scripts/security/scan.sh` before every commit (via `.githooks/pre-commit`) and before any push or release. Any unresolved real leak blocks publication. Never add a real secret to an allowlist. Scan reports stay in ignored `security-reports/` and must not quote raw credentials.

## Evidence review

A human must review public videos, screenshots and evaluation JSON before commit. Strip terminal output, connection details and deployment logs; retain scientific content (task, steps, success, model id, seed). Raw episode recordings stay outside the repository.
