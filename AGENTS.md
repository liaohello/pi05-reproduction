# AGENTS.md — public repository policy

Binding rules for any agent or human working in this repository. They exist because the
repository is public while the machines it runs on are private.

## 1. Scope

Migrate and extend only what is verified today: the Phase 1 LIBERO inference baseline, the
scripts it needs, configuration templates and sanitised experiment evidence. RoboTwin,
dual-SO101, RTC and multi-model abstractions stay in `docs/roadmap.md` until a phase starts.
Do not create empty implementation directories that imply support that does not exist.

## 2. Evidence integrity

- P1.0 is an **inference smoke test**, recorded as such.
- Never phrase smoke results as a success rate. The canonical sentence is:
  `4/4 smoke-test episodes succeeded; not a benchmark success-rate estimate.`
- M0 / Phase 1 stays **in progress** until a full protocol evaluation and a training
  reproduction exist.
- Every number in `reports/` must be reproducible from a run in `experiments/ledger.jsonl`.

## 3. Source provenance

- Third-party source (OpenPI, LIBERO, simulators) is **not** committed.
- Record file counts, byte totals, aggregate SHA-256, image digests and how to obtain the
  source in `third_party/PROVENANCE.md`.
- Do not claim a git commit for code that was exported from an image and has no verifiable
  upstream revision.

## 4. Sensitive data protection

Forbidden in any tracked file, commit message, screenshot, video, log or report:

- public IPv4/IPv6 addresses, private domains, internal host names
- SSH or service ports, port mappings
- SSH user names, private absolute paths, workdir names
- SSH keys, passwords, tokens, cookies, registry credentials, private registry addresses

Use environment variables and the placeholders in `configs/hosts/example.env`. Real values live
only in `scripts/deploy/config.local.json` or the shell environment, both outside Git.

## 5. Clean migration

- The repository history starts fresh. Do not merge, cherry-pick or re-import the internal
  repository's history, and do not copy its `.git` directory.
- Migrate only files on the reviewed whitelist.
- Keep a written record of the source revision in `docs/migration-notes.md`.

## 6. Security gate

- `scripts/security/scan.sh` runs before every commit (`.githooks/pre-commit`) and before any
  push or release.
- Any unresolved real leak blocks publication. Never add a real secret to an allowlist.
- Scan reports stay in `security-reports/` (git-ignored) and must not quote raw credentials.

## 7. Evidence review

- Public videos, screenshots and evaluation JSON must be reviewed by a human before commit.
- Strip terminal output, connection details and deployment logs; keep the scientific content
  (task, steps, success, model id, seed).
- Raw episode recordings stay outside the repository.

## 8. Licence compliance

- Original code: MIT.
- Keep copyright notices of any adapted third-party file, mark modifications, and list the
  component, source and licence in `THIRD_PARTY_LICENSES.md`.
- Check the model weights' and datasets' own terms before redistributing anything derived
  from them. This repository does not redistribute weights or datasets.

## 9. Lean development

Implement the smallest thing the current phase needs. No framework-building ahead of results:
contracts may be defined (`inference/CONTRACT.md`), implementations may not.

## 10. Release gate

A public release requires, in order: security scan clean → licence check → migrated-file review →
current experiment regression green → tag. Until then work stays local.
