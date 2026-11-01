# Reproducibility

Binding rules from AGENTS.md.

## Levels

Every claim about this project carries a level. Do not mix them.

- **R1 structural** — a fresh clone, with no private server, key, GPU or image, can read every protocol, config and recorded evidence file, and the pins are machine-readable.
- **R2 environmental** — the pinned images and source trees are present and match their recorded identities, so the environment can be rebuilt.
- **R3 numerical** — a recorded run can be re-executed and compared against the ledger.

## Rules

- R1 is the minimum acceptable state of this repository. `./scripts/run.sh repro doctor` must reach R1 on a clean clone. A commit that breaks R1 blocks release.
- Third-party source, weights and datasets stay uncommitted. **Uncommitted is not the same as unobtainable**: every external dependency needs a line in `configs/repro/manifest.json` carrying a digest, a SHA-256, or at minimum file count plus byte total plus aggregate digest.
- An acquisition path that cannot be pinned does not get documented as if it existed. Write `null` and `status: blocked`; `repro doctor` must report it as unresolved. Silent degradation is forbidden.
- Fill the manifest first, then the prose. Documentation cites the manifest; it never restates a pin that the manifest does not hold.
- Every report states its level. A number that is not R3 is labelled reproduce-pending, using the sentence form in [scope and evidence](scope-evidence.md).
- A registry address is never committed. Images are resolved through `PI05_REGISTRY` at run time.

## Forbidden

- Weakening a check, skipping a SHA-256, or flipping `blocked` to `pinned` without the bytes that justify it.
- Fetching unverified archives. `repro fetch-source` refuses an archive that has no pinned hash.
- Claiming external reproducibility for assets whose terms forbid redistribution. Say the level, not the wish.
