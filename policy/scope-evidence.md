# Scope and evidence integrity

Binding rules from AGENTS.md.

## Scope

Migrate and extend only what is verified today: the Phase 1 LIBERO inference baseline, its scripts, configuration templates and sanitised experiment evidence. RoboTwin, dual-SO101, RTC and multi-model abstractions stay in [`docs/roadmap.md`](../docs/roadmap.md) until a phase starts. Do not create empty implementation directories that imply unsupported capabilities.

## Evidence integrity

- P1.0 is an **inference smoke test**, recorded as such.
- Never phrase smoke results as a success rate. Canonical sentence: `4/4 smoke-test episodes succeeded; not a benchmark success-rate estimate.`
- M0 / Phase 1 stays **in progress** until a full protocol evaluation and a training reproduction exist.
- Every number in `reports/` must be reproducible from a run in `experiments/ledger.jsonl`.
- Every report states its reproducibility level (R1/R2/R3, see [reproducibility](reproducibility.md)). A result below R3 is written as reproduce-pending, never as reproduced.
