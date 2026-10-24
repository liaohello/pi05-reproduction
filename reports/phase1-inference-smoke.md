# Phase 1 · Inference smoke test (P1.0)

**Date:** 2026-10-23
**Status:** passed — 4/4 smoke-test episodes succeeded; **not** a benchmark success-rate estimate.

## What was tested

The full closed loop on the public `pi05_libero` checkpoint:

```
task instruction → policy server (WebSocket) → LIBERO simulator → success predicate → replay video
```

One episode per suite, task 0, one rollout each. This proves the plumbing (containers, mounting,
rendering, action execution, success checking, recording) is correct. It says nothing about
average performance: four episodes cannot estimate a success rate.

## Protocol

| Parameter | Value |
| --- | --- |
| Policy config / checkpoint | `pi05_libero` (16 files, ≈11.58 GB) |
| Seed | 7 |
| Replan steps | 10 |
| Settle steps | 10 |
| Policy input / render size | 224 / 256 |
| Camera rotation | 180° |
| Action mode | relative end-effector, 7-D |
| Images | `openpi-server` `405af520…`, `openpi-client` `5dc5543f…` |
| `norm_stats` SHA-256 | `b3a44bb2810436fb…` (matches checkpoint) |

## Results

| Suite | Task 0 instruction | Steps | Episode time | Policy calls | First / mean inference | Result |
| --- | --- | --- | --- | --- | --- | --- |
| `libero_spatial` | pick up the black bowl between the plate and the ramekin and place it on the plate | 77 | 24.9 s | 8 | 20.8 s / 2.74 s | success |
| `libero_object` | pick up the alphabet soup and place it in the basket | 138 | 27.0 s | 14 | 20.8 s / 1.63 s | success |
| `libero_goal` | open the middle drawer of the cabinet | 115 | 26.7 s | 12 | 21.7 s / 1.95 s | success |
| `libero_10` | put both the alphabet soup and the tomato sauce in the basket | 280 | 33.2 s | 28 | 20.5 s / 0.87 s | success |

Total: 4 episodes, 4 successes, 610 environment steps.

## Reading the timings honestly

The first inference of every process costs ~20 s — that is JAX compilation, not model latency.
Mean inference per policy call ranged 0.87–2.74 s because the GPUs were shared with other
workloads at ~99% utilisation during the runs. **These numbers are not throughput measurements**
and must not be quoted as inference speed.

## Evidence

- Ledger entries: `experiments/ledger.jsonl` (5 rows, `kind=smoke`, `phase=D3`): the four original suite smoke episodes and one migration regression rerun.
- Public video: `reports/assets/pi05_libero_smoke_grid.mp4` (2×2 grid, 512×600, 14 s, 558 KB).
  Each quadrant is one suite; shorter episodes hold their last frame.
- Raw per-episode videos and `evaluation.json` files stay on the experiment machine, outside Git.

The video was reviewed before commit: it contains only simulator frames and task labels — no
terminal, no paths, no connection details.

## Reproduce

```bash
./scripts/run.sh smoke --suite libero_spatial --tasks 0 --episodes 1
./scripts/run.sh status
```

`smoke.py` picks the GPU with the most free memory, refuses to start below a free-memory
threshold, records the run, and removes its containers afterwards.

## What this does not establish

1. **No success rate.** Four episodes, four suites, one task each.
2. **No training reproduction.** The checkpoint is the published one; Phase 1's training half
   needs the `pi05_base` weights (≈12.44 GB) that are not present in this environment.
3. **No RTC / async comparison.** The executor waits for every chunk; batching and latency
   hiding are Phase 3 work.
4. **No ACT baseline.** Comparison against ACT needs the same protocol on the same tasks.

## Next gates

| Gate | Requirement |
| --- | --- |
| P1.1 | 4 suites × 10 tasks × 50 rollouts = 2,000 episodes, protocol locked, per-suite SR/SPL reported |
| P1.2 | training reproduction from `pi05_base`, with data version and optimiser settings recorded |
| P1.3 | ACT under the identical protocol, one comparison report |
