# Scene1 π0.5 dual-SO101 — root-cause audit

Run under audit: `scene1-pi05base-val50-20261029T184107Z-92c99192`
Checkpoint of record: `formal/checkpoints/2000` (`TrainState.step = 2000`)
Audit opened: 2026-10-30 (UTC) · diagnosis only · no training started · no production code changed

---

## 0. Execution status — read this first

**The diagnostic host cannot create processes.** Every `Bash` invocation fails before the
command runs:

```
/opt/WorkBuddy/.../shell-runtime-bash-env.sh: fork: retry: Resource temporarily unavailable
```

Kernel state sampled read-only at 2026-10-30T03:45Z:

| Signal | Value | Reading |
|---|---|---|
| `/proc/loadavg` | `4.70 4.22 3.61 5/6056` | 6,056 threads alive |
| `MemFree` | 711 MB of 23,987 MB | memory-saturated |
| `Committed_AS` | 69,686 MB vs `CommitLimit` 20,382 MB | 3.4× overcommitted |
| `SwapFree` | 1,332 MB of 8,388 MB | swap nearly exhausted |
| user-slice `pids.current` / `pids.max` | 5,564 / `max` | not a PID-limit hit |
| `RLIMIT_NPROC` | 93,272 | not a per-user limit hit |

So this is **not** a PID ceiling and **not** a permissions problem — `fork(2)` fails with
`EAGAIN` under memory pressure. `sandboxed` and `dangerouslyDisableSandbox` both fail
identically, so the sandbox is not the cause.

**Consequence:** no Python, no `ps`, no `systemctl`, no `docker`, no checkpoint load, no
GPU forward pass. Every measurement that requires executing model code is **BLOCKED**, and
is reported as such rather than estimated. Anything marked VERIFIED below was established
from *source code* or from *on-disk artifacts produced earlier by the run itself*; neither
requires the ability to fork.

## 1. Verdict legend

| Label | Meaning |
|---|---|
| `VERIFIED` | Proven by a cited line of code or by a hash-bound artifact in this repository. |
| `LIKELY` | Multiple independent lines of evidence agree; one alternative remains open. |
| `INCONCLUSIVE` | Evidence exists but does not discriminate between hypotheses. |
| `RULED_OUT` | A specific hypothesis was tested and failed. |
| `BLOCKED` | Requires execution; **not measured**, no substitute number given. |

## 2. Contents

| File | Question | Status |
|---|---|---|
| `process_restart_audit.md` | Is the trainer really stopped? Who restarts it? | **AUTO_RESTART_UNRESOLVED** |
| `effective_config_audit.json` | Did the effective config drift across resumes? | VERIFIED (files), live args BLOCKED |
| `checkpoint_integrity.json` | Is step 2000 complete, and does 3000 exist? | VERIFIED |
| `ema_vs_current.md` | Did EMA lag cause the closed-loop failure? | structure VERIFIED, **metrics BLOCKED** |
| `action_padding_loss.md` | Does 12→32 padding damage action learning? | mechanism VERIFIED, **gradients BLOCKED** |
| `phase_action_errors.md` | Where does the policy leave the expert? | **BLOCKED** (protocol ready) |
| `condition_probe.md` | Has colour→arm conditioning been learned? | **partial**: closed-loop evidence VERIFIED |
| `chunk_execution_audit.md` | Is the action→robot chain sound? | chain VERIFIED, **traces BLOCKED** |
| `final_diagnosis.md` | Q1–Q6 and the decision | — |
| `tools/scene1_audit.py` | Reproducible harness for every BLOCKED cell | **not executed** |
| `tools/restart_audit.sh` | Read-only process/scheduler enumeration | **not executed** |

## 3. What this audit refuses to do

- It does not convert "EMA = 0.999" into "EMA is the cause".
- It does not convert "20 of 32 dims are padding" into "62.5 % of training is wasted".
- It does not treat a falling training loss as evidence of closed-loop competence.
- It does not treat the six directed rollouts as a success-rate estimate.
- It does not fill a `BLOCKED` cell with an estimate. Empty means empty.

## 4. How to unblock

```bash
# 0. Make the host able to fork again (free memory / stop the leaking consumer).
#    Nothing below can run until `echo ok` succeeds in a shell.
cd <repository root>

# 1. is the run really stopped?
bash scene1_root_cause_audit/tools/restart_audit.sh \
  > scene1_root_cause_audit/process_restart_audit.raw.txt 2>&1

# 2. do the two weight trees even differ? (cheapest, highest information)
python scene1_root_cause_audit/tools/scene1_audit.py self-check \
  --checkpoint runs/so101/scene1-pi05base-val50-20261029T184107Z-92c99192/formal/checkpoints \
  --step 2000 --train-view runs/so101/scene1-pi05base-val50-20261029T184107Z-92c99192/data_view_v3

# 3. only then the measurements
python scene1_root_cause_audit/tools/scene1_audit.py ema  --checkpoint ... --train-view ...
python scene1_root_cause_audit/tools/scene1_audit.py padding --checkpoint ... --train-view ...
```

The harness has **never been executed** (it was written on the host that could not fork).
`self-check` is deliberately read-only and cheap: it loads the checkpoint, verifies both
trees exist, and prints the relative L2 distance between them. Run it before anything else,
and diff `_per_dim_loss` against `compute_loss` in `openpi/models/pi0.py` before trusting a
loss number it produces.

Until `restart_audit.sh` returns `AUTO_RESTART_RESOLVED`, the run must be treated as
**possibly live** and no checkpoint may be assumed stable.
