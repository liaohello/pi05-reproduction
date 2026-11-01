# Final diagnosis — Scene1 π0.5 dual-SO101, step 2000

Run `scene1-pi05base-val50-20261029T184107Z-92c99192` · checkpoint `2000` · diagnosis only

> **The host could not execute a single process during this audit** (`fork: EAGAIN`, see
> `README.md §0`). Everything below is either read from source, read from hash-bound run
> artifacts, or explicitly marked `BLOCKED`. No number in this document is estimated.

---

## Q1. Is there a definite training-code bug?

**`RULED_OUT` for the objective and the optimiser. Two real defects found, both in how the
run is *measured*, not in how it *learns*.**

Checked and clean:

| Item | Result |
|---|---|
| Loss definition | unmasked mean over 32 dims — intentional upstream behaviour, not a defect (`pi0.py:214`) |
| Pad order | normalise → pad; padded target is exactly 0 (`data_loader.py:186-189`, `transforms.py:328-337`) |
| Optimiser / clipping / LR schedule | standard AdamW + global-norm clip, schedule matches config (`optimizer.py:65-85`) |
| Train/resume config consistency | no drift across the two snapshots (`effective_config_audit.json`) |
| Checkpoint completeness | step 2000 has params + EMA + 60 optimiser tensors + step agreement (`checkpoint_integrity.json`) |

Defects that are real:

**D1 — validation and deployment use different weight trees.** `scene1_validation.py:166`
merges `state.params`; `policy_config.py:57` loads `params/`, which `_split_params`
(`checkpoints.py:145-152`) fills from `ema_params`. Every validation record says so itself
(`val_loss_full_2000.json:43`). Minimal fix, diagnostic only: parameterise `weights_source`
and emit both. **No retraining required.** Consequence today: the loss curve that justified
"training looks healthy" has never described the robot's weights.

**D2 — the resume contract does not pin `ema_decay`.** `train_so101.py:119-145` pins model
scope and `state_mode` only. Because `params/` *is* the EMA tree, a resume with a different
`--ema-decay` silently redefines the deployed policy. Not exercised in this run (both
snapshots carry 0.999), but one forgotten flag away.

**D3 (evaluation, not training) — the six rollouts are not one experiment.** ep01 ran on
`rollout_six`, ep05 on `rollout_six_v2`, with 9.9 vs 1.67 frames per request
(`chunk_execution_audit.md §2`). They must not be pooled.

**Not auditable:** `/opt/openpi/scripts/train.py` — the file that actually runs the loop —
is not archived in this repository (`train_so101.py:31-42` loads it from disk). EMA
initialisation, EMA update order, and the checkpoint gate live there. That is a
reproducibility gap under `policy/reproducibility.md`, independent of this failure.

## Q2. Is EMA an important contributor to the closed-loop failure?

**`INCONCLUSIVE` — and it cannot be raised above that without the measurement in
`ema_vs_current.md §4`.**

What is settled: the asymmetry exists (`VERIFIED`), the deployed tree is EMA (`VERIFIED`),
and the deployed tree is **not** corrupt — it produced 1 m of coherent goal-directed motion
in ep05, which a broken or zero-initialised EMA cannot do (`RULED_OUT`).

What is not settled: the sign and size of `L_ema − L_current`. The argument that EMA lags
("effective window 1000 updates, LR still at peak, so EMA ≈ a step-1000–1200 policy") makes
a material gap *plausible*; it does not make it causal. Notably, if the same red/left bias
appears under both trees in the prompt-swap probe, EMA is irrelevant to the observed failure
mode — that is the cheapest discriminating test available, and it is not the loss table.

## Q3. Does 32-dim padding damage action learning?

**`LIKELY` no — and the "62.5 % wasted" framing is `RULED_OUT` by analysis.**

For a padded dim, `a = 0` exactly, so `x_t = t·ε` and `u_t = ε`; given the input,
`ε = x_t/t` is deterministic. The padding objective is a trivially learnable copy, **not**
irreducible noise, and its loss can reach zero. With AdamW, a constant factor on the loss
does not change the update (`m/√v` is invariant), so dividing by 32 does not shrink the step
on the real dims.

What padding *does* break is the **metric**: `0.008662` is a 32-dim average with ~20
near-zero terms, so the real 12-dim loss is roughly `0.008662 × 32/12 ≈ 0.023` if padding has
converged. That is why a mediocre policy looked healthy on a dashboard. This must be fixed
before any further loss comparison, whatever else is decided.

The one live mechanism — gradient *variance* injected into shared parameters by 20
pure-noise channels — is `BLOCKED` (`action_padding_loss.md §5`). It is a candidate for
option D, not a finding.

## Q4. At which phase does the policy leave the expert?

**Per-phase numbers: `BLOCKED`. But the closed-loop record already excludes the late phases
for the two episodes examined.**

In both blue episodes the cube was **never touched** (`robot_contacts: []`, `contained:
false`, `sorted: false` after 8.5 s and 20.6 s). Grasp, lift, transport, place and release
therefore **never occurred**. `RULED_OUT`: this is not a contact-rich-manipulation failure,
not a placement-geometry failure, and not compounding drift after a good grasp.

The divergence begins **at or before reach**: the arm left for the wrong side of the table
on the first chunk and kept going. That is a goal-selection failure, not a tracking failure.
The phase table in `phase_action_errors.md` is still worth producing — but its expected
result is now "flat and small through reach, then meaningless", because the policy is being
asked the wrong question.

## Q5. Has colour conditioning / arm selection been learned?

**`NOT LEARNED` — `VERIFIED` on two independent blue episodes, and it is not a
generalisation problem.**

| Episode | Colour | Arm that moved | Where it ended | Cube |
|---|---|---|---|---|
| 05 (validation) | **blue** | left | `red_tray_1` at y = +0.251 | untouched at y = −0.121 |
| 01 (novel) | **blue** | left | table at y = +0.100 | untouched at y = −0.046 |

Red tray is at `y ∈ [0.22, 0.28]`, blue at `y ∈ [−0.28, −0.22]`. Both blue episodes drove
**left** and toward **+y**: the correct behaviour for a *red* instruction. Training episodes
(03, 04) failed as well, so memorisation does not save it.

The dataset is `RULED_OUT` as the cause: 32/32 colour balance, 52/12 split, per-episode
prompts verified against the scene (`data_audit.json:477-500`), `red→left / blue→right`
mapping consistent with the expert.

## Q6. What next?

**Recommended: option E — one more minimal, isolated diagnostic round. Nothing else is
admissible yet.**

Not A: no code bug was found that would change the learned function; "fix and retrain" would
burn 25 k updates on a hypothesis the audit did not support.
Not B: resuming is forbidden until `AUTO_RESTART_RESOLVED`, and resuming the same objective
cannot fix a conditional the model never learned at 8 % of schedule — though it *might* if the
cause is simply under-training, which E1 tests cheaply.
Not C: the EMA question is answerable for free inside E1; acting on it before measuring it
would repeat the original mistake (trusting a plausible mechanism instead of a number).
Not D: padding has not been shown to be causal; a masked-loss ablation now would confound a
method change with an unresolved conditioning failure.

| # | Action | Cost | Answers | Pass/fail criterion |
|---|---|---|---|---|
| **E0** | `restart_audit.sh` twice, 10 min apart; confirm checkpoint mtime stable | minutes | is the run stopped? | `AUTO_RESTART_RESOLVED` both times, else stop here |
| **E1** | prompt-swap probe, fixed scene/state/noise, **both trees**, ≥8 scenes × 3 seeds | ~1 GPU-hour, no training | Q2 + Q5 jointly | if arm-selection flips with the prompt → conditioning OK, failure is execution → go to E3; if it does not flip under **either** tree → EMA `RULED_OUT` for this failure, conditioning is the root cause → go to E2 |
| **E2** | start one blue episode **at pregrasp** (expert state) and run closed loop from there | ~0.5 GPU-hour | can it execute once it is in the right place? | success from pregrasp + failure from start ⇒ reach/goal-selection is the whole story |
| **E3** | re-score the six rollouts with `penetration − budget` distribution and chunk-id traces | ~0 | is the guard mis-calibrated? are the harnesses comparable? | if expert replays also reach 0.3–0.5 mm, the 0.5 mm budget must be recalibrated before any success statistic is quoted |

Only after E1–E3 may one of A–D be selected, and each carries a precondition:

| Option | Becomes admissible when | Risk | Verification |
|---|---|---|---|
| A fix + retrain | E1/E2 finds a deterministic defect | 25 k updates spent on a misdiagnosis | 12-dim loss **and** arm-selection rate both move |
| B resume | E1 shows prompt sensitivity at 2000 but weak; and restart is resolved | continues a possibly-correct trajectory; no new information about conditioning | arm-selection rate at 6 k/9 k |
| C EMA strategy | E1 shows a large, consistent `Δ` between trees | optimises the wrong tree; hides a conditioning failure | paired 12-dim loss + arm-selection rate |
| D masked loss | E1 rules out conditioning **and** `cos(∇L_real, ∇L_pad)` turns out large | method change confounded with the real bug | 12-dim loss only, never the 32-dim number |

## Stop condition

This audit stops here. It did not modify production code, did not start or resume training,
did not disable the collision guard, and did not overwrite any experiment. The next action
belongs to a human decision on E0–E3.
