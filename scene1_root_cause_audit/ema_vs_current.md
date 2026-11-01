# A. EMA vs current parameters

**Verdict: `LIKELY` that the two trees differ materially; `RULED_OUT` that the deployed
tree is corrupt; `BLOCKED` on every number that would settle whether the difference
explains the closed-loop failure.**

---

## 1. The five hops, with citations

| # | Fact | Evidence |
|---|---|---|
| 1 | `TrainState` carries both `params` and `ema_params`, plus a static `ema_decay` | `openpi/training/utils.py:13-23` |
| 2 | `ema_decay = 0.999` was actually in force for both the initial and the resumed process | `formal/config_initial.json:74`, `formal/config_resume.json:74` |
| 3 | On save, **the `params` item is written from `ema_params`** whenever EMA exists; `train_state` keeps the live tree and stores `ema_params=None` | `openpi/training/checkpoints.py:145-152` |
| 4 | On restore, `_merge_params` re-attaches the saved `params` item as `ema_params` | `openpi/training/checkpoints.py:155-159` |
| 5 | Inference (`create_trained_policy`) loads `checkpoint_dir/"params"` — i.e. **the EMA tree** | `openpi/policies/policy_config.py:57` |
| 6 | Validation (`validation_batch_losses`) merges `state.params` — i.e. **the live tree** | `scripts/so101/scene1_validation.py:166`; declared at `:254` (`"weights_source": "TrainState.params"`) |
| 7 | Every validation record on disk says so in its own metadata | `formal/val_loss_full_2000.json:43` |

So the pipeline validates one set of weights and deploys another. That is not an inference
from the config file; hop 7 is a field the validation job wrote about itself.

## 2. Why the asymmetry is dangerous *here* (not in general)

An EMA is normally a free win: it denoises SGD noise. At step 2000 with `β = 0.999` the
effective averaging window is `1/(1-β) = 1000` optimizer updates, so the deployed weights
are approximately "the policy averaged over updates 1000–2000". Two properties of *this*
run make that lag unusually costly:

1. **The LR is still at its peak.** Warmup ends at step 1000 and cosine decay runs to
   step 25000 (`formal/config_resume.json:61-66`). At step 2000 the LR is ≈ `5e-5`, i.e.
   the single largest displacement-per-step in the whole schedule. An EMA trailing a
   fast-moving parameter is furthest behind precisely when the parameter is moving fastest.
2. **Only 2000 updates exist.** The EMA has seen 2000 samples of a 1000-step window; it is
   structurally a "step ~1000–1200" policy wearing a step-2000 label.

That is an argument that the gap is *non-trivial*. It is **not** an argument that the gap
causes the failure. Those are different claims and this audit keeps them apart.

## 3. What the rollouts already rule out

`RULED_OUT: the deployed tree is corrupt or zero-initialised.`

A zero- or garbage-initialised EMA would emit near-constant actions. Instead, in a **blue**
episode the left arm traversed ~1 m of table and struck the **red** tray within 20.6 s
(`evaluation_2000/episodes/05_validation_blue_ep11/closed_loop_result.json:94-108`).
Coherent, goal-directed, wrong-goal motion cannot come from a broken weight tree. It comes
from a working policy with the wrong conditional.

This matters: it removes the one EMA hypothesis that would have been catastrophic, and
leaves only the "lagging but functional" hypothesis, which is much weaker.

## 4. The measurement (not performed)

Both trees, same checkpoint, same windows, same noise, same flow time, same seed, same
unnormalisation — differing only in `state.params` vs `state.ema_params`:

| Metric | current (`params`) | EMA (`ema_params`) |
|---|---|---|
| Validation flow-matching loss, 32-dim mean | 0.008662 *(known, live tree only)* | **BLOCKED** |
| Real 12-dim loss | **BLOCKED** | **BLOCKED** |
| Padding 20-dim loss | **BLOCKED** | **BLOCKED** |
| Action MAE (normalised) | **BLOCKED** | **BLOCKED** |
| TCP position error (mm) | **BLOCKED** | **BLOCKED** |
| TCP orientation error (deg) | **BLOCKED** | **BLOCKED** |
| Gripper error | **BLOCKED** | **BLOCKED** |

The single known number in that table is the live-tree validation loss at step 2000. It is
reproduced here only to fix the reference point; it is **not** an EMA result.

Why all cells are blocked: `scene1_validation.py:166` hard-codes `state.params`, and
`register_plan(..., fresh=False)` refuses to run unless the saved plan is byte-identical
(`scene1_validation.py:284-294`) — a plan that declares `weights_source: "TrainState.params"`.
An EMA evaluation therefore *cannot* be produced by the production validator without
violating its own contract. It has to be a separate diagnostic path, which is what
`tools/scene1_audit.py ema` implements.

## 5. Decision rule (fixed in advance, so the result cannot be reinterpreted afterwards)

Let `Δ = L_ema − L_current` on the 12 real dims, with a paired bootstrap over the 8300
windows (1000 resamples, seed 42).

| Outcome | Reading |
|---|---|
| `Δ ≤ 0` (EMA equal or better) | EMA is **not** a contributor → `RULED_OUT`; the failure lies in conditioning, phase behaviour, or the execution chain. |
| `0 < Δ`, CI excludes 0, but `Δ` small relative to the current-vs-expert MAE | EMA is a **second-order** contributor → at most `INCONCLUSIVE`, do not act on it. |
| `0 < Δ`, CI excludes 0, and `Δ` large enough to cross a phase boundary (e.g. pregrasp → grasp) | EMA is a **first-order** contributor → `LIKELY`, and option C in `final_diagnosis.md` becomes admissible. |

## 6. One more thing the resume path does not protect

`train_so101.py:119-145` pins model scope and `state_mode` on resume but **not**
`ema_decay`. Because `params/` *is* the EMA tree, a resume with a different `--ema-decay`
silently redefines what gets deployed, with no hash break and no error. In this run both
snapshots carry 0.999, so it did not happen — but the guard is missing, and the next resume
is one forgotten flag away from changing the deployed policy definition. See
`effective_config_audit.json → resume_contract_gap`.

## 7. Reproduce

```bash
python scene1_root_cause_audit/tools/scene1_audit.py ema \
  --checkpoint runs/so101/scene1-pi05base-val50-20261029T184107Z-92c99192/formal/checkpoints/2000 \
  --train-view  runs/so101/scene1-pi05base-val50-20261029T184107Z-92c99192/data_view_v3 \
  --samples 512 --seed 42 \
  --out scene1_root_cause_audit/ema_vs_current.measured.json
```
