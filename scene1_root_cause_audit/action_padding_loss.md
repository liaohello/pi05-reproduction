# B. 12 → 32 action padding and the flow-matching loss

**Verdict: the mechanism is `VERIFIED`; the "62.5 % of training is wasted" reading is
`RULED_OUT` by analysis; the residual concern (gradient noise on shared parameters) is
`BLOCKED` and must be measured, not argued.**

---

## 1. What the code actually does

| Step | Behaviour | Citation |
|---|---|---|
| Normalise first | `Normalize` runs **before** the model transforms, so it sees 12 real dims | `openpi/training/data_loader.py:186-189` |
| Pad second | `PadStatesAndActions` zero-pads `state` and `actions` along the last axis to `model_action_dim = 32`; no mask, no length metadata, no `action_dim` field | `openpi/transforms.py:328-337` |
| Loss | `compute_loss` draws noise over the **full** `actions.shape`, builds `x_t`, `u_t`, and returns `jnp.mean(jnp.square(v_t - u_t), axis=-1)` — a mean over all 32 dims, per horizon step | `openpi/models/pi0.py:189-214` |
| No masking anywhere | no mask argument, no `valid_dim`, no reduction weight | same |

Because padding happens *after* normalisation, the padded target is **exactly zero**, not
"zero in raw units mapped through a normaliser". That detail decides everything below.

## 2. What the padding objective really is

For a padded dimension, `a = 0` exactly, therefore:

```
x_t = t·ε + (1−t)·0 = t·ε
u_t = ε − 0        = ε
```

Given `x_t` and `t`, `ε = x_t / t` **deterministically**. So the Bayes-optimal velocity
prediction is `v̂ = x_t / t`, and the achievable loss on those dimensions is **zero**.

That single line invalidates the usual framing. The 20 padding dimensions are not an
irreducible-noise tax; they are a trivially learnable **copy task**. A model that has
learned it pays almost nothing for them.

## 3. Three claims that do not survive

**(a) "The loss is 62.5 % noise."** False. The target is a deterministic function of the
input. Padding loss → 0 as training proceeds.

**(b) "Only 12/32 of the gradient does useful work, so the real dims learn 2.67× slower."**
False for this optimiser. With `AdamW` (`optimizer.py:65-85`), a constant factor `c` applied
to the whole loss leaves the update direction unchanged and rescales `m` and `√v` together,
so `m/√v` is invariant. The real dims do not get a smaller step because the loss is divided
by 32. Only two second-order effects survive: the `eps = 1e-8` floor, and weight decay,
both negligible at this scale.

**(c) "The reported validation loss means what it says."** This one is true and is the real
damage. `0.008662` is an average over 32 dims of which 20 are near-zero once converged.
If padding has converged, the 12-dim loss is approximately `0.008662 × 32/12 ≈ 0.0231`;
if padding has *not* converged, the 12-dim loss is *lower* than that. Either way the number
in the dashboards is **not** the action-modelling error, and comparing it against a 12-dim
run from any other project is meaningless. This is a reporting defect, not a training bug —
but it is the defect that let "val loss looks fine" coexist with "every rollout fails".

## 4. What is genuinely lost (and is measurable)

| Cost | Size | Status |
|---|---|---|
| Output head widened 12 → 32 rows | ~20 × 1024 params, negligible | VERIFIED by arithmetic |
| 20 constant zero **state** entries padded to 32 before tokenisation | with `discrete_state_input: true` these enter the prefix alongside the 12 real state tokens | LIKELY (transform order + config flag); token count not verified |
| Pure-noise velocity targets fed into the suffix tokens | the model must learn to ignore 20 uninformative channels | mechanism VERIFIED, magnitude BLOCKED |
| Extra gradient variance on **shared** parameters (action expert, projection, LoRA) | the only mechanism that could slow real-dim learning | **BLOCKED** |

The fourth row is the only one that could matter, and it is exactly what the task asked to
be measured. It cannot be argued into existence from a dimension ratio.

## 5. The measurement (not performed)

On one fixed batch, with identical rng/noise/time, compute three losses and their gradients
w.r.t. the same parameters:

```
L_real  = mean over dims 0..11   of (v̂ − u)²
L_pad   = mean over dims 12..31  of (v̂ − u)²
L_total = mean over dims 0..31   of (v̂ − u)²     ← the production loss
```

Report, per parameter group (`*.llm.*_1.*` = action expert, `*action_out_proj*` /
`*action_in_proj*` = projection, `*lora*` = language LoRA — the same partition used by
`make_freeze_filter`, `openpi_so101.py:134-152`):

- `‖∇L_real‖`, `‖∇L_pad‖`, `‖∇L_total‖`
- `cos(∇L_real, ∇L_pad)` — if this is near 0, the padding objective is orthogonal noise on
  shared weights; if near 1, it is a redundant copy of the same signal
- share of `‖∇L_total‖²` attributable to each term
- the *loss values* `L_real`, `L_pad`, `L_total` on the same batch

Predicted, and to be falsified: `L_pad ≪ L_real` and `cos ≈ 0` once converged. If that
holds, padding is `RULED_OUT` as a cause and the only action needed is to fix the reported
metric (divide by 12, or mask). If `cos` is high and `‖∇L_pad‖` is comparable, padding is
injecting structured competition into the shared trunk and option D (masked-loss ablation)
becomes admissible.

## 6. Note on "just add a mask"

Adding a mask to `compute_loss` changes the training objective. That is a **method change**,
not a bug fix, and the task forbids it on the production path. It is correctly gated behind
option D in `final_diagnosis.md`, and it must be evaluated on the 12-dim loss, not on the
32-dim number it replaces.

## 7. Reproduce

```bash
python scene1_root_cause_audit/tools/scene1_audit.py padding \
  --checkpoint runs/so101/scene1-pi05base-val50-20261029T184107Z-92c99192/formal/checkpoints/2000 \
  --train-view  runs/so101/scene1-pi05base-val50-20261029T184107Z-92c99192/data_view_v3 \
  --batch 8 --seed 42 \
  --out scene1_root_cause_audit/action_padding_loss.measured.json
```
