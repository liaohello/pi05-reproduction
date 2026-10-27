# SO101 inference contract

This contract is separate from the LIBERO contract in [`CONTRACT.md`](CONTRACT.md). It does not
change, reinterpret or extend the LIBERO 8-value state / 7-DoF end-effector delta contract.

## Robot interface (dataset, policy server and rollout share it)

| Field | Type | Shape | Meaning |
| --- | --- | --- | --- |
| `observation.state` | `float32` | `(12,)` | left arm six, then right arm six |
| `observation.images.central` | `uint8` | `(H, W, 3)` | fixed third-person RGB, H×W = 480×640 |
| `observation.images.left_wrist` | `uint8` | `(H, W, 3)` | left wrist RGB |
| `observation.images.right_wrist` | `uint8` | `(H, W, 3)` | right wrist RGB |
| `prompt` | `str` | — | task instruction |

Per arm the six values are `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`,
`wrist_roll` and `gripper`. The first five use degrees, the gripper uses percent (0 = closed,
100 = open). Both state and action use the same order and units.

## Policy output

| Field | Type | Shape | Meaning |
| --- | --- | --- | --- |
| `actions` | `float32` | `(10, 12)` | absolute joint-position targets, same order/units as state |
| `policy_timing` | `dict` | — | optional server-side latency |

The executor applies one row per 50 Hz control period, requests a new chunk after
`replan_steps = 10` actions, and clips every value to the actuator limits. The clip is owned by
the single environment stepper (`PolicyRunner`); the policy never writes actuators directly.

## Dataset label rule

For control frame `t`, a training row contains:

* observation at `t` **before** the step (RGB + measured joint state), and
* the actuator target that is actually applied at `t` after the external robot layer clamps it.

The next measured state is never used as the label. Object poses, contacts, expert stage and
success labels are diagnostics only and are not policy inputs.

## Normalisation

The policy server owns state/action normalisation. It loads exactly the `assets/<asset_id>/norm_stats.json`
written by the training run, which was computed from the 16 training episodes only. The client
never rescales actions or states.

## Time base

* physics: 500 Hz, ten physics substeps per control frame
* control/recording: 50 Hz
* one action row advances exactly one control frame (ten physics substeps); a contact stop may end the last frame after fewer substeps
* nominal 50 Hz is a configuration value, not a claim of real-time wall-clock performance

## Images

Images are stored uint8 HWC at 480×640. Policy preprocessing applies OpenPI `ResizeImages`
(pad to square, 224×224) consistently in training and inference. Rotation, camera assignment
and resize are fixed by this contract; changing any of them requires a new dataset and checkpoint.

## Invariants

1. Three cameras are required; no camera may be replaced by a zero image or another view.
2. State and action are exactly 12 finite values; padding to the model's internal 32 dimensions
   happens inside the model pipeline and is never exposed as robot semantics.
3. The action is absolute, not a delta and not a velocity.
4. A checkpoint from one camera layout, unit set or control rate is invalid for another.
5. Rollout success is judged by the external physical task predicate, never by the policy or by
   end-effector proximity alone.
