# D. Colour conditioning and arm selection

**Verdict: at step 2000 the policy has `NOT` learned colour→arm selection.
`VERIFIED` on two independent blue episodes, corroborated by 6/6 reports. The dataset is
`RULED_OUT` as the cause. The counterfactual probe itself is `BLOCKED`.**

This is the one substantive positive finding of the audit, and it rests on artifacts, not
on speculation.

---

## 1. The data are not the problem (`RULED_OUT`)

| Property | Value | Source |
|---|---|---|
| Episodes by colour | 32 red / 32 blue | `experiments/so101/scene1-preflight/data_audit.json:477-480` |
| Split | 52 train / 12 validation | `:498-500` |
| Train colour balance | 26 / 26 | consistent with the above |
| Prompt is per-episode, not a constant | `task_per_episode_matches_scene: true` | `:496` |
| Prompt template | `Put the {color} block into the {color} tray.` | `patches/so101_scene1_single_random.patch:18` |
| Arm mapping | `red → left`, `blue → right` | same patch, `:30-32` |
| Training loader emits the episode's own `task` string | `SO101EpisodeDataset.__getitem__` returns `"task": episode["task"]` | `scripts/so101/openpi_so101.py:326-331` |

The training prompt is therefore *not* the constant `TASK` string at
`openpi_so101.py:41`. Colour, tray, arm and prompt are balanced and consistent. Whatever is
broken, it is not the labels.

## 2. The closed loop says the conditional was ignored (`VERIFIED`, n=2 blue)

Episode 11 — **validation, blue**, a clean scene with a passing first-observation audit
(camera MAE ≈ 0.8–1.0 px, state error 0.0):

- blue cube at `(0.2025, −0.1208)`, untouched, `robot_contacts: []`
- stopped at frame 1011 after 20.6 s
- `stop_contacts[0]`: `left_jaw_collision_band_0` vs **`red_tray_1`** at `(0.295, 0.251)`

Episode 01 — **novel, blue**:

- blue cube at `(0.2017, −0.0464)`, untouched
- stopped at frame 405 after 8.5 s
- `stop_contacts[0]`: `left_jaw_collision_band_0` vs **`table`** at `(0.240, 0.100)`

Read the geometry. The red tray lives at `y ∈ [0.22, 0.28]`, the blue tray at
`y ∈ [−0.28, −0.22]` (`patches/…:33-40`). In both blue episodes the **left** jaw travelled
to positive `y` — the red half — and never touched the cube it was told to move. That is not
imprecision. That is the correct behaviour for a red instruction, executed in a blue scene.

Two episodes is a small n, and this audit will not report a rate. But the direction agrees
in both, and it agrees with the 6/6 report that all rollouts showed left-arm / red-tray
bias. A timing accident does not produce that.

## 3. Why this is not a generalisation story

`evaluation_2000/episodes/03_train_red_ep62` and `04_train_blue_ep13` are **training**
episodes and they failed too. Whatever is wrong survives memorisation of the training set.
That eliminates "new scene, unseen layout" as the explanation, which was one of the more
comfortable hypotheses available.

## 4. The cause is *not* yet identified

Balanced data + correct prompts + a policy that always answers "red/left" leaves three live
explanations, and this audit cannot separate them without a forward pass:

1. **Under-training.** 2000 of 25 000 updates; the language branch may still be dominated by
   a prior. Cheapest to test: run the counterfactual probe at step 2000 and at a later
   checkpoint, and see whether the arm-selection ratio moves.
2. **Language is not reaching the action expert.** With `discrete_state_input: true`, state
   is tokenised into the same prefix as the instruction; 32 state entries (12 real + 20
   padding zeros, see B §4) sit next to a short instruction. If attention is dominated by
   state tokens, the colour token can be effectively invisible.
3. **The colour is being read from the image, and the wrong tray is being read.** Both trays
   are present; a model that latches onto "tray" without binding "the matching one" produces
   exactly this failure. Note ep11 drove to the red tray in a scene whose *blue* tray was
   also present.

Only (1) is testable with what exists today.

## 5. Counterfactual probe (designed, `BLOCKED`)

Hold everything except the prompt fixed — same scene, same both-tray geometry, same robot
state, same sampled noise, same flow time — and swap only the colour token. Vary scene,
phase, seed, and weight tree.

Do **not** report mean |Δaction| over 10 steps; it is dominated by magnitude and cannot
distinguish "different arm" from "same arm, jitter". Report instead:

| Metric | Why |
|---|---|
| active-arm identification (which arm has larger integrated motion) | directly tests selection |
| sign of Δ motion in the inactive arm | catches the observed failure mode |
| cosine between predicted TCP displacement and the vector to the *named* cube | tests binding |
| gripper command at grasp phase | tests whether it commits to a grasp at all |
| fraction of scenes where the two prompts produce the *same* arm | a sensitivity rate, not a distance |

A finding of "insensitive" requires ≥ 8 scenes spanning both colours and ≥ 3 seeds, and must
hold for both trees. One counterfactual is an anecdote, and the task is right to forbid
concluding from it.

## 6. Reproduce

```bash
python scene1_root_cause_audit/tools/scene1_audit.py condition \
  --checkpoint runs/so101/scene1-pi05base-val50-20261029T184107Z-92c99192/formal/checkpoints/2000 \
  --train-view  runs/so101/scene1-pi05base-val50-20261029T184107Z-92c99192/data_view_v3 \
  --episodes 62,13,11,2 --pairs 8 --seeds 3 --tree current,ema \
  --out scene1_root_cause_audit/condition_probe.measured.json
```
