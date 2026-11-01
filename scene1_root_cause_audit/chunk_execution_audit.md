# E. Action chunk and closed-loop execution chain

**Verdict: the nominal chain is `VERIFIED` and self-consistent; the *observed* chain is not
homogeneous across the six episodes, and the contact guard is operating at the physics
solver's noise floor. Both are `LIKELY` contributors to "six failures" — neither explains
"wrong arm".**

---

## 1. Nominal chain, verified

```
expert obs → policy(data_transforms → Normalize → Pad → model)
           → 32-D action → Unnormalize → SO101Outputs[..., :12]
           → clip → controller @50 Hz → 10 physics substeps @500 Hz
```

| Property | Expected | Observed | Check |
|---|---|---|---|
| Chunk horizon | 10 | 10 (`action_horizon: 10`) | config |
| Control rate | 50 Hz | `control_tick_s` min/max = 0.01999999999 / 0.020000000006 | `data_audit.json:470-474` |
| Substeps / control frame | 10 | `executed_substeps: 4060` for `frames: 406` | ep01 |
| Chunk reuse | 10 frames / request | `41 requests` for `406 frames` → 9.9 | ep01 |
| 32 → 12 truncation happens **after** unnormalisation | yes | output transform order: `model_transforms.outputs → Unnormalize → data_transforms.outputs` (`policy_config.py:84-89`), truncation in `SO101Outputs` (`openpi_so101.py:91-96`) | code |
| Action-clip accounting | recorded | `clipped_frames` / `clipped_values` present | ep01: 0/0, ep05: 126/126 |

Episode 01 is a textbook-clean execution of the intended design: 406 frames from 41
requests, 10 substeps each, zero clipping. The chain is not structurally broken.

## 2. The six episodes are not one experiment (`VERIFIED`, n=2 checked)

| Episode | Output root | requests | frames | frames/request |
|---|---|---|---|---|
| 01 novel blue | `rollout_six/` | 41 | 406 | **9.9** |
| 05 validation blue | `rollout_six_v2/` | 605 | 1012 | **1.67** |

A ratio of 1.67 means the harness asked for a fresh 10-action chunk roughly every 1.7
control frames — it consumed ~17 % of each chunk and threw the rest away. If that is what
`rollout_six_v2` does, it is a **different controller** (receding-horizon k≈2, not k=10):
smoother in principle, but with 6× the number of chunk boundaries, 6× the opportunity for
boundary discontinuity, and a completely different disturbance spectrum.

Two hypotheses, not yet separable from disk:

1. `requests` counts something else in v2 (e.g. per-frame inference because a cache was
   disabled), and the chunking was unchanged;
2. v2 really re-planned every ~1.7 frames, because 126 clipped frames caused the queue to be
   flushed.

Decisive check: log `(frame_index, chunk_id, chunk_position)` per control step. Without it,
**the six-episode set must not be pooled**, and any statement of the form "5 of 6 failed the
same way" is comparing two harnesses. That alone may explain part of the "one episode ran to
the time limit while five hit the guard" split.

## 3. The contact guard fires at the solver's noise floor (`LIKELY`)

| Episode | penetration | budget | ratio | stop after |
|---|---|---|---|---|
| 01 | 0.5032 mm | 0.5000 mm | **1.006×** | 10 substeps |
| 05 | 0.7354 mm | 0.5000 mm | 1.47× | 6 substeps |

Episode 01 was killed by **6 micrometres** over budget — 0.6 % — with `partial_frame: false`
and zero clipped actions. A 0.5 mm penetration budget is at or below the contact-slop
tolerance typical of a 500 Hz MuJoCo/Isaac step. When the abort threshold and the solver's
own penetration allowance are the same order of magnitude, the guard stops measuring
"the robot hit the tray" and starts measuring "the solver took a normal step".

This does **not** rescue the policy: in ep05 the jaw was 30 cm from where it should have
been. But in ep01 the marginal trigger means the run was one solver iteration away from
continuing, and any success-rate or "contact rate" statistic computed from these rollouts is
contaminated by an abort criterion that is tighter than the controller's tracking error.

Recommended (not applied — the task forbids touching the guard): record
`penetration_m − budget` as a continuous distribution over *successful* expert replays. If
expert replays also reach 0.3–0.5 mm, the budget is mis-set and the guard needs recalibration
before it can be used as a failure classifier.

## 4. The latency instrumentation is not trustworthy (`VERIFIED`, but harmless)

`ep05`: `inference_ms_mean 11.78`, `inference_ms_p50 0.0083`, `p95 115.26`.

A p50 of 8 µs is not a 3 B-parameter forward pass; it is a dictionary lookup. The
distribution is bimodal — most calls returned instantly from a queue/cache, a minority paid
the real cost. Therefore:

- no argument about "the policy is too slow / timing causes drift" can be built on these
  numbers;
- the 115 ms p95 is the only plausible model-time figure in the record, and 115 ms against a
  20 ms control period is a **5.8× overrun**. If that p95 is real, the controller was
  periodically starved, which would produce exactly the kind of state divergence that makes
  an offline-good policy fail closed-loop.

This is the single most under-examined number in the run. It needs a timestamped latency
trace, not a summary.

## 5. Not measured (`BLOCKED`)

| Question | Why blocked |
|---|---|
| observation timestamp vs action-apply timestamp (delay) | no per-frame trace retained |
| chunk-boundary discontinuity (Δaction at frames 10, 20, …) | pre-clip raw actions were never logged — the gap the brief already flags |
| whether unnormalise uses 12-D or 32-D stats | `norm_stats.json` is not synced locally; inference ran, so the path is shape-compatible, but the convention was not verified |
| inactive-arm drift under a fixed observation | needs forward pass |

## 6. Reproduce

```bash
# once a trace exists
python scene1_root_cause_audit/tools/scene1_audit.py chunk \
  --trace runs/so101/.../evaluation_2000/.../trace.jsonl \
  --out scene1_root_cause_audit/chunk_execution.measured.json
```

Required trace fields per control step: `frame`, `t_obs`, `t_apply`, `chunk_id`,
`chunk_pos`, `raw_action[12]`, `clipped_action[12]`, `applied_action[12]`,
`gripper_cmd`, `tcp_pose`, `contacts`.
