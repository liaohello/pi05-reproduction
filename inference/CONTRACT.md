# Inference contract

Defined now, implemented per model later. The point is that a policy server — π0.5 today,
SmolVLA or X-VLA in Phase 2 — is interchangeable as long as it honours this contract. No
abstraction layer exists yet, only this document and the WebSocket convention it describes.

## Request (client → server)

One observation per call, sent over WebSocket as JSON:

| Key | Type | Shape | Meaning |
| --- | --- | --- | --- |
| `observation/image` | `uint8` | `(224, 224, 3)` | third-person camera, RGB, already resized and rotated to match training |
| `observation/wrist_image` | `uint8` | `(224, 224, 3)` | wrist camera |
| `observation/state` | `float32` | `(8,)` | end-effector pose (3) + axis-angle (3) + gripper (2) |
| `prompt` | `str` | — | the task instruction, verbatim from the benchmark |

Images are sent as nested lists or base64 as the client library chooses; the contract fixes the
semantic content, not the encoding detail.

## Response (server → client)

| Key | Type | Shape | Meaning |
| --- | --- | --- | --- |
| `actions` | `float32` | `(H, 7)` | action chunk: 6-DoF delta end-effector + gripper, unnormalised to robot units |
| `policy_timing` | `dict` | — | optional server-side latency report |

The client executes the first `replan_steps` actions, then re-plans. `H` is model-specific
(π0.5 predicts a 50-step chunk by default); the executor is agnostic as long as
`H >= replan_steps`.

## Invariants

1. **Normalisation is the server's job.** Statistics come from the checkpoint's
   `norm_stats.json`; the client never rescales actions.
2. **Action semantics are fixed.** Relative end-effector deltas in the robot's control frame,
   gripper in `[-1, 1]`. Absolute joint control is out of contract.
3. **Time base is the environment's.** One action advances the simulator one control step
   (20 Hz in LIBERO); RTC in Phase 3 must preserve this mapping or declare a different one.
4. **No hidden state.** A server may cache an action chunk, but a request with identical
   observations must yield the same actions under the same seed.

## What Phase 2 and 3 will add

- A `model_id` and `protocol_version` field in the response, so comparison runs can assert
  which model produced which chunk.
- An asynchronous variant of the client loop for RTC (Phase 3); the request/response schema
  above stays unchanged.
