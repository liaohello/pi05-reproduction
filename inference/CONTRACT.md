# LIBERO inference contract

Contract only; no multi-model abstraction exists. SO101 is separate ([contract](SO101_CONTRACT.md)).

## Request

One JSON observation per WebSocket call: `observation/image` and `observation/wrist_image` are uint8 `(224,224,3)` RGB; `observation/state` is float32 `(8,)` = end-effector position (3), axis-angle (3), gripper (2); `prompt` is benchmark instruction verbatim. Images may be nested lists or base64; semantics are fixed.

## Response

`actions` is float32 `(H,7)`: unnormalised 6-DoF relative end-effector delta plus gripper. Optional `policy_timing` reports server latency. Require `H >= replan_steps`; execute that many rows then request again (π0.5 defaults to 50).

Further [invariants and planned fields](CONTRACT_DETAILS.md).
