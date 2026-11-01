# SO101 inference contract

Separate from [LIBERO](CONTRACT.md); does not alter its state/action semantics.

State: float32 `(12,)`, left arm then right; each order is shoulder pan, shoulder lift, elbow flex, wrist flex, wrist roll, gripper. First five are degrees; gripper percent (0 closed, 100 open). Three uint8 HWC RGB images: central and both wrists, each 480×640. Prompt is task instruction.

Output: float32 `(10,12)` absolute joint targets in state order/units; optional `policy_timing`. Apply one row per 50 Hz control frame and request every 10 rows. `PolicyRunner` alone clips actuator limits; policy never writes actuators.

Time base, labels and preprocessing: [details](SO101_CONTRACT_DETAILS.md).
