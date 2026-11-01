# Scene2 engineering notes

- External `SingleBlockPlaceTask` hard-codes `cube_0`; scene2 uses its own dual-block task and an expert wrapper for the sampled slot. External source unchanged.
- Pick descend grazes the target CAD band by 0.6-1.3 mm. Only the target's legal bands are admitted; distractor, trays, table and arm-arm keep the 0.5 mm guard.
- Scan: open 70, yaw [30,75]; 0/15 and 90 hit the tray wall. Not pilot data.
- Layout 240/240 is a pre-screen, not an expert success rate.
- Smoke: layout 0 red placed the target, release hit the red tray wall by 0.83 mm; layout 0 blue placed it, withdraw had no collision-free direction. Distractor drift ~3e-9 m.
- Discarded: one protocol-binding run with no completed episode; baseline seeds 21000-21004. See protocol amendments.
- Full attempts: [pilot](../experiments/so101/scene2-dual-random-pilot.json). Raw traces and videos are ignored.

R3 local; reproduce-pending from a clean clone.
