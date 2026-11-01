# Scene2 v3 per-scene search pilot

Claim: v3 4 layouts x 2 instructions = 8 condition episodes; 3/8; 1 layout both; R3 local; reproduce-pending.

v3 changes: 0.10 m block-to-tray outer clearance; per-scene release open 40/50/30 and four withdraw directions at the actual pose. Predicate and 0.5 mm guard unchanged.

Result: 3/8 (v1 4/8, v2 2/8); 3 place_release at 0.50-0.57 mm tray wall; 2 place_withdraw where the fixed 60 mm vertical waypoint is unreachable. Layout 2 passed both. v3 is a partial per-scene success; v1 remains the reference.

Conclusion: parameter/search iterations plateau at 3-4/8; the tray inner clearance plus gripper envelope is structural. Next: redesign the tray/place geometry with the external model owner, or restrict the formal family to the measured feasible subset; re-pilot.

[v3 pilot](../experiments/so101/scene2-dual-random-v3-pilot.json) · [images](assets/so101-scene2-dual-random-v3/layout_02_red_vs_blue.png).
