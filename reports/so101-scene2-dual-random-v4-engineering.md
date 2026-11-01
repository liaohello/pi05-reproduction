# Scene2 v4 engineering notes

v4 narrows the v1/v3 success region: yaw 33-67; tray x 0.214-0.270; positive y 0.263-0.274; negative y -0.274 to -0.235; block-tray outer 0.105 m; v1 reference expert/slots; parked-arm/tray initial check. A preliminary episode with positive y 0.235 was excluded after arm_not_clear; the positive lower was raised to 0.263.

v4 19/24; 7 layouts both; 12/12 initial fingerprints; 5 failures (4 place_withdraw ik, 1 direct_carry ik). The gate 20/24 fails by one. The narrowed family plus the v1 expert is the best result (v1 4/8, v2 2/8, v3 3/8) but not enough for formal collection.

The remaining failure is the v1 expert withdraw/carry IK, not target containment or the distractor. Next: v5 v4 family + v3 per-scene release/withdraw search, or a shared tray/place redesign; do not register formal collection.

R3 local; reproduce-pending from a clean clone.
