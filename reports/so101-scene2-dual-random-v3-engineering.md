# Scene2 v3 engineering notes

v3 adds a 0.10 m outer clearance and per-scene release/withdraw search; predicate and guard unchanged.

v3 3/8 (v1 4/8, v2 2/8). Layout 2 passed both; layout 3 red passed. Failures: 3 place_release at 0.50-0.57 mm tray wall; 2 place_withdraw where the fixed 60 mm vertical waypoint is unreachable for every direction (the search changes direction, not lift height). The release contact appears at the 500 Hz substep guard even when the control-frame static screen passed; the 40/50/30 openings do not clear this tray.

Conclusion: the tray inner half 0.055 and the gripper jaw envelope are the limiting geometry; more parameter search is not converging. The next change is a shared external tray/place redesign (larger inner clearance or lower release/approach) or a formal family restricted to the measured feasible subset. Do not start formal collection.

R3 local; reproduce-pending from a clean clone.
