# Scene2 pilot data quality

8 episodes, 4503 control frames at 50 Hz; dt 0.020 s, no long/short/unexplained gaps. Three failure episodes end with terminal partial frames (7, 9 and 10 substeps); trace time plus partial substeps equals the summary sim time within 1e-14 s. applied == commanded exactly (max 0.0). State/action shapes (N,12), finite; joint jumps <=4.52 deg; gripper 70 percent at the open stage. Three cameras at 640x480, 50 fps; decoded frames equal trace frames; no central-camera duplicate run; wrist cameras show 2-3 frame repeats during slow carry/lower. Initial state hashes match 4/4. Target paths 0.15-0.218 m on successes; distractor net 0.0 m in all episodes. The recorded predicate and summary agree on all 8. Raw traces and videos stay in the ignored run directory; the tracked audit is [here](../experiments/so101/scene2-dual-random-audit.json).

R3 local; reproduce-pending from a clean clone.
