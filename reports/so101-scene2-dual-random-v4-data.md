# Scene2 v4 data quality

24 episodes, 16927 control frames at 50 Hz; dt 0.020 s, no long/short/unexplained gaps; no terminal partial frames. applied == commanded exactly (max 0.0); state/action finite; joint jumps <=5.31 deg; gripper 70 percent at open. Three cameras 640x480/50 fps; decoded frames equal trace frames; central no duplicate; wrist repeats 2-3 frames. Initial states 12/12. Target paths 0.154-0.236 m; distractor max 3.3e-9 m. Predicate and summary agree on all 24. Raw data ignored; the tracked audit is [here](../experiments/so101/scene2-dual-random-v4-audit.json).
