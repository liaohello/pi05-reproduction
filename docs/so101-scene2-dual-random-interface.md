# Scene2 data interface

- 50 Hz control, 500 Hz physics, 10 substeps per tick; raw sim time and wall-clock start/end. A synthetic index never proves no frames were dropped.
- state: 12 values, left six then right six; joints degrees, grippers percent; measured before the action.
- action: 12 absolute joint targets actually applied after clamping; never the next measured state.
- cameras: central, left_wrist, right_wrist; 640x480 RGB; before the action.
- objects: both blocks, position/quaternion/velocity per tick; stage, contacts, events, initial/final state.
- pilot: 4 layouts x 2 instructions = 8 condition episodes; pilot-only, never validation.
- family: physical_sha256; a layout's red/blue pair never crosses a split.
- chunk length: model line decides; raw trajectories are contiguous.
- formal: counts/split/norm later; no clean60/clean100 norm or checkpoint.
- raw evidence in the ignored run directory; tracked summaries in experiments/so101.
