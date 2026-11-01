# SO101 preprocessing and invariants

The server loads `assets/<asset_id>/norm_stats.json`, computed from 16 training episodes; client never rescales. Images are uint8 HWC 480×640 and OpenPI `ResizeImages` pads/resizes to 224² consistently in training and inference.

All three camera views are required and fixed; no zero-image substitutions. State/action each have exactly 12 finite values; internal model padding to 32 dimensions is not robot semantics. Actions are absolute positions, never deltas or velocities. A checkpoint from another camera layout, unit set or control rate is invalid. Evaluate with the external physical task predicate, never policy output or end-effector proximity alone.

See the [SO101 interface contract](SO101_CONTRACT.md).
