# E4-T patch provenance

`patches/so101/e4t-mid-perturbation.patch` is a 45,330-byte diff, SHA-256 `0c134360c3a84cde582920542c290f5ae3d0d9fb6350275e4633f6b103575891`, applied with `patch -p0` to a private pinned external checkout. It contains no complete upstream files.

The patch modifies `control/mid_perturbation.py`, `control/motion.py`, `recording/episode.py` and `recording/task_replay.py`. Certificate rebuilt per snapshot: 237 left / 251 right pick points, geometrically identical to E1. Patch licensing inherits external project terms; see [licence notes](licence-notes.md).

Patch generation compares private baseline and working copies. Twenty-one private snapshots (v1–v21) were built during pilot convergence; none is committed. Each run records its aggregate. Dataset view binds both hashes in `experiments/so101/e4t-data.json`. See [snapshot hashes](provenance-e4t-hashes.md).
