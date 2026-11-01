# LIBERO environment adapter

LIBERO is upstream; this page documents wiring. See [runner](../../evaluation/simulation/README.md), [contract](../../inference/CONTRACT.md) and [provenance](../../third_party/PROVENANCE.md).

`openpi-server` loads `pi05_libero` and serves WebSocket actions; `openpi-client` constructs scenes, steps simulation, checks success, writes JSON/video. Repository mounts at `/workspace`; checkpoint is read-only at `/checkpoint`.

Suites: spatial, object, goal, libero_10. Episode n uses stored initial state n; apply 10 dummy settle steps. Rotate images 180°, render at 256², resize to 224². Success uses LIBERO predicate. Budgets: 220 / 280 / 300 / 520 steps.

Set EGL `MUJOCO_GL`, `PYOPENGL_PLATFORM`, dummy `PYNPUT_BACKEND`, writable `HOME` and `NUMBA_CACHE_DIR`. See [runner](../../evaluation/simulation/libero_episode_runner.py), [probe](../../evaluation/simulation/probe_env.py), [protocol](../../configs/libero/protocol.json).
