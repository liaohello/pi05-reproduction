# LIBERO environment adapter

LIBERO is used as published; this directory only records how it is wired up, because the
adapter is configuration plus a container contract, not a reimplementation.

## Runtime shape

| Role | Image | What it does |
| --- | --- | --- |
| policy server | `openpi-server` | loads `pi05_libero`, serves actions over WebSocket on port 8000, internal Docker network only |
| episode runner | `openpi-client` | builds the LIBERO scene, steps the simulator, checks success, writes video and JSON |

The repository is mounted at `/workspace` in both containers; the checkpoint is mounted
read-only at `/checkpoint` in the server. No GPU is requested by this repository's tooling
beyond the two containers above.

## Task construction (must not drift)

- Suite selection from `benchmark.get_benchmark_dict()`; suites used here:
  `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`.
- Each episode starts from the task's stored initial state, index = episode number.
- 10 settle steps with a dummy action before the first policy call.
- Camera images rotated 180° before being sent to the policy — this was part of training and is
  mandatory for reproducing behaviour.
- Rendering at 256×256, resized to 224×224 for the policy.
- Success is the LIBERO predicate for the task; the policy never sees it.

## Max steps per suite

`libero_spatial` 220, `libero_object` 280, `libero_goal` 300, `libero_10` 520. These are the
episode budgets used by the upstream example and are recorded per run.

## Offscreen rendering

`MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, `PYNPUT_BACKEND=dummy`, writable `HOME` and
`NUMBA_CACHE_DIR` inside the mounted workspace. Missing any of these turns into a silent
render failure or an interactive prompt in the container.

## Where the code lives

- Runner: [`evaluation/simulation/libero_episode_runner.py`](../../evaluation/simulation/libero_episode_runner.py)
- Environment probe: [`evaluation/simulation/probe_env.py`](../../evaluation/simulation/probe_env.py)
- Protocol constants: [`configs/libero/protocol.json`](../../configs/libero/protocol.json)
