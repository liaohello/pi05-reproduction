# Simulation evaluation

## `libero_episode_runner.py`

Runs N episodes of one LIBERO suite against a running policy server and writes a machine-readable
report plus one replay video per episode.

```bash
python3 evaluation/simulation/libero_episode_runner.py \
  --host <policy-container> --port 8000 \
  --suite libero_spatial --task-ids 0 --episodes 1 \
  --seed 7 --replan-steps 10 \
  --output /workspace/runs/<run-id>
```

Options that matter for comparability:

| Flag | Default | Why it is explicit |
| --- | --- | --- |
| `--task-ids` | `0` | `all` runs the whole suite; a comma list runs a subset |
| `--episodes` | `1` | episodes per task, each starting from a different stored initial state |
| `--seed` | `7` | seeds NumPy before scene construction |
| `--replan-steps` | `10` | how many actions of each chunk are executed before re-planning |
| `--output` | required | everything for the run goes here: `evaluation.json`, videos, LIBERO config |

## Output schema (`evaluation.json`)

```json
{
  "status": "complete",
  "backend": "OpenPI",
  "suite": "libero_spatial",
  "task_ids": [0],
  "episodes_per_task": 1,
  "seed": 7,
  "replan_steps": 10,
  "settle_steps": 10,
  "max_steps": 220,
  "action_mode": "relative_end_effector",
  "camera_rotation_degrees": 180,
  "policy_image_size": 224,
  "render_image_size": 256,
  "policy_metadata": {"..." : "server-reported metadata"},
  "completed_episodes": 1,
  "successes": 1,
  "success_rate": 1.0,
  "episodes": [
    {
      "task_id": 0,
      "episode": 0,
      "initial_state_index": 0,
      "language": "pick up the black bowl ...",
      "success": true,
      "steps": 77,
      "duration_s": 24.9,
      "policy_calls": 8,
      "first_inference_s": 20.8,
      "mean_inference_s": 2.7,
      "video": "task_00_episode_000_success.mp4"
    }
  ]
}
```

Before publishing such a file, check it for private paths. The ledger records the same numbers
with placeholders instead of machine paths.

## `probe_env.py`

Verifies that a client container can actually render offscreen and find its LIBERO assets,
without loading a policy. Run it when the simulator misbehaves before blaming the model:

```bash
python3 evaluation/simulation/probe_env.py --output /workspace/runs/probe
```

## Derivation

`libero_episode_runner.py` is adapted from OpenPI's `examples/libero/main.py` (Apache-2.0).
Modifications: single-task selection, explicit seed, JSON reporting with timing fields,
per-episode success-named video capture, and no dependency on the example's CLI defaults.
