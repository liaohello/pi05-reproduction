# Simulation evaluation

Runs LIBERO against a policy server. See [environment](../../environments/libero/README.md) and [inference](../../inference/CONTRACT.md).

```bash
python3 evaluation/simulation/libero_episode_runner.py \
  --host <policy-container> --port 8000 --suite libero_spatial \
  --task-ids 0 --episodes 1 --seed 7 --replan-steps 10 \
  --output <run-directory>
```

Task ids accept `all` or comma lists; episodes use distinct stored initial states. Seed applies before scene construction. Output includes `evaluation.json`, videos, config, protocol and per-episode diagnostics. Review JSON for private paths before publishing; ledger uses placeholders. Smoke results are not benchmark success-rate estimates.

`probe_env.py --output <dir>` checks rendering/assets without loading policy. Runner derives from OpenPI `examples/libero/main.py` (Apache-2.0); changes add task selection, seed, timing JSON and episode videos. See [licence](../../THIRD_PARTY_LICENSES.md).
