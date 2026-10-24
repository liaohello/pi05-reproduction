# pi05-reproduction

**π0.5 Reproduction & Bimanual VLA Research** — reproducing and benchmarking π0.5, SmolVLA and X-VLA across LIBERO, RoboTwin, a MuJoCo dual-SO101 setup and real robots, covering expert data generation, LeRobot fine-tuning, RTC inference, evaluation and sim-to-real transfer.

This is a research repository, not a fork of the upstream projects. It keeps the third-party
simulators and model code out of Git, pins them by content hash and image digest, and adds only
the adapters, protocols, evaluation harness and experiment records needed to compare models
under one protocol.

> Status: **early Phase 1**. One inference pipeline is verified end to end; nothing here is a
> published benchmark number yet.

## Scope and honesty of claims

The project separates three layers of claims, and the README never mixes them:

| Layer | Claim | Current state |
| --- | --- | --- |
| **Baseline Reproduction** | Re-run public checkpoints and evaluate them under a fixed protocol | Inference path verified; full evaluation not run |
| **Method Experiments** | Compare action chunking, RTC, expert-data scale and generalisation | Not started |
| **Robot Adaptation** | MuJoCo dual SO101 and real-robot validation | Not started |

## Phases

| Phase | Content | Status |
| --- | --- | --- |
| **Phase 1 — LIBERO Benchmark** | π0.5 official baseline, fine-tuning, checkpoints, simulation evaluation | **P1.0 inference smoke test done**; P1.1 full evaluation and P1.2 training reproduction pending |
| **Phase 2 — RoboTwin Benchmark** | Bimanual expert data, π0.5 / SmolVLA / X-VLA comparison, generalisation | Roadmap only |
| **Phase 3 — MuJoCo Dual-SO101** | Automatic expert trajectories, LeRobot conversion, bimanual fine-tuning, RTC execution | Roadmap only |
| **Phase 4 — Real-World Sim-to-Real** | SO101 data collection, real-scene fine-tuning, deployment, sim-vs-real gap analysis | Roadmap only |

Milestones and gates are defined in [`docs/roadmap.md`](docs/roadmap.md).

### What P1.0 actually is

Four LIBERO suites were run once each (task 0, one rollout per suite) against the public
`pi05_libero` checkpoint, in a closed loop: instruction → policy server → simulator → success
check → replay video. All four episodes succeeded.

> `4/4 smoke-test episodes succeeded; not a benchmark success-rate estimate.`

Step counts: `libero_spatial` 77, `libero_object` 138, `libero_goal` 115, `libero_10` 280.
Fixed protocol: `seed=7`, `replan_steps=10`, 10 settle steps, 256×256 rendering, 224 policy input,
camera rotated 180°. Details and caveats: [`reports/phase1-inference-smoke.md`](reports/phase1-inference-smoke.md).

## Repository layout

```
configs/          protocol definitions and normalisation statistics
  libero/         evaluation protocol + norm_stats.json
environments/     adapter notes per simulator (no upstream code is vendored)
  libero/
inference/        observation/action contract shared by all models
evaluation/       offline, simulation and real-robot evaluation
  simulation/     LIBERO episode runner and environment probe
experiments/      append-only ledger of runs and phase gates
reports/          human-readable experiment reports and public evidence
scripts/          deploy, tracking, verification and security tooling
third_party/      provenance of pinned third-party source (code itself is not committed)
docs/             roadmap, Phase 1 guide, migration notes, reading guide
```

## Quick start

Deployment targets are resolved from environment variables (or a git-ignored
`scripts/deploy/config.local.json`); no host, port, user or key path is stored in this repository.

```bash
cp configs/hosts/example.env .env.local     # fill in your own machine
set -a && source .env.local && set +a

./scripts/run.sh preflight            # config, source lock, git, disk
./scripts/run.sh preflight-remote     # ssh, workdir, images, checkpoint, GPU
./scripts/run.sh deploy               # sync this repository to the remote workdir
./scripts/run.sh verify               # remote must equal local byte for byte
./scripts/run.sh smoke --suite libero_spatial --tasks 0 --episodes 1
./scripts/run.sh status               # phase gates + ledger summary
./scripts/run.sh release              # give the GPU back when finished
```

The remote side needs two pinned images (`openpi-server`, `openpi-client`) and a `pi05_libero`
checkpoint. See [`third_party/PROVENANCE.md`](third_party/PROVENANCE.md) for what they must
contain and how to reproduce the same content.

## Reproducibility

Everything needed to re-create an experiment is recorded per run in
[`experiments/ledger.jsonl`](experiments/ledger.jsonl): policy config, checkpoint id, both image
digests, `norm_stats` hash, git revision, protocol parameters, command, metrics and artifact
paths (private prefixes replaced with placeholders).

Third-party source is not vendored. Its provenance — file counts, byte totals, aggregate SHA-256
and image digests — is published in `third_party/PROVENANCE.md`; a full per-file manifest can be
regenerated from the images and compared against the aggregate.

## Security policy

This is a public repository and the deployment target is private. Before every commit:

```bash
./scripts/run.sh scan        # or: git config core.hooksPath .githooks
```

The gate blocks real IP addresses, host names, ports, user names, private absolute paths,
key material, tokens and registry addresses. Rules and the full policy live in
[`AGENTS.md`](AGENTS.md). Never weaken the scan to make a commit pass.

## Licences

Original code here is MIT (see `LICENSE`). Third-party components keep their own licences:
OpenPI is Apache-2.0 (plus `LICENSE_GEMMA.txt`), LIBERO code is MIT and its datasets are
CC BY 4.0. See [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md) for details and obligations.
Model weights and datasets are not redistributed here.

## Not affiliated

This is an independent reproduction effort. π0.5, OpenPI, LIBERO, RoboTwin and LeRobot are
projects of their respective authors; this repository is not endorsed by them.
