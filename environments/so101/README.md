# SO101 environment binding

Dual-SO101 MuJoCo is external: source, meshes, caches and recordings are not vendored. This repo supplies adapters, training-view export, launchers and records. See [inference contract](../../inference/SO101_CONTRACT.md) and [fingerprints](../../third_party/PROVENANCE.md).

## Source identity

Set `SO101_SOURCE_ROOT` and `SO101_PYTHON` at runtime via ignored local config or environment, then run `scripts/so101/source_fingerprint.py`. No private absolute path belongs in tracked files. Fingerprints include source/expert/config hashes and package versions; each experiment must identify its tree. E1 episode provenance records the same fingerprint.

## Reused modules

External `bimanual.sim`/`robots`: scene/model; `experts`/`control`/`tasks`: scripted demos; LeRobot v3: dataset; `recording.episode`/`collection`: runs/audits; `policy_runner`: single-owner stepping.

Dataset conversion and claim boundaries: [details](README-details.md).
