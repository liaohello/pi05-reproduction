# SO101 source fingerprints

Dual-SO101 source is external. E0–E3 use the reviewed snapshot, not the older deployed checkout. Reproduce via `scripts/so101/source_fingerprint.py`; `provenance()` must match each episode or mixed-source collections fail verification.

| Item | Verified value |
|---|---|
| Python / packages | Python 3.12; mujoco 3.11.0; numpy 2.2.6; lerobot 0.6.1; av 15.1.0; CPU PyTorch 2.11.0 |
| Source SHA-256 / files (`src/bimanual/**/*.py`) | `201898ff7f01eac078ac708b27788f0360e227b02be4c4abae143b679e946fe2` / 109 |
| Expert SHA-256 / files | `200dcd12513a31af15a1dc915526fd79b319924c2cb3ead2ae85f4e0e6e7421e` / 33 |
| 26 YAML config SHA-256 | `1ada3f568aae4da606946287657ce2b7248dcaf676bb129297adbc7aadfab859` |
| SO101 model SHA-256 | `6d18dad52363423314dde71d1e87d05d43e4ad5b71e9a723c6e766cd44bbb1c8` |

E1 episodes store the fingerprint; it binds replay.

Runtime pin details: [SO101 OpenPI model runtime](provenance-so101-runtime.md).
