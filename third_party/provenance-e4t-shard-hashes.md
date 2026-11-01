# E4-T shard fingerprints

| Snapshot | Verified value |
|---|---|
| Patched source v17, main shard | `d80a7ee755fb2bdf617a3651280e672714194f2dc52a4fef6102bfe0b40fcb67` |
| Patched source v19, top-ups/audits/replay | `ec6ebbddc0704594e8b8602d1d7b0b0cb8ca533f59559e479f630e9350fb766a` |
| Patched experts v17/v19, 34 files | `37694cfa6a6597206f268a57cc9e19515f6980e66ff6d66298ec9a56be2d3a90` |

Twenty-one private snapshots (v1–v21) were created during pilot convergence and mechanism re-registration; none is committed. Collection, audit and replay records identify the actual aggregate hash. The dataset view binds source aggregates in `collections` and `source_sha256` (`experiments/so101/e4t-data.json`).
