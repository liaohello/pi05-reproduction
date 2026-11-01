# Runtime fingerprint commands

```bash
PI05_OPENPI_SRC=/path/to/openpi \
PI05_SOURCE_MANIFEST=/path/to/manifest.json \
python3 scripts/verify/source_lock.py
python3 scripts/verify/source_lock.py --server
```

The server command fingerprints images read-only without GPU or model load. Full per-file manifests are ignored under `third_party/lock/`.

| Component | Version |
|---|---|
| Python inference / simulation | 3.11.9 / 3.8.20 |
| JAX / JAXLIB | 0.5.3 |
| Flax / Optax | 0.10.2 / 0.2.4 |
| PyTorch inference / simulation | 2.7.1 / 1.11.0+cu113 |
| MuJoCo / robosuite / NumPy | 3.2.3 / 1.4.1 / 1.22.4 |

Absent by design: model weights (~12.4 GB), datasets (32.5 GB LeRobot; 100 GB raw HDF5), private registry, deployment hosts/ports/credentials.

## Acquisition

[`configs/repro/manifest.json`](../configs/repro/manifest.json) is the machine-readable form of this page. Four fields are still null, and they are what keeps R2 out of reach for any machine but the one that built the images.

| Component | Pinned identity | Retrieval | Still missing |
|---|---|---|---|
| openpi-server | image id `sha256:405af520…` | registry via `PI05_REGISTRY` | no public mirror recorded |
| openpi-client | image id `sha256:5dc5543f…` | registry via `PI05_REGISTRY` | no public mirror recorded |
| OpenPI tree | 140 files / 1,606,857 B / `e9a694731905fb9d…` | image export | archive URL, archive SHA-256 |
| LIBERO tree | 1,116 files / 443,644,276 B / `e6d16c4a17b0a567…` | image export | archive URL, archive SHA-256 |
| π0.5 weights | ~12.4 GB, not redistributable | operator | — |

Fill `sources[].archive_url` and `sources[].archive_sha256` with the archive actually used, then re-run `./scripts/run.sh repro doctor`. Until then the fields stay `null` and the doctor reports them as unresolved. That is the intended state, not an oversight: a blank is a task, a guessed URL is a defect.
