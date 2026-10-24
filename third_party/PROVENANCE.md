# Third-party source provenance

The simulator and the model code are **not** committed. This file records what they must
contain, so a third party can verify that they are running the same content.

## 1. What the two images provide

| Image | Digest | Contents used here | Files | Bytes | Aggregate SHA-256 |
| --- | --- | --- | --- | --- | --- |
| `openpi-server` (inference/runtime) | `sha256:405af52051682d5e13169d1de1523fd0adc116eceae4fc314cadfacd45b1ad52` | OpenPI source at `/opt/openpi` | 140 | 1,606,857 | `e9a694731905fb9d22a56ca5267d71a8…` |
| `openpi-client` (LIBERO simulation) | `sha256:5dc5543f91a61a109181de1e1f1a50e771e40aaa1058207d1d5a3fab2bfb377a` | LIBERO source and assets at `/opt/openpi/third_party/libero` | 1,116 | 443,644,276 | `e6d16c4a17b0a56708bb550273b321ef…` |

Aggregate SHA-256 is computed over `"{relative_path}\0{size}\0{sha256}"` lines, sorted by path and
joined with `\n`. It is a single number that changes if any file is added, removed or modified.

LIBERO upstream commit for the source in the client image:
`8f1084e3132a39270c3a13ebe37270a43ece2a01`.

## 2. Why there is no upstream commit for OpenPI

The OpenPI tree in the server image came from a release archive rather than a Git checkout, so
there is no verifiable commit to pin. Pinning a `git` commit that cannot be checked would be
worse than pinning content. Hence: image digest + per-file hash + Python/package versions.

## 3. How to obtain equivalent content

Either of these is acceptable, as long as the resulting aggregate hash matches:

1. **Rebuild from upstream** at a revision whose OpenPI/LIBERO files hash to the aggregates above,
   installing the dependency set recorded in the image (see the repository's own lock files).
2. **Use your own images** and regenerate the manifest, accepting that your run is only
   comparable, not identical, to the one reported here.

```bash
# fingerprint a tree you already have
PI05_OPENPI_SRC=/path/to/openpi \
PI05_SOURCE_MANIFEST=/path/to/manifest.json \
python3 scripts/verify/source_lock.py

# fingerprint the images directly (read-only, no GPU, no model load)
python3 scripts/verify/source_lock.py --server
```

The full per-file manifest is intentionally git-ignored (`third_party/lock/`): it is machine
bookkeeping and can be regenerated. Only the aggregates are published, and they are enough to
detect any content drift.

## 4. Runtime versions

| Component | Version |
| --- | --- |
| Python (inference image) | 3.11.9 |
| Python (simulation image) | 3.8.20 |
| JAX / JAXLIB | 0.5.3 |
| Flax / Optax | 0.10.2 / 0.2.4 |
| PyTorch (inference / simulation) | 2.7.1 / 1.11.0+cu113 |
| MuJoCo / robosuite / NumPy (simulation) | 3.2.3 / 1.4.1 / 1.22.4 |

## 5. What is deliberately absent

- Model weights (≈12.4 GB) and datasets (32.5 GB LeRobot, 100 GB raw HDF5).
- The private registry these images were built into; no registry address is published.
- Any host, port, path or credential used by the authors' deployment.
