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

## 6. SO101 branch (E0–E3) additional pinned content

The external dual-SO101 project is not committed. E0–E3 used an immutable content snapshot of the
reviewed tree, not the older deployed checkout:

| Item | Verified value |
| --- | --- |
| `source_sha256` (`src/bimanual/**/*.py`) | `201898ff7f01eac078ac708b27788f0360e227b02be4c4abae143b679e946fe2` |
| Source files | 109 |
| `expert_sha256` (control/experts/tasks/robots/sim) | `200dcd12513a31af15a1dc915526fd79b319924c2cb3ead2ae85f4e0e6e7421e` |
| Expert files | 33 |
| Config SHA-256 (26 YAML files) | `1ada3f568aae4da606946287657ce2b7248dcaf676bb129297adbc7aadfab859` |
| SO101 model SHA-256 | `6d18dad52363423314dde71d1e87d05d43e4ad5b71e9a723c6e766cd44bbb1c8` |
| External Python | 3.12, `mujoco==3.11.0`, `numpy==2.2.6`, `lerobot==0.6.1`, `av==15.1.0`, CPU PyTorch 2.11.0 |

Reproduce the fingerprint with `scripts/so101/source_fingerprint.py --root <external-checkout>
--python <external-python>`. The external project's own `provenance()` must report the same
`source_sha256` for every episode; `verify_collection` rejects mixed revisions.

The training/serving runtime is the already-pinned `openpi-server` image
(`sha256:405af52051682d5e13169d1de1523fd0adc116eceae4fc314cadfacd45b1ad52`). Its installed OpenPI
source has 55 files and aggregate SHA-256
`d8ee60bcf03530a14986374a5a33e00915929368cfe5f9fa613ebbd8ee02c186`. A host `openpi-main` copy
differed slightly (`20a1b543a434fc2c4b3496253a4865754d4cb96a0c00323fbadaea027ff2a658`, 497,543 vs
497,539 bytes) and was not used as the runtime.

Additional pinned content for this branch:

| Item | Verified value |
| --- | --- |
| `pi05_libero` checkpoint | 16 files, 12,439,085,481 bytes, aggregate `7c81971fafcdbc677b0e8fd25b3bffd3d624abe4635323784f1918abdcef6f15` |
| `pi05_libero` parameter tree | 51 tensors, tree SHA-256 `bd475f88ac7e9e9134e1283ca1eff5001159701556578b48e1c68b5f97c33aa3` |
| PaliGemma tokenizer | 4,264,023 bytes, SHA-256 `8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6` |

The checkpoint and tokenizer are not redistributed. The tokenizer is copied into a writable
`OPENPI_DATA_HOME` for offline runs; the image does not bundle it.

## E4-T patch against the external dual-SO101 expert

The mid-episode perturbation work does not vendor any external source. It ships one unified diff,
`patches/so101/e4t-mid-perturbation.patch` (currently 45,330 bytes, SHA-256
`0c134360c3a84cde582920542c290f5ae3d0d9fb6350275e4633f6b103575891`), applied with `patch -p0` to a
private copy of the pinned external checkout. The revision that was in force while the main shard
was collected is `a756fe169adeec6886c0e18358b3e39ff34ea471376c053f5e683c3b4ca38d38`; an earlier
pilot-stage revision measured 43,113 bytes (`a4cd614d0377905490403eaae18171371425d9554bc1d5099aab9df39f1e4d26`).
Both are recorded here because the dataset mixes snapshots:

| Item | Verified value |
| --- | --- |
| Baseline source aggregate (unpatched) | `201898ff7f01eac078ac708b27788f0360e227b02be4c4abae143b679e946fe2` |
| Baseline expert packages (unpatched) | `200dcd12513a31af15a1dc915526fd79b319924c2cb3ead2ae85f4e0e6e7421e` |
| Patched source aggregate (v13, 110 files) | `cceb2865cb767b375db063cc63e95c84597d1b9052ee848ecabc4c8a58866b97` |
| Patched expert packages (v13, 34 files) | `4c0ae2445fa980edfea515bd5615abb6944fa3340eb22e2f82d1b64820fae8ac` |
| Patched source aggregate (v17, main collection shard) | `d80a7ee755fb2bdf617a3651280e672714194f2dc52a4fef6102bfe0b40fcb67` |
| Patched source aggregate (v19, top-up shards, audits and replay) | `ec6ebbddc0704594e8b8602d1d7b0b0cb8ca533f59559e479f630e9350fb766a` |
| Patched expert packages (v17 and v19, 34 files) | `37694cfa6a6597206f268a57cc9e19515f6980e66ff6d66298ec9a56be2d3a90` |
| Files touched | `src/bimanual/control/motion.py`, `src/bimanual/control/mid_perturbation.py`, `src/bimanual/recording/episode.py`, `src/bimanual/recording/task_replay.py` |
| Certificate | rebuilt per patched snapshot; 237 left / 251 right pick points, geometrically identical to the E1 certificate |

Twenty-one private snapshots (v1–v21) were built while the pilot converged and while the
mechanisms were re-registered; every collection, audit and replay run records the aggregate hash of
the snapshot it actually used, and none of the snapshots is committed. The view binds both source
aggregates above (`collections` and `source_sha256` in `experiments/so101/e4t-data.json`). The patch is generated with
`diff -u` between the private baseline copy and the working copy, so it carries no upstream file
contents beyond the modified hunks.
