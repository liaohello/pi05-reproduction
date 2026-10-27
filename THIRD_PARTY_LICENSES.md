# Third-party licences and provenance

This repository's original code is MIT (see `LICENSE`). The OpenPI-derived runner remains under
Apache-2.0 and is accompanied by that licence. The project MIT licence does not replace the
licence on adapted third-party code. Model weights and datasets remain under their own terms and
are not included here.

| Component | Licence | How this repo uses it | Obligations |
| --- | --- | --- | --- |
| [OpenPI](https://github.com/Physical-Intelligence/openpi) | Apache-2.0; OpenPI also distributes separate Gemma terms | Runtime source is not vendored. `evaluation/simulation/libero_episode_runner.py` is adapted from `examples/libero/main.py` | The adapted runner retains Apache-2.0 terms; its full licence is in `third_party/licenses/Apache-2.0.txt`; its header identifies the adaptation and changes |
| [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | MIT (code), CC BY 4.0 (datasets) | Not vendored. Task definitions, initial states and assets are read from the simulator image | Attribution for code and dataset use |
| [Gym](https://github.com/openai/gym) / robosuite / MuJoCo | MIT / MIT / Apache-2.0 | Runtime dependencies inside the client image only | None for this repository, since they are not redistributed |
| π0.5 model weights (`pi05_libero`) | Terms of the upstream model release | Downloaded/kept outside the repository; identified by checkpoint id and `norm_stats` hash | Do not redistribute weights without checking upstream terms |
| `configs/libero/norm_stats.json` | Derived statistics from the public checkpoint | 2 KB of state/action normalisation statistics, committed so evaluation is verifiable | Treat as derived from the upstream checkpoint; see the note below |
| [FFmpeg](https://ffmpeg.org/legal.html) (local tooling only) | LGPL/GPL depending on build | Used locally to composite the public demo video, never vendored | None, since binaries are not distributed |
| [OpenPI](https://github.com/Physical-Intelligence/openpi) `msgpack_numpy` wire format | Apache-2.0 | `scripts/so101/so101_policy_client.py` implements a compatible packer/unpacker so the external SO101 environment can talk to the OpenPI policy server | The file header identifies the adaptation; the Apache-2.0 text is in `third_party/licenses/Apache-2.0.txt` |
| External dual-SO101 project | The external project's own terms; its SO101 URDF/mesh assets carry Apache-2.0 (`UPSTREAM_LICENSE`) | Imported at runtime through `SO101_SOURCE_ROOT`; no source, mesh, dataset or recording is copied into this repository | Do not redistribute the external project from here; retain its own notices when using it |
| [LeRobot](https://github.com/huggingface/lerobot) | Apache-2.0 | Runtime dataset writer/reader in the external SO101 environment; only the v3→training-view conversion is original code here | No LeRobot source is vendored |

## Notes

- `evaluation/simulation/libero_episode_runner.py` is derived from OpenPI's
  `examples/libero/main.py`; its header records the source and modifications (single-task
  selection, JSON reporting, deterministic seeding, per-episode video capture). The full Apache-2.0
  text is included at `third_party/licenses/Apache-2.0.txt`.
- `configs/libero/norm_stats.json` is a small set of normalisation statistics required to
  reproduce action scaling. It is treated as derived data from the upstream checkpoint. If the
  upstream terms tighten, remove the file and document how to regenerate it from the checkpoint.
- `scripts/so101/so101_policy_client.py` contains small `_pack_array`/`_unpack_array` functions
  compatible with OpenPI's `openpi_client.msgpack_numpy`; the adaptation and the changed context
  (SO101 client adapter) are noted in the file header.
- The external dual-SO101 code, meshes and recordings remain outside this repository. The public
  records only contain fingerprints, counts and sanitised diagnostics.
- `patches/so101/e4t-mid-perturbation.patch` is a **modification of that external dual-SO101
  expert** (adds `control/mid_perturbation.py` and touches `control/motion.py`,
  `recording/episode.py` and `recording/task_replay.py`). It is distributed as a diff only, inherits the external project's
  own terms, and is not original standalone code; the baseline and patched aggregate hashes are
  recorded in `third_party/PROVENANCE.md`. Nothing from the external repository is re-licensed here.
- No third-party `NOTICE` file currently accompanies OpenPI. If a future upstream revision adds
  one, it must be copied unmodified into `third_party/`.
