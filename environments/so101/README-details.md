# SO101 data boundary and claims

External writer emits LeRobot v3; pinned OpenPI expects older LeRobot v2.1 API. `scripts/so101/export_training_view.py` verifies frame/action alignment and exports JPG, NPZ, split and train-only stats; it never edits `codebase_version`.

E0–E3 is an engineering chain: scripted demos, 100 π0.5 updates, checkpoint restore and 20-scene diagnostic. It is not formal training, a benchmark success rate, LIBERO P1.1/P1.2 completion or real-robot validation.

Runtime: Python 3.12, MuJoCo 3.11.0, LeRobot 0.6.1 and CPU PyTorch. Physics uses CPU and rendering uses EGL.

See the [environment binding](README.md) and [source fingerprint record](../../third_party/provenance-so101.md).
