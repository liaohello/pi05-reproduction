# Adaptation and derived-data notes

The adapted [`libero_episode_runner.py`](../evaluation/simulation/libero_episode_runner.py) derives from OpenPI `examples/libero/main.py`; changes add task selection, seeding, JSON timing and episode videos. Its header and Apache-2.0 text are retained.

[`so101_policy_client.py`](../scripts/so101/so101_policy_client.py) implements compatible `_pack_array`/`_unpack_array` helpers from OpenPI `msgpack_numpy`; its header identifies the adaptation. Apache text: [`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt).

`configs/libero/norm_stats.json` is ~2 KB of checkpoint-derived scaling statistics. If upstream terms tighten, remove it and document regeneration. No OpenPI NOTICE is currently included; copy any future NOTICE unmodified into `third_party/`.

The E4-T patch is a diff against the external SO101 expert, not standalone original code; external terms apply. It contains changed hunks only. See [patch provenance](provenance-e4t.md).
