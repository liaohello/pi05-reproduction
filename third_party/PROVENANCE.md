# Third-party source fingerprints

Simulator and model source are external, not committed. These pages document pinned content and retrieval. No git revision is claimed for source exported from an image.

- [Container images, source acquisition and runtime versions](provenance-runtime.md)
- [SO101 source and training runtime](provenance-so101.md)
- [E4-T patch and dataset snapshots](provenance-e4t.md)

The ignored `third_party/lock/` per-file manifest can be regenerated with `scripts/verify/source_lock.py`; published aggregates detect drift. The machine-readable pins these pages describe live in [`configs/repro/manifest.json`](../configs/repro/manifest.json), which `./scripts/run.sh repro doctor` reads.
