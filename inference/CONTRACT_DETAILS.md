# LIBERO inference invariants

Server normalises using checkpoint `norm_stats.json`; client never rescales. Gripper is `[-1,1]`; absolute joint control is out of contract. One row advances one environment step (LIBERO 20 Hz). Same observation and seed yield same actions; chunk caching is allowed.

Phase 2 may add `model_id` and `protocol_version`. Phase 3 may add asynchronous RTC while preserving schema or declaring its time mapping.

See the [request/response contract](CONTRACT.md).
