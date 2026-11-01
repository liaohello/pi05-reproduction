# Scene2 v2 reproduction

v2 uses the v2 wrappers and protocol; the v1 files remain.

    export SO101_EGL_DEVICE=0
    R=$SO101_SOURCE_ROOT; P=$SO101_PYTHON; O=<run>
    Q=configs/so101/scene2_dual_random_protocol_v2.json
    $P scripts/so101/scene2_certify_v2.py --root "$R" --protocol "$Q" --output <cert_v2>
    $P scripts/so101/scene2_layout_pilot_v2.py --root "$R" --protocol "$Q" --output "$O/layout"
    $P scripts/so101/scene2_settle_baseline_v2.py --root "$R" --protocol "$Q" --output "$O/baseline"
    $P scripts/so101/scene2_pregenerate_v2.py --root "$R" --protocol "$Q" --output "$O/pilot_scenes.json"
    $P scripts/so101/scene2_collect_v2.py --root "$R" --protocol "$Q" --scenes "$O/pilot_scenes.json" --output "$O/pilot"
    $P scripts/so101/scene2_audit_v2.py --run "$O/pilot" --protocol "$Q"

Hashes: protocol ae7dbd06, profile 862e145b, certificate 154819d6, adapter c8b5d503, scenes 4edbd0cd. v1: protocol 5a1b72c4, adapter e1f826b3. R3 local; reproduce-pending.
