# Scene2 v4 narrowed-family reproduction

    export SO101_EGL_DEVICE=0
    R=$SO101_SOURCE_ROOT; P=$SO101_PYTHON; O=<run>
    Q=configs/so101/scene2_dual_random_protocol_v4.json
    $P scripts/so101/scene2_certify_v4.py --root "$R" --protocol "$Q" --output <cert_v4>
    $P scripts/so101/scene2_layout_pilot_v4.py --root "$R" --protocol "$Q" --output "$O/layout"
    $P scripts/so101/scene2_settle_baseline_v4.py --root "$R" --protocol "$Q" --output "$O/baseline"
    $P scripts/so101/scene2_pregenerate_v4.py --root "$R" --protocol "$Q" --output "$O/pilot_scenes.json"
    $P scripts/so101/scene2_collect_v4.py --root "$R" --protocol "$Q" --scenes "$O/pilot_scenes.json" --output "$O/pilot"
    $P scripts/so101/scene2_audit_v4.py --run "$O/pilot" --protocol "$Q"

Hashes: protocol be908cb2, profile 5d9073db, certificate 1c004f24, adapter a6b9867b, scenes c32b2759. v4 19/24; 7 layouts both; gate 20/24 not met. R3 local; reproduce-pending.
