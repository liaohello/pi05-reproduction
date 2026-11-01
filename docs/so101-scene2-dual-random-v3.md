# Scene2 v3 reproduction

v3 uses the v3 protocol, profile and wrappers; the baseline is the full v3 sampler script.

    export SO101_EGL_DEVICE=0
    R=$SO101_SOURCE_ROOT; P=$SO101_PYTHON; O=<run>
    Q=configs/so101/scene2_dual_random_protocol_v3.json
    $P scripts/so101/scene2_certify_v3.py --root "$R" --protocol "$Q" --output <cert_v3>
    $P scripts/so101/scene2_layout_pilot_v3.py --root "$R" --protocol "$Q" --output "$O/layout"
    $P scripts/so101/scene2_settle_baseline_v3.py --root "$R" --protocol "$Q" --output "$O/baseline"
    $P scripts/so101/scene2_pregenerate_v3.py --root "$R" --protocol "$Q" --output "$O/pilot_scenes.json"
    $P scripts/so101/scene2_collect_v3.py --root "$R" --protocol "$Q" --scenes "$O/pilot_scenes.json" --output "$O/pilot"
    $P scripts/so101/scene2_audit_v3.py --run "$O/pilot" --protocol "$Q"

Hashes: protocol 48117757, profile 58bc734e, certificate f593c042, adapter a63f6021, scenes 0a3d78e3. v1 4/8, v2 2/8, v3 3/8. R3 local; reproduce-pending.
