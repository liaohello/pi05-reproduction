# SO101 scene2 dual-block reproduction

External dual-SO101 source is not vendored; bind an independent snapshot.

    R=$SO101_SOURCE_ROOT; P=$SO101_PYTHON; O=<run>
    export SO101_EGL_DEVICE=0
    $P scripts/so101/scene2_certify.py --root "$R" --output <cert>
    $P scripts/so101/scene2_layout_pilot.py --root "$R" --output "$O/layout"
    $P scripts/so101/scene2_settle_baseline.py --root "$R" --output "$O/baseline"
    $P scripts/so101/scene2_pregenerate.py --root "$R" --output "$O/pilot_scenes.json"
    $P scripts/so101/scene2_collect.py --root "$R" --scenes "$O/pilot_scenes.json" --output "$O/pilot"
    $P scripts/so101/scene2_audit.py --run "$O/pilot"

Hashes: source 908d940d, expert 113847db, model 6d18dad5, profile 260e65f6, certificate 11511f03, adapter e1f826b3, frozen protocol 5a1b72c4.

R3 local; external snapshot not redistributed; reproduce-pending on a clean clone.
