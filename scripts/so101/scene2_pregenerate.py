#!/usr/bin/env python3
"""Freeze the four pilot layouts and their eight counterfactual conditions.

Nothing is executed here.  The output file is written once and then becomes the
immutable scene registry used by the physical pilot.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene2_common as s2  # noqa: E402


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--protocol", type=Path,
                        default=s2.REPO_ROOT / "configs/so101/scene2_dual_random_protocol.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = s2.configure(args.root)
    protocol = s2.load_protocol(args.protocol)
    s2.set_protocol_id(protocol["protocol_id"])
    certificate_path = (root / protocol["source_binding"]["certificate"]).resolve()
    config = s2.load_config(root, protocol, certificate=certificate_path)
    certificate, certificate_path = s2.load_certificate(config)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite frozen pilot scenes: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance = s2.provenance(root, args.protocol, config, certificate_path)

    settings = protocol["collection"]["physical_pilot"]
    seed_start, _ = protocol["collection"]["seed_registry"]["physical_pilot_layouts"]
    layouts_needed = int(settings["layouts"])
    candidates_max = int(settings["max_layout_candidates"])
    max_attempts = int(protocol["collection"]["layout_pilot"]["maximum_attempts_per_layout"])

    BimanualEnv = s2._b("BimanualEnv")
    certification = s2._b("certification")
    started = time.time()
    attempts, layouts = [], []
    with BimanualEnv(config, images=False) as env:
        stale = certification.check_certificate_current(certificate, env)
        if stale:
            raise SystemExit("certificate stale: " + "; ".join(stale))
        s2.install_runtime(config, env)
        wanted = {"red_positive_y": layouts_needed // 2,
                  "red_negative_y": layouts_needed - layouts_needed // 2}
        got = {"red_positive_y": 0, "red_negative_y": 0}
        for offset in range(candidates_max):
            if all(got[key] >= wanted[key] for key in wanted):
                break
            seed = int(seed_start) + offset
            arrangement = "red_positive_y" if offset % 2 == 0 else "red_negative_y"
            if got[arrangement] >= wanted[arrangement]:
                attempts.append({"seed": seed, "arrangement": arrangement,
                                 "status": "skipped_quota_full"})
                continue
            trace = []
            started_seed = time.time()
            try:
                layout = s2.generate_layout(
                    config, certificate, seed, arrangement, env=env,
                    provenance_hashes=provenance, max_attempts=max_attempts, trace=trace)
            except s2._b("SceneGenerationError") as error:
                attempts.append({
                    "seed": seed, "arrangement": arrangement, "status": "rejected",
                    "reason": getattr(error, "message", str(error))[:400],
                    "attempts": len(trace), "wall_time_s": time.time() - started_seed})
                continue
            got[arrangement] += 1
            layouts.append(layout)
            attempts.append({
                "seed": seed, "arrangement": arrangement, "status": "accepted",
                "physical_sha256": layout["physical_sha256"],
                "layout_sha256": layout["layout_sha256"],
                "sampler_attempts": layout["checks"]["sampler_attempts"],
                "wall_time_s": time.time() - started_seed})
    scenes = {}
    for layout in layouts:
        scenes[layout["physical_sha256"]] = {}
        for color in ("red", "blue"):
            scene = s2.build_scene(config, layout, color)
            scenes[layout["physical_sha256"]][color] = scene.as_dict()
    payload = {
        "schema_version": 1,
        "created": _now(),
        "phase": "pilot_scene_freeze",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": s2.common.sha256_file(args.protocol),
        "profile_sha256": provenance["profile_sha256"],
        "adapter_sha256": provenance["adapter_sha256"],
        "certificate_sha256": provenance["certificate_sha256"],
        "source_fingerprint": {key: provenance[key] for key in
                               ("source_sha256", "source_files", "expert_sha256",
                                "model_sha256", "config_sha256", "packages")},
        "seed_registry": protocol["collection"]["seed_registry"],
        "layouts_needed": layouts_needed,
        "layouts_accepted": len(layouts),
        "arrangement_balance": got,
        "complete": len(layouts) == layouts_needed,
        "layouts": layouts,
        "scenes": scenes,
        "attempts": attempts,
        "wall_time_s": time.time() - started,
        "note": "four layouts x two instructions = eight condition episodes; not eight independent scenes",
    }
    if not payload["complete"]:
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise SystemExit(f"only {len(layouts)}/{layouts_needed} layouts were accepted; registry written for audit")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items()
                      if key not in ("layouts", "scenes", "attempts")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
