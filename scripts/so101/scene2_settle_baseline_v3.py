#!/usr/bin/env python3
"""V3 baseline: distractor drift on 12 v3 layouts.

Unlike the v1/v2 baseline this uses the actual v3 layout sampler (release and
withdraw search included), because the 0.10 m outer clearance makes one-draw
sampling too sparse.  The measured hold, threshold rule and success predicate
are unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene2_common_v3 as s2  # noqa: E402


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--protocol", type=Path,
                        default=s2.REPO_ROOT / "configs/so101/scene2_dual_random_protocol_v3.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=12)
    parser.add_argument("--seed-start", type=int, default=23800)
    parser.add_argument("--hold-s", type=float, default=2.0)
    args = parser.parse_args()

    root = s2.configure(args.root)
    protocol = s2.load_protocol(args.protocol)
    s2.set_protocol_id(protocol["protocol_id"])
    certificate_path = (root / protocol["source_binding"]["certificate"]).resolve()
    config = s2.load_config(root, protocol, certificate=certificate_path)
    certificate, certificate_path = s2.load_certificate(config)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to reuse existing baseline directory: {output}")
    output.mkdir(parents=True)
    provenance = s2.provenance(root, args.protocol, config, certificate_path)
    BimanualEnv = s2._b("BimanualEnv")
    certification = s2._b("certification")
    contacts = s2._b("contacts")
    mujoco = s2._b("mujoco")
    started = time.time()
    records, rejected = [], []
    with BimanualEnv(config, images=False) as env:
        stale = certification.check_certificate_current(certificate, env)
        if stale:
            raise SystemExit("certificate stale: " + "; ".join(stale))
        s2.install_runtime(config, env)
        for offset in range(args.scenes):
            seed = args.seed_start + offset
            arrangement = "red_positive_y" if offset % 2 == 0 else "red_negative_y"
            try:
                layout = s2.generate_layout(
                    config, certificate, seed, arrangement, env=env,
                    provenance_hashes=provenance, max_attempts=400)
            except s2._b("SceneGenerationError") as error:
                rejected.append({"seed": seed, "arrangement": arrangement,
                                 "reason": getattr(error, "message", str(error))[:300]})
                continue
            trays, blocks = layout["trays"], layout["blocks"]
            s2.set_scene_state(env, config, trays, blocks,
                               settle_s=float(config["runtime"]["settle_s"]))
            initial = s2.measure_initial(env, config, blocks)
            if not initial["passed"]:
                rejected.append({"seed": seed, "arrangement": arrangement,
                                 "reason": "initial_" + ",".join(
                                     name for name, ok in initial["checks"].items() if not ok)})
                continue
            baseline = {name: contacts.object_pose(env.model, env.data, name)[0].copy()
                        for name in ("cube_0", "cube_1")}
            max_displacement = {name: 0.0 for name in baseline}
            max_speed = {name: 0.0 for name in baseline}
            robot_contacts, false_grasp = [], []
            steps = round(float(args.hold_s) * float(config["runtime"]["physics_hz"]))
            for _ in range(steps):
                mujoco.mj_step(env.model, env.data)
                for name in baseline:
                    position = contacts.object_pose(env.model, env.data, name)[0]
                    max_displacement[name] = max(
                        max_displacement[name],
                        float(np.linalg.norm(position - baseline[name])))
                    linear, angular, _, _ = contacts.object_speed(env.model, env.data, name)
                    max_speed[name] = max(max_speed[name], max(linear, angular))
                    touching = contacts.robot_contacts_with(env.model, env.data, name)
                    if touching:
                        robot_contacts.append({"object": name, "contacts": touching[:2]})
                    if any(contacts.bilateral_grasp(env.model, env.data, name, arm)[0]
                           for arm in ("left", "right")):
                        false_grasp.append({"object": name})
            mujoco.mj_forward(env.model, env.data)
            records.append({
                "seed": seed, "arrangement": arrangement,
                "physical_sha256": layout["physical_sha256"],
                "layout_sha256": layout["layout_sha256"],
                "sampler_attempts": layout["checks"]["sampler_attempts"],
                "trays": trays, "blocks": blocks,
                "initial_checks": initial["checks"],
                "hold_s": float(args.hold_s),
                "max_displacement_m": {name: round(value, 9)
                                       for name, value in max_displacement.items()},
                "max_speed": {name: round(value, 9) for name, value in max_speed.items()},
                "robot_contact_samples": len(robot_contacts),
                "false_grasp_samples": len(false_grasp),
            })
    flat = [value for row in records for value in row["max_displacement_m"].values()]
    maximum = max(flat) if flat else None
    threshold = None if maximum is None else round(max(0.002, maximum + 0.001), 6)
    payload = {
        "schema_version": 1, "created": _now(),
        "phase": "distractor_settle_baseline_v3",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": s2.common.sha256_file(args.protocol),
        "certificate_sha256": provenance["certificate_sha256"],
        "adapter_sha256": provenance["adapter_sha256"],
        "source_sha256": provenance["source_sha256"],
        "seed_range": [args.seed_start, args.seed_start + args.scenes - 1],
        "scenes_requested": args.scenes, "scenes_measured": len(records),
        "rejected": rejected, "hold_s": float(args.hold_s),
        "max_hold_displacement_m": maximum,
        "threshold_rule": "max(0.002, measured_max + 0.001)",
        "threshold_m": threshold,
        "any_robot_contact": any(row["robot_contact_samples"] for row in records),
        "any_false_grasp": any(row["false_grasp_samples"] for row in records),
        "claims": "v3 layouts from the actual sampler; arms parked; no expert execution",
        "reproducibility": "R3 local; external snapshot pinned by hash and not redistributed; reproduce-pending from a clean clone",
        "records": records,
    }
    s2.common.write_json(output / "baseline.json", payload)
    print(json.dumps({key: value for key, value in payload.items()
                      if key not in ("records", "rejected")}, indent=2))
    return 0 if records and not payload["any_robot_contact"] and not payload["any_false_grasp"] else 1


if __name__ == "__main__":
    sys.exit(main())
