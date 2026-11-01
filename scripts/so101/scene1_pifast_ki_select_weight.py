#!/usr/bin/env python3
"""Select and justify fast_ce_weight from the observed KI gradient matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

FLOW_GROUPS = {"action_expert_lora", "action_time_projections"}
FAST_GROUPS = {"vision", "language_lora", "fast_adapter"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group_norms(scenario: dict) -> dict[str, float]:
    return {
        name: float(values["norm"])
        for name, values in scenario["groups"].items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gradient-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.gradient_audit.read_text(encoding="utf-8"))
    scenarios = audit["gradient_scenarios"]
    flow = scenarios["flow_ki_true"]
    fast = scenarios["fast_ki_true"]
    no_ki = scenarios["flow_ki_false"]
    joint = scenarios["joint_ki_true"]
    flow_norms = group_norms(flow)
    fast_norms = group_norms(fast)
    no_ki_norms = group_norms(no_ki)

    for name in FAST_GROUPS:
        if flow_norms.get(name, 0.0) != 0.0:
            raise ValueError(f"Flow KI leak into FAST group {name}")
    for name in FLOW_GROUPS:
        if fast_norms.get(name, 0.0) != 0.0:
            raise ValueError(f"FAST objective changed Flow group {name}")
    for name in ("vision", "language_lora"):
        if no_ki_norms.get(name, 0.0) <= 0.0:
            raise ValueError(f"KI-off control did not activate {name}")
    if no_ki_norms.get("fast_adapter", 0.0) != 0.0:
        raise ValueError("Flow KI-off control unexpectedly touched FAST adapter")

    flow_norm = math.sqrt(sum(
        flow_norms.get(name, 0.0) ** 2 for name in FLOW_GROUPS))
    fast_norm = math.sqrt(sum(
        fast_norms.get(name, 0.0) ** 2 for name in FAST_GROUPS))
    if flow_norm <= 0.0 or fast_norm <= 0.0:
        raise ValueError("a training branch has zero aggregate gradient")
    ratio = flow_norm / fast_norm
    low = ratio * math.sqrt(0.4 / 0.6)
    high = ratio * math.sqrt(0.6 / 0.4)
    candidate = max(low, min(high, ratio))
    selected = round(candidate, 3)
    if not low <= selected <= high:
        selected = round(candidate, 4)
    if not low <= selected <= high:
        raise ValueError(
            f"no rounded candidate in gradient balance range "
            f"[{low}, {high}]")

    parts = {}
    for weight in (0.1, 0.25, 0.5, 0.75, 1.0):
        flow_energy = flow_norm**2
        fast_energy = (weight * fast_norm) ** 2
        parts[str(weight)] = {
            "global_norm_before_clip": math.sqrt(
                flow_energy + fast_energy),
            "global_clip_coefficient": min(
                1.0, 1.0 / math.sqrt(flow_energy + fast_energy)),
            "flow_energy_fraction": flow_energy / (
                flow_energy + fast_energy),
            "fast_energy_fraction": fast_energy / (
                flow_energy + fast_energy),
        }
    predicted_joint = math.sqrt(
        flow_norm**2 + fast_norm**2)
    measured_joint = float(joint["global_norm"])
    if not math.isclose(
            predicted_joint, measured_joint, rel_tol=2e-5, abs_tol=1e-4):
        raise ValueError(
            "observed joint norm is inconsistent with disjoint branch groups")

    payload = {
        "schema_version": 1,
        "status": "pass",
        "claim": (
            "Weight chosen from real-data branch gradient norms and KI isolation; "
            "not from loss magnitude alone"),
        "evidence": {
            "gradient_audit": str(args.gradient_audit),
            "gradient_audit_sha256": sha256_file(args.gradient_audit),
            "global_batch_size": audit["global_batch_size"],
            "fsdp_devices": audit["fsdp_devices"],
            "device_count": audit["device_count"],
            "sample_indices_sha256": audit["sample_indices_sha256"],
        },
        "branch_losses": {
            "flow_mse_12d": float(flow["loss"]),
            "fast_ce": float(fast["loss"]),
        },
        "branch_gradient_norms": {
            "flow": flow_norm,
            "fast": fast_norm,
            "flow_to_fast_ratio": ratio,
        },
        "selection_rule": {
            "flow_energy_fraction_min": 0.4,
            "flow_energy_fraction_max": 0.6,
            "fast_energy_fraction_min": 0.4,
            "fast_energy_fraction_max": 0.6,
            "candidate_lower_bound": low,
            "candidate_upper_bound": high,
            "rounding": "ratio rounded to three decimals within accepted interval",
        },
        "candidate_weights": parts,
        "selected_fast_ce_weight": selected,
        "selected_weight_effect": {
            "global_norm_before_clip": math.sqrt(
                flow_norm**2 + (selected * fast_norm) ** 2),
            "global_clip_coefficient": min(
                1.0, 1.0 / math.sqrt(
                    flow_norm**2 + (selected * fast_norm) ** 2)),
            "flow_energy_fraction": flow_norm**2 / (
                flow_norm**2 + (selected * fast_norm) ** 2),
            "fast_energy_fraction": (selected * fast_norm) ** 2 / (
                flow_norm**2 + (selected * fast_norm) ** 2),
        },
        "ki_checks": {
            "flow_ki_true_shared_group_norms": {
                name: flow_norms.get(name, 0.0) for name in FAST_GROUPS
            },
            "flow_ki_false_shared_group_norms": {
                name: no_ki_norms.get(name, 0.0) for name in FAST_GROUPS
            },
            "fast_flow_group_norms": {
                name: fast_norms.get(name, 0.0) for name in FLOW_GROUPS
            },
            "joint_norm_prediction": predicted_joint,
            "joint_norm_measured_at_weight_1": measured_joint,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "pass",
        "selected_fast_ce_weight": selected,
        "candidate_interval": [low, high],
        "flow_gradient_norm": flow_norm,
        "fast_gradient_norm": fast_norm,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
