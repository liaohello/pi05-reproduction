"""V4 narrowed feasible family: v1 expert and slots on the measured subset.

The v4 profile narrows yaw, tray and block-to-tray outer clearance to the region
covered by the v1/v3 successes.  This module deliberately reuses the v2 wrapper
with **no v2 execution keys** in the v4 profile, so pick/place/expert behaviour
is the v1 reference behaviour; only the v2 block outer-clearance sampler check
and the v1 slot rule are active.
"""
from __future__ import annotations

from pathlib import Path

import math

import scene2_common as _v1
import scene2_common_v2 as _v2
from scene2_common_v2 import *  # noqa: F401,F403

_b = _v1._b
_ORIGINAL_MEASURE_INITIAL = _v1.measure_initial


def measure_initial(env, config, blocks):
    result = _ORIGINAL_MEASURE_INITIAL(env, config, blocks)
    regions = _v1._b("regions")
    outer = regions.tray_inner_half(config) + 0.006
    margin = float(config["sorting"].get("arm_clearance_m", 0.01))
    distances = {}
    for arm in ("left", "right"):
        tcp = env.data.site_xpos[env.model.site(f"{arm}_tcp").id]
        for color in ("red", "blue"):
            body = env.model.body(f"{color}_tray").id
            center = env.model.body_pos[body]
            dx = max(abs(float(tcp[0]) - float(center[0])) - float(outer[0]), 0.0)
            dy = max(abs(float(tcp[1]) - float(center[1])) - float(outer[1]), 0.0)
            distances[f"{arm}_{color}"] = math.hypot(dx, dy)
    result["checks"]["parked_arms_clear_of_trays"] = all(
        value >= margin for value in distances.values())
    result["parked_arm_tray_clearance_m"] = distances
    result["passed"] = all(result["checks"].values())
    return result


def configure(root, *, python=None):
    root = _v2.configure(root, python=python)
    _v1.measure_initial = measure_initial
    return root


def provenance(root, protocol_path, config, certificate_path, adapter_path=None):
    result = _v1.provenance(root, protocol_path, config, certificate_path,
                            adapter_path=Path(__file__).resolve())
    protocol = _v1.load_protocol(protocol_path)
    profile = _v1.REPO_ROOT / protocol["source_binding"]["profile"]
    result["profile_sha256"] = _v1.common.sha256_file(profile)
    result["adapter_sha256"] = _v1.common.sha256_file(Path(__file__).resolve())
    return result
