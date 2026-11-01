"""Shared per-device GPU admission policy for probes and formal training."""
from __future__ import annotations

import math


def memory_plan(
    devices: list[dict],
    *,
    demand_mib: int,
    margin_mib: int,
    explicit_fraction: float | None = None,
    jax_pool_peak_mib: int | None = None,
    pool_headroom: float = 1.05,
) -> dict:
    """Admit on free memory and choose one cap for the whole JAX process.

    ``demand_mib`` is the measured whole-process GPU increase, including CUDA
    context and other non-pool allocations.  Once the JAX pool peak is known,
    the cap is based on that pool plus bounded headroom rather than all free
    memory.  The least-free device binds every device's shared fraction.
    """
    if not devices:
        raise ValueError("at least one GPU is required")
    if demand_mib < 0 or margin_mib < 0:
        raise ValueError("GPU demand and memory margin must be non-negative")
    if pool_headroom < 1.0:
        raise ValueError("JAX pool headroom must be at least 1.0")

    required = demand_mib + margin_mib
    for device in devices:
        free = int(device["memory_free_mib"])
        total = int(device["memory_total_mib"])
        if total <= 0 or free < 0 or free > total:
            raise ValueError(f"invalid GPU memory readings on device {device.get('index')}")
        if free < required:
            raise RuntimeError(
                f"GPU {device.get('index')} has {free} MiB free; "
                f"requires {required} MiB including the {margin_mib} MiB margin")

    available = min(
        int(device["memory_free_mib"]) - margin_mib for device in devices)
    caps: dict[int, float] = {}
    for device in devices:
        index = int(device["index"])
        total = int(device["memory_total_mib"])
        free = int(device["memory_free_mib"])
        cap_mib = float(available)
        if jax_pool_peak_mib is not None:
            if jax_pool_peak_mib <= 0:
                raise ValueError("JAX pool peak must be positive when supplied")
            non_pool = max(0, demand_mib - jax_pool_peak_mib)
            capped_pool = min(
                jax_pool_peak_mib * pool_headroom,
                free - margin_mib - non_pool,
            )
            cap_mib = min(cap_mib, max(0.0, capped_pool))
        if cap_mib <= 0:
            raise RuntimeError(f"no safe JAX pool headroom on GPU {index}")
        caps[index] = cap_mib / total

    computed = min(caps.values())
    if explicit_fraction is None:
        fraction = computed
    else:
        fraction = float(explicit_fraction)
        if not 0.0 < fraction <= 1.0 or not math.isfinite(fraction):
            raise ValueError("JAX memory fraction must be finite and in (0, 1]")
        if fraction > computed:
            raise RuntimeError(
                f"JAX fraction {fraction:.6f} exceeds the safe shared cap {computed:.6f}")

    plans = []
    for device in devices:
        total = int(device["memory_total_mib"])
        free = int(device["memory_free_mib"])
        pool_cap = fraction * total
        plans.append({
            "index": int(device["index"]),
            "uuid": device.get("uuid"),
            "memory_total_mib": total,
            "memory_free_mib": free,
            "memory_used_mib": int(device.get("memory_used_mib", total - free)),
            "required_free_mib": required,
            "jax_pool_cap_mib": round(pool_cap, 3),
            "margin_after_pool_cap_mib": round(free - pool_cap, 3),
        })
    return {
        "demand_mib": int(demand_mib),
        "margin_mib": int(margin_mib),
        "required_free_mib": required,
        "jax_memory_fraction": fraction,
        "jax_pool_cap_mib_by_device": plans,
        "jax_preallocate": False,
        "utilisation_is_not_an_admission_gate": True,
        "non_pool_demand_included": True,
        "jax_pool_peak_mib": jax_pool_peak_mib,
        "pool_headroom": pool_headroom,
    }


def sampled_margin_problems(devices: list[dict], *, margin_mib: int) -> list[str]:
    """Reject a completed probe if any observed physical free margin was too small."""
    problems = []
    for device in devices:
        total = int(device["memory_total_mib"])
        used = int(device.get("peak_mib_sampled", 0))
        if used <= 0:
            problems.append(f"GPU {device.get('index')} has no sampled peak")
            continue
        observed = total - used
        if observed < margin_mib:
            problems.append(
                f"GPU {device.get('index')} observed only {observed} MiB free at peak; "
                f"requires {margin_mib}")
    return problems
