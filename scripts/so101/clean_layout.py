"""Repository-local rejection filter over the external certified scene sampler."""
from __future__ import annotations

import math


def inspect_layout(scene, minimum_distance_m: float) -> dict:
    objects = list(scene.objects)
    if len(objects) != 4:
        raise ValueError("normal sorting requires exactly four objects")
    if scene.extra.get("perturbation"):
        raise ValueError("clean scene contains a perturbation schedule")
    if any(obj.route != "direct" for obj in objects):
        raise ValueError("clean sorting requires direct routes")
    points = [tuple(float(v) for v in obj.xy) for obj in objects]
    if any(len(p) != 2 or not all(math.isfinite(v) for v in p) for p in points):
        raise ValueError("invalid object positions")
    distance = min(math.dist(a, b) for i, a in enumerate(points) for b in points[i + 1:])
    if distance < minimum_distance_m:
        raise ValueError("objects do not meet the registered minimum separation")
    return {"minimum_pairwise_distance_m": distance,
            "required_minimum_distance_m": minimum_distance_m,
            "perturbation_present": False, "objects": [obj.as_dict() for obj in objects]}


def install_filter(collection_module, protocol: dict):
    """Retain certified sampling and IK; reject layouts before physical execution."""
    from bimanual.tasks.sorting.scene_sampling import SceneGenerationError

    original = collection_module.generate_scene
    minimum = float(protocol["scene_layout"]["minimum_pairwise_distance_m"])
    if minimum <= 0:
        raise ValueError("minimum separation must be positive")

    def generate(config, certificate, profile, seed, **kwargs):
        if kwargs.get("env") is None:
            raise ValueError("spread layouts require the external sampler's live IK check")
        scene = original(config, certificate, profile, seed, **kwargs)
        try:
            audit = inspect_layout(scene, minimum)
        except ValueError as error:
            raise SceneGenerationError(str(error)) from error
        scene.extra["clean_layout"] = audit
        scene.sha256 = scene.digest()
        return scene

    collection_module.generate_scene = generate
    return generate
