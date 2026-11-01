"""Pure checks shared by the clean100 merge and audit scripts (no numpy)."""
from __future__ import annotations

import math

FPS = 50.0


def all_finite(values) -> bool:
    for value in values:
        if isinstance(value, (list, tuple)):
            if not all_finite(value):
                return False
            continue
        try:
            if not math.isfinite(float(value)):
                return False
        except (TypeError, ValueError):
            return False
    return True


def frame_index_continuous(frame_index, frames: int) -> bool:
    if len(frame_index) != frames:
        return False
    return all(int(value) == expected for expected, value in enumerate(frame_index))


def timestamps_nominal(timestamp, frames: int, fps: float = FPS,
                       atol: float = 1e-4) -> bool:
    if len(timestamp) != frames:
        return False
    if frames and abs(float(timestamp[0])) > 1e-7:
        return False
    step = 1.0 / float(fps)
    return all(abs(float(timestamp[index]) - index * step) <= atol
               for index in range(frames))


def control_clock_aligned(trace_time, timestamp, frames: int,
                          atol: float = 1e-4) -> bool:
    if len(trace_time) != frames or len(timestamp) != frames:
        return False
    if frames < 2:
        return True
    first_time, first_stamp = float(trace_time[0]), float(timestamp[0])
    return all(abs((float(trace_time[index]) - first_time)
                   - (float(timestamp[index]) - first_stamp)) <= atol
               for index in range(frames))


def control_clock_regular(trace_time, frames: int, step: float = 0.02,
                          atol: float = 1e-6) -> bool:
    if len(trace_time) != frames:
        return False
    return all(abs(float(trace_time[index]) - float(trace_time[index - 1]) - step) <= atol
               for index in range(1, frames))


def resolve_split(rows: list[dict], old_split: dict[int, str],
                  families: list[list[int]], target_train: int,
                  target_validation: int) -> tuple[dict[int, str], dict]:
    """Assign new episodes to sides while keeping every family atomic.

    ``rows`` contain episode_index, collection ('old' or 'new') and a default
    split.  The preserved rows must already be on the side recorded in
    ``old_split``; only new rows may move.
    """
    new_indices = [int(row["episode_index"]) for row in rows if row["collection"] == "new"]
    new_set = set(new_indices)
    assignment = {int(row["episode_index"]): row["split"] for row in rows}
    for index, side in old_split.items():
        if assignment.get(index) != side:
            raise ValueError(f"preserved episode {index} is not on its registered side")
    constraints: dict[int, str] = {}
    impossible = []
    for family in families:
        old_sides = {old_split[index] for index in family if index not in new_set}
        if len(old_sides) > 1:
            impossible.append(family)
            continue
        if not old_sides:
            continue
        side = next(iter(old_sides))
        for index in family:
            if index in new_set:
                constraints[index] = side
                assignment[index] = side
    if impossible:
        raise ValueError(f"preserved family crosses the registered split: {impossible[:3]}")
    order = {index: rank for rank, index in enumerate(new_indices)}
    blocks = []
    for family in families:
        members = [index for index in family if index in new_set]
        if not members:
            continue
        if any(index in constraints for index in members):
            side = constraints[members[0]]
        else:
            side = assignment[min(members, key=lambda index: order[index])]
        for index in members:
            assignment[index] = side
        blocks.append(members)
    train_count = sum(assignment[index] == "train" for index in new_indices)
    target = int(target_train)
    if sum(assignment[index] == "validation" for index in new_indices) + train_count != len(new_indices):
        raise ValueError("new split does not cover every new episode")
    adjustments = []
    if train_count != target:
        from_side = "train" if train_count > target else "validation"
        to_side = "validation" if from_side == "train" else "train"
        need = abs(train_count - target)
        movable = sorted(
            (min(order[index] for index in members), members)
            for members in blocks
            if not any(index in constraints for index in members)
            and assignment[members[0]] == from_side)
        # A family block may be larger than one episode, so flipping the first
        # movable block can overshoot.  Pick an exact-size subset deterministically.
        reachable: dict[int, tuple[int, int] | None] = {0: None}
        for position, (_, members) in enumerate(movable):
            size = len(members)
            updated = dict(reachable)
            for total in reachable:
                if total + size <= need and total + size not in updated:
                    updated[total + size] = (total, position)
            reachable = updated
        if need not in reachable:
            raise ValueError("cannot reach the registered split without splitting a family")
        chosen = []
        total = need
        while total:
            previous, position = reachable[total]
            chosen.append(movable[position][1])
            total = previous
        for members in chosen:
            for index in members:
                assignment[index] = to_side
            adjustments.append({"members": members, "to": to_side})
        train_count = sum(assignment[index] == "train" for index in new_indices)
    crossing = [family for family in families
                if len({assignment[index] for index in family}) > 1]
    if crossing:
        raise ValueError(f"family split check failed: {crossing[:3]}")
    if len([index for index in new_indices if assignment[index] == "train"]) != target:
        raise ValueError("new train count differs from the registered target")
    if sum(assignment[index] == "validation" for index in new_indices) != int(target_validation):
        raise ValueError("new validation count differs from the registered target")
    return assignment, {"adjustments": adjustments, "crossing": crossing}
