"""Deterministic G1 inputs, shared by the reference oracle and the candidate tests.

Everything here is plain data: no NumPy RNG, no torch, no tokenizer. The oracle
turns these descriptions into reference outputs; the tests feed the identical
description to the candidate. Because both sides read the same literal, a
disagreement can only come from behaviour, never from input drift.
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Recipes
# --------------------------------------------------------------------------- #
DEFAULT_RECIPE = {
    "blend": {
        "high_level_subtask": {
            "weight": 0.30,
            "messages": [
                {"role": "user", "content": "${task}", "stream": "high_level"},
                {
                    "role": "assistant",
                    "content": "${subtask}",
                    "stream": "high_level",
                    "target": True,
                    "if_present": "subtask",
                },
            ],
        },
        "low_level_execution": {
            "weight": 0.70,
            "messages": [
                {"role": "user", "content": "${subtask}", "stream": "low_level", "if_present": "subtask"}
            ],
        },
    }
}

LOW_LEVEL_ONLY_RECIPE = {
    "messages": [
        {"role": "user", "content": "${task}", "stream": "low_level"},
        {"role": "assistant", "content": "${subtask}", "stream": "high_level", "target": True},
    ]
}

VQA_BLEND_RECIPE = {
    "blend": {
        "high_level_subtask": {
            "weight": 0.50,
            "messages": [
                {"role": "user", "content": "${task}", "stream": "high_level"},
                {"role": "assistant", "content": "${subtask}", "stream": "high_level", "target": True},
            ],
        },
        "low_level_execution": {
            "weight": 0.50,
            "messages": [{"role": "user", "content": "${subtask}", "stream": "low_level"}],
        },
        "vqa_left": {
            "weight": 1.0,
            "route": "vqa",
            "messages": [
                {"role": "user", "content": "${vqa_query}", "stream": "high_level"},
                {"role": "assistant", "content": "${vqa}", "stream": "high_level", "target": True},
            ],
        },
    }
}

# --------------------------------------------------------------------------- #
# Language rows
# --------------------------------------------------------------------------- #
PERSISTENT_ROWS = [
    {"style": "subtask", "role": "user", "content": "pick up the red cube", "timestamp": 0.0},
    {"style": "subtask", "role": "user", "content": "place it in the bowl", "timestamp": 2.0},
    {"style": "memory", "role": "assistant", "content": "the bowl is on the left", "timestamp": 1.0},
]

TASK_AUG_ROWS = [
    {"style": "task_aug", "role": "user", "content": "sort the red cube", "timestamp": 0.0},
    {"style": "task_aug", "role": "user", "content": "move the red block", "timestamp": 0.0},
    {"style": "task_aug", "role": "user", "content": "put the cube away", "timestamp": 0.0},
]

VQA_EVENT_ROWS = [
    {
        "style": "vqa",
        "role": "user",
        "content": "what is on the left?",
        "camera": "base",
        "timestamp": 1.5,
    },
    {
        "style": "vqa",
        "role": "assistant",
        "content": '{"label": "bowl", "point_format": "xy", "point": [0.25, 0.5]}',
        "camera": "base",
        "timestamp": 1.5,
    },
]

#: sample indices used for every routing case.
ROUTING_SAMPLE_INDICES = list(range(8))

ROUTING_CASES = [
    {
        "name": "high_low_mixed",
        "recipe": DEFAULT_RECIPE,
        "persistent": PERSISTENT_ROWS,
        "events": [],
        "task": "sort the cube",
        "timestamp": 1.5,
        "default_task": "sort the cube",
    },
    {
        "name": "missing_subtask",
        "recipe": DEFAULT_RECIPE,
        "persistent": [],
        "events": [],
        "task": "sort the cube",
        "timestamp": 0.0,
        "default_task": "sort the cube",
    },
    {
        "name": "task_aug_rotation",
        "recipe": DEFAULT_RECIPE,
        "persistent": TASK_AUG_ROWS + [PERSISTENT_ROWS[0]],
        "events": [],
        "task": None,
        "timestamp": 0.5,
        "default_task": "canonical task",
    },
    {
        "name": "recipe_none_fallback",
        "recipe": None,
        "persistent": [],
        "events": [],
        "task": "sort the cube",
        "timestamp": 0.0,
        "default_task": "sort the cube",
    },
    {
        "name": "low_level_only",
        "recipe": LOW_LEVEL_ONLY_RECIPE,
        "persistent": PERSISTENT_ROWS,
        "events": [],
        "task": "sort the cube",
        "timestamp": 2.5,
        "default_task": "sort the cube",
    },
    {
        "name": "vqa_routed",
        "recipe": VQA_BLEND_RECIPE,
        "persistent": PERSISTENT_ROWS,
        "events": VQA_EVENT_ROWS,
        "task": "sort the cube",
        "timestamp": 1.55,
        "default_task": "sort the cube",
    },
]

# --------------------------------------------------------------------------- #
# Prompt / label cases
# --------------------------------------------------------------------------- #
PROMPT_CASES = [
    {
        "name": "single_turn_target",
        "messages": [{"role": "user", "content": "pick up the cube"}],
        "streams": ["high_level"],
        "targets": [],
        "state": None,
    },
    {
        "name": "assistant_target",
        "messages": [
            {"role": "user", "content": "pick up the cube"},
            {"role": "assistant", "content": "grasp the red cube"},
        ],
        "streams": ["high_level", "high_level"],
        "targets": [1],
        "state": None,
    },
    {
        "name": "low_level_with_state",
        "messages": [{"role": "user", "content": "place it in the bowl"}],
        "streams": ["low_level"],
        "targets": [],
        "state": [0.0, -1.0, 1.0, 0.5, -0.5, 0.25, 0.75, -0.75],
    },
    {
        "name": "multiline_and_spaces",
        "messages": [
            {"role": "user", "content": "line one\nline two  spaced"},
            {"role": "assistant", "content": "answer  with   spaces"},
        ],
        "streams": ["high_level", "high_level"],
        "targets": [1],
        "state": None,
    },
    {
        "name": "runtime_query",
        "messages": [{"role": "user", "content": "what is on the left?"}],
        "streams": [],
        "targets": [],
        "state": None,
        "runtime": True,
    },
]

# --------------------------------------------------------------------------- #
# State discretisation and locations
# --------------------------------------------------------------------------- #
STATE_BIN_CASES = [
    [0.0, -1.0, 1.0],
    [-1.0, -0.9921875, -0.9921876, 0.0, 0.9921875, 0.9921876, 1.0],
    [-2.0, -1.5, 2.0, 1.5],
    [0.00390625, -0.00390625, 0.5],
]

LOC_POINT_CASES = [
    {"label": "cup", "point_format": "xy", "point": [0.5, 0.25]},
    {"label": "bowl", "point_format": "xy", "point": [0.0, 1.0]},
    {"label": "block", "point_format": "xy", "point": [0.25, 0.5]},
    {"label": "", "point_format": "xy", "point": [0.5, 0.5]},
]

LOC_BOX_CASES = [
    {"detections": [{"label": "cup", "bbox_format": "xyxy", "bbox": [0.0, 0.25, 0.5, 1.0]}]},
    {
        "detections": [
            {"label": "cup", "bbox_format": "xyxy", "bbox": [0.0, 0.0, 0.25, 0.25]},
            {"label": "bowl", "bbox_format": "xyxy", "bbox": [0.5, 0.5, 1.0, 1.0]},
        ]
    },
]

# --------------------------------------------------------------------------- #
# FAST cases
# --------------------------------------------------------------------------- #
FAST_ACTIONS = {
    "ramp6": [[round(0.01 * i + 0.001 * d, 6) for d in range(6)] for i in range(50)],
    "ramp12": [[round(0.02 * i - 0.003 * d, 6) for d in range(12)] for i in range(50)],
    "mixed14": [
        [round(0.5 * ((i * 7 + d * 3) % 11) / 11.0 - 0.5, 6) for d in range(14)] for i in range(50)
    ],
}

FAST_SKIP_TOKENS_UNDER_TEST = [1152, 128]

# --------------------------------------------------------------------------- #
# Normalisation and relative actions
# --------------------------------------------------------------------------- #
NORMALIZATION_DIMS = [6, 12, 14, 32]

NORMALIZATION_STATS = {
    "q01": [-1.0, -0.5, 0.0, 0.1, -2.0, 0.3],
    "q99": [1.0, 0.5, 0.0, 0.9, 2.0, 0.7],
    "min": [-1.2, -0.6, 0.0, 0.05, -2.2, 0.2],
    "max": [1.2, 0.6, 0.0, 0.95, 2.2, 0.8],
    "mean": [0.0, 0.0, 0.0, 0.5, 0.0, 0.5],
    "std": [0.5, 0.25, 1.0, 0.2, 1.0, 0.1],
}

#: Dimension 2 is constant in the fixture, so a zero spread must be handled.
ACTION_NAMES_6 = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# --------------------------------------------------------------------------- #
# Image cases
# --------------------------------------------------------------------------- #
IMAGE_CASES = [
    {"name": "single_nchw_224", "keys": ["observation.images.base"], "shape": (3, 224, 224), "layout": "nchw"},
    {"name": "single_nhwc_240x320", "keys": ["observation.images.base"], "shape": (240, 320, 3), "layout": "nhwc"},
    {"name": "two_cameras", "keys": ["observation.images.base", "observation.images.wrist"], "shape": (3, 200, 200), "layout": "nchw"},
    {
        "name": "three_cameras_mixed_order",
        "keys": ["observation.images.left", "observation.images.base", "observation.images.right"],
        "shape": (3, 180, 300),
        "layout": "nchw",
    },
    # Partial camera absence. ``present`` is the subset actually supplied in the
    # batch; ``keys`` is the configured feature order used by both frameworks.
    {
        "name": "two_missing_first",
        "keys": ["observation.images.base", "observation.images.wrist"],
        "present": ["observation.images.wrist"],
        "shape": (3, 224, 224),
        "layout": "nchw",
    },
    {
        "name": "two_missing_last_resized",
        "keys": ["observation.images.base", "observation.images.wrist"],
        "present": ["observation.images.base"],
        "shape": (3, 200, 200),
        "layout": "nchw",
    },
    {
        "name": "three_missing_middle",
        "keys": ["observation.images.left", "observation.images.base", "observation.images.right"],
        "present": ["observation.images.left", "observation.images.right"],
        "shape": (3, 224, 224),
        "layout": "nchw",
    },
    {
        "name": "three_missing_two",
        "keys": ["observation.images.left", "observation.images.base", "observation.images.right"],
        "present": ["observation.images.base"],
        "shape": (3, 224, 224),
        "layout": "nchw",
    },
    {
        "name": "all_missing",
        "keys": ["observation.images.left", "observation.images.base", "observation.images.right"],
        "present": [],
        "shape": (3, 224, 224),
        "layout": "nchw",
        "expect_error": "ValueError",
    },
]

# --------------------------------------------------------------------------- #
# FIT fixture: a deterministic LeRobot-shaped action table
# --------------------------------------------------------------------------- #
FIT_EPISODES = 6
FIT_FRAMES_PER_EPISODE = 64
FIT_ACTION_DIM = 7
FIT_EXCLUDE_EPISODES = [1]
FIT_CHUNK_SIZE = 8
FIT_SAMPLES = 24
FIT_VALIDATION_SAMPLES = 8


def fit_action_table():
    """Deterministic, non-degenerate actions with one excluded episode.

    Values come from a fixed arithmetic pattern rather than an RNG so the table
    can be described in one line and regenerated anywhere.
    """
    rows = []
    episodes = []
    states = []
    for episode in range(FIT_EPISODES):
        for frame in range(FIT_FRAMES_PER_EPISODE):
            action = [
                round(0.4 * ((episode * 3 + frame + 2 * dim) % 13) / 13.0 - 0.2, 6)
                for dim in range(FIT_ACTION_DIM)
            ]
            state = [round(0.1 * ((episode + frame + dim) % 7) - 0.3, 6) for dim in range(FIT_ACTION_DIM)]
            rows.append(action)
            states.append(state)
            episodes.append(episode)
    return rows, states, episodes


# --------------------------------------------------------------------------- #
# Truncation cases
# --------------------------------------------------------------------------- #
TRUNCATION_LONG_PROMPT = "describe the scene in detail: " + " ".join(
    f"object{i}" for i in range(400)
)

# --------------------------------------------------------------------------- #
# Shared numeric fixtures (identical bytes on both sides)
# --------------------------------------------------------------------------- #
def norm_case_values(dim: int):
    """Action block used for every normalisation case, with a constant dimension."""
    values = [[round(0.3 * ((i * 3 + d) % 9) / 9.0 - 0.15, 6) for d in range(dim)] for i in range(6)]
    if dim >= 3:
        for row in values:
            row[2] = 0.0
    return values


def norm_case_stats(dim: int) -> dict:
    """Per-dimension statistics, tiled from NORMALIZATION_STATS, constant at dim 2."""
    stats = {}
    for key, base in NORMALIZATION_STATS.items():
        repeats = (dim + len(base) - 1) // len(base)
        column = (base * repeats)[:dim]
        if dim >= 3 and key in ("q01", "q99", "min", "max", "std"):
            column[2] = 0.0
        stats[key] = column
    return stats


def relative_case(dim: int):
    """Actions/state pair for the relative-action cases."""
    actions = [[round(0.05 * (i + d), 6) for d in range(dim)] for i in range(4)]
    state = [round(0.1 * d, 6) for d in range(dim)]
    return actions, state


def image_case_batch(case: dict) -> dict:
    """Deterministic pixels for one image case, already in the case's layout."""
    batch = {}
    shape = tuple(case["shape"])
    total = 1
    for size in shape:
        total *= size
    for index, key in enumerate(case["keys"]):
        # Different cameras get different, non-constant patterns. Both the shift
        # and the scale stay inside [0, 1] so swapping two cameras can never be
        # mistaken for a rounding difference after the [-1, 1] rescale.
        sequence = np.arange(total, dtype=np.float32) + np.float32(17 * index)
        base = (sequence % 251.0) / 251.0
        base = (base * np.float32(0.9 - 0.1 * index)).astype(np.float32)
        batch[key] = base.reshape(shape)[None, ...]
    return batch
