#!/usr/bin/env python3
"""E4-T loss curve: the training flow-matching loss and the validation flow loss, log scale.

The delivery figure `curve_flow_loss.png` is drawn on a linear axis, where the validation loss falls
from 2.45 to 0.003 in the first two points and every later point sits on the zero line — unreadable.
This script draws the same quantity on a log axis and adds the training loss the optimiser actually
saw, so the two can be compared without pretending the training loss is an action metric.

  * top panel: training flow-matching loss per optimizer step (the raw line plus a per-pixel
    minimum/mean envelope), and the gradient norm on a second log axis;
  * bottom panel: validation flow loss at each saved checkpoint, true-12 and padding-20, for the
    clean-active and post-event-active groups, with the initialisation reference at step 1.

Reads only the training log and the evaluation reports; writes PNG and PDF. No learned action is
executed and no threshold is involved.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_common as common  # noqa: E402

WIDTH, HEIGHT = 1400, 980
MARGIN, GAP = 96, 86
# the launcher tees the container output with a monotonic-time prefix, so the record is
# matched anywhere on the line rather than at its start
TRAIN_RE = re.compile(r"Step (\d+): grad_norm=([0-9.eE+-]+),\s+loss=([0-9.eE+-]+)")


def read_training(path: Path) -> list[tuple[int, float, float]]:
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    return [(int(step), float(grad), float(loss))
            for step, grad, loss in TRAIN_RE.findall(text)
            if float(loss) > 0 and float(grad) > 0]


def read_validation(root: Path, split: str = "validation") -> list[dict]:
    rows = []
    for path in sorted(root.glob("*/report.json")):
        if "superseded" in path.parent.name:
            continue
        report = common.load_json(path)
        if report.get("split") != split:
            continue
        if (report.get("window_set") or "frozen_plan") != "frozen_plan":
            continue
        aggregates = report.get("aggregates") or {}
        rows.append({
            "step": int(report.get("checkpoint_train_state_step", -1)),
            "label": report.get("label"),
            "clean_active": aggregates.get("clean_active") or {},
            "recovery_post_active": aggregates.get("recovery_post_active") or {},
        })
    return sorted(rows, key=lambda row: row["step"])


def envelope(points: list[tuple[float, float]], bins: int) -> list[tuple[float, float, float]]:
    """Per-pixel minimum, mean and maximum, so 30,000 steps fit and no spike is smoothed away."""
    if not points:
        return []
    low, high = points[0][0], points[-1][0]
    span = max(1e-9, high - low)
    buckets: dict[int, list[float]] = {}
    for step, value in points:
        buckets.setdefault(min(bins - 1, int((step - low) / span * bins)), []).append(value)
    def percentile(values: list[float], fraction: float) -> float:
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, max(0, int(fraction * (len(ordered) - 1))))]
    return [(low + (index + 0.5) * span / bins, percentile(values, 0.10), sum(values) / len(values),
             percentile(values, 0.90)) for index, values in sorted(buckets.items())]


def log_axis(values: list[float], floor: float | None = None) -> tuple[float, float]:
    positives = [value for value in values if value and value > 0]
    low = math.log10(min(positives)) if positives else -6.0
    high = math.log10(max(positives)) if positives else 0.0
    if floor is not None:
        low = min(low, math.log10(floor))
    if high - low < 1.0:
        high = low + 1.0
    return math.floor(low * 2) / 2, math.ceil(high * 2) / 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--training-config", type=Path,
                        default=common.REPO_ROOT / "configs/so101/e4t_training.json")
    parser.add_argument("--png", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, default=None)
    args = parser.parse_args()

    train = read_training(args.train_log)
    validation = read_validation(args.reports)
    if not train:
        raise SystemExit(f"no 'Step N: grad_norm=… loss=…' lines in {args.train_log}")
    budgets = common.load_json(args.training_config)["optimisation"]

    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    left, right = MARGIN, WIDTH - 40
    top_a, bottom_a = 96, 470
    top_b, bottom_b = 606, HEIGHT - 96
    draw.text((left, 8), "E4-T flow-matching loss: training (top) and validation (bottom), log scale", fill=(20, 20, 20))
    draw.text((left, 26), "the training curve is what the optimiser saw; it is a 32-dim objective and not a "
                          "substitute for the 12-dim action errors", fill=(70, 70, 70))
    draw.text((left, 44), "top: solid = per-pixel mean over each pixel column, band = 10th..90th percentile of the "
                          "steps inside it", fill=(70, 70, 70))
    draw.text((left, 62), "bottom: true-12 and padding-20 flow loss for the two active groups at every saved checkpoint",
              fill=(70, 70, 70))

    # ---- top panel: training loss
    steps = [row[0] for row in train]
    losses = [row[2] for row in train]
    grads = [row[1] for row in train]
    lo, hi = log_axis(losses)
    g_lo, g_hi = log_axis(grads)
    draw.rectangle([left, top_a, right, bottom_a], outline=(60, 60, 60), width=2)
    for exponent in range(int(math.ceil(lo)), int(math.floor(hi)) + 1):
        value = 10.0 ** exponent
        y = bottom_a - (math.log10(value) - lo) / (hi - lo) * (bottom_a - top_a)
        draw.line([left - 6, y, left, y], fill=(90, 90, 90))
        draw.text((left - 92, y - 6), f"{value:.1e}", fill=(60, 60, 60))
    for fraction in range(0, 6):
        step_value = int(max(steps) * fraction / 5)
        x = left + (step_value / max(1, max(steps))) * (right - left)
        draw.line([x, bottom_a, x, bottom_a + 6], fill=(90, 90, 90))
        draw.text((x - 20, bottom_a + 10), str(step_value), fill=(60, 60, 60))
    draw.text((right - 110, bottom_a + 44), "optimizer step", fill=(20, 20, 20))
    draw.text((left, top_a - 14), "training loss (MSE)", fill=(20, 20, 20))

    def to_y(value: float, low: float, high: float, top: int, bottom: int) -> float:
        return bottom - (math.log10(value) - low) / (high - low) * (bottom - top)

    band = envelope(list(zip(steps, losses)), right - left)
    for x_step, value_min, _, value_max in band:
        x = left + (x_step / max(1, max(steps))) * (right - left)
        draw.line([x, to_y(value_min, lo, hi, top_a, bottom_a),
                   x, to_y(value_max, lo, hi, top_a, bottom_a)], fill=(236, 236, 236), width=1)
    mean_points = [(left + (x_step / max(1, max(steps))) * (right - left),
                    to_y(value_mean, lo, hi, top_a, bottom_a)) for x_step, _, value_mean, _ in band]
    for a, b in zip(mean_points, mean_points[1:]):
        draw.line([a, b], fill=(31, 119, 180), width=2)
    # ---- bottom panel: validation flow loss
    entries = []
    for row in validation:
        for group, colour in (("clean_active", (31, 119, 180)), ("recovery_post_active", (214, 39, 40))):
            aggregates = row[group] or {}
            for key, dash in (("true12_mse", False), ("padding20_mse", True)):
                value = aggregates.get(key)
                if value:
                    entries.append((row["step"], float(value), colour, dash, group, key))
    if entries:
        v_lo, v_hi = log_axis([item[1] for item in entries])
        draw.rectangle([left, top_b, right, bottom_b], outline=(60, 60, 60), width=2)
        for exponent in range(int(math.ceil(v_lo)), int(math.floor(v_hi)) + 1):
            value = 10.0 ** exponent
            y = to_y(value, v_lo, v_hi, top_b, bottom_b)
            draw.line([left - 6, y, left, y], fill=(90, 90, 90))
            draw.text((left - 92, y - 6), f"{value:.1e}", fill=(60, 60, 60))
        for fraction in range(0, 6):
            step_value = int(max(item[0] for item in entries) * fraction / 5)
            x = left + (step_value / max(1, max(item[0] for item in entries))) * (right - left)
            draw.line([x, bottom_b, x, bottom_b + 6], fill=(90, 90, 90))
            draw.text((x - 20, bottom_b + 10), str(step_value), fill=(60, 60, 60))
        draw.text((right - 110, bottom_b + 44), "TrainState.step", fill=(20, 20, 20))
        draw.text((left, top_b - 14), "validation flow loss (MSE)", fill=(20, 20, 20))
        for index, (step, value, colour, dash, group, key) in enumerate(entries):
            x = left + (step / max(1, max(item[0] for item in entries))) * (right - left)
            y = to_y(value, v_lo, v_hi, top_b, bottom_b)
            if dash:
                draw.rectangle([x - 3, y - 3, x + 3, y + 3], outline=colour, width=1)
            else:
                draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=colour)
        # series labels, drawn from the last point of each series
        legend = [("clean active, true-12", (31, 119, 180), False),
                  ("recovery post-event active, true-12", (214, 39, 40), False),
                  ("padding-20 (same groups, open markers)", (120, 120, 120), True)]
        for index, (name, colour, _) in enumerate(legend):
            draw.text((left + 12 + (index % 2) * 470, top_b + 10 + (index // 2) * 16), f"- {name}",
                      fill=colour)

    args.png.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.png)
    if args.pdf:
        image.convert("RGB").save(args.pdf, "PDF", resolution=200.0)
    summary = {
        "training_points": len(train), "first_loss": losses[0], "last_loss": losses[-1],
        "min_loss": min(losses), "validation_points": len(validation),
        "validation_true12": {row["label"]: (row["clean_active"] or {}).get("true12_mse")
                              for row in validation},
        "planned_updates": budgets.get("planned_updates"),
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
