#!/usr/bin/env python3
"""E4-T T2: render the open-loop curves and per-episode overlays as PNG and PDF.

Neither the host nor the pinned OpenPI image carries matplotlib, and this round
installs nothing, so the figures are drawn with PIL, which is already present.
The numbers are read from the reports and arrays written by
evaluate_e4t_openloop.py; nothing here recomputes an error metric.  The overlay event
guides use only the fields that evaluator wrote into its own records.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_common as common  # noqa: E402

WIDTH, HEIGHT, MARGIN = 1400, 780, 74
PALETTE = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (227, 119, 32),
           (188, 189, 34), (148, 103, 189), (23, 190, 207), (174, 121, 11)]
JOINT_LABELS = ["L sh_pan", "L sh_lift", "L elbow", "L wrist_flex", "L wrist_roll", "L grip",
                "R sh_pan", "R sh_lift", "R elbow", "R wrist_flex", "R wrist_roll", "R grip"]
JOINT_LANES = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
GRIPPER_LANES = [5, 11]
MARKERS = {
    "disturbance": ((106, 61, 154), "disturbance window"),
    "expert_close": ((44, 160, 44), "expert close"),
    "expert_release": ((23, 190, 207), "expert release"),
    "predicted_close": ((227, 119, 32), "predicted close"),
    "predicted_release": ((31, 119, 180), "predicted release"),
    "detection": ((148, 103, 189), "expert replan (detection)"),
    "correction": ((174, 121, 11), "restarted plan (correction)"),
    "regrasp": ((140, 86, 75), "open-hand catch (regrasp)"),
}
EVENT_FIELDS = (("expert_close", "gt_close_events"), ("expert_release", "gt_release_events"),
                ("predicted_close", "pred_close_events"),
                ("predicted_release", "pred_release_events"))


def _ramp(value: float) -> tuple[int, int, int]:
    low, high = np.array([247, 247, 245]), np.array([5, 48, 97])
    clipped = min(max(float(value), 0.0), 1.0)
    return tuple(int(x) for x in low + (high - low) * clipped)


def _pale(colour: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(int(255 - (255 - channel) * 0.35) for channel in colour)


def marker_legend(draw: ImageDraw.ImageDraw, markers: list) -> None:
    counts: dict[str, list[str]] = {}
    for _, _, kind, note in markers:
        counts.setdefault(kind, []).append(note)
    x = MARGIN
    for kind, notes in counts.items():
        colour, text = MARKERS[kind]
        label = f"{text} x{len(notes)}" if len(notes) > 1 else f"{text} {notes[0]}".rstrip()
        if x > WIDTH - 240:
            break
        draw.line([x, 38, x + 16, 38], fill=colour, width=3)
        draw.text((x + 20, 32), label, fill=colour)
        x += 32 + int(draw.textlength(label))


FRAME_TOP = 78  # room for the title, the unit line and the legend


def _frame(draw: ImageDraw.ImageDraw, title: str, xlabel: str, ylabel: str,
           ymax: float, xmax: float, xticks: list, yticks: list, top: int = FRAME_TOP) -> tuple:
    left, right, bottom = MARGIN, WIDTH - 26, HEIGHT - MARGIN
    draw.rectangle([left, top, right, bottom], outline=(60, 60, 60), width=2)
    draw.text((left, 8), title, fill=(20, 20, 20))
    draw.text((left, 26), ylabel, fill=(20, 20, 20))
    draw.text((right - 190, bottom + 44), xlabel, fill=(20, 20, 20))
    for value in yticks:
        y = bottom - (float(value) / ymax) * (bottom - top)
        draw.line([left - 6, y, left, y], fill=(90, 90, 90), width=1)
        draw.text((left - 58, y - 6), f"{float(value):.4g}", fill=(60, 60, 60))
    for value in xticks:
        x = left + (float(value) / xmax) * (right - left)
        draw.line([x, bottom, x, bottom + 6], fill=(90, 90, 90), width=1)
        label = f"{value:.0f}" if xmax >= 100 else f"{float(value):.3g}"
        draw.text((x - 18, bottom + 10), label, fill=(60, 60, 60))
    return left, top, right, bottom



def legend_rows(draw: ImageDraw.ImageDraw, names: list, colours: list) -> None:
    """Legend above the axes, three columns, so no entry can cover a data line."""
    for index, (name, colour) in enumerate(zip(names, colours)):
        x = MARGIN + (index % 3) * 420
        y = 44 + (index // 3) * 15
        draw.text((x, y), f"- {name}", fill=colour)


def line_chart(path: Path, title: str, xlabel: str, ylabel: str,
                series: dict[str, list[tuple[float, float]]], gate: float | None = None) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    points = [p for values in series.values() for p in values if p[1] is not None]
    if not points:
        image.save(path)
        return
    xs = [p[0] for p in points]
    ys = [p[1] for p in points] + ([gate] if gate is not None else [])
    xmax = max(max(xs), min(max(xs) + 1, max(xs) * 1.05)) or 1.0
    ymax = max(ys) * 1.15 if max(ys) > 0 else 1.0
    left, top, right, bottom = _frame(draw, title, xlabel, ylabel, ymax, xmax,
                                      [round(v, 3) for v in np.linspace(0, xmax, 6)],
                                      [round(v, 3) for v in np.linspace(0, ymax, 6)])
    legend_rows(draw, list(series), [PALETTE[index % len(PALETTE)]
                                     for index in range(len(series))])
    if gate is not None:
        y = bottom - (gate / ymax) * (bottom - top)
        draw.line([left, y, right, y], fill=(214, 39, 40), width=2)
        # the curves converge at the right edge, so the threshold is labelled on the left
        draw.text((left + 8, y - 16), f"gate {gate:g}", fill=(214, 39, 40))
    for index, (name, values) in enumerate(series.items()):
        colour = PALETTE[index % len(PALETTE)]
        clean = [(x, y) for x, y in values if y is not None]
        pixels = [(left + (x / xmax) * (right - left), bottom - (y / ymax) * (bottom - top))
                  for x, y in clean]
        for a, b in zip(pixels, pixels[1:]):
            draw.line([a, b], fill=colour, width=3)
        for point in pixels:
            draw.ellipse([point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4], fill=colour)
    image.save(path)


def bar_chart(path: Path, title: str, ylabel: str, groups: list[str], values: list[float],
              gate: float | None = None) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    if not values:
        image.save(path)
        return
    ymax = max(max(values), gate or 0.0) * 1.2 or 1.0
    left, top, right, bottom = _frame(draw, title, "", ylabel, ymax, len(values),
                                      list(range(len(values) + 1)),
                                      [round(v, 3) for v in np.linspace(0, ymax, 6)])
    width = (right - left) / max(len(values), 1)
    for index, value in enumerate(values):
        x0 = left + index * width + width * 0.15
        x1 = x0 + width * 0.7
        y1 = bottom - (value / ymax) * (bottom - top)
        draw.rectangle([x0, y1, x1, bottom], fill=PALETTE[index % len(PALETTE)])
        draw.text((x0, bottom + 12), groups[index][:28], fill=(40, 40, 40))
        draw.text((x0 + 4, y1 - 16), f"{value:.2f}", fill=(40, 40, 40))
    if gate is not None:
        y = bottom - (gate / ymax) * (bottom - top)
        draw.line([left, y, right, y], fill=(214, 39, 40), width=2)
    image.save(path)


def lane_banner(draw: ImageDraw.ImageDraw, lanes: list[int], gripper: bool, text: str) -> None:
    rows = [row for row, lane in enumerate(lanes) if (lane in GRIPPER_LANES) == gripper]
    if rows:
        draw.text((MARGIN + 6, 46 + rows[0] * 118), text, fill=(90, 90, 90))


def overlay(path: Path, title: str, truth: np.ndarray, prediction: np.ndarray,
            lanes: list[int], markers: list | None = None) -> None:
    markers = markers or []
    image = Image.new("RGB", (WIDTH, 60 + 118 * len(lanes)), "white")
    draw = ImageDraw.Draw(image)
    left, right = MARGIN, WIDTH - 26
    body_top, body_bottom = 58, 60 + 118 * len(lanes) - 16
    step = (right - left) / max(len(truth) - 1, 1)
    draw.text((MARGIN, 14), title, fill=(20, 20, 20))
    marker_legend(draw, markers)
    for x0, x1, kind, _note in markers:
        if x1 > x0:
            draw.rectangle([left + x0 * step, body_top, left + x1 * step, body_bottom],
                           fill=_pale(MARKERS[kind][0]))
    for row, lane in enumerate(lanes):
        top = 60 + row * 118
        bottom = top + 100
        gripper = lane in GRIPPER_LANES
        series = [truth[:, lane], prediction[:, lane]]
        high = (max(float(np.max(s)) for s in series) if gripper
                else max(float(np.max(np.abs(s))) for s in series)) or 1.0
        zero, gain = ((bottom, (bottom - top) * 0.9 / high) if gripper
                      else ((top + bottom) / 2, (bottom - top) / 2.2 / high))
        draw.rectangle([left, top, right, bottom], outline=(200, 200, 200), width=1)
        draw.line([left, zero, right, zero], fill=(225, 225, 225), width=1)
        draw.text((4, top + 34), JOINT_LABELS[lane], fill=(30, 30, 30))
        for colour, values in zip(((30, 30, 30), (214, 39, 40)), series):
            pixels = [(left + i / max(len(values) - 1, 1) * (right - left), zero - float(v) * gain)
                      for i, v in enumerate(values)]
            for a, b in zip(pixels, pixels[1:]):
                draw.line([a, b], fill=colour, width=2)
        scale = f"scale 0-{high:.1f} %" if gripper else f"scale +/-{high:.1f}"
        draw.text((right - 250, top - 2), scale, fill=(120, 120, 120))
    for x0, x1, kind, _note in markers:
        for x in (x0, x1):
            draw.line([left + x * step, body_top, left + x * step, body_bottom],
                      fill=MARKERS[kind][0], width=2)
    lane_banner(draw, lanes, False, "joint lanes, absolute degrees")
    lane_banner(draw, lanes, True, "gripper lanes, percent of travel")
    image.save(path)


def record_markers(rows: list[dict], horizon_index: int) -> list[tuple[float, float, str, str]]:
    """Event guides for one episode page, positioned in window-index units.

    The disturbance band is the only frame bound the records support: the exporter drops windows
    that straddle the perturbation, so the last pre-event window ends before it and the first
    post-event window starts ten frames after it.  Close and release moments come from the event
    lists the evaluator wrote per window; a moment absent from the records is not drawn.
    """
    starts = [int(row["start"]) for row in rows]

    def position(frame: float) -> float:
        return float(np.interp(frame - horizon_index, starts, range(len(starts))))

    marks = []
    phases = [row.get("disturbance_phase") for row in rows]
    pre = [start for start, phase in zip(starts, phases) if phase == "pre_event"]
    post = [start for start, phase in zip(starts, phases) if phase == "post_event"]
    if pre and post:
        low, high = max(pre) + 10, min(post) - 10
        if low <= high:
            marks.append((position(low), position(high), "disturbance",
                          f"f{low}" if low == high else f"f{low}-f{high}"))
    for row in rows:
        for kind, field in EVENT_FIELDS:
            for event in row.get(field) or []:
                frame = int(row["start"]) + int(event["frame"])
                marks.append((position(frame), position(frame), kind, ""))
    return marks


def diagnostic_markers(events_path: Path | None, rows: list[dict], horizon_index: int,
                       fps: float) -> list:
    """Guides for the moments the expert recorded in its own episode diagnostics.

    The evaluator only sees frozen observations, so detection/correction/regrasp come from the
    recorded event stream of the collection the episode came from.  When that stream is not on this
    host the figure simply carries the record-derived guides.
    """
    if events_path is None or not events_path.is_file() or not rows:
        return []
    events = [json.loads(line) for line in
              events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    event = next((row for row in events if row.get("type") == "perturbation"), None)
    if event is None:
        return []
    starts = sorted(int(row["start"]) for row in rows)

    def frame_of(row: dict) -> int:
        return int(round(float(row.get("sim_time", row.get("sim_time_s", 0.0))) * fps))

    def position(frame: int) -> float:
        return float(np.interp(frame - horizon_index, starts, range(len(starts))))

    after = float(event.get("sim_time_s", event.get("sim_time", 0.0)))
    marks = []
    for kind, wanted in (("detection", "expert_replan"), ("regrasp", "expert_recovery_settle")):
        for row in events:
            if row.get("type") == wanted and float(row.get("sim_time", row.get("sim_time_s", 0.0))) >= after:
                frame = frame_of(row)
                marks.append((position(frame), position(frame), kind, f"f{frame}"))
    for row in events:
        if (row.get("type") == "plan_start" and row.get("object") == event.get("object")
                and float(row.get("sim_time", 0.0)) >= after):
            frame = frame_of(row)
            marks.append((position(frame), position(frame), "correction", f"f{frame}"))
    return marks


def heatmap(path: Path, title: str, matrix: np.ndarray, rows: list[str], cols: list[str]) -> None:
    cell = 52
    image = Image.new("RGB", (220 + cell * len(cols), 120 + cell * len(rows)), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 16), title, fill=(20, 20, 20))
    high = float(np.max(matrix)) if matrix.size else 1.0
    for r, row_name in enumerate(rows):
        draw.text((18, 70 + r * cell + 16), row_name, fill=(30, 30, 30))
        for c in range(len(cols)):
            value = float(matrix[r, c]) / high if high else 0.0
            colour = _ramp(value)
            x0, y0 = 220 + c * cell, 70 + r * cell
            draw.rectangle([x0, y0, x0 + cell - 3, y0 + cell - 3], fill=colour)
            draw.text((x0 + 6, y0 + 16), f"{matrix[r, c]:.1f}",
                      fill=(255, 255, 255) if value > 0.55 else (30, 30, 30))
    for c, column in enumerate(cols):
        draw.text((220 + c * cell + 10, 48), column, fill=(30, 30, 30))
    image.save(path)


def pdf(path: Path, pages: list[Path]) -> None:
    images = [Image.open(page).convert("RGB") for page in pages]
    if images:
        images[0].save(path, save_all=True, append_images=images[1:])


def reports(root: Path, split: str = "validation",
            per_copy_dir: Path | None = None) -> list[dict]:
    """Every report that belongs on the curve: one split, the frozen window set, no superseded run.

    A re-scored directory left beside the live tree would otherwise draw a second line for the same
    checkpoint step, and the event-anchored pass would add a step that is not a checkpoint.
    """
    rows = []
    per_copy_dir = per_copy_dir or (root / "per_copy")
    for path in sorted(root.glob("*/report.json")):
        if "superseded" in path.parent.name:
            continue
        report = common.load_json(path)
        per_copy_path = per_copy_dir / f"{path.parent.name}.json"
        if not per_copy_path.is_file():
            per_copy_path = path.parent / "per_copy.json"
        if per_copy_path.is_file():
            report["per_copy"] = common.load_json(per_copy_path)
        if report.get("split") != split:
            continue
        if (report.get("window_set") or "frozen_plan") != "frozen_plan":
            continue
        rows.append((report, path.parent))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None,
                        help="summarize_e4t_openloop output, used to name the best checkpoint")
    parser.add_argument("--gate", type=float, default=2.0)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--lr-trace", type=Path, default=None,
                        help="e4t_lr_trace.py output; plotted in units of 1e-5")
    parser.add_argument("--run-config-sha256", default=None,
                        help="refuse an LR trace that was built from another run config")
    parser.add_argument("--collection", type=Path, action="append", default=None,
                        help="collection directories holding diagnostics/<attempt_id>/events.jsonl")
    parser.add_argument("--view", type=Path, default=None,
                        help="frozen training view; lets the overlays mark the expert's own "
                             "detection/correction/regrasp moments from the episode diagnostics")
    parser.add_argument("--curves-only", action="store_true",
                        help="write aggregate curves and diagnostics without per-episode overlays")
    parser.add_argument("--per-copy-dir", type=Path, default=None,
                        help="directory holding <label>.json per-copy re-derivations")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    found = reports(args.eval_root, args.split, args.per_copy_dir)
    if not found:
        raise SystemExit(f"no reports under {args.eval_root}")
    best_step = None
    if args.summary and args.summary.is_file():
        summary = common.load_json(args.summary)
        selected = summary.get("best_checkpoint") or summary.get("diagnostic_checkpoint") or {}
        best_step = selected.get("step")

    def value(report, group, key):
        per_copy = ((report.get("per_copy") or {}).get("aggregates") or {}).get(group) or {}
        return per_copy.get(key) if key in per_copy else (
            (report.get("aggregates") or {}).get(group) or {}).get(key)

    steps = [int(r.get("checkpoint_train_state_step", i)) for i, (r, _) in enumerate(found)]
    ordered = sorted(zip(steps, found), key=lambda pair: pair[0])
    series = {
        "clean active MAE": [(step, value(r, "clean_active", "joint_macro_mae_deg"))
                             for step, (r, _) in ordered],
        "recovery post-event active MAE": [(step, value(r, "recovery_post_active", "joint_macro_mae_deg"))
                                           for step, (r, _) in ordered],
        "clean active P95": [(step, value(r, "clean_active", "joint_p95_deg"))
                             for step, (r, _) in ordered],
        "recovery post-event P95": [(step, value(r, "recovery_post_active", "joint_p95_deg"))
                                    for step, (r, _) in ordered],
        "gripper MAE (clean)": [(step, value(r, "clean_active", "gripper_macro_mae_pct"))
                                for step, (r, _) in ordered],
        "gripper MAE (recovery)": [(step, value(r, "recovery_post_active", "gripper_macro_mae_pct"))
                                   for step, (r, _) in ordered],
    }
    line_chart(args.output / "curve_error_steps.png", "open-loop action error vs optimizer step",
               "TrainState.step", "degrees / percent", series, gate=args.gate)
    measured = [key for key in ("A", "B", "C")
                if int(value(found[-1][0], f"recovery_{key}_post_active", "windows") or 0) > 0]
    bar_chart(args.output / "curve_per_category.png",
              "post-disturbance active MAE by perturbation category", "degrees",
              [f"{key}" for key in measured],
              [float(value(found[-1][0], f"recovery_{key}_post_active", "joint_macro_mae_deg") or 0.0)
               for key in measured], gate=args.gate)
    line_chart(args.output / "curve_flow_loss.png",
               "flow-matching loss vs optimizer step (not a substitute for the action metrics)",
               "TrainState.step", "MSE",
               {"clean active true-12": [(step, value(r, "clean_active", "true12_mse"))
                                          for step, (r, _) in ordered],
                "recovery post-event active true-12": [
                    (step, value(r, "recovery_post_active", "true12_mse")) for step, (r, _) in ordered],
                "recovery post-event active padding-20": [
                    (step, value(r, "recovery_post_active", "padding20_mse"))
                    for step, (r, _) in ordered]})
    if args.lr_trace and Path(args.lr_trace).is_file():
        trace = common.load_json(Path(args.lr_trace).expanduser().resolve())
        if trace.get("run_config_sha256") and args.run_config_sha256 and \
                trace["run_config_sha256"] != args.run_config_sha256:
            raise SystemExit(f"the LR trace was built from a different run config "
                             f"({trace['run_config_sha256']} != {args.run_config_sha256})")
        line_chart(args.output / "curve_lr.png",
                   "learning rate vs optimizer step (evaluated with the schedule object the trainer "
                   "builds, inside the training image)",
                   "TrainState.step", "learning rate x 1e-5",
                   {"lr": [(int(row["step"]), float(row["lr"]) * 1e5) for row in trace["points"]]})

    horizon_step = best_step if best_step in {step for step, _ in ordered} else ordered[-1][0]
    horizon_report = next(report for step, (report, _) in ordered if step == horizon_step)
    horizon_groups = ((horizon_report.get("per_copy") or {}).get("horizon_metrics_by_group") or {})
    horizon_series = {}
    horizon_p95_series = {}
    gripper_horizon_series = {}
    for key, label in (("clean_active", "clean active"),
                       ("recovery_post_active", "recovery post-event active")):
        group = horizon_groups.get(key) or {}
        means = group.get("joint_macro_mae_deg") or {}
        p95s = group.get("joint_p95_deg") or {}
        grip_mae = group.get("gripper_macro_mae_pct") or {}
        if means:
            horizon_series[label] = [(h, means.get(str(h))) for h in range(10)]
            horizon_p95_series[label] = [(h, p95s.get(str(h))) for h in range(10)]
            gripper_horizon_series[label] = [(h, grip_mae.get(str(h))) for h in range(10)]
    if not horizon_series:
        raise SystemExit("selected checkpoint lacks per-copy horizon metrics")
    line_chart(args.output / "curve_horizon.png",
               f"episode-equal joint MAE by prediction horizon (step {horizon_step})",
               "action horizon (control frames)", "degrees", horizon_series)
    line_chart(args.output / "curve_horizon_p95.png",
               f"pooled joint P95 by prediction horizon (step {horizon_step})",
               "action horizon (control frames)", "degrees", horizon_p95_series)
    line_chart(args.output / "curve_horizon_gripper.png",
               f"episode-equal gripper MAE by prediction horizon (step {horizon_step})",
               "action horizon (control frames)", "percent of travel", gripper_horizon_series)

    view_rows: dict[int, dict] = {}
    view_fps = 50.0
    if args.view:
        view_meta = common.load_json(Path(args.view).expanduser().resolve() / "meta.json")
        view_rows = {int(row["episode_index"]): row for row in view_meta["episodes"]}
        view_fps = float(view_meta.get("fps", 50))

    def episode_events(episode: int) -> Path | None:
        """The diagnostics directory is named by attempt id, which the view metadata carries."""
        attempt = str(view_rows.get(episode, {}).get("attempt_id") or "")
        if not attempt:
            return None
        for root in args.collection or []:
            candidate = Path(root).expanduser().resolve() / "diagnostics" / attempt / "events.jsonl"
            if candidate.is_file():
                return candidate
        return None

    by_step = {step: (report, folder) for step, (report, folder) in ordered}
    chosen_step = best_step if best_step in by_step else ordered[-1][0]
    report, folder = by_step[chosen_step]
    with gzip.open(folder / "records.jsonl.gz", "rt", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    arrays = np.load(folder / "arrays.npz")
    truth = np.asarray(arrays["ground_truth"])
    all_predictions = np.asarray(arrays["predictions"])
    # the evaluator stores one ground-truth window per row and the noise copies behind it
    if truth.ndim == 4:
        truth = truth[:, 0]
    if all_predictions.ndim == 4:
        # per-window error averaged over sampled copies (never over the trajectory itself)
        window_joint_mae = np.abs(
            all_predictions - truth[:, None, :, :])[:, :, :, JOINT_LANES].mean(axis=(1, 2, 3))
        prediction = all_predictions[:, 0]
    else:
        window_joint_mae = np.abs(
            all_predictions - truth)[:, :, JOINT_LANES].mean(axis=(1, 2))
        prediction = all_predictions
    horizon_groups = ((report.get("per_copy") or {}).get("horizon_metrics_by_group") or {})
    for key, suffix, label in (("clean_active", "clean", "clean active"),
                               ("recovery_post_active", "recovery", "recovery post-event active")):
        group = horizon_groups.get(key) or {}
        by_joint = group.get("joint_mae_deg_by_joint") or {}
        if not by_joint:
            continue
        matrix = np.asarray([by_joint[str(h)] for h in range(10)], dtype=float).T
        filename = ("joint_horizon_heatmap.png" if suffix == "clean"
                    else f"joint_horizon_heatmap_{suffix}.png")
        heatmap(args.output / filename,
                f"episode-equal mean absolute error per joint and horizon: {label} "
                f"(step {report['checkpoint_train_state_step']})",
                matrix, [JOINT_LABELS[lane] for lane in JOINT_LANES],
                [f"h{h}" for h in range(10)])
        grip_by_joint = group.get("gripper_mae_pct_by_joint") or {}
        if grip_by_joint:
            grip_matrix = np.asarray([grip_by_joint[str(h)] for h in range(10)], dtype=float).T
            filename = ("gripper_horizon_heatmap.png" if suffix == "clean"
                        else f"gripper_horizon_heatmap_{suffix}.png")
            heatmap(args.output / filename,
                    f"episode-equal gripper error by horizon: {label} "
                    f"(step {report['checkpoint_train_state_step']})",
                    grip_matrix, ["L grip", "R grip"], [f"h{h}" for h in range(10)])

    def mean(values):
        values = [float(value) for value in values]
        return sum(values) / len(values) if values else None

    def group(rows, predicate):
        values = [float(window_joint_mae[index]) for index, row in enumerate(rows)
                  if predicate(row) and index < len(window_joint_mae)]
        return mean(values)

    phase_series = {
        "static windows": group(records, lambda row: row["phase"] == "static"),
        "active windows": group(records, lambda row: row["phase"] == "active"),
        "active + expert close inside": group(
            records, lambda row: row["phase"] == "active" and bool(row.get("gt_close_events"))),
        "active + expert release inside": group(
            records, lambda row: row["phase"] == "active" and bool(row.get("gt_release_events"))),
        "clean active": group(records, lambda row: row["category"] == "clean"
                              and row["phase"] == "active"),
        "recovery post-event active": group(
            records, lambda row: row["category"] != "clean"
            and row["phase"] == "active" and row.get("disturbance_phase") == "post_event"),
        "recovery pre-event active": group(
            records, lambda row: row["category"] != "clean"
            and row["phase"] == "active" and row.get("disturbance_phase") == "pre_event"),
    }
    phase_labels = [key for key, value in phase_series.items() if value is not None]
    bar_chart(args.output / "curve_phase_compare.png",
              f"joint MAE by activity and key phase (step "
              f"{report['checkpoint_train_state_step']})", "degrees",
              phase_labels, [float(phase_series[key]) for key in phase_labels], gate=args.gate)

    per_episode: dict[int, list[int]] = {}
    for index, record in enumerate(records):
        if index < truth.shape[0]:
            per_episode.setdefault(int(record["episode"]), []).append(index)
    if not args.curves_only:
        overlay_pages = []
        lanes = JOINT_LANES + GRIPPER_LANES
        for episode, indices in sorted(per_episode.items()):
            indices.sort(key=lambda index: int(records[index]["start"]))
            rows = [records[index] for index in indices]
            for horizon_index in (0, 4, 9):
                page = args.output / f"episode_{episode:03d}_h{horizon_index}.png"
                overlay(page, f"episode {episode} horizon {horizon_index}  (black ground truth, "
                              f"red sampled copy 0) step {report['checkpoint_train_state_step']}",
                        truth[indices][:, horizon_index, :], prediction[indices][:, horizon_index, :],
                        lanes, record_markers(rows, horizon_index)
                        + diagnostic_markers(episode_events(episode), rows, horizon_index, view_fps))
                overlay_pages.append(page)
        pdf(args.output / f"overlays_{args.split}.pdf", overlay_pages)
    pdf(args.output / "curves.pdf", [args.output / name for name in
                                     ("curve_error_steps.png", "curve_horizon.png",
                                      "curve_horizon_p95.png", "curve_horizon_gripper.png",
                                      "curve_per_category.png",
                                      "joint_horizon_heatmap.png",
                                      "joint_horizon_heatmap_recovery.png",
                                      "gripper_horizon_heatmap.png",
                                      "gripper_horizon_heatmap_recovery.png")
                                     if (args.output / name).exists()])
    manifest = {
        "schema_version": 1, "branch": "so101", "phase": "E3", "substage": "E4-T-T2-curves",
        "split": args.split, "checkpoints": steps, "best_step_reported": best_step,
        "figure_root": args.output.name,
        "curves_only": bool(args.curves_only),
        "overlay_episode_ids": (sorted(per_episode) if not args.curves_only else []),
        "overlay_episode_splits": ({str(episode): args.split for episode in sorted(per_episode)}
                                    if not args.curves_only else {}),
        "overlay_episode_split_source": "evaluation reports filtered by the requested --split",
        "overlay_horizons": ([0, 4, 9] if not args.curves_only else []),
        "overlay_files_by_episode": ({str(episode): [
            f"episode_{episode:03d}_h{horizon}.png" for horizon in (0, 4, 9)]
            for episode in sorted(per_episode)} if not args.curves_only else {}),
        "figures": sorted(p.name for p in args.output.glob("*")),
        "gate_line_deg": args.gate,
        "claim": "open-loop prediction curves on recorded expert observations; "
                 "no policy action was executed in simulation",
    }
    common.write_json(args.output / "figure_manifest.json", manifest)
    print(json.dumps({"checkpoints": steps, "figures": manifest["figures"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
