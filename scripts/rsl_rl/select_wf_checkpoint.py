#!/usr/bin/env python3
"""Rank saved WF Wheel checkpoints using rolling TensorBoard stability metrics."""

from __future__ import annotations

import argparse
import math
import pathlib
import re
import statistics
from dataclasses import dataclass

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TAGS = {
    "base_contact": "Episode/Episode_Termination/base_contact",
    "timeout": "Episode/Episode_Termination/time_out",
    "height_rmse": "Episode/Metrics/body_height/height_tracking_rmse_m",
    "xy_error": "Episode/Metrics/base_velocity/error_vel_xy",
    "yaw_error": "Episode/Metrics/base_velocity/error_vel_yaw",
    "terrain_level": "Episode/Curriculum/terrain_levels",
}
MODEL_PATTERN = re.compile(r"model_(\d+)\.pt$")


@dataclass(frozen=True)
class CheckpointMetrics:
    iteration: int
    path: pathlib.Path
    base_contact: float
    timeout: float
    height_rmse: float
    xy_error: float
    yaw_error: float
    terrain_level: float


def _rolling_mean_at(events, iteration: int, window: int) -> float:
    eligible = [event.value for event in events if event.step <= iteration]
    if not eligible:
        return math.nan
    return statistics.fmean(eligible[-window:])


def load_checkpoint_metrics(run_dir: pathlib.Path, window: int) -> list[CheckpointMetrics]:
    """Read metrics at every saved model iteration in ``run_dir``."""
    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags()["scalars"])
    missing = [tag for tag in TAGS.values() if tag not in available]
    if missing:
        raise RuntimeError(f"Missing required TensorBoard scalars: {missing}")
    events = {name: accumulator.Scalars(tag) for name, tag in TAGS.items()}

    rows = []
    for path in sorted(run_dir.glob("model_*.pt")):
        match = MODEL_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        iteration = int(match.group(1))
        if iteration == 0:
            continue
        values = {name: _rolling_mean_at(series, iteration, window) for name, series in events.items()}
        if not all(math.isfinite(value) for value in values.values()):
            continue
        rows.append(CheckpointMetrics(iteration=iteration, path=path, **values))
    return rows


def is_eligible(row: CheckpointMetrics, args: argparse.Namespace) -> bool:
    """Require acceptable task performance before minimizing fall rate."""
    return (
        row.height_rmse <= args.max_height_rmse
        and row.xy_error <= args.max_xy_error
        and row.yaw_error <= args.max_yaw_error
        and row.terrain_level >= args.min_terrain_level
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select the lowest-base-contact WF Wheel checkpoint after applying minimum height, velocity, "
            "and terrain-performance requirements."
        )
    )
    parser.add_argument("run_dir", type=pathlib.Path, help="Completed WF Wheel run directory")
    parser.add_argument("--window", type=int, default=100, help="Rolling iteration window (default: 100)")
    parser.add_argument("--max_height_rmse", type=float, default=0.08, help="Maximum height RMSE in metres")
    parser.add_argument("--max_xy_error", type=float, default=0.35, help="Maximum planar velocity error")
    parser.add_argument("--max_yaw_error", type=float, default=0.40, help="Maximum yaw-rate error")
    parser.add_argument("--min_terrain_level", type=float, default=3.5, help="Minimum mean terrain level")
    parser.add_argument("--top", type=int, default=10, help="Number of eligible checkpoints to show")
    parser.add_argument(
        "--write_selection",
        action="store_true",
        help="Write selected_stability_checkpoint.txt for automatic MuJoCo deployment",
    )
    args = parser.parse_args()

    if args.window <= 0 or args.top <= 0:
        parser.error("--window and --top must be positive")
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        parser.error(f"Run directory does not exist: {run_dir}")

    rows = load_checkpoint_metrics(run_dir, args.window)
    if not rows:
        raise RuntimeError(f"No usable model checkpoints found in {run_dir}")
    eligible = [row for row in rows if is_eligible(row, args)]
    candidates = eligible if eligible else rows
    candidates.sort(key=lambda row: (row.base_contact, row.height_rmse, row.xy_error, row.yaw_error))

    if not eligible:
        print("WARNING: no checkpoint met every performance threshold; showing fallback stability ranking.")
    print("iteration  base_contact  timeout  height_rmse  xy_error  yaw_error  terrain")
    for row in candidates[: args.top]:
        print(
            f"{row.iteration:9d}  {row.base_contact:12.4f}  {row.timeout:7.4f}  "
            f"{row.height_rmse:11.4f}  {row.xy_error:8.4f}  {row.yaw_error:9.4f}  "
            f"{row.terrain_level:7.3f}"
        )
    selected = candidates[0]
    print(f"\nRecommended checkpoint: {selected.path}")
    if args.write_selection:
        marker = run_dir / "selected_stability_checkpoint.txt"
        marker.write_text(f"{selected.path.name}\n", encoding="utf-8")
        print(f"Wrote deployment selection: {marker}")


if __name__ == "__main__":
    main()
