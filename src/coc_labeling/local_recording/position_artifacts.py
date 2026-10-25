# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Write localization trajectory and motion audits."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path

import cv2
import numpy as np

from .localization import sample_position_at
from .types import (
    LOCALIZATION_RESAMPLE_INTERVAL_SEC,
    MAX_LOCALIZATION_GAP_SEC,
    POSITION_DERIVATIVE_WINDOW_SEC,
    PositionMotionSample,
    ResampledPosition,
    TimedPosition,
)


def write_position_motion_audit(
    samples: Sequence[PositionMotionSample],
    output_dir: Path,
    record_id: str,
    event_timestamp: Decimal,
    *,
    trajectory_samples: Sequence[ResampledPosition] | None = None,
    window_sec: Decimal = POSITION_DERIVATIVE_WINDOW_SEC,
    max_gap_sec: Decimal = MAX_LOCALIZATION_GAP_SEC,
) -> tuple[Path, Path]:
    """写出位置差分、车体速度审计 JSON 和 map-XY 相对轨迹图。"""
    if not samples:
        raise ValueError("Position motion audit requires at least one target sample")
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "position_motion_audit.json"
    plot_path = output_dir / "position_trajectory.png"

    audit_payload = {
        "record_id": record_id,
        "time_origin": "localization_filename_sampling_timestamp",
        "event_unix_timestamp_sec": str(event_timestamp),
        "position_source": "localization JSON position.x and position.y",
        "velocity_method": (
            "central_difference_on_10hz_resampled_positions_then_inverse_orientation_rotation"
        ),
        "map_frame": "localization position map-XY; axis labels are not stored in JSON",
        "body_frame": (
            "inferred FLU: x=forward, y=left, z=up; localization JSON has no frame label"
        ),
        "orientation_source": (
            "localization JSON orientation quaternion, stored as wxyz and interpreted as "
            "body-FLU-to-map-XY"
        ),
        "orientation_interpolation": "shortest_path_slerp_at_each_velocity_timestamp",
        "body_velocity_conversion": (
            "v_body_flu = inverse(q_body_flu_to_map_xy) * [v_map_x, v_map_y, 0]"
        ),
        "agent_velocity_mapping": (
            "v_long=body_forward; v_lat=-body_left (right-positive)"
        ),
        "recorded_velocity_audit": (
            "The four checked records strongly support interpreting localization "
            "linear_velocity x/y as body-FLU forward/left: rotating those components by "
            "orientation agrees much better with position-derived map velocity than treating "
            "them as map-XY. Compare derived body forward/left components against that "
            "working hypothesis for audit only; producer frame metadata is absent, and this "
            "field is not sent to the agent."
        ),
        "recorded_linear_velocity_frame_hypothesis": (
            "body-FLU-like horizontal frame (x=forward, y=left) is strongly supported by the "
            "four checked records; z-up is assumed, and no frame is declared by producer metadata"
        ),
        "derivative_window_sec": float(window_sec),
        "resampled_position_interval_sec": float(LOCALIZATION_RESAMPLE_INTERVAL_SEC),
        "maximum_raw_interpolation_gap_sec": float(max_gap_sec),
        "maximum_resampled_interpolation_gap_sec": float(
            LOCALIZATION_RESAMPLE_INTERVAL_SEC
        ),
        "coordinate_warning": (
            "Localization JSON has no explicit coordinate-frame labels. For these four records, "
            "map motion bearing closely follows quaternion local-x direction; planning theta "
            "and front-camera scene flow in 20261019153941_record support interpreting local x "
            "as forward (FLU). Independently, rotating linear_velocity x/y as body-FLU by the "
            "quaternion gives substantially lower direction and vector residuals against "
            "position-derived map velocity than treating it as map-XY. This is a strong "
            "dataset-level inference, not producer metadata."
        ),
        "samples": [
            {
                "relative_time_sec": float(sample.relative_time),
                "target_unix_timestamp_sec": str(sample.target_timestamp),
                "status": (
                    "valid"
                    if sample.position_valid and sample.motion_valid
                    else "position_only"
                    if sample.position_valid
                    else "invalid"
                ),
                "relative_position_map_xy_m": (
                    {"x": sample.relative_x, "y": sample.relative_y}
                    if sample.position_valid
                    else None
                ),
                "velocity_map_xy_mps": (
                    {"x": sample.velocity_x, "y": sample.velocity_y}
                    if sample.motion_valid
                    else None
                ),
                "orientation_body_flu_to_map_xy_wxyz": (
                    {
                        "w": sample.orientation_wxyz[0],
                        "x": sample.orientation_wxyz[1],
                        "y": sample.orientation_wxyz[2],
                        "z": sample.orientation_wxyz[3],
                    }
                    if sample.orientation_wxyz is not None
                    else None
                ),
                "velocity_body_flu_mps": (
                    {
                        "forward": sample.body_velocity_forward,
                        "left": -sample.body_velocity_right,
                    }
                    if sample.body_velocity_right is not None
                    and sample.body_velocity_forward is not None
                    else None
                ),
                "v_long_forward_positive_mps": sample.body_velocity_forward,
                "v_lat_right_positive_mps": sample.body_velocity_right,
                "recorded_linear_velocity_xyz_candidate_flu_mps": (
                    {
                        "x": sample.recorded_linear_velocity_xyz[0],
                        "y": sample.recorded_linear_velocity_xyz[1],
                        "z": sample.recorded_linear_velocity_xyz[2],
                    }
                    if sample.recorded_linear_velocity_xyz is not None
                    else None
                ),
                "body_velocity_residual_vs_recorded_xy_mps": (
                    sample.body_velocity_residual_mps
                ),
                "recorded_linear_velocity_as_map_xy_residual_mps": (
                    math.hypot(
                        sample.recorded_linear_velocity_xyz[0] - sample.velocity_x,
                        sample.recorded_linear_velocity_xyz[1] - sample.velocity_y,
                    )
                    if sample.recorded_linear_velocity_xyz is not None
                    and sample.velocity_x is not None
                    and sample.velocity_y is not None
                    else None
                ),
                "ground_speed_mps": sample.ground_speed,
                "trajectory_bearing_map_xy_deg": sample.trajectory_bearing_deg,
                "connected_to_previous": sample.connected_to_previous,
                "position_error": sample.position_error,
                "motion_error": sample.motion_error,
                "body_motion_error": sample.body_motion_error,
            }
            for sample in samples
        ],
    }
    audit_path.write_text(json.dumps(audit_payload, indent=2) + "\n", encoding="utf-8")

    # 使用 OpenCV 绘制独立审计图，避免为简单轨迹图增加 matplotlib 依赖。
    canvas_width, canvas_height = 1000, 760
    left, top, right, bottom = 110, 90, 950, 650
    canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)
    cv2.putText(
        canvas,
        (
            f"Position-derived trajectory: {record_id} (10 Hz)"
            if trajectory_samples is not None
            else f"Position-derived trajectory: {record_id}"
        ),
        (left, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )

    plot_points: list[tuple[Decimal, float, float, bool]] = []
    if trajectory_samples is None:
        plot_points = [
            (
                sample.relative_time,
                float(sample.relative_x),
                float(sample.relative_y),
                sample.connected_to_previous,
            )
            for sample in samples
            if sample.position_valid
            and sample.relative_x is not None
            and sample.relative_y is not None
        ]
    else:
        event_position = next(
            (
                sample
                for sample in trajectory_samples
                if sample.relative_time == 0 and sample.position_valid
            ),
            None,
        )
        if (
            event_position is not None
            and event_position.x is not None
            and event_position.y is not None
        ):
            plot_points = [
                (
                    sample.relative_time,
                    sample.x - event_position.x,
                    sample.y - event_position.y,
                    sample.connected_to_previous,
                )
                for sample in trajectory_samples
                if sample.position_valid and sample.x is not None and sample.y is not None
            ]

    if plot_points:
        x_values = [point[1] for point in plot_points]
        y_values = [point[2] for point in plot_points]
        raw_x_min, raw_x_max = min(x_values), max(x_values)
        raw_y_min, raw_y_max = min(y_values), max(y_values)
        x_span = max(raw_x_max - raw_x_min, 0.5) * 1.24
        y_span = max(raw_y_max - raw_y_min, 0.5) * 1.24
        # 统一 x/y 的米/像素比例，避免把实际轨迹方向画歪。
        meters_per_pixel = max(
            x_span / (right - left), y_span / (bottom - top)
        )
        x_center = (raw_x_min + raw_x_max) / 2
        y_center = (raw_y_min + raw_y_max) / 2
        x_half_span = meters_per_pixel * (right - left) / 2
        y_half_span = meters_per_pixel * (bottom - top) / 2
        x_min, x_max = x_center - x_half_span, x_center + x_half_span
        y_min, y_max = y_center - y_half_span, y_center + y_half_span

        def to_pixel(x: float, y: float) -> tuple[int, int]:
            px = left + round((x - x_min) / (x_max - x_min) * (right - left))
            py = bottom - round((y - y_min) / (y_max - y_min) * (bottom - top))
            return px, py

        # 淡色网格和边框帮助判断轨迹的相对位移尺度。
        for grid_index in range(5):
            x = left + round(grid_index * (right - left) / 4)
            y = top + round(grid_index * (bottom - top) / 4)
            cv2.line(canvas, (x, top), (x, bottom), (230, 230, 230), 1)
            cv2.line(canvas, (left, y), (right, y), (230, 230, 230), 1)
        cv2.rectangle(canvas, (left, top), (right, bottom), (80, 80, 80), 2)
        cv2.putText(
            canvas,
            "Relative localization map x (m)",
            (left + 250, bottom + 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Relative map y (m)",
            (18, top + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

        previous: tuple[int, int] | None = None
        first_relative_time = plot_points[0][0]
        for relative_time, relative_x, relative_y, connected_to_previous in plot_points:
            point = to_pixel(relative_x, relative_y)
            if previous is not None and connected_to_previous:
                cv2.line(canvas, previous, point, (180, 110, 40), 2, cv2.LINE_AA)
            is_event = relative_time == 0
            is_gap_break = (
                trajectory_samples is not None
                and relative_time != first_relative_time
                and not connected_to_previous
            )
            point_color = (
                (30, 150, 30)
                if is_event
                else (20, 20, 220)
                if is_gap_break
                else (190, 90, 20)
            )
            cv2.circle(canvas, point, 6 if is_event or is_gap_break else 3, point_color, -1)
            show_label = not trajectory_samples or relative_time % Decimal("1.0") == 0
            if show_label:
                label = f"{relative_time:+.1f}s"
                cv2.putText(
                    canvas,
                    label,
                    (point[0] + 7, point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (35, 35, 35),
                    1,
                    cv2.LINE_AA,
                )
            previous = point
    else:
        cv2.putText(
            canvas,
            "No target positions could be sampled.",
            (left, top + 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (20, 20, 200),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        "Map-XY path direction is not vehicle heading or body lateral motion.",
        (left, 715),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (50, 50, 50),
        1,
        cv2.LINE_AA,
    )
    if trajectory_samples is not None:
        cv2.putText(
            canvas,
            "Red points mark raw localization gaps over 32 ms.",
            (left, 742),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (40, 40, 180),
            1,
            cv2.LINE_AA,
        )
    if not cv2.imwrite(str(plot_path), canvas):
        raise RuntimeError(f"Could not write position trajectory plot: {plot_path}")
    return audit_path, plot_path


def write_resampled_position_trajectory(
    trajectory: Sequence[ResampledPosition],
    raw_samples: Sequence[TimedPosition],
    output_dir: Path,
    record_id: str,
    event_timestamp: Decimal,
    *,
    max_raw_gap_sec: Decimal = MAX_LOCALIZATION_GAP_SEC,
) -> Path:
    """写出 10 Hz 位置轨迹，并单独列出网格点之间的原始定位大间隔。"""
    if not trajectory:
        raise ValueError("Resampled position trajectory requires at least one point")
    if not raw_samples:
        raise ValueError("Raw localization samples are required for trajectory audit")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "position_trajectory_10hz.json"
    try:
        event_x, event_y = sample_position_at(
            raw_samples, event_timestamp, max_gap_sec=max_raw_gap_sec
        )
        event_position_error = None
    except ValueError as exc:
        event_x = event_y = None
        event_position_error = str(exc)

    range_start = trajectory[0].target_timestamp
    range_end = trajectory[-1].target_timestamp
    raw_gaps = []
    for left, right in zip(raw_samples, raw_samples[1:]):
        gap = right.timestamp - left.timestamp
        if gap <= max_raw_gap_sec or right.timestamp <= range_start or left.timestamp >= range_end:
            continue
        distance = math.hypot(right.x - left.x, right.y - left.y)
        raw_gaps.append(
            {
                "start_relative_time_sec": float(left.timestamp - event_timestamp),
                "end_relative_time_sec": float(right.timestamp - event_timestamp),
                "start_unix_timestamp_sec": str(left.timestamp),
                "end_unix_timestamp_sec": str(right.timestamp),
                "gap_ms": float(gap * 1000),
                "position_delta_m": distance,
                "implied_interval_speed_mps": distance / float(gap),
            }
        )

    payload = {
        "record_id": record_id,
        "time_origin": "localization_filename_sampling_timestamp",
        "event_unix_timestamp_sec": str(event_timestamp),
        "grid_interval_sec": float(LOCALIZATION_RESAMPLE_INTERVAL_SEC),
        "resampling_method": "linear_interpolation_from_raw_localization_positions",
        "maximum_raw_interpolation_gap_sec": float(max_raw_gap_sec),
        "position_reference": "map x/y relative to the event position",
        "event_position_error": event_position_error,
        "raw_sampling_gaps_over_limit": raw_gaps,
        "samples": [
            {
                "relative_time_sec": float(sample.relative_time),
                "target_unix_timestamp_sec": str(sample.target_timestamp),
                "status": "valid" if sample.position_valid else "not_available",
                "relative_position_map_xy_m": (
                    {
                        "x": sample.x - event_x,
                        "y": sample.y - event_y,
                    }
                    if sample.position_valid
                    and sample.x is not None
                    and sample.y is not None
                    and event_x is not None
                    and event_y is not None
                    else None
                ),
                "connected_to_previous": sample.connected_to_previous,
                "position_error": sample.position_error,
            }
            for sample in trajectory
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path
