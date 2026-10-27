# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""本地记录的纵向 meta-action 分类。"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from meta_action.data_structures.ego_meta_action import clean_longitudinal_segments
from meta_action.processor.longitudinal.classification import (
    GentleAcceleration as _PaiGentleAcceleration,
)
from meta_action.processor.longitudinal.classification import (
    Reverse as _PaiReverse,
)
from meta_action.processor.longitudinal.classification import (
    Stop as _PaiStop,
)
from meta_action.processor.longitudinal.classification import (
    StrongAcceleration as _PaiStrongAcceleration,
)
from meta_action.processor.longitudinal.classification import (
    classify_longitudinal,
)
from meta_action.processor.longitudinal.config import (
    DEFAULT_LONGITUDINAL_MIN_LEN_TICKS,
    LONGITUDINAL_MIN_LEN_TICKS_BY_CAPTION,
)
from meta_action.processor.longitudinal.segmentation import segment_longitudinal
from meta_action.utils.constant import DELTA_TIMESTAMP

from .types import PositionMotionSample

# 使用与 PAI prompt 一致的显示名称。
NOT_AVAILABLE = "Not Available"
REVERSE = "Reverse"
STOP = "Stop"
STRONG_ACCELERATION = "Strong Acceleration"
GENTLE_ACCELERATION = "Gentle Acceleration"
STRONG_DECELERATION = "Strong Deceleration"
GENTLE_DECELERATION = "Gentle Deceleration"
MAINTAIN_SPEED = "Maintain Speed"

# PAI 分类阈值与区间长度；本地结果仍须人工核对，不能视作动作真值。
ACTION_GRID_INTERVAL_SEC = Decimal(str(DELTA_TIMESTAMP))
DEFAULT_MIN_ACTION_DURATION_SEC = (
    Decimal(DEFAULT_LONGITUDINAL_MIN_LEN_TICKS) * ACTION_GRID_INTERVAL_SEC
)
MAINTAIN_SPEED_MIN_DURATION_SEC = (
    Decimal(LONGITUDINAL_MIN_LEN_TICKS_BY_CAPTION["MaintainSpeed"])
    * ACTION_GRID_INTERVAL_SEC
)


def _action_threshold(action: type, property_name: str, occurrence: int = 0) -> float:
    """从 PAI 动作类的阈值表读取审计中展示的阈值。"""
    matches = [
        float(threshold.threshold)
        for threshold in action.THRESHOLDS
        if threshold.property_name == property_name
    ]
    return matches[occurrence]


REVERSE_SPEED_MPS = abs(
    _action_threshold(_PaiReverse, "average_speed_along_heading")
)
STOP_SPEED_MPS = _action_threshold(_PaiStop, "average_speed")
GENTLE_ACCELERATION_MPS2 = _action_threshold(
    _PaiGentleAcceleration, "average_acceleration"
)
STRONG_ACCELERATION_MPS2 = _action_threshold(
    _PaiStrongAcceleration, "average_acceleration"
)
MIN_ACTION_DELTA_VELOCITY_MPS = _action_threshold(
    _PaiGentleAcceleration, "delta_velocity"
)

# 与 PAI segment_longitudinal 的默认分段参数一致。
SMOOTHING_WINDOW_SAMPLES = 5
ACCELERATION_HYSTERESIS_FRACTION = 0.15
MAX_ACTION_SAMPLE_TIME_ERROR_SEC = Decimal("0.05")

_PAI_TO_LOCAL_LABEL = {
    "Reverse": REVERSE,
    "Stop": STOP,
    "StrongAcceleration": STRONG_ACCELERATION,
    "GentleAcceleration": GENTLE_ACCELERATION,
    "StrongDeceleration": STRONG_DECELERATION,
    "GentleDeceleration": GENTLE_DECELERATION,
    "MaintainSpeed": MAINTAIN_SPEED,
}


@dataclass(frozen=True)
class LongitudinalActionPoint:
    """一个 10 Hz 轨迹点及其规则分类结果。"""

    relative_time: Decimal
    label: str | None
    v_long_mps: float | None
    speed_2d_mps: float | None
    acceleration_mps2: float | None
    valid: bool
    connected_to_previous: bool


@dataclass(frozen=True)
class LongitudinalActionSpan:
    """短动作清理后的连续动作区间。"""

    label: str
    start_time: Decimal
    end_time: Decimal


@dataclass(frozen=True)
class LongitudinalActionResult:
    """10 Hz 分类点及汇总后的动作区间。"""

    points: tuple[LongitudinalActionPoint, ...]
    spans: tuple[LongitudinalActionSpan, ...]

    def labels_at(self, target_times: Sequence[Decimal]) -> list[str | None]:
        """按请求时刻查找最近的有效动作标签。

        CoC 图片是 2 Hz，分类器使用 10 Hz。比如 ``t=+1.5`` 的图片会取
        ``t=+1.5`` 的 10 Hz 标签；目标附近没有有效点时返回 ``None``。
        """
        valid_points = [point for point in self.points if point.valid and point.label]
        result: list[str | None] = []
        for target in target_times:
            if not valid_points:
                result.append(None)
                continue
            point = min(valid_points, key=lambda item: abs(item.relative_time - target))
            result.append(
                point.label
                if abs(point.relative_time - target) <= MAX_ACTION_SAMPLE_TIME_ERROR_SEC
                else None
            )
        return result


def format_local_meta_action_text(
    target_times: Sequence[Decimal],
    result: LongitudinalActionResult | None,
) -> str:
    """格式化与 PAI 相同结构的本地纵向动作文本。

    例如，图片时刻为 ``[-0.5, 0.0, 0.5]`` 时，会生成三行纵向动作；
    港口数据没有 lane graph，因此 lateral 和 lane 保持 ``Not Available``。
    """
    labels = (
        result.labels_at(target_times)
        if result is not None
        else [None] * len(target_times)
    )
    lines = ["Meta-actions by sampled timestamp (relative to event, unit: seconds):"]
    for target, label in zip(target_times, labels):
        lines.append(
            f"(t={target:+.1f}s) longitudinal: {label or NOT_AVAILABLE}, "
            f"lateral: {NOT_AVAILABLE}, lane: {NOT_AVAILABLE}"
        )
    return "\n".join(lines)


def _valid_runs(samples: Sequence[PositionMotionSample]) -> list[list[PositionMotionSample]]:
    """按无效运动样本拆分序列，避免 PAI 分段跨越数据缺口。"""
    runs: list[list[PositionMotionSample]] = []
    current: list[PositionMotionSample] = []
    for sample in samples:
        values = (sample.body_velocity_forward, sample.ground_speed)
        if any(value is None or not np.isfinite(value) for value in values):
            if current:
                runs.append(current)
                current = []
            continue
        current.append(sample)
    if current:
        runs.append(current)
    return runs


def _classify_valid_run(
    run: Sequence[PositionMotionSample],
) -> list[LongitudinalActionPoint]:
    """调用 PAI 的纵向分段、分类和短段清理，处理一段连续的 10 Hz 数据。"""
    # 本地地速和车体前向速度分别对应 PAI 的这两个纵向输入字段。
    agent_speed = np.asarray([float(sample.ground_speed) for sample in run])
    agent_speed_along_heading = np.asarray(
        [float(sample.body_velocity_forward) for sample in run]
    )
    if len(run) < 2:
        sample = run[0]
        return [
            LongitudinalActionPoint(
                relative_time=sample.relative_time,
                label=None,
                v_long_mps=agent_speed_along_heading[0],
                speed_2d_mps=agent_speed[0],
                acceleration_mps2=None,
                valid=True,
                connected_to_previous=sample.connected_to_previous,
            )
        ]
    scenario = SimpleNamespace(
        all_agents_names=["ego"],
        agent_trajdata={
            "agent_speed": agent_speed[None, :],
            "agent_speed_along_heading": agent_speed_along_heading[None, :],
        },
    )

    segmented = segment_longitudinal(
        scenario,
        "ego",
        accel_smoothing_window=SMOOTHING_WINDOW_SAMPLES,
        hysteresis_fraction=ACCELERATION_HYSTERESIS_FRACTION,
    )
    classified = classify_longitudinal(segmented)
    cleaned = clean_longitudinal_segments(classified)

    # PAI 的加速度是地速的一阶前向差分，首点沿用首个差分值。
    dt = float(DELTA_TIMESTAMP)
    speed_acceleration = np.diff(agent_speed) / dt
    speed_acceleration = np.concatenate(([speed_acceleration[0]], speed_acceleration))

    action_segments = [
        (
            float(row.segment_start_time),
            float(row.segment_start_time + row.duration),
            _PAI_TO_LOCAL_LABEL.get(str(row.caption)),
        )
        for row in cleaned.itertuples(index=False)
    ]
    last_sample_time = (len(run) - 1) * dt
    if action_segments and abs(action_segments[-1][1] - last_sample_time) < 1e-7:
        start, end, label = action_segments[-1]
        action_segments[-1] = (start, end + dt, label)

    points: list[LongitudinalActionPoint] = []
    for index, sample in enumerate(run):
        elapsed = index * dt
        label = next(
            (
                segment_label
                for start, end, segment_label in action_segments
                if start <= elapsed < end
            ),
            None,
        )
        points.append(
            LongitudinalActionPoint(
                relative_time=sample.relative_time,
                label=label,
                v_long_mps=agent_speed_along_heading[index],
                speed_2d_mps=agent_speed[index],
                acceleration_mps2=float(speed_acceleration[index]),
                valid=True,
                connected_to_previous=sample.connected_to_previous,
            )
        )
    return points


def _label_runs(
    labels: Sequence[str | None],
) -> Iterator[tuple[int, int, str | None]]:
    """逐段返回相同标签的半开区间 ``[start, end)``。"""
    start = 0
    while start < len(labels):
        label = labels[start]
        end = start + 1
        while end < len(labels) and labels[end] == label:
            end += 1
        yield start, end, label
        start = end


def _make_spans(points: Sequence[LongitudinalActionPoint]) -> tuple[LongitudinalActionSpan, ...]:
    """把连续点标签汇总成人类可读的时间区间。"""
    spans: list[LongitudinalActionSpan] = []
    labels = [point.label for point in points]
    for index, end, label in _label_runs(labels):
        if label is None:
            continue
        if end < len(points):
            end_time = points[end].relative_time
        elif len(points) > 1:
            end_time = points[end - 1].relative_time + ACTION_GRID_INTERVAL_SEC
        else:
            end_time = points[index].relative_time
        spans.append(
            LongitudinalActionSpan(
                label=label,
                start_time=points[index].relative_time,
                end_time=end_time,
            )
        )
    return tuple(spans)


def classify_longitudinal_series(
    samples: Sequence[PositionMotionSample],
) -> LongitudinalActionResult:
    """把本地 10 Hz 运动数据送入 PAI 的纵向动作处理流程。

    输入按无效样本切成连续段；每段分别进行 PAI 分段、分类和短段清理，
    因此分类结果不会跨越定位数据缺口。PAI 使用地速和车体前向速度。
    """
    ordered = list(samples)
    if any(
        later.relative_time <= earlier.relative_time
        for earlier, later in zip(ordered, ordered[1:])
    ):
        raise ValueError("Longitudinal motion timestamps must be strictly increasing")

    valid_by_time: dict[Decimal, LongitudinalActionPoint] = {}
    for run in _valid_runs(ordered):
        valid_by_time.update(
            (point.relative_time, point) for point in _classify_valid_run(run)
        )

    points = [
        valid_by_time.get(
            sample.relative_time,
            LongitudinalActionPoint(
                relative_time=sample.relative_time,
                label=None,
                v_long_mps=None,
                speed_2d_mps=None,
                acceleration_mps2=None,
                valid=False,
                connected_to_previous=False,
            ),
        )
        for sample in ordered
    ]
    return LongitudinalActionResult(tuple(points), _make_spans(points))


def write_longitudinal_action_audit(
    result: LongitudinalActionResult,
    output_dir: Path,
    record_id: str,
    event_timestamp: Decimal,
) -> Path:
    """把 10 Hz 动作点和汇总区间写成可审计 JSON。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "longitudinal_meta_action.json"
    payload = {
        "record_id": record_id,
        "event_unix_timestamp_sec": str(event_timestamp),
        "source": "position-derived body-forward velocity from localization",
        "acceleration_source": (
            "finite difference of 10 Hz ground_speed, matching PAI longitudinal preparation"
        ),
        "coordinate_frame": "vehicle FLU interpreted as x=forward, y=left; v_long=body_x",
        "v_lat_and_lane": "not generated for local port recordings",
        "rules": {
            "grid_interval_sec": float(ACTION_GRID_INTERVAL_SEC),
            "acceleration_smoothing_window_samples": SMOOTHING_WINDOW_SAMPLES,
            "acceleration_hysteresis_fraction": ACCELERATION_HYSTERESIS_FRACTION,
            "stop_speed_mps": STOP_SPEED_MPS,
            "reverse_speed_mps": REVERSE_SPEED_MPS,
            "gentle_acceleration_mps2": GENTLE_ACCELERATION_MPS2,
            "strong_acceleration_mps2": STRONG_ACCELERATION_MPS2,
            "minimum_action_duration_sec": float(DEFAULT_MIN_ACTION_DURATION_SEC),
            "maintain_speed_minimum_duration_sec": float(MAINTAIN_SPEED_MIN_DURATION_SEC),
            "minimum_delta_velocity_mps": MIN_ACTION_DELTA_VELOCITY_MPS,
        },
        "spans": [
            {
                "label": span.label,
                "start_time_sec": float(span.start_time),
                "end_time_sec": float(span.end_time),
            }
            for span in result.spans
        ],
        "points": [
            {
                "relative_time_sec": float(point.relative_time),
                "label": point.label,
                "v_long_mps": point.v_long_mps,
                "speed_2d_mps": point.speed_2d_mps,
                "acceleration_mps2": point.acceleration_mps2,
                "valid": point.valid,
                "connected_to_previous": point.connected_to_previous,
            }
            for point in result.points
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path
