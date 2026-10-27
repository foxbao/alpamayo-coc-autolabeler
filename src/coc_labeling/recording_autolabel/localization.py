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

"""Load localization positions, resample them to 10 Hz, and derive motion."""

from __future__ import annotations

import bisect
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import replace
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from functools import partial
from pathlib import Path
from typing import TypeVar

from .types import (
    LOCALIZATION_RESAMPLE_INTERVAL_SEC,
    MAX_LOCALIZATION_GAP_SEC,
    POSITION_DERIVATIVE_WINDOW_SEC,
    PositionMotionSample,
    ResampledPosition,
    TimedPosition,
)

_BracketSample = TypeVar("_BracketSample")
_SampleResult = TypeVar("_SampleResult")


def load_localization_positions(localization_dir: Path) -> list[TimedPosition]:
    """读取定位 JSON 中的 map-XY 位置、FLU 到 map 姿态和速度审计字段。

    例如，``1792395581.795200472.json`` 的文件名给出采样时刻，JSON 中的
    ``position.x`` 和 ``position.y`` 给出该时刻的位置坐标。
    """
    if not localization_dir.is_dir():
        raise FileNotFoundError(f"Localization directory not found: {localization_dir}")
    paths = sorted(
        path
        for path in localization_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".json"
    )
    if not paths:
        raise FileNotFoundError(f"No localization JSON files found in {localization_dir}")

    samples: list[TimedPosition] = []
    for path in paths:
        try:
            timestamp = Decimal(path.stem)
        except InvalidOperation as exc:
            raise ValueError(
                f"Localization filename is not a numeric timestamp: {path.name}"
            ) from exc
        if not timestamp.is_finite():
            raise ValueError(f"Localization timestamp must be finite: {path.name}")

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read localization JSON: {path}") from exc
        position = payload.get("position") if isinstance(payload, dict) else None
        if not isinstance(position, dict):
            raise ValueError(f"Localization JSON has no position object: {path}")  # noqa: TRY004

        coordinates: list[float] = []
        for axis in ("x", "y"):
            value = position.get(axis)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(  # noqa: TRY004
                    f"Localization position.{axis} must be numeric: {path}"
                )
            coordinate = float(value)
            if not math.isfinite(coordinate):
                raise ValueError(f"Localization position.{axis} must be finite: {path}")
            coordinates.append(coordinate)
        orientation_wxyz = _read_optional_orientation(payload, path)
        linear_velocity_xyz = _read_optional_linear_velocity(payload, path)
        samples.append(
            TimedPosition(
                timestamp=timestamp,
                x=coordinates[0],
                y=coordinates[1],
                source_path=path,
                orientation_wxyz=orientation_wxyz,
                linear_velocity_xyz=linear_velocity_xyz,
            )
        )

    samples.sort(key=lambda sample: (sample.timestamp, sample.source_path.name))
    for previous, current in zip(samples, samples[1:]):
        if current.timestamp <= previous.timestamp:
            raise ValueError(
                "Localization timestamps must be strictly increasing; "
                f"duplicate timestamp {current.timestamp} in {previous.source_path.name} "
                f"and {current.source_path.name}"
            )
    return samples


def _read_optional_orientation(
    payload: object, path: Path
) -> tuple[float, float, float, float] | None:
    """读取并归一化 JSON 四元数；内部顺序为 wxyz。"""
    if not isinstance(payload, dict):
        return None
    orientation = payload.get("orientation")
    if orientation is None:
        return None
    if not isinstance(orientation, dict):
        raise ValueError(  # noqa: TRY004
            f"Localization orientation must be an object: {path}"
        )

    values: list[float] = []
    for axis in ("w", "x", "y", "z"):
        value = orientation.get(axis)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(  # noqa: TRY004
                f"Localization orientation.{axis} must be numeric: {path}"
            )
        component = float(value)
        if not math.isfinite(component):
            raise ValueError(f"Localization orientation.{axis} must be finite: {path}")
        values.append(component)

    magnitude = math.sqrt(sum(component * component for component in values))
    if magnitude <= 1e-12:
        raise ValueError(f"Localization orientation quaternion must be non-zero: {path}")
    return tuple(component / magnitude for component in values)  # type: ignore[return-value]


def _read_optional_linear_velocity(
    payload: object, path: Path
) -> tuple[float, float, float] | None:
    """读取 linear_velocity 原值，仅供审计；四条记录强支持车体 FLU 解释。"""
    if not isinstance(payload, dict):
        return None
    velocity = payload.get("linear_velocity")
    if velocity is None:
        return None
    if not isinstance(velocity, dict):
        raise ValueError(  # noqa: TRY004
            f"Localization linear_velocity must be an object: {path}"
        )

    components: list[float] = []
    for axis in ("x", "y", "z"):
        value = velocity.get(axis, 0.0 if axis == "z" else None)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(  # noqa: TRY004
                f"Localization linear_velocity.{axis} must be numeric: {path}"
            )
        component = float(value)
        if not math.isfinite(component):
            raise ValueError(f"Localization linear_velocity.{axis} must be finite: {path}")
        components.append(component)
    return tuple(components)  # type: ignore[return-value]


def _sample_bracket(
    samples: Sequence[_BracketSample],
    timestamp: Decimal,
    *,
    max_gap_sec: Decimal,
    get_timestamp: Callable[[_BracketSample], Decimal],
) -> tuple[_BracketSample, _BracketSample, float]:
    """返回目标时刻两侧样本和插值比例；不外推，也不跨越长采样间隔。"""
    if not samples:
        raise ValueError("No localization samples are available")
    timestamps = [get_timestamp(sample) for sample in samples]
    index = bisect.bisect_left(timestamps, timestamp)
    if index < len(samples) and timestamps[index] == timestamp:
        sample = samples[index]
        return sample, sample, 0.0
    if index == 0 or index == len(samples):
        raise ValueError(
            f"Target timestamp {timestamp} is outside localization coverage "
            f"[{timestamps[0]}, {timestamps[-1]}]"
        )

    left, right = samples[index - 1], samples[index]
    left_timestamp, right_timestamp = timestamps[index - 1], timestamps[index]
    gap = right_timestamp - left_timestamp
    if gap > max_gap_sec:
        raise ValueError(
            f"Localization sampling gap {gap * 1000} ms exceeds "
            f"{max_gap_sec * 1000} ms between {left_timestamp} and {right_timestamp}"
        )
    fraction = float((timestamp - left_timestamp) / gap)
    return left, right, fraction


def _slerp_orientation(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
    fraction: float,
) -> tuple[float, float, float, float]:
    """在 wxyz 顺序的两个单位四元数间做最短路径球面插值。"""
    dot = sum(a * b for a, b in zip(left, right))
    if dot < 0.0:
        right = tuple(-component for component in right)  # type: ignore[assignment]
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        values = tuple(
            a + fraction * (b - a)
            for a, b in zip(left, right)
        )
    else:
        angle = math.acos(dot)
        sine = math.sin(angle)
        left_weight = math.sin((1.0 - fraction) * angle) / sine
        right_weight = math.sin(fraction * angle) / sine
        values = tuple(
            left_weight * a + right_weight * b
            for a, b in zip(left, right)
        )
    magnitude = math.sqrt(sum(component * component for component in values))
    if magnitude <= 1e-12:
        raise ValueError("Interpolated localization orientation is degenerate")
    return tuple(component / magnitude for component in values)  # type: ignore[return-value]


def sample_orientation_at(
    samples: Sequence[TimedPosition],
    timestamp: Decimal,
    *,
    max_gap_sec: Decimal = MAX_LOCALIZATION_GAP_SEC,
) -> tuple[float, float, float, float]:
    """读取或 SLERP 插值车体 FLU 到 map-XY 的姿态四元数。"""
    left, right, fraction = _sample_bracket(
        samples,
        timestamp,
        max_gap_sec=max_gap_sec,
        get_timestamp=lambda sample: sample.timestamp,
    )
    if left.orientation_wxyz is None or right.orientation_wxyz is None:
        raise ValueError(f"Localization orientation is unavailable at {timestamp}")
    if left is right:
        return left.orientation_wxyz
    return _slerp_orientation(left.orientation_wxyz, right.orientation_wxyz, fraction)


def sample_recorded_linear_velocity_at(
    samples: Sequence[TimedPosition],
    timestamp: Decimal,
    *,
    max_gap_sec: Decimal = MAX_LOCALIZATION_GAP_SEC,
) -> tuple[float, float, float]:
    """插值原始速度供审计；工作假设为车体 FLU，尚无生产端坐标契约。"""
    left, right, fraction = _sample_bracket(
        samples,
        timestamp,
        max_gap_sec=max_gap_sec,
        get_timestamp=lambda sample: sample.timestamp,
    )
    if left.linear_velocity_xyz is None or right.linear_velocity_xyz is None:
        raise ValueError(f"Localization linear_velocity is unavailable at {timestamp}")
    if left is right:
        return left.linear_velocity_xyz
    return tuple(
        a + fraction * (b - a)
        for a, b in zip(left.linear_velocity_xyz, right.linear_velocity_xyz)
    )  # type: ignore[return-value]


def rotate_map_velocity_to_body_flu(
    velocity_map_xyz: tuple[float, float, float],
    orientation_wxyz: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """用 FLU→map 姿态的逆旋转，把 map-XY 速度变为车体 Forward/Left/Up。"""
    w, x, y, z = orientation_wxyz
    magnitude = math.sqrt(w * w + x * x + y * y + z * z)
    if magnitude <= 1e-12:
        raise ValueError("Localization orientation quaternion must be non-zero")
    # q 描述 body→map；其共轭四元数执行 map→body 的逆旋转。
    inverse = (w / magnitude, -x / magnitude, -y / magnitude, -z / magnitude)
    qw, qx, qy, qz = inverse
    vx, vy, vz = velocity_map_xyz
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )

def position_interval_is_continuous(
    samples: Sequence[TimedPosition],
    start_timestamp: Decimal,
    end_timestamp: Decimal,
    max_gap_sec: Decimal,
) -> bool:
    """判断区间是否有完整定位覆盖，且没有超过阈值的样本间隔。"""
    if not samples:
        return False
    if (
        start_timestamp > end_timestamp
        or start_timestamp < samples[0].timestamp
        or end_timestamp > samples[-1].timestamp
    ):
        return False

    timestamps = [sample.timestamp for sample in samples]
    start_index = max(0, bisect.bisect_right(timestamps, start_timestamp) - 1)
    end_index = min(len(samples) - 1, bisect.bisect_left(timestamps, end_timestamp))
    for index in range(start_index, end_index):
        left = samples[index].timestamp
        right = samples[index + 1].timestamp
        # 只检查与请求区间内部相交的样本间隔；恰好在端点之外的间隔不影响窗口。
        if right <= start_timestamp or left >= end_timestamp:
            continue
        gap = right - left
        if gap > max_gap_sec:
            return False
    return True


def _select_motion_samples(
    dense_samples: Sequence[PositionMotionSample],
    offsets: Sequence[Decimal],
) -> list[PositionMotionSample] | None:
    """从密集运动序列抽取目标时刻，并保留目标间隔的连续性语义。

    若 CoC 采样时刻不在 10 Hz 网格上，返回 ``None``，由调用方单独计算。
    ``connected_to_previous`` 汇总两个抽取时刻之间所有密集网格段的状态，
    因此仍表示整个 CoC 间隔（例如 0.5 秒）是否连续。
    """
    if any(later <= earlier for earlier, later in zip(offsets, offsets[1:])):
        raise ValueError("Motion sample offsets must be strictly increasing")

    samples_by_time = {
        sample.relative_time: (index, sample)
        for index, sample in enumerate(dense_samples)
    }
    if any(offset not in samples_by_time for offset in offsets):
        return None

    selected: list[PositionMotionSample] = []
    previous_index: int | None = None
    for offset in offsets:
        index, sample = samples_by_time[offset]
        connected = previous_index is not None and all(
            dense_samples[step].connected_to_previous
            for step in range(previous_index + 1, index + 1)
        )
        selected.append(replace(sample, connected_to_previous=connected))
        previous_index = index
    return selected


def sample_position_at(
    samples: Sequence[TimedPosition],
    timestamp: Decimal,
    *,
    max_gap_sec: Decimal = MAX_LOCALIZATION_GAP_SEC,
) -> tuple[float, float]:
    """在目标采样时刻取位置；只对短间隔线性插值，不外推。

    例如，目标时刻处于相距 10 ms 的两个定位样本之间时会线性插值；
    若两样本相距 424 ms，则返回错误，不跨过记录开头的长间隔。
    """
    left, right, fraction = _sample_bracket(
        samples,
        timestamp,
        max_gap_sec=max_gap_sec,
        get_timestamp=lambda sample: sample.timestamp,
    )
    return (
        left.x + fraction * (right.x - left.x),
        left.y + fraction * (right.y - left.y),
    )


def _sample_resampled_position_at(
    valid_samples: Sequence[ResampledPosition],
    timestamp: Decimal,
    *,
    max_gap_sec: Decimal,
) -> tuple[float, float]:
    """在有效的重采样位置间插值，不跨越缺失网格点。"""
    left, right, fraction = _sample_bracket(
        valid_samples,
        timestamp,
        max_gap_sec=max_gap_sec,
        get_timestamp=lambda sample: sample.target_timestamp,
    )
    # valid_samples 已筛掉没有坐标的点。
    assert left.x is not None and left.y is not None
    assert right.x is not None and right.y is not None
    return (
        left.x + fraction * (right.x - left.x),
        left.y + fraction * (right.y - left.y),
    )


def _optional_sample(
    sample: Callable[[], _SampleResult],
) -> tuple[_SampleResult | None, str | None]:
    """将一次可选采样统一转换为“结果或错误信息”。"""
    try:
        return sample(), None
    except ValueError as exc:
        return None, str(exc)


def _derive_velocity(
    sample_position: Callable[[Decimal], tuple[float, float]],
    timestamp: Decimal,
    *,
    window_sec: Decimal,
) -> tuple[float, float, float, float]:
    """只从中心差分两端估算 map-XY 速度、速度大小和方向。"""
    if window_sec <= 0:
        raise ValueError("Position derivative window must be positive")
    half_window = window_sec / 2
    before_x, before_y = sample_position(timestamp - half_window)
    after_x, after_y = sample_position(timestamp + half_window)
    duration = float(window_sec)
    velocity_x = (after_x - before_x) / duration
    velocity_y = (after_y - before_y) / duration
    ground_speed = math.hypot(velocity_x, velocity_y)
    trajectory_bearing_deg = math.degrees(math.atan2(velocity_y, velocity_x))
    return velocity_x, velocity_y, ground_speed, trajectory_bearing_deg


def derive_position_motion(
    samples: Sequence[TimedPosition],
    timestamp: Decimal,
    *,
    window_sec: Decimal = POSITION_DERIVATIVE_WINDOW_SEC,
    max_gap_sec: Decimal = MAX_LOCALIZATION_GAP_SEC,
) -> tuple[float, float, float, float, float, float]:
    """用中心差分返回位置、map-XY 速度、地面速度大小和轨迹方向。

    例如，对匀速位置 ``x(t)=2t, y(t)=-t``，返回的速度约为
    ``vx=2 m/s``、``vy=-1 m/s``，而轨迹方向约为 ``-26.6°``。
    只检查中心位置和差分两端各自的插值间隔；窗口内部与这三个取值点无关的
    原始采样缺口会单独保留在轨迹审计里，不会自动使中心差分失效。
    """
    if window_sec <= 0:
        raise ValueError("Position derivative window must be positive")
    position_x, position_y = sample_position_at(
        samples, timestamp, max_gap_sec=max_gap_sec
    )
    velocity_x, velocity_y, ground_speed, trajectory_bearing_deg = _derive_velocity(
        lambda target: sample_position_at(
            samples, target, max_gap_sec=max_gap_sec
        ),
        timestamp,
        window_sec=window_sec,
    )
    return (
        position_x,
        position_y,
        velocity_x,
        velocity_y,
        ground_speed,
        trajectory_bearing_deg,
    )


def _position_trajectory_offsets(
    history_sec: Decimal,
    future_sec: Decimal,
    interval_sec: Decimal = LOCALIZATION_RESAMPLE_INTERVAL_SEC,
    *,
    derivative_window_sec: Decimal = POSITION_DERIVATIVE_WINDOW_SEC,
) -> list[Decimal]:
    """生成事件相对 10 Hz 位置网格，并在两端多留半个差分窗口。

    例如 CoC 窗口为 ``[-2, +6]`` 秒、差分窗口为 0.5 秒时，10 Hz 网格覆盖
    ``[-2.3, +6.3]`` 秒，以便在最早和最晚的 CoC 时刻也能取到差分两端。
    """
    if interval_sec <= 0 or history_sec < 0 or future_sec <= 0:
        raise ValueError("Position trajectory history/future and interval are invalid")
    if derivative_window_sec <= 0:
        raise ValueError("Position derivative window must be positive")

    margin = derivative_window_sec / 2
    start_steps = int(
        ((history_sec + margin) / interval_sec).to_integral_value(
            rounding=ROUND_CEILING
        )
    )
    end_steps = int(
        ((future_sec + margin) / interval_sec).to_integral_value(
            rounding=ROUND_CEILING
        )
    )
    return [interval_sec * index for index in range(-start_steps, end_steps + 1)]

def resample_position_trajectory(
    samples: Sequence[TimedPosition],
    event_timestamp: Decimal,
    offsets: Sequence[Decimal],
    *,
    max_raw_gap_sec: Decimal = MAX_LOCALIZATION_GAP_SEC,
) -> list[ResampledPosition]:
    """将原始 localization 位置插值到事件相对的规则时间网格。

    每个网格点只在它自己的原始定位样本括区不超过 ``max_raw_gap_sec`` 时插值。
    两个网格点之间若存在更大的原始采样缺口，会在 ``connected_to_previous`` 中留痕。
    """
    if not offsets:
        raise ValueError("Position trajectory requires at least one grid offset")
    if any(later <= earlier for earlier, later in zip(offsets, offsets[1:])):
        raise ValueError("Position trajectory offsets must be strictly increasing")

    result: list[ResampledPosition] = []
    previous_valid_sample: ResampledPosition | None = None
    for offset in offsets:
        target_timestamp = event_timestamp + offset
        try:
            x, y = sample_position_at(
                samples, target_timestamp, max_gap_sec=max_raw_gap_sec
            )
            position_error = None
            position_valid = True
        except ValueError as exc:
            x = y = None
            position_error = str(exc)
            position_valid = False

        connected_to_previous = False
        if (
            position_valid
            and previous_valid_sample is not None
            and previous_valid_sample.position_valid
        ):
            connected_to_previous = position_interval_is_continuous(
                samples,
                previous_valid_sample.target_timestamp,
                target_timestamp,
                max_raw_gap_sec,
            )

        item = ResampledPosition(
            relative_time=offset,
            target_timestamp=target_timestamp,
            position_valid=position_valid,
            x=x,
            y=y,
            position_error=position_error,
            connected_to_previous=connected_to_previous,
        )
        result.append(item)
        previous_valid_sample = item if position_valid else None

    return result

def _position_motion_offsets(
    history_sec: Decimal, future_sec: Decimal, interval_sec: Decimal
) -> list[Decimal]:
    """生成含事件时刻的速度时间网格，例如默认配置得到 17 个时刻。"""
    if interval_sec <= 0 or history_sec < 0 or future_sec <= 0:
        raise ValueError("Position motion history/future and interval are invalid")
    history_count = history_sec / interval_sec
    future_count = future_sec / interval_sec
    if history_count != history_count.to_integral_value():
        raise ValueError("History duration must be divisible by the motion interval")
    if future_count != future_count.to_integral_value():
        raise ValueError("Future duration must be divisible by the motion interval")
    return [
        -history_sec + index * interval_sec for index in range(int(history_count) + 1)
    ] + [interval_sec * (index + 1) for index in range(int(future_count))]

def build_position_motion_samples(
    samples: Sequence[TimedPosition],
    event_timestamp: Decimal,
    offsets: Sequence[Decimal],
    *,
    resampled_positions: Sequence[ResampledPosition] | None = None,
    window_sec: Decimal = POSITION_DERIVATIVE_WINDOW_SEC,
    max_gap_sec: Decimal = MAX_LOCALIZATION_GAP_SEC,
) -> list[PositionMotionSample]:
    """从 10 Hz 位置差分及同一时刻的姿态生成 CoC 运动样本。

    地图 XY 速度由 10 Hz 位置网格估计，再用定位姿态逆旋转成车体 FLU。
    缺失的 10 Hz 网格点会形成超过 0.1 秒的间隔，速度插值不会跨过这些缺口。
    """
    if resampled_positions is None:
        motion_position_at = lambda timestamp: sample_position_at(
            samples, timestamp, max_gap_sec=max_gap_sec
        )
    else:
        valid_motion_positions = [
            sample
            for sample in resampled_positions
            if sample.position_valid and sample.x is not None and sample.y is not None
        ]
        motion_position_at = lambda timestamp: _sample_resampled_position_at(
            valid_motion_positions,
            timestamp,
            max_gap_sec=LOCALIZATION_RESAMPLE_INTERVAL_SEC,
        )

    event_position, event_position_error = _optional_sample(
        lambda: sample_position_at(samples, event_timestamp, max_gap_sec=max_gap_sec)
    )
    event_x, event_y = (
        event_position if event_position is not None else (None, None)
    )

    result: list[PositionMotionSample] = []
    previous_timestamp: Decimal | None = None
    previous_position_valid = False
    for offset in offsets:
        target_timestamp = event_timestamp + offset
        position, position_error = _optional_sample(
            partial(
                sample_position_at,
                samples,
                target_timestamp,
                max_gap_sec=max_gap_sec,
            )
        )
        position_x, position_y = position if position is not None else (None, None)

        relative_x = (
            position_x - event_x
            if position_x is not None and event_x is not None
            else None
        )
        relative_y = (
            position_y - event_y
            if position_y is not None and event_y is not None
            else None
        )
        if event_position_error and position_error is None:
            position_error = f"Event position is unavailable: {event_position_error}"

        velocity, motion_error = _optional_sample(
            partial(
                _derive_velocity,
                motion_position_at,
                target_timestamp,
                window_sec=window_sec,
            )
        )
        if velocity is None:
            velocity_x = velocity_y = ground_speed = trajectory_bearing_deg = None
        else:
            velocity_x, velocity_y, ground_speed, trajectory_bearing_deg = velocity

        orientation_wxyz: tuple[float, float, float, float] | None = None
        body_velocity_right: float | None = None
        body_velocity_forward: float | None = None
        body_motion_error: str | None = None
        recorded_linear_velocity_xyz: tuple[float, float, float] | None = None
        body_velocity_residual_mps: float | None = None
        orientation_wxyz, body_motion_error = _optional_sample(
            partial(
                sample_orientation_at,
                samples,
                target_timestamp,
                max_gap_sec=max_gap_sec,
            )
        )

        # 原始速度仅用于残差审计；采样失败不影响位置差分速度。
        recorded_linear_velocity_xyz, _ = _optional_sample(
            partial(
                sample_recorded_linear_velocity_at,
                samples,
                target_timestamp,
                max_gap_sec=max_gap_sec,
            )
        )

        if velocity_x is None or velocity_y is None:
            body_motion_error = (
                f"Position-derived map velocity is unavailable: {motion_error}"
            )
        elif orientation_wxyz is not None:
            body_velocity_flu = rotate_map_velocity_to_body_flu(
                (velocity_x, velocity_y, 0.0), orientation_wxyz
            )
            body_velocity_forward = body_velocity_flu[0]
            body_velocity_right = -body_velocity_flu[1]
            body_motion_error = None
            if recorded_linear_velocity_xyz is not None:
                body_velocity_left = -body_velocity_right
                body_velocity_residual_mps = math.hypot(
                    body_velocity_forward - recorded_linear_velocity_xyz[0],
                    body_velocity_left - recorded_linear_velocity_xyz[1],
                )

        connected_to_previous = False
        if (
            previous_timestamp is not None
            and previous_position_valid
            and position_x is not None
        ):
            connected_to_previous = position_interval_is_continuous(
                samples, previous_timestamp, target_timestamp, max_gap_sec
            )

        result.append(
            PositionMotionSample(
                relative_time=offset,
                target_timestamp=target_timestamp,
                position_valid=position_x is not None and event_x is not None,
                relative_x=relative_x,
                relative_y=relative_y,
                position_error=position_error,
                motion_valid=ground_speed is not None,
                velocity_x=velocity_x,
                velocity_y=velocity_y,
                ground_speed=ground_speed,
                trajectory_bearing_deg=trajectory_bearing_deg,
                motion_error=motion_error,
                connected_to_previous=connected_to_previous,
                body_velocity_right=body_velocity_right,
                body_velocity_forward=body_velocity_forward,
                body_motion_error=body_motion_error,
                orientation_wxyz=orientation_wxyz,
                recorded_linear_velocity_xyz=recorded_linear_velocity_xyz,
                body_velocity_residual_mps=body_velocity_residual_mps,
            )
        )
        previous_timestamp = target_timestamp
        previous_position_valid = position_x is not None

    return result
