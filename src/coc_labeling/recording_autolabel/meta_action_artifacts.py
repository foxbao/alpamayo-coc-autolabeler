# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""完整本地记录 meta-action 结果的读写。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .actions import (
    ACCELERATION_HYSTERESIS_FRACTION,
    ACTION_GRID_INTERVAL_SEC,
    DEFAULT_MIN_ACTION_DURATION_SEC,
    GENTLE_ACCELERATION_MPS2,
    MAINTAIN_SPEED_MIN_DURATION_SEC,
    MIN_ACTION_DELTA_VELOCITY_MPS,
    REVERSE_SPEED_MPS,
    SMOOTHING_WINDOW_SAMPLES,
    STOP_SPEED_MPS,
    STRONG_ACCELERATION_MPS2,
    LongitudinalActionResult,
)
from .types import LocalMotionChunk


@dataclass(frozen=True)
class FullRecordMetaAction:
    """从完整记录动作文件读取的筛选阶段输入。"""

    record_id: str
    localization_start: Decimal
    action_motions: tuple[LocalMotionChunk, ...]
    point_count: int


def write_full_record_meta_action(
    result: LongitudinalActionResult,
    output_path: Path,
    record_id: str,
    localization_start: Decimal,
) -> Path:
    """保存 10 Hz 动作点和区间，并附上绝对时间戳。"""
    payload = {
        "record_id": record_id,
        "time_origin": "first localization filename timestamp",
        "localization_start_unix_timestamp_sec": str(localization_start),
        "source": "position-derived body-forward velocity from localization",
        "coordinate_frame": ("vehicle FLU interpreted as x=forward, y=left; v_long=body_x"),
        "scope_note": (
            "Rule-based longitudinal hints for review; they are not ground-truth labels. "
            "Lateral and lane actions are not generated."
        ),
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
                "start_time_sec": str(span.start_time),
                "end_time_sec": str(span.end_time),
                "start_unix_timestamp_sec": str(localization_start + span.start_time),
                "end_unix_timestamp_sec": str(localization_start + span.end_time),
                "end_is_exclusive": True,
            }
            for span in result.spans
        ],
        "points": [
            {
                "relative_time_sec": str(point.relative_time),
                "unix_timestamp_sec": str(localization_start + point.relative_time),
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path


def load_full_record_meta_action(input_path: Path) -> FullRecordMetaAction:
    """读取动作生成阶段的 JSON，供候选筛选阶段独立使用。"""
    if not input_path.is_file():
        raise FileNotFoundError(
            f"找不到 meta-action 文件：{input_path}；请先运行 "
            "python -m coc_labeling.local_data_labeling"
        )

    try:
        payload: Any = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 meta-action JSON：{input_path}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"Meta-action 文件顶层必须是 JSON object：{input_path}")

    record_id = payload.get("record_id")
    raw_start = payload.get("localization_start_unix_timestamp_sec")
    raw_spans = payload.get("spans")
    raw_points = payload.get("points", [])
    if not isinstance(record_id, str) or not record_id:
        raise ValueError(f"Meta-action 文件缺少有效 record_id：{input_path}")
    if not isinstance(raw_spans, list):
        raise TypeError(f"Meta-action 文件的 spans 必须是 JSON array：{input_path}")
    if not isinstance(raw_points, list):
        raise TypeError(f"Meta-action 文件的 points 必须是 JSON array：{input_path}")

    localization_start = _decimal_field(
        raw_start, "localization_start_unix_timestamp_sec", input_path
    )

    if any(not isinstance(point, dict) for point in raw_points):
        raise TypeError(f"Meta-action 文件中的 points 项必须是 JSON object：{input_path}")

    action_motions: list[LocalMotionChunk] = []
    for index, raw_span in enumerate(raw_spans):
        if not isinstance(raw_span, dict):
            raise TypeError(f"spans[{index}] 必须是 JSON object：{input_path}")
        action_type = raw_span.get("label")
        if not isinstance(action_type, str) or not action_type:
            raise ValueError(f"spans[{index}] 缺少有效 label：{input_path}")
        start_time = _decimal_field(
            raw_span.get("start_time_sec"), f"spans[{index}].start_time_sec", input_path
        )
        end_time = _decimal_field(
            raw_span.get("end_time_sec"), f"spans[{index}].end_time_sec", input_path
        )
        if end_time <= start_time:
            raise ValueError(f"spans[{index}] 的起止时间无效：{input_path}")
        action_motions.append(
            LocalMotionChunk(
                agent_token="ego",
                action_type=action_type,
                start_time=start_time,
                end_time=end_time,
            )
        )

    return FullRecordMetaAction(
        record_id=record_id,
        localization_start=localization_start,
        action_motions=tuple(action_motions),
        point_count=len(raw_points),
    )


def _decimal_field(value: object, field_name: str, input_path: Path) -> Decimal:
    """解析 JSON 中的有限十进制数，错误信息指出无效字段。"""
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"Meta-action 字段 {field_name} 不是有效数字：{input_path}") from exc
    if not number.is_finite():
        raise ValueError(f"Meta-action 字段 {field_name} 必须是有限数：{input_path}")
    return number
