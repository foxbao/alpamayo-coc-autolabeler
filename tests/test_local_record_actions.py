# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from coc_labeling.recording_autolabel.actions import (
    GENTLE_ACCELERATION,
    MAINTAIN_SPEED,
    REVERSE,
    STOP,
    STRONG_ACCELERATION,
    STRONG_DECELERATION,
    classify_longitudinal_series,
    format_local_meta_action_text,
    write_longitudinal_action_audit,
)
from coc_labeling.recording_autolabel.agent import build_agent_data
from coc_labeling.recording_autolabel.types import PositionMotionSample, SelectedFrame, TimedImage


def _samples(
    v_long_values: list[float],
    *,
    speed_2d: float | None = None,
    start_time: Decimal = Decimal("0.0"),
) -> list[PositionMotionSample]:
    return [
        PositionMotionSample(
            relative_time=start_time + Decimal(index) / 10,
            target_timestamp=Decimal("100.0") + start_time + Decimal(index) / 10,
            position_valid=True,
            relative_x=None,
            relative_y=None,
            position_error=None,
            motion_valid=True,
            velocity_x=None,
            velocity_y=None,
            ground_speed=abs(value) if speed_2d is None else speed_2d,
            trajectory_bearing_deg=None,
            motion_error=None,
            connected_to_previous=index > 0,
            body_velocity_right=0.0,
            body_velocity_forward=value,
            body_motion_error=None,
        )
        for index, value in enumerate(v_long_values)
    ]


def test_local_longitudinal_classifier_emits_gentle_acceleration() -> None:
    # 0.2 m/s² 持续 5 秒，速度变化超过 PAI 风格的 0.8 m/s 门槛。
    result = classify_longitudinal_series(
        _samples([1.0 + index * 0.02 for index in range(51)])
    )

    assert any(point.label == GENTLE_ACCELERATION for point in result.points)
    assert any(span.label == GENTLE_ACCELERATION for span in result.spans)


def test_local_longitudinal_classifier_handles_strong_motion_and_reverse() -> None:
    strong_acceleration = classify_longitudinal_series(
        _samples([1.0 + index * 0.2 for index in range(31)])
    )
    strong_deceleration = classify_longitudinal_series(
        _samples([5.0 - index * 0.2 for index in range(31)])
    )
    reverse = classify_longitudinal_series(_samples([-1.0] * 31))

    assert any(point.label == STRONG_ACCELERATION for point in strong_acceleration.points)
    assert any(point.label == STRONG_DECELERATION for point in strong_deceleration.points)
    assert {point.label for point in reverse.points} == {REVERSE}


def test_local_action_spans_ignore_raw_gap_flag_when_tick_is_valid() -> None:
    samples = _samples([3.0] * 20)
    samples[10] = PositionMotionSample(
        **{**samples[10].__dict__, "connected_to_previous": False}
    )

    result = classify_longitudinal_series(samples)

    assert len(result.spans) == 1
    assert result.spans[0].start_time == Decimal("0.0")
    assert result.spans[0].end_time == Decimal("2.0")


def test_local_action_spans_do_not_bridge_an_invalid_motion_sample() -> None:
    samples = _samples([3.0] * 20)
    samples[10] = PositionMotionSample(
        **{
            **samples[10].__dict__,
            "ground_speed": None,
            "body_velocity_forward": None,
        }
    )

    result = classify_longitudinal_series(samples)

    assert result.points[10].label is None
    assert len(result.spans) == 2
    assert result.spans[0].end_time == Decimal("1.0")
    assert result.spans[1].start_time == Decimal("1.1")


def test_single_valid_tick_is_preserved_without_an_action_label() -> None:
    result = classify_longitudinal_series(_samples([2.0]))

    assert result.points[0].valid is True
    assert result.points[0].v_long_mps == 2.0
    assert result.points[0].label is None
    assert result.spans == ()


def test_local_stop_uses_two_dimensional_speed_guard() -> None:
    stopped = classify_longitudinal_series(_samples([0.0] * 20))
    sideways = classify_longitudinal_series(_samples([0.0] * 20, speed_2d=1.0))

    assert {point.label for point in stopped.points} == {STOP}
    assert {point.label for point in sideways.points} == {MAINTAIN_SPEED}


def test_local_classifier_uses_pai_ground_speed_acceleration() -> None:
    samples = _samples([3.0] * 41, speed_2d=1.0)
    samples = [
        replace(sample, ground_speed=1.0 + index * 0.1)
        for index, sample in enumerate(samples)
    ]

    result = classify_longitudinal_series(samples)

    assert {point.label for point in result.points} == {GENTLE_ACCELERATION}
    assert result.points[0].acceleration_mps2 == pytest.approx(1.0)


def test_local_meta_action_text_aligns_to_image_times(tmp_path: Path) -> None:
    result = classify_longitudinal_series(_samples([3.0] * 20))
    text = format_local_meta_action_text(
        [Decimal("-0.2"), Decimal("0.0"), Decimal("0.5")], result
    )

    assert "(t=+0.0s) longitudinal: Maintain Speed" in text
    assert "lateral: Not Available, lane: Not Available" in text
    audit_path = write_longitudinal_action_audit(
        result, tmp_path, "sample_record", Decimal("100.0")
    )
    assert audit_path.is_file()
    assert "Maintain Speed" in audit_path.read_text(encoding="utf-8")


def test_local_classifier_rejects_unsorted_duplicate_timestamps() -> None:
    samples = _samples([1.0] * 3)
    samples[1] = PositionMotionSample(
        **{**samples[1].__dict__, "relative_time": samples[0].relative_time}
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        classify_longitudinal_series(samples)


def test_build_agent_data_adds_local_longitudinal_actions_to_input() -> None:
    frames = [
        SelectedFrame(
            relative_time=Decimal(index) / 2 - 2,
            target_timestamp=Decimal(2000) + Decimal(index) / 2 - 2,
            image=TimedImage(
                Path(f"{index}.jpg"),
                Decimal(2000) + Decimal(index) / 2 - 2,
            ),
            error_sec=Decimal(0),
        )
        for index in range(16)
    ]
    images = [np.zeros((8, 8, 3)) for _ in frames]
    actions = classify_longitudinal_series(
        _samples([3.0] * 81, start_time=Decimal("-2.0"))
    )

    data = build_agent_data(
        "sample_record",
        Decimal("10.0"),
        frames,
        images,
        longitudinal_actions=actions,
    )

    assert data["all_meta_action_str"].count("Maintain Speed") == 16
    assert data["all_meta_action_str"].count("lane: Not Available") == 16
    assert "rule-based labels derived" in data["meta_action_prompt"]
