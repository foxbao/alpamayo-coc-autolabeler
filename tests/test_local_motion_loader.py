# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from decimal import Decimal
from pathlib import Path

import pytest

from coc_labeling.recording_autolabel import motion_loader
from coc_labeling.recording_autolabel.motion_loader import LocalMotionLoader
from coc_labeling.recording_autolabel.types import TimedPosition


def test_local_motion_loader_builds_dense_actions_and_coc_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_time = Decimal("10.0")
    localization_positions = [
        TimedPosition(
            timestamp=event_time + Decimal(index) / 100,
            x=2.0 * index / 100,
            y=0.0,
            source_path=Path(f"{index}.json"),
            orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
        for index in range(-100, 101)
    ]
    monkeypatch.setattr(
        motion_loader,
        "load_localization_positions",
        lambda _directory: localization_positions,
    )

    loader = LocalMotionLoader(
        Decimal("0.5"),
        Decimal("0.5"),
        Decimal("0.5"),
    )
    result = loader.load(tmp_path, event_time)
    reused_positions = loader.load_from_positions(localization_positions, event_time)

    assert reused_positions.coc_motion_samples == result.coc_motion_samples
    assert reused_positions.longitudinal_actions == result.longitudinal_actions
    assert len(result.resampled_positions) == 17
    assert len(result.coc_motion_samples) == 3
    assert len(result.longitudinal_actions.points) == 11
    assert [sample.relative_time for sample in result.coc_motion_samples] == [
        Decimal("-0.5"),
        Decimal("0.0"),
        Decimal("0.5"),
    ]
    assert [sample.body_velocity_forward for sample in result.coc_motion_samples] == pytest.approx(
        [2.0, 2.0, 2.0]
    )

    fallback_result = LocalMotionLoader(
        Decimal("0.5"),
        Decimal("0.5"),
        Decimal("0.25"),
    ).load(tmp_path, event_time)
    assert len(fallback_result.coc_motion_samples) == 5
    assert [
        sample.body_velocity_forward for sample in fallback_result.coc_motion_samples
    ] == pytest.approx([2.0] * 5)
