# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from decimal import Decimal
from pathlib import Path

import pytest

from coc_labeling.recording_autolabel import scenario as scenario_module
from coc_labeling.recording_autolabel.actions import (
    LongitudinalActionPoint,
    LongitudinalActionResult,
    LongitudinalActionSpan,
)
from coc_labeling.recording_autolabel.scenario import LocalTemporalScenario
from coc_labeling.recording_autolabel.types import TimedPosition


def _positions() -> list[TimedPosition]:
    return [
        TimedPosition(Decimal("100.0"), 0.0, 0.0, Path("100.0.json")),
        TimedPosition(Decimal("100.1"), 0.1, 0.0, Path("100.1.json")),
    ]


def test_local_temporal_scenario_returns_motion_for_requested_action_and_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = LongitudinalActionResult(
        points=(
            LongitudinalActionPoint(Decimal("0.0"), "Stop", 0.0, 0.0, 0.0, True, False),
            LongitudinalActionPoint(Decimal("0.1"), "Stop", 0.0, 0.0, 0.0, True, True),
        ),
        spans=(LongitudinalActionSpan("Stop", Decimal("0.0"), Decimal("0.2")),),
    )
    classify_calls = 0

    def classify(_samples: object) -> LongitudinalActionResult:
        nonlocal classify_calls
        classify_calls += 1
        return result

    monkeypatch.setattr(scenario_module, "resample_position_trajectory", lambda *a, **k: [])
    monkeypatch.setattr(scenario_module, "build_position_motion_samples", lambda *a, **k: [])
    monkeypatch.setattr(scenario_module, "classify_longitudinal_series", classify)

    scenario = LocalTemporalScenario("record", _positions())

    motions = scenario.get_tag_motions("Stop")
    assert len(motions) == 1
    assert motions[0].agent_token == "ego"
    assert motions[0].action_type == "Stop"
    assert motions[0].start_time == Decimal("0.0")
    assert motions[0].end_time == Decimal("0.2")
    assert scenario.get_tag_motions("Maintain Speed") == []
    assert classify_calls == 1
    assert scenario.localization_start == Decimal("100.0")


def test_local_temporal_scenario_rejects_unsupported_action_type() -> None:
    scenario = LocalTemporalScenario("record", _positions())

    with pytest.raises(ValueError, match="Turn Left"):
        scenario.get_tag_motions("Turn Left")
