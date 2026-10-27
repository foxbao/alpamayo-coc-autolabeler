# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from decimal import Decimal
from pathlib import Path

import pytest

from coc_labeling.recording_autolabel.actions import (
    LongitudinalActionPoint,
    LongitudinalActionResult,
    LongitudinalActionSpan,
)
from coc_labeling.recording_autolabel.meta_action_artifacts import (
    load_full_record_meta_action,
    write_full_record_meta_action,
)


def test_full_record_meta_action_round_trip(tmp_path: Path) -> None:
    result = LongitudinalActionResult(
        points=(
            LongitudinalActionPoint(Decimal("0.0"), "Stop", 0.0, 0.0, 0.0, True, False),
            LongitudinalActionPoint(Decimal("0.1"), "Stop", 0.0, 0.0, 0.0, True, True),
        ),
        spans=(LongitudinalActionSpan("Stop", Decimal("0.0"), Decimal("0.2")),),
    )
    output_path = tmp_path / "record" / "full_record_meta_action.json"

    write_full_record_meta_action(result, output_path, "record", Decimal("1792395581.795200472"))
    loaded = load_full_record_meta_action(output_path)

    assert loaded.record_id == "record"
    assert loaded.localization_start == Decimal("1792395581.795200472")
    assert loaded.point_count == 2
    assert len(loaded.action_motions) == 1
    assert loaded.action_motions[0].agent_token == "ego"
    assert loaded.action_motions[0].action_type == "Stop"
    assert loaded.action_motions[0].start_time == Decimal("0.0")
    assert loaded.action_motions[0].end_time == Decimal("0.2")


def test_full_record_meta_action_loader_rejects_invalid_span(tmp_path: Path) -> None:
    input_path = tmp_path / "invalid.json"
    input_path.write_text(
        json.dumps(
            {
                "record_id": "record",
                "localization_start_unix_timestamp_sec": "100.0",
                "spans": [{"label": "Stop", "start_time_sec": "1.0"}],
                "points": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"spans\[0\]\.end_time_sec"):
        load_full_record_meta_action(input_path)
