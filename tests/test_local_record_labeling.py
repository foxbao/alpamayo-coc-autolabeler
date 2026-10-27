# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from decimal import Decimal
from pathlib import Path

import pytest

from coc_labeling.local_record_labeling import _load_selected_candidates


def test_load_selected_candidates_filters_and_reads_camera_relative_time(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "segment_candidates.json"
    manifest_path.write_text(
        json.dumps(
            {
                "record_id": "record",
                "candidates": [
                    {
                        "candidate_id": "candidate_001",
                        "camera_relative_time_sec": "2.073690377",
                        "status": "selected",
                    },
                    {
                        "candidate_id": "candidate_002",
                        "camera_relative_time_sec": "7",
                        "status": "suppressed",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    record_id, candidates = _load_selected_candidates(manifest_path, {"candidate_001"})

    assert record_id == "record"
    assert candidates == [("candidate_001", Decimal("2.073690377"))]


def test_load_selected_candidates_rejects_non_selected_requested_id(tmp_path: Path) -> None:
    manifest_path = tmp_path / "segment_candidates.json"
    manifest_path.write_text(
        json.dumps(
            {
                "record_id": "record",
                "candidates": [
                    {
                        "candidate_id": "candidate_002",
                        "camera_relative_time_sec": "7",
                        "status": "suppressed",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="不存在或不是 selected 状态"):
        _load_selected_candidates(manifest_path, {"candidate_002"})


def test_load_selected_candidates_reads_grouped_v2_pools_in_time_order(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "segment_candidates.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "record_id": "record",
                "action_candidates": [
                    {
                        "candidate_id": "action_001",
                        "camera_relative_time_sec": "7",
                        "status": "selected",
                    }
                ],
                "coverage_candidates": [
                    {
                        "candidate_id": "coverage_001",
                        "camera_relative_time_sec": "2",
                        "status": "selected",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    record_id, candidates = _load_selected_candidates(manifest_path, None)

    assert record_id == "record"
    assert candidates == [
        ("coverage_001", Decimal(2)),
        ("action_001", Decimal(7)),
    ]


@pytest.mark.parametrize(
    ("candidate_type", "expected"),
    [
        ("action_span", [("action_001", Decimal(7))]),
        ("coverage", [("coverage_001", Decimal(2))]),
    ],
)
def test_load_selected_candidates_filters_v2_pool(
    tmp_path: Path,
    candidate_type: str,
    expected: list[tuple[str, Decimal]],
) -> None:
    manifest_path = tmp_path / "segment_candidates.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "record_id": "record",
                "action_candidates": [
                    {
                        "candidate_id": "action_001",
                        "camera_relative_time_sec": "7",
                        "status": "selected",
                    }
                ],
                "coverage_candidates": [
                    {
                        "candidate_id": "coverage_001",
                        "camera_relative_time_sec": "2",
                        "status": "selected",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    record_id, candidates = _load_selected_candidates(
        manifest_path,
        None,
        candidate_type=candidate_type,
    )

    assert record_id == "record"
    assert candidates == expected


def test_load_selected_candidates_rejects_requested_id_from_other_pool(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "segment_candidates.json"
    manifest_path.write_text(
        json.dumps(
            {
                "record_id": "record",
                "candidates": [
                    {
                        "candidate_id": "coverage_001",
                        "camera_relative_time_sec": "2",
                        "source": "periodic_coverage",
                        "status": "selected",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="不属于 --candidate-type action_span"):
        _load_selected_candidates(
            manifest_path,
            {"coverage_001"},
            candidate_type="action_span",
        )
