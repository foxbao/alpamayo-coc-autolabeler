# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from coc_labeling.recording_autolabel.segment_candidates import (
    CandidateSeed,
    SegmentCandidate,
    apply_candidate_separation,
    balance_action_candidates,
    balance_and_select_candidates,
    candidate_manifest_payload,
    select_segment_candidates,
)
from coc_labeling.recording_autolabel.types import LocalMotionChunk, TimedImage


def _camera_images() -> list[TimedImage]:
    return [
        TimedImage(
            path=Path(f"{Decimal(100) + Decimal(index) / 10}.jpg"),
            timestamp=Decimal(100) + Decimal(index) / 10,
        )
        for index in range(301)
    ]


def _action_candidate(
    candidate_id: str,
    start: int,
    end: int,
    *,
    action: str = "Stop",
) -> SegmentCandidate:
    seed = CandidateSeed(
        candidate_id=candidate_id,
        camera_time_sec=Decimal(start),
        absolute_timestamp=Decimal(1000 + start),
        localization_time_sec=Decimal(start),
        source="action_span_start",
        action_label=action,
        action_end_localization_time_sec=Decimal(end),
    )
    return SegmentCandidate(seed, "selected", selected_frames=[])


def _coverage_candidate(candidate_id: str, time_sec: int) -> SegmentCandidate:
    return SegmentCandidate(
        CandidateSeed(
            candidate_id=candidate_id,
            camera_time_sec=Decimal(time_sec),
            absolute_timestamp=Decimal(1000 + time_sec),
            localization_time_sec=Decimal(time_sec),
            source="periodic_coverage",
            action_label=None,
        ),
        "selected",
        selected_frames=[],
    )


def test_candidates_convert_time_origins_and_prioritize_action_starts() -> None:
    actions = [
        LocalMotionChunk("ego", "Maintain Speed", Decimal(0), Decimal(8)),
        LocalMotionChunk("ego", "Stop", Decimal(8), Decimal(9)),
    ]

    candidates = select_segment_candidates(
        actions,
        localization_start=Decimal("100.4"),
        camera_images=_camera_images(),
        history_sec=Decimal(2),
        future_sec=Decimal(6),
        interval_sec=Decimal("0.5"),
    )

    initial_action = next(item for item in candidates if item.seed.source == "action_span_start")
    assert initial_action.seed.camera_time_sec == Decimal("0.4")
    assert initial_action.status == "rejected"
    assert initial_action.reason == "insufficient_camera_history"

    stop_candidate = next(item for item in candidates if item.seed.action_label == "Stop")
    assert stop_candidate.seed.camera_time_sec == Decimal("8.4")
    assert stop_candidate.seed.absolute_timestamp == Decimal("108.4")
    assert stop_candidate.seed.localization_time_sec == Decimal(8)
    assert stop_candidate.seed.action_end_localization_time_sec == Decimal(9)
    assert stop_candidate.seed.action_duration_sec == Decimal(1)
    assert stop_candidate.status == "selected"
    assert stop_candidate.selected_frames is not None
    assert len(stop_candidate.selected_frames) == 16

    nearby_periodic = next(
        item
        for item in candidates
        if item.seed.source == "periodic_coverage" and item.seed.camera_time_sec == Decimal(7)
    )
    assert nearby_periodic.status == "suppressed"
    assert nearby_periodic.merged_into == stop_candidate.seed.candidate_id

    selected_times = sorted(
        item.seed.camera_time_sec for item in candidates if item.status == "selected"
    )
    assert all(
        later - earlier >= Decimal(2) for earlier, later in zip(selected_times, selected_times[1:])
    )


def test_action_pool_filters_keep_coverage_candidates_unlabeled() -> None:
    actions = [
        LocalMotionChunk("ego", "Stop", Decimal(8), Decimal("8.8")),
        LocalMotionChunk("ego", "Gentle Acceleration", Decimal(10), Decimal(12)),
        LocalMotionChunk("ego", "Reverse", Decimal(15), Decimal(17)),
    ]

    candidates = select_segment_candidates(
        actions,
        localization_start=Decimal(100),
        camera_images=_camera_images(),
        history_sec=Decimal(2),
        future_sec=Decimal(6),
        interval_sec=Decimal("0.5"),
        minimum_action_duration_sec=Decimal(1),
        action_types={"Stop", "Gentle Acceleration"},
    )

    short_stop = next(item for item in candidates if item.seed.action_label == "Stop")
    assert short_stop.status == "filtered"
    assert short_stop.reason == "action_span_below_minimum_duration"

    excluded_reverse = next(item for item in candidates if item.seed.action_label == "Reverse")
    assert excluded_reverse.status == "filtered"
    assert excluded_reverse.reason == "action_type_not_selected"

    for coverage in (item for item in candidates if item.seed.source == "periodic_coverage"):
        assert coverage.seed.action_label is None
        assert coverage.seed.action_end_localization_time_sec is None
    assert any(
        item.seed.source == "periodic_coverage" and item.status == "selected" for item in candidates
    )


def test_default_minimum_action_duration_matches_ten_hz_one_second() -> None:
    candidates = select_segment_candidates(
        [
            LocalMotionChunk("ego", "Stop", Decimal(8), Decimal("8.8")),
            LocalMotionChunk("ego", "Gentle Deceleration", Decimal(10), Decimal(11)),
        ],
        localization_start=Decimal(100),
        camera_images=_camera_images(),
        history_sec=Decimal(2),
        future_sec=Decimal(6),
        interval_sec=Decimal("0.5"),
    )

    short_action = next(item for item in candidates if item.seed.action_label == "Stop")
    one_second_action = next(
        item for item in candidates if item.seed.action_label == "Gentle Deceleration"
    )
    assert short_action.status == "filtered"
    assert short_action.seed.action_duration_sec == Decimal("0.8")
    assert one_second_action.status == "selected"
    assert one_second_action.seed.action_duration_sec == Decimal(1)


def test_manifest_separates_action_intervals_from_coverage_points() -> None:
    camera_images = _camera_images()
    candidates = select_segment_candidates(
        [LocalMotionChunk("ego", "Stop", Decimal(10), Decimal(12))],
        localization_start=Decimal(100),
        camera_images=camera_images,
        history_sec=Decimal(2),
        future_sec=Decimal(6),
        interval_sec=Decimal("0.5"),
    )

    manifest = candidate_manifest_payload(
        "record",
        candidates,
        camera_images[0].timestamp,
        camera_images[-1].timestamp,
        Decimal(100),
        Decimal(2),
        Decimal(6),
        Decimal("0.5"),
        Decimal(5),
        Decimal(2),
        action_target_count=2,
    )

    assert manifest["schema_version"] == 2
    assert manifest["window"]["action_target_count_per_type"] == 2
    assert manifest["window"]["action_target_selection"] == "longest_action_span_first"
    assert len(manifest["action_candidates"]) == 1
    assert len(manifest["coverage_candidates"]) == 5
    action = manifest["action_candidates"][0]
    assert action["candidate_type"] == "action_span"
    assert action["longitudinal_action"] == "Stop"
    assert action["action_span"] == {
        "start_localization_time_sec": "10",
        "end_localization_time_sec": "12",
        "duration_sec": "2",
        "start_unix_timestamp_sec": "110",
        "end_unix_timestamp_sec": "112",
        "end_is_exclusive": True,
    }
    coverage = manifest["coverage_candidates"][0]
    assert coverage["candidate_type"] == "coverage"
    assert coverage["longitudinal_action"] is None
    assert coverage["action_span"] is None


def test_action_candidates_are_balanced_globally_by_action_and_duration() -> None:
    def action_candidate(
        candidate_id: str,
        action: str,
        start: int,
        end: int,
    ) -> SegmentCandidate:
        seed = CandidateSeed(
            candidate_id=candidate_id,
            camera_time_sec=Decimal(start),
            absolute_timestamp=Decimal(1000 + start),
            localization_time_sec=Decimal(start),
            source="action_span_start",
            action_label=action,
            action_end_localization_time_sec=Decimal(end),
        )
        return SegmentCandidate(seed, "selected", selected_frames=[])

    short_stop = action_candidate("candidate_001", "Stop", 5, 7)
    long_stop = action_candidate("candidate_002", "Stop", 6, 11)
    long_stop.status = "suppressed"
    long_stop.reason = "within_minimum_candidate_separation"
    acceleration = action_candidate("candidate_003", "Gentle Acceleration", 20, 23)
    coverage = SegmentCandidate(
        CandidateSeed(
            "coverage_001",
            Decimal(4),
            Decimal(1004),
            Decimal(4),
            "periodic_coverage",
            None,
        ),
        "suppressed",
        reason="within_minimum_candidate_separation",
        merged_into=short_stop.seed.candidate_id,
        selected_frames=[],
    )

    summary = balance_action_candidates(
        [short_stop, coverage, acceleration, long_stop], target_count=1
    )
    apply_candidate_separation([short_stop, coverage, acceleration, long_stop], Decimal(2))

    assert summary == {"Gentle Acceleration": (1, 1), "Stop": (2, 1)}
    assert long_stop.status == "selected"
    assert long_stop.selected_frames == []
    assert short_stop.status == "filtered"
    assert short_stop.reason == "action_type_target_count_exceeded"
    assert short_stop.selected_frames is None
    assert acceleration.status == "selected"
    assert coverage.status == "selected"
    assert coverage.merged_into is None


def test_action_candidate_balance_requires_positive_target_count() -> None:
    try:
        balance_action_candidates([], target_count=0)
    except ValueError as exc:
        assert "at least one" in str(exc)
    else:
        raise AssertionError("expected a positive action candidate target count")


def test_balancing_refills_from_distant_candidates_after_strict_separation() -> None:
    def action_candidate(candidate_id: str, start: int, end: int) -> SegmentCandidate:
        seed = CandidateSeed(
            candidate_id=candidate_id,
            camera_time_sec=Decimal(start),
            absolute_timestamp=Decimal(1000 + start),
            localization_time_sec=Decimal(start),
            source="action_span_start",
            action_label="Stop",
            action_end_localization_time_sec=Decimal(end),
        )
        return SegmentCandidate(seed, "selected", selected_frames=[])

    longest = action_candidate("candidate_a", 10, 15)
    nearby = action_candidate("candidate_b", 11, 15)
    distant = action_candidate("candidate_c", 30, 33)

    summary = balance_and_select_candidates(
        [[longest, nearby, distant]],
        target_count=2,
        minimum_separation_sec=Decimal(2),
    )

    assert summary == {"Stop": (3, 2)}
    assert longest.status == "selected"
    assert nearby.status == "suppressed"
    assert distant.status == "selected"
    selected_times = [
        candidate.seed.camera_time_sec
        for candidate in (longest, nearby, distant)
        if candidate.status == "selected"
    ]
    assert selected_times == [Decimal(10), Decimal(30)]


def test_cross_record_backfill_obeys_global_action_target_count() -> None:
    longest = _action_candidate("longest", 10, 20)
    nearby = _action_candidate("nearby", 11, 20)
    distant = _action_candidate("distant", 40, 48)
    extra = _action_candidate("extra", 50, 57)

    summary = balance_and_select_candidates(
        [[longest, nearby], [distant, extra]],
        target_count=2,
        minimum_separation_sec=Decimal(3),
    )

    selected = [item for item in (longest, nearby, distant, extra) if item.status == "selected"]
    assert summary == {"Stop": (4, 2)}
    assert selected == [longest, distant]


def test_backfilled_action_candidate_suppresses_nearby_coverage_candidate() -> None:
    longest = _action_candidate("longest", 10, 18)
    nearby = _action_candidate("nearby", 11, 18)
    backfill = _action_candidate("backfill", 20, 26)
    coverage = _coverage_candidate("coverage", 21)

    balance_and_select_candidates(
        [[longest, nearby, backfill, coverage]],
        target_count=2,
        minimum_separation_sec=Decimal(3),
    )

    assert longest.status == "selected"
    assert backfill.status == "selected"
    assert coverage.status == "suppressed"
    assert coverage.merged_into == backfill.seed.candidate_id


def test_greedy_backfill_can_leave_target_underfilled_to_keep_longer_candidates() -> None:
    shorter_early = _action_candidate("shorter_early", 0, 3)
    longest_clustered = _action_candidate("longest_clustered", 2, 9)
    medium_clustered = _action_candidate("medium_clustered", 4, 8)
    longest_distant = _action_candidate("longest_distant", 15, 23)

    summary = balance_and_select_candidates(
        [[shorter_early, longest_clustered, medium_clustered, longest_distant]],
        target_count=3,
        minimum_separation_sec=Decimal(3),
    )

    assert summary == {"Stop": (4, 2)}
    assert shorter_early.status == "filtered"
    assert longest_clustered.status == "selected"
    assert medium_clustered.status == "suppressed"
    assert longest_distant.status == "selected"


def test_balancing_rejects_pending_action_candidates() -> None:
    candidate = _action_candidate("pending_action", 10, 12)
    candidate.status = "pending"

    with pytest.raises(ValueError, match="must pass initial window and separation checks"):
        balance_and_select_candidates(
            [[candidate]],
            target_count=1,
            minimum_separation_sec=Decimal(2),
        )
