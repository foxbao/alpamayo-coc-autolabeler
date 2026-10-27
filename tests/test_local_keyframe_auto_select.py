# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path

import cv2
import numpy as np
import pytest

import coc_labeling.local_keyframe_auto_select as selector
from coc_labeling.local_data_labeling import _discover_records as discover_labeled_records
from coc_labeling.recording_autolabel.segment_candidates import CandidateSeed, SegmentCandidate


def test_discover_records_rejects_duplicate_requested_names() -> None:
    with pytest.raises(SystemExit, match="duplicate record names"):
        selector._discover_records(Path("unused"), ["record_a", "record_a"])

    with pytest.raises(SystemExit, match="duplicate record names"):
        discover_labeled_records(Path("unused"), ["record_a", "record_a"])


def test_candidate_writer_removes_stale_generated_contact_sheets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record_output = tmp_path / "record"
    contact_sheet_dir = record_output / "contact_sheets"
    contact_sheet_dir.mkdir(parents=True)
    current_sheet = contact_sheet_dir / "candidate_001.jpg"
    stale_sheet = contact_sheet_dir / "candidate_002.jpg"
    unrelated_file = contact_sheet_dir / "notes.txt"
    current_sheet.write_bytes(b"old image")
    stale_sheet.write_bytes(b"stale image")
    unrelated_file.write_text("keep", encoding="utf-8")

    seed = CandidateSeed(
        candidate_id="candidate_001",
        camera_time_sec=Decimal(10),
        absolute_timestamp=Decimal(110),
        localization_time_sec=Decimal(10),
        source="action_span_start",
        action_label="Stop",
        action_end_localization_time_sec=Decimal(12),
    )
    candidate = SegmentCandidate(seed, "selected", selected_frames=None)
    prepared = selector._PreparedCandidateRecord(
        record_id="record",
        record_output=record_output,
        camera_dir=tmp_path / "camera",
        meta_action_path=tmp_path / "meta_action.json",
        localization_start=Decimal(100),
        camera_start=Decimal(100),
        camera_end=Decimal(130),
        action_tick_count=30,
        action_span_count=1,
        candidates=[candidate],
    )

    reselected: list[Decimal] = []
    monkeypatch.setattr(selector, "load_timed_images", lambda _: [])
    monkeypatch.setattr(selector, "sample_offsets", lambda *_: [Decimal(0)])

    def select_frames(images: list[object], event_time: Decimal, offsets: list[Decimal]) -> list[object]:
        reselected.append(event_time)
        return []

    monkeypatch.setattr(selector, "select_frames", select_frames)
    monkeypatch.setattr(selector, "load_rgb_frames", lambda _: [])

    def write_contact_sheet(
        selected_frames: list[object],
        decoded_frames: list[object],
        output_path: Path,
        *,
        header_text: str,
    ) -> None:
        output_path.write_bytes(b"current image")

    monkeypatch.setattr(selector, "write_contact_sheet", write_contact_sheet)
    selector._write_candidate_outputs(
        prepared,
        history_sec=Decimal(2),
        future_sec=Decimal(6),
        interval_sec=Decimal("0.5"),
        periodic_interval_sec=Decimal(5),
        minimum_separation_sec=Decimal(2),
        minimum_action_duration_sec=Decimal(1),
        action_types=None,
        action_target_count=500_000,
    )

    assert current_sheet.read_bytes() == b"current image"
    assert not stale_sheet.exists()
    assert unrelated_file.read_text(encoding="utf-8") == "keep"
    assert reselected == [Decimal(10)]


def test_cli_reads_meta_action_from_separate_root_and_writes_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_root = tmp_path / "recordings"
    camera_dir = input_root / "record_a" / "camera" / "front"
    camera_dir.mkdir(parents=True)
    for index in range(29):
        timestamp = Decimal(1000) + Decimal(index) / 2
        image = np.full((24, 32, 3), index, dtype=np.uint8)
        assert cv2.imwrite(str(camera_dir / f"{timestamp}.jpg"), image)

    meta_action_root = tmp_path / "meta_actions"
    meta_action_dir = meta_action_root / "record_a"
    meta_action_dir.mkdir(parents=True)
    meta_action_path = meta_action_dir / "full_record_meta_action.json"
    meta_action_path.write_text(
        json.dumps(
            {
                "record_id": "record_a",
                "localization_start_unix_timestamp_sec": "1000",
                "spans": [
                    {
                        "label": "Stop",
                        "start_time_sec": "5",
                        "end_time_sec": "7",
                    }
                ],
                "points": [{"relative_time_sec": "5"}],
            }
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "selected_candidates"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "local_keyframe_auto_select",
            "--input-root",
            str(input_root),
            "--meta-action-root",
            str(meta_action_root),
            "--output-dir",
            str(output_root),
            "--record",
            "record_a",
        ],
    )

    selector.main()

    manifest_path = output_root / "record_a" / "segment_candidates.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected_actions = [
        candidate
        for candidate in manifest["action_candidates"]
        if candidate["status"] == "selected"
    ]
    assert len(selected_actions) == 1
    assert meta_action_path.is_file()
    assert not (output_root / "record_a" / "full_record_meta_action.json").exists()
    assert (output_root / "record_a" / selected_actions[0]["contact_sheet"]).is_file()
    assert "available 1 -> selected 1 (cap 500000; separation applied)" in capsys.readouterr().out
