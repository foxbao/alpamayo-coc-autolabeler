# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import cv2
import numpy as np
import pytest

from coc_labeling.data_loader.sample_dataset import SampleDataset


def _make_record(root: Path, batch: str, record: str, *, complete: bool = True) -> Path:
    record_dir = root / "sample" / batch / record
    (record_dir / "camera" / "front_image").mkdir(parents=True)
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    assert cv2.imwrite(str(record_dir / "camera" / "front_image" / "100.0.jpg"), image)
    if complete:
        localization_dir = record_dir / "localization"
        localization_dir.mkdir()
        (localization_dir / "100.0.json").write_text(
            '{"position":{"x":1,"y":2},"orientation":{"w":1,"x":0,"y":0,"z":0}}',
            encoding="utf-8",
        )
    return record_dir


def test_sample_dataset_discovers_records_and_uses_shared_loaders(tmp_path: Path) -> None:
    first = _make_record(tmp_path, "batch_b", "record_2")
    _make_record(tmp_path, "batch_a", "record_1")
    _make_record(tmp_path, "batch_a", "incomplete", complete=False)

    dataset = SampleDataset("vla_kl", tmp_path)
    records = dataset.discover_records()

    assert [record.qualified_id for record in records] == [
        "batch_a/record_1",
        "batch_b/record_2",
    ]
    record = dataset.get_record("batch_b/record_2")
    assert record.record_dir == first
    assert len(record.load_front_images()) == 1
    positions = record.load_localization()
    assert len(positions) == 1
    assert (positions[0].x, positions[0].y) == (1.0, 2.0)


def test_sample_dataset_rejects_invalid_or_missing_record_id(tmp_path: Path) -> None:
    dataset = SampleDataset("kl_8", tmp_path)
    _make_record(tmp_path, "batch", "record")

    with pytest.raises(ValueError, match="<batch>/<record>"):
        dataset.get_record("record")
    with pytest.raises(KeyError, match="not found"):
        dataset.get_record("batch/missing")


def test_sample_dataset_rejects_unknown_dataset(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown dataset"):
        SampleDataset("unknown", tmp_path)
