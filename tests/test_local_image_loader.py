# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from decimal import Decimal
from pathlib import Path

import cv2
import numpy as np

from coc_labeling.recording_autolabel.camera import (
    ImgLoader,
    front_camera_directory,
    load_timed_images,
)


def test_img_loader_selects_and_decodes_event_window(tmp_path: Path) -> None:
    camera_dir = front_camera_directory(tmp_path)
    camera_dir.mkdir(parents=True)
    for timestamp in ("100.0", "100.5", "101.0"):
        frame = np.full((4, 6, 3), 127, dtype=np.uint8)
        assert cv2.imwrite(str(camera_dir / f"{timestamp}.jpg"), frame)

    loader = ImgLoader(
        history_sec=Decimal("0.5"),
        future_sec=Decimal("0.5"),
        interval_sec=Decimal("0.5"),
    )
    loaded = loader.load(camera_dir, Decimal("1.0"))
    reused_images = loader.load_from_images(load_timed_images(camera_dir), Decimal("1.0"))

    assert [frame.image for frame in reused_images.selected_frames] == [
        frame.image for frame in loaded.selected_frames
    ]
    assert reused_images.event_timestamp == loaded.event_timestamp
    assert [frame.image.path.name for frame in loaded.selected_frames] == [
        "100.5.jpg",
        "101.0.jpg",
    ]
    assert [frame.relative_time for frame in loaded.selected_frames] == [
        Decimal("-0.5"),
        Decimal("0.0"),
    ]
    assert loaded.event_timestamp == Decimal("101.0")
    assert [frame.shape for frame in loaded.decoded_frames] == [(4, 6, 3), (4, 6, 3)]


def test_front_camera_directory_uses_dataset_camera_name(tmp_path: Path) -> None:
    assert front_camera_directory(tmp_path) == tmp_path / "camera" / "front_image"
