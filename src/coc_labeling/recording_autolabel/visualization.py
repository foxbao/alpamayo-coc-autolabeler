# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Write camera previews and CoC contact sheets."""

from __future__ import annotations

import math
import textwrap
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import yaml

from .types import PREVIEW_COLUMNS, THUMBNAIL_WIDTH, SelectedFrame


def _annotate_frame(frame: np.ndarray, label: str) -> np.ndarray:
    """在仅供检查的帧副本顶部添加时间戳和源文件名。

    例如，横幅文字可以是 ``t=-0.5s | source=1792395581.2.jpg``。
    """
    annotated = frame.copy()
    banner_height = 36
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], banner_height), (0, 0, 0), -1)
    cv2.putText(
        annotated,
        label[:150],
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return annotated


def _render_contact_sheet(
    selected: Sequence[SelectedFrame],
    frames: Sequence[np.ndarray],
    *,
    header_text: str | None = None,
) -> np.ndarray:
    """把选中帧排成网格；可选标题用于显示候选时间和动作提示。"""
    if len(selected) != len(frames) or not frames:
        raise ValueError("Selected-frame metadata and image arrays must have equal non-zero length")
    source_height, source_width = frames[0].shape[:2]
    thumbnail_height = max(1, round(THUMBNAIL_WIDTH * source_height / source_width))
    cell_height = thumbnail_height + 44
    header_lines = textwrap.wrap(header_text, width=105) if header_text else []
    header_height = 44 + 30 * len(header_lines) if header_lines else 0
    sheet_shape = (
        header_height + cell_height * math.ceil(len(frames) / PREVIEW_COLUMNS),
        THUMBNAIL_WIDTH * PREVIEW_COLUMNS,
        3,
    )
    sheet = np.zeros(sheet_shape, dtype=np.uint8)
    for line_index, line in enumerate(header_lines):
        cv2.putText(
            sheet,
            line[:130],
            (12, 30 + line_index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    for index, (item, frame) in enumerate(zip(selected, frames)):
        if frame.shape[:2] != (source_height, source_width):
            frame = cv2.resize(frame, (source_width, source_height))
        row, column = divmod(index, PREVIEW_COLUMNS)
        x, y = column * THUMBNAIL_WIDTH, header_height + row * cell_height
        thumbnail = cv2.resize(frame, (THUMBNAIL_WIDTH, thumbnail_height))
        sheet[y : y + thumbnail_height, x : x + THUMBNAIL_WIDTH] = thumbnail
        label = (
            f"t={item.relative_time:+.1f}s  error={item.error_sec * 1000:.1f}ms  "
            f"{item.image.path.name}"
        )
        cv2.rectangle(
            sheet,
            (x, y + thumbnail_height),
            (x + THUMBNAIL_WIDTH, y + cell_height),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            sheet,
            label[:75],
            (x + 8, y + thumbnail_height + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return sheet


def write_contact_sheet(
    selected: Sequence[SelectedFrame],
    frames: Sequence[np.ndarray],
    output_path: Path,
    *,
    header_text: str | None = None,
) -> Path:
    """只生成联系表 JPEG，可选标题说明候选时刻和纵向动作。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet = _render_contact_sheet(selected, frames, header_text=header_text)
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"Could not write contact sheet: {output_path}")
    return output_path


def write_preview_video(
    selected: Sequence[SelectedFrame],
    frames: Sequence[np.ndarray],
    output_path: Path,
    *,
    fps: float,
    header_text: str | None = None,
) -> Path:
    """Write selected frames as a timestamped, playable MP4 preview."""
    if len(selected) != len(frames) or not frames:
        raise ValueError("Selected-frame metadata and image arrays must have equal non-zero length")
    if fps <= 0:
        raise ValueError("Preview frame rate must be positive")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_height, source_width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (source_width, source_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create preview video: {output_path}")
    try:
        for item, frame in zip(selected, frames):
            if frame.shape[:2] != (source_height, source_width):
                frame = cv2.resize(frame, (source_width, source_height))
            label_parts = []
            if header_text:
                label_parts.append(header_text)
            label_parts.extend(
                (f"t={item.relative_time:+.1f}s", f"source={item.image.path.name}")
            )
            writer.write(_annotate_frame(frame, " | ".join(label_parts)))
    finally:
        writer.release()
    return output_path


def write_previews(
    selected: Sequence[SelectedFrame], frames: Sequence[np.ndarray], output_dir: Path
) -> tuple[Path, Path]:
    """生成带说明的联系表和预览视频。

    默认 16 帧排成每行 4 张的联系表；视频按 2 fps 写入，因此总时长为 8 秒。
    两种预览中的标签都来自相对事件时间和源图片文件名。
    """
    if len(selected) != len(frames) or not frames:
        raise ValueError("Selected-frame metadata and image arrays must have equal non-zero length")
    # 这两个文件都用于人工检查；标记只画在图片副本上，
    # 不会改动随后传给模型的源图片数组。
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_frames: list[np.ndarray] = []
    source_height, source_width = frames[0].shape[:2]
    for frame in frames:
        if frame.shape[:2] != (source_height, source_width):
            frame = cv2.resize(frame, (source_width, source_height))
        normalized_frames.append(frame)

    contact_sheet = output_dir / "contact_sheet.jpg"
    sheet = _render_contact_sheet(selected, normalized_frames)
    if not cv2.imwrite(str(contact_sheet), sheet):
        raise RuntimeError(f"Could not write contact sheet: {contact_sheet}")

    preview_video = output_dir / "preview.mp4"
    # 默认每 0.5 秒选一帧，因此 16 帧预览视频是 2 fps、8 秒。
    write_preview_video(selected, normalized_frames, preview_video, fps=2.0)
    return contact_sheet, preview_video


def write_contact_sheet_with_coc(
    contact_sheet_path: Path, result_yaml_path: Path, output_path: Path
) -> Path:
    """在原联系表上方添加 CoC 描述，并写出新的 JPEG 图片。

    例如，YAML 中的 ``effect_on_ego_behavior`` 是模型生成的一句话，
    函数会把它换行后放在 4×4 选帧网格上方，不修改原始联系表。
    """
    contact_sheet = cv2.imread(str(contact_sheet_path), cv2.IMREAD_COLOR)
    if contact_sheet is None:
        raise ValueError(f"Could not read contact sheet: {contact_sheet_path}")

    result = yaml.safe_load(result_yaml_path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"CoC YAML must contain a mapping: {result_yaml_path}")  # noqa: TRY004
    final_content = result.get("final_content")
    if not isinstance(final_content, dict):
        raise ValueError("CoC YAML final_content must be a mapping")  # noqa: TRY004
    ego_schema = final_content.get("ego_behavior_schema")
    if not isinstance(ego_schema, dict):
        raise ValueError(  # noqa: TRY004
            "CoC YAML final_content.ego_behavior_schema must be a mapping"
        )
    effect = ego_schema.get("effect_on_ego_behavior")
    if not isinstance(effect, str) or not effect.strip():
        raise ValueError("CoC YAML effect_on_ego_behavior is missing or empty")

    # 标题与描述放在单独的顶部区域，既不遮挡画面，也不覆盖原联系表。
    lines = textwrap.wrap(" ".join(effect.split()), width=110) or [effect]
    line_height = 32
    panel_height = 68 + line_height * len(lines)
    panel = np.zeros((panel_height, contact_sheet.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        panel,
        "CoC effect_on_ego_behavior",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )
    for line_index, line in enumerate(lines):
        cv2.putText(
            panel,
            line,
            (16, 68 + line_index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet_with_coc = np.vstack((panel, contact_sheet))
    if not cv2.imwrite(str(output_path), sheet_with_coc):
        raise RuntimeError(f"Could not write CoC contact sheet: {output_path}")
    return output_path
