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

"""Read timestamped camera images and select frames around an event."""

from __future__ import annotations

import argparse
import bisect
from collections.abc import Sequence
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path

import cv2
import numpy as np

from .types import MAX_FRAME_ERROR_SEC, SelectedFrame, TimedImage


def _parse_filename_timestamp(path: Path) -> Decimal:
    """从图片文件名解析以秒为单位的时间戳。

    例如，``1792395581.796917231.jpg`` 会解析为
    ``Decimal("1792395581.796917231")``。
    """
    try:
        timestamp = Decimal(path.stem)
    except InvalidOperation as exc:
        raise ValueError(f"Camera image filename is not a numeric timestamp: {path.name}") from exc
    if not timestamp.is_finite():
        raise ValueError(f"Camera image timestamp must be finite: {path.name}")
    return timestamp

def _parse_event_time(value: str) -> Decimal:
    """解析非负事件时间，并在命令行参数无效时给出明确错误。

    例如，字符串 ``"21.6"`` 会变成 ``Decimal("21.6")``；负数或非数字会被拒绝。
    """
    try:
        event_time = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError("event time must be a number of seconds") from exc
    if not event_time.is_finite() or event_time < 0:
        raise argparse.ArgumentTypeError("event time must be finite and non-negative")
    return event_time

def load_timed_images(camera_dir: Path) -> list[TimedImage]:
    """读取 front 相机目录中的 JPEG，并按文件名时间戳排序。

    例如，目录中有 ``103.0.jpg``、``101.0.jpg`` 时，返回顺序是 101.0、103.0。
    """
    if not camera_dir.is_dir():
        raise FileNotFoundError(f"Front camera directory not found: {camera_dir}")
    paths = sorted(
        path for path in camera_dir.iterdir() if path.is_file() and path.suffix.lower() == ".jpg"
    )
    if not paths:
        raise FileNotFoundError(f"No JPEG images found in {camera_dir}")
    # 使用 Decimal 保留图片文件名中亚毫秒级的时间精度。
    images = [TimedImage(path=path, timestamp=_parse_filename_timestamp(path)) for path in paths]
    return sorted(images, key=lambda image: (image.timestamp, image.path.name))

def _sample_offsets(
    history_sec: Decimal, future_sec: Decimal, interval_sec: Decimal
) -> list[Decimal]:
    """生成历史和未来采样时刻；事件时刻 t=0 只包含一次。

    例如，历史 2 秒、未来 1 秒、间隔 0.5 秒时，结果是
    ``[-2, -1.5, -1, -0.5, 0, 0.5]``。未来窗口不包含右端点 1 秒。
    """
    if interval_sec <= 0 or history_sec <= 0 or future_sec <= 0:
        raise ValueError("History, future, and sampling interval must be positive")
    history_count = history_sec / interval_sec
    future_count = future_sec / interval_sec
    if history_count != history_count.to_integral_value():
        raise ValueError("History length must be divisible by the sampling interval")
    if future_count != future_count.to_integral_value():
        raise ValueError("Future length must be divisible by the sampling interval")

    # 历史帧止于 t=0 之前；未来帧从 t=0 开始，因此事件帧只出现一次。
    # 默认配置会生成 4 张历史帧和 12 张未来帧。
    offsets = [
        -history_sec + interval_sec * index for index in range(int(history_count))
    ]
    offsets.extend(interval_sec * index for index in range(int(future_count)))
    return offsets

def select_frames(
    images: Sequence[TimedImage],
    event_time_sec: Decimal,
    offsets: Sequence[Decimal],
    *,
    max_error_sec: Decimal = MAX_FRAME_ERROR_SEC,
) -> list[SelectedFrame]:
    """为每个相对事件时刻选择最近且不重复的源图片。

    相对时间换算公式是 ``图片起始时间 + 事件秒数 + 相对偏移``。例如，
    起始时间为 1000 秒、事件时间为 5 秒、偏移为 -1.5 秒时，目标时间是 1003.5 秒。
    若最近图片时间为 1003.52 秒，误差 20 毫秒，可以选中；误差超过 50 毫秒则报错。
    """
    if not images:
        raise ValueError("No camera images are available")
    if not event_time_sec.is_finite() or event_time_sec < 0:
        raise ValueError("Event time must be a finite, non-negative number of seconds")

    timestamps = [image.timestamp for image in images]
    # 用 front 相机第一张图片文件名中的时间戳作为相对时间起点。
    camera_start = timestamps[0]
    selected: list[SelectedFrame] = []
    used_paths: set[Path] = set()

    for offset in offsets:
        # 将请求的相对时刻换算成绝对 Unix 时间戳。图片已经排序，
        # 因此用二分查找，只需比较目标点前后的两张候选图片。
        target = camera_start + event_time_sec + offset
        insertion_index = bisect.bisect_left(timestamps, target)
        candidate_indices = [
            index
            for index in (insertion_index - 1, insertion_index)
            if 0 <= index < len(images)
        ]
        image = min(
            (images[index] for index in candidate_indices),
            key=lambda candidate: (
                abs(candidate.timestamp - target),
                candidate.timestamp,
                candidate.path.name,
            ),
        )
        error_sec = abs(image.timestamp - target)
        if error_sec > max_error_sec:
            raise ValueError(
                f"No camera image within {max_error_sec * 1000} ms of t={offset:+.1f}s; "
                f"nearest is {image.path.name} ({error_sec * 1000:.3f} ms away)"
            )
        if image.path in used_paths:
            raise ValueError(
                f"Camera image would be reused at t={offset:+.1f}s: {image.path.name}"
            )
        used_paths.add(image.path)
        selected.append(
            SelectedFrame(
                relative_time=offset,
                target_timestamp=target,
                image=image,
                error_sec=error_sec,
            )
        )
    return selected

def _timestamp_micros(timestamp_sec: Decimal) -> int:
    """将 Decimal 类型的 Unix 秒时间戳四舍五入为整数微秒。

    例如，1000.25 秒会变成 1,000,250,000 微秒。
    """
    return int((timestamp_sec * Decimal(1_000_000)).to_integral_value(rounding=ROUND_HALF_UP))

def _event_timestamp(selected: Sequence[SelectedFrame]) -> Decimal:
    """从相对时刻 t=0 的目标时间取得事件的绝对相机时间戳。

    选帧里同时有 -0.5、0、+0.5 秒时，返回中间那帧的目标绝对时间。
    """
    for frame in selected:
        if frame.relative_time == 0:
            return frame.target_timestamp
    raise ValueError("Selected CoC timestamps must include the event time t=0.0s")

def _load_rgb_frames(selected: Sequence[SelectedFrame]) -> list[np.ndarray]:
    """用 OpenCV 将选中的 JPEG 读成现有 VLM 封装所需的 BGR 数组。

    例如，一张 1920×1080 的图片会成为形状为 ``(1080, 1920, 3)`` 的数组。
    OpenCV 使用 BGR 通道顺序；函数名中的 RGB 是历史命名。
    """
    frames: list[np.ndarray] = []
    for item in selected:
        frame = cv2.imread(str(item.image.path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Could not decode camera image: {item.image.path}")
        frames.append(frame)
    return frames
