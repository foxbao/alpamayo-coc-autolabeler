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

"""Load local localization data and prepare motion evidence for CoC labeling."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from .actions import LongitudinalActionResult, classify_longitudinal_series
from .localization import (
    _position_motion_offsets,
    _position_trajectory_offsets,
    _select_motion_samples,
    build_position_motion_samples,
    load_localization_positions,
    resample_position_trajectory,
)
from .types import (
    LOCALIZATION_RESAMPLE_INTERVAL_SEC,
    POSITION_DERIVATIVE_WINDOW_SEC,
    PositionMotionSample,
    ResampledPosition,
    TimedPosition,
)


@dataclass(frozen=True)
class LocalMotionData:
    """本地定位处理结果，供 Agent 输入和审计产物共同使用。"""

    localization_positions: list[TimedPosition]
    resampled_positions: list[ResampledPosition]
    coc_motion_samples: list[PositionMotionSample]
    longitudinal_actions: LongitudinalActionResult


class LocalMotionLoader:
    """从 localization JSON 构造轨迹、CoC 运动样本和纵向动作先验。

    10 Hz 运动样本在内部共享给动作分类器和 CoC 时间点抽取，避免重复求导。
    """

    def __init__(
        self,
        history_sec: Decimal,
        future_sec: Decimal,
        coc_interval_sec: Decimal,
    ) -> None:
        self.history_sec = history_sec
        self.future_sec = future_sec
        self.coc_interval_sec = coc_interval_sec

    def load(
        self,
        localization_dir: Path,
        event_timestamp: Decimal,
    ) -> LocalMotionData:
        """读取一条记录的定位数据，并生成 Agent 所需的运动证据。"""
        localization_positions = load_localization_positions(localization_dir)
        return self.load_from_positions(localization_positions, event_timestamp)

    def load_from_positions(
        self,
        localization_positions: list[TimedPosition],
        event_timestamp: Decimal,
    ) -> LocalMotionData:
        """复用已读取的定位样本，为一个事件计算运动证据。"""
        trajectory_offsets = _position_trajectory_offsets(
            self.history_sec,
            self.future_sec,
            derivative_window_sec=POSITION_DERIVATIVE_WINDOW_SEC,
        )
        resampled_positions = resample_position_trajectory(
            localization_positions,
            event_timestamp,
            trajectory_offsets,
        )

        coc_offsets = _position_motion_offsets(
            self.history_sec,
            self.future_sec,
            self.coc_interval_sec,
        )
        motion_offsets_10hz = _position_motion_offsets(
            self.history_sec,
            self.future_sec,
            LOCALIZATION_RESAMPLE_INTERVAL_SEC,
        )
        motion_samples_10hz = build_position_motion_samples(
            localization_positions,
            event_timestamp,
            motion_offsets_10hz,
            resampled_positions=resampled_positions,
        )
        coc_motion_samples = _select_motion_samples(
            motion_samples_10hz,
            coc_offsets,
        )
        if coc_motion_samples is None:
            # 若 CoC 时间间隔不再与 10 Hz 网格对齐，单独计算目标时刻。
            coc_motion_samples = build_position_motion_samples(
                localization_positions,
                event_timestamp,
                coc_offsets,
                resampled_positions=resampled_positions,
            )

        longitudinal_actions = classify_longitudinal_series(motion_samples_10hz)
        return LocalMotionData(
            localization_positions=localization_positions,
            resampled_positions=resampled_positions,
            coc_motion_samples=coc_motion_samples,
            longitudinal_actions=longitudinal_actions,
        )
