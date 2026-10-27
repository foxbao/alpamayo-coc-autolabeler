# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""本地车辆记录的完整时间状态和动作区间。"""

from __future__ import annotations

from .actions import (
    GENTLE_ACCELERATION,
    GENTLE_DECELERATION,
    MAINTAIN_SPEED,
    REVERSE,
    STOP,
    STRONG_ACCELERATION,
    STRONG_DECELERATION,
    LongitudinalActionResult,
    classify_longitudinal_series,
)
from .localization import (
    build_position_motion_samples,
    resample_position_trajectory,
)
from .types import LOCALIZATION_RESAMPLE_INTERVAL_SEC, LocalMotionChunk, TimedPosition

LOCAL_LONGITUDINAL_ACTION_TYPES = (
    REVERSE,
    STOP,
    STRONG_ACCELERATION,
    GENTLE_ACCELERATION,
    STRONG_DECELERATION,
    GENTLE_DECELERATION,
    MAINTAIN_SPEED,
)


class LocalTemporalScenario:
    """保存单条本地记录的轨迹状态，并按动作类别提供时间区间。

    结构对应 PAI 的 ``TemporalScenario``。本地记录只包含一辆自车，输入是
    localization JSON；派生出来的 10 Hz 纵向动作结果缓存在 ``segment_cache``。
    """

    def __init__(
        self,
        record_id: str,
        localization_positions: list[TimedPosition],
    ) -> None:
        if len(localization_positions) < 2:
            raise ValueError("本地 temporal scenario 至少需要两个定位样本")

        self.record_id = record_id
        self.all_agents_names = ("ego",)
        self.localization_positions = tuple(localization_positions)
        self.localization_start = localization_positions[0].timestamp
        self.segment_cache: dict[str, LongitudinalActionResult] = {}

    @property
    def longitudinal_action_result(self) -> LongitudinalActionResult:
        """返回缓存的整条记录 10 Hz 纵向动作点和区间。"""
        result = self.segment_cache.get("longitudinal_action_result")
        if result is None:
            duration = self.localization_positions[-1].timestamp - self.localization_start
            tick_count = int(duration / LOCALIZATION_RESAMPLE_INTERVAL_SEC)
            offsets = [
                LOCALIZATION_RESAMPLE_INTERVAL_SEC * index for index in range(tick_count + 1)
            ]
            resampled_positions = resample_position_trajectory(
                list(self.localization_positions),
                self.localization_start,
                offsets,
            )
            motion_samples = build_position_motion_samples(
                list(self.localization_positions),
                self.localization_start,
                offsets,
                resampled_positions=resampled_positions,
            )
            result = classify_longitudinal_series(motion_samples)
            self.segment_cache["longitudinal_action_result"] = result
        return result

    def get_tag_motions(self, action_type: str) -> list[LocalMotionChunk]:
        """返回一种纵向动作对应的自车区间。

        ``action_type`` 使用与 PAI prompt 相同的显示名称，例如 ``"Stop"`` 或
        ``"Maintain Speed"``。调用多个类别时会复用 ``segment_cache`` 中的分类结果。
        """
        if action_type not in LOCAL_LONGITUDINAL_ACTION_TYPES:
            raise ValueError(f"不支持的本地纵向动作类型：{action_type}")

        return [
            LocalMotionChunk(
                agent_token="ego",
                action_type=action_type,
                start_time=span.start_time,
                end_time=span.end_time,
            )
            for span in self.longitudinal_action_result.spans
            if span.label == action_type
        ]
