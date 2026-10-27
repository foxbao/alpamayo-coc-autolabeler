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

"""Shared data structures and settings for local-record labeling."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

MAX_FRAME_ERROR_SEC = Decimal("0.050")
MAX_LOCALIZATION_GAP_SEC = Decimal("0.032")
LOCALIZATION_RESAMPLE_INTERVAL_SEC = Decimal("0.1")
POSITION_DERIVATIVE_WINDOW_SEC = Decimal("0.5")
PREVIEW_COLUMNS = 4
THUMBNAIL_WIDTH = 480
LOCAL_POSITION_EGO_STATE_PROMPT = (
    "These ego motion states are estimated from localization positions resampled to 10 Hz, "
    "then differentiated and rotated from map XY into the vehicle Forward/Left/Up (FLU) "
    "frame using the localization orientation. For this dataset, the orientation's local x "
    "axis is interpreted as forward and local y as left, based on trajectory and front-camera "
    "evidence. v_long is forward-positive; v_lat is right-positive, so it is the negative of "
    "body-left velocity. v_lat is instantaneous body-frame sideways velocity, not lane-change "
    "direction. A missing v_long or v_lat means the position derivative or orientation could "
    "not be sampled reliably. Use camera frames to determine actions and lane changes."
)
LOCAL_META_ACTION_PROMPT = (
    "These local meta-actions are rule-based labels derived from the localization position "
    "trajectory and body-forward velocity. Longitudinal labels are PAI-style hints for "
    "acceleration, deceleration, maintaining speed, stopping, or reversing; they are not "
    "vehicle control commands and may be noisy. This port recording has no lane graph, so "
    "lateral and lane are intentionally Not Available. Use the camera frames as the primary "
    "evidence and do not infer a lane change from the missing fields."
)
LOCAL_SCENE_CONTEXT_PROMPT = (
    "Scene context: These recordings show an IGV operating in a Qingdao port. This dataset "
    "does not provide conventional traffic-lane annotations or a lane graph. Do not assume "
    "the vehicle is following or changing a road lane, and do not use ‘keep lane’, ‘stay in "
    "lane’, or ‘lane change’ as a generic description or fallback. Describe the visible path "
    "using port aisles, open drivable space, container rows, obstacles, and relative direction. "
    "For routine travel without a notable maneuver, describe continuing along the open path or "
    "aisle at a steady speed. Mention a lane only when a clearly visible marked lane is "
    "relevant to the observed action."
)


@dataclass(frozen=True)
class TimedImage:
    """一张相机图片，以及从文件名解析出的 Unix 时间戳。"""

    path: Path
    timestamp: Decimal


@dataclass(frozen=True)
class SelectedFrame:
    """为某个相对 CoC 事件时刻选中的源图片及其时间误差。"""

    relative_time: Decimal
    target_timestamp: Decimal
    image: TimedImage
    error_sec: Decimal


@dataclass(frozen=True)
class TimedPosition:
    """一条定位样本；时间来自文件名，位置在 map-XY，姿态按车体-FLU 到 map-XY 解释。"""

    timestamp: Decimal
    x: float
    y: float
    source_path: Path
    orientation_wxyz: tuple[float, float, float, float] | None = None
    linear_velocity_xyz: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class LocalMotionChunk:
    """本地自车动作区间，对应 PAI 的 ``TemporalMotionChunk``。"""

    agent_token: str
    action_type: str
    start_time: Decimal
    end_time: Decimal


@dataclass(frozen=True)
class ResampledPosition:
    """一条事件相对 10 Hz 网格上的定位位置；缺失点保留错误原因。"""

    relative_time: Decimal
    target_timestamp: Decimal
    position_valid: bool
    x: float | None
    y: float | None
    position_error: str | None
    connected_to_previous: bool


@dataclass(frozen=True)
class PositionMotionSample:
    """一个相对事件时刻的位置差分运动估计及车体坐标速度。"""

    relative_time: Decimal
    target_timestamp: Decimal
    position_valid: bool
    relative_x: float | None
    relative_y: float | None
    position_error: str | None
    motion_valid: bool
    velocity_x: float | None
    velocity_y: float | None
    ground_speed: float | None
    trajectory_bearing_deg: float | None
    motion_error: str | None
    connected_to_previous: bool
    body_velocity_right: float | None = None
    body_velocity_forward: float | None = None
    body_motion_error: str | None = None
    orientation_wxyz: tuple[float, float, float, float] | None = None
    recorded_linear_velocity_xyz: tuple[float, float, float] | None = None
    body_velocity_residual_mps: float | None = None
