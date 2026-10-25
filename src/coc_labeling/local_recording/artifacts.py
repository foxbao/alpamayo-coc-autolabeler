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

"""Write run manifests and agent input previews."""

from __future__ import annotations

import json
from collections.abc import Sequence
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path
from typing import Any

from .camera import _event_timestamp, _timestamp_micros
from .types import SelectedFrame


def _write_frame_manifest(
    manifest_path: Path,
    record_id: str,
    event_time_sec: Decimal,
    selected: Sequence[SelectedFrame],
) -> None:
    """将选帧来源、目标时刻和实际误差写入 JSON 清单。

    例如，一项记录会同时保存目标时间 1003.5 秒、源图时间 1003.52 秒和 20 毫秒误差。
    """
    # 同时保存目标时间和源图片实际时间，方便检查采样误差，
    # 无需重新打开图片或重复计算选帧结果。
    payload = {
        "record_id": record_id,
        "time_origin": "first_front_camera_filename_timestamp",
        "event_time_sec": float(event_time_sec),
        "event_start_timestamp": _timestamp_micros(_event_timestamp(selected)),
        "event_start_frame": int(
            (event_time_sec * Decimal(10)).to_integral_value(rounding=ROUND_FLOOR)
        ),
        "inference_status": "prepared_only",
        "frames": [
            {
                "relative_time_sec": float(frame.relative_time),
                "target_unix_timestamp_sec": str(frame.target_timestamp),
                "source_unix_timestamp_sec": str(frame.image.timestamp),
                "source_image": str(frame.image.path),
                "absolute_error_ms": float(frame.error_sec * 1000),
            }
            for frame in selected
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_agent_input_preview(data: dict[str, Any], output_path: Path) -> Path:
    """保存将发送给 agent 的运动文本和提示说明，供推理前人工检查。"""
    payload = {
        "clip_id": data["clip_id"],
        "event_start_timestamp": data["event_start_timestamp"],
        "ego_text": data["ego_text"],
        "ego_state_prompt": data.get("ego_state_prompt"),
        "all_meta_action_str": data["all_meta_action_str"],
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path
