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

"""Prepare the CoC input dictionary and call the configured VLM agent."""

from __future__ import annotations

import os
from collections.abc import Sequence
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml

from coc_labeling.data_loader.video_loader import FrameIdentifier

from .actions import LongitudinalActionResult, format_local_meta_action_text
from .camera import event_timestamp, timestamp_micros
from .types import (
    LOCAL_META_ACTION_PROMPT,
    LOCAL_POSITION_EGO_STATE_PROMPT,
    LOCAL_SCENE_CONTEXT_PROMPT,
    PositionMotionSample,
    SelectedFrame,
)


def _format_position_velocity(samples: Sequence[PositionMotionSample]) -> str:
    """将车体前向和右向速度格式化成 CoC 可读的时序文本。"""
    if not samples:
        raise ValueError("Position-velocity text requires at least one motion sample")
    pairs = []
    for sample in samples:
        v_long = (
            f"{sample.body_velocity_forward:.3f} m/s"
            if sample.body_velocity_forward is not None
            else "Not Available"
        )
        v_lat = (
            f"{sample.body_velocity_right:.3f} m/s"
            if sample.body_velocity_right is not None
            else "Not Available"
        )
        pairs.append(
            f"(t={sample.relative_time:+.1f}s, v_long={v_long}, v_lat={v_lat})"
        )
    return (
        "Position-derived body-frame ego velocity time series "
        "(relative to event time t=0.0s):\n"
        "- v_long is forward-positive and v_lat is right-positive: ["
        + ", ".join(pairs)
        + "]\n"
        "Missing values are not interpolated. Body-frame components require valid position "
        "differences and localization orientation."
    )


def _apply_local_prompt_overrides(
    messages: list[dict[str, Any]], data: dict[str, Any], prompt_module: Any
) -> list[dict[str, Any]]:
    """仅在本地适配层覆盖动作说明，并追加场景语义。"""
    meta_action_prompt = data.get("meta_action_prompt")
    if meta_action_prompt:
        for message in messages:
            if (
                message.get("role") == "user"
                and message.get("m_type") == "text"
                and message.get("content") == prompt_module.meta_action_prompt
            ):
                message["content"] = meta_action_prompt
                break

    scene_context = data.get("local_scene_context_prompt")
    if scene_context:
        messages.append({"role": "user", "m_type": "text", "content": scene_context})
    return messages


def build_agent_data(
    record_id: str,
    event_time_sec: Decimal,
    selected: Sequence[SelectedFrame],
    frames: Sequence[np.ndarray],
    position_motion: Sequence[PositionMotionSample] | None = None,
    longitudinal_actions: LongitudinalActionResult | None = None,
) -> dict[str, Any]:
    """组装 ``VLMCoCAgent`` 所需的最小输入字典。

    默认 16 帧会拆成 4 张历史帧和 12 张未来帧；港口只提供纵向动作，
    lateral/lane 保持 ``Not Available``。如果提供位置差分结果，``ego_text``
    只包含车体前向、右向速度分量。
    """
    if len(selected) != len(frames) or len(selected) < 2:
        raise ValueError("Selected-frame metadata and image arrays must have equal non-zero length")
    # 现有 agent 要求历史帧和未来帧分开传入，在 t=0 处分割。
    history_frames = sum(frame.relative_time < 0 for frame in selected)
    if history_frames == 0 or history_frames == len(selected):
        raise ValueError("Selected frames must include history and future images")
    if position_motion is not None:
        if len(position_motion) != 17:
            raise ValueError("Position-derived ego motion must contain exactly 17 samples")
        if any(
            later.relative_time <= earlier.relative_time
            for earlier, later in zip(position_motion, position_motion[1:])
        ):
            raise ValueError("Position-derived ego motion samples must be time ordered")
        if not any(sample.relative_time == 0 for sample in position_motion):
            raise ValueError("Position-derived ego motion samples must include event time t=0.0s")

    # 用 10 Hz 规则结果给每张 2 Hz 图片配一个纵向先验；横向和 lane 仍不可用。
    meta_action_text = format_local_meta_action_text(
        [frame.relative_time for frame in selected], longitudinal_actions
    )
    event_timestamp_us = timestamp_micros(event_timestamp(selected))
    all_frame_info = [
        FrameIdentifier(
            video_filename=frame.image.path.name,
            index_in_video=0,
            timestamp_micros=timestamp_micros(frame.image.timestamp),
        )
        for frame in selected
    ]
    if position_motion is None:
        ego_text = (
            "Ego velocity time series (relative to event time t=0.0s):\n"
            "- v_long (forward-positive): Not Available\n"
            "- v_lat (right-positive): Not Available\n"
            "Motion state is unavailable for this image-input smoke test."
        )
    else:
        ego_text = _format_position_velocity(position_motion)

    return {
        "clip_id": record_id,
        "event_start_timestamp": event_timestamp_us,
        # agent 的数据格式还要求提供 10 Hz 帧索引。上面的绝对相机时间戳
        # 才是精确事件时刻；这里提供的是向下取整得到的帧索引。
        "event_start_frame": int(
            (event_time_sec * Decimal(10)).to_integral_value(rounding=ROUND_FLOOR)
        ),
        "hist_fpv_frames": list(frames[:history_frames]),
        "fut_fpv_frames": list(frames[history_frames:]),
        "all_fpv_frames_info": all_frame_info,
        "all_meta_action_str": "\n" + meta_action_text,
        "ego_text": ego_text,
        **(
            {"meta_action_prompt": LOCAL_META_ACTION_PROMPT}
            if longitudinal_actions is not None
            else {}
        ),
        **(
            {"ego_state_prompt": LOCAL_POSITION_EGO_STATE_PROMPT}
            if position_motion is not None
            else {}
        ),
        "local_scene_context_prompt": LOCAL_SCENE_CONTEXT_PROMPT,
    }

def _configure_openai_environment(api_config_path: Path) -> None:
    """加载兼容 OpenAI 的 API key 和 base URL，不打印其具体值。

    配置文件可使用 ``api_key``、``base_url`` 两个字段；已经存在的
    ``OPENAI_API_KEY`` 或 ``OPENAI_BASE_URL`` 环境变量优先使用。
    """
    # 只有执行推理时才读取凭据，并且绝不将凭据内容写入日志。
    payload: object = {}
    if api_config_path.is_file():
        loaded = yaml.safe_load(api_config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"API config must contain a YAML mapping: {api_config_path}")
        payload = loaded
    elif not os.environ.get("OPENAI_API_KEY"):
        raise FileNotFoundError(f"API config not found: {api_config_path}")

    config = payload if isinstance(payload, dict) else {}
    api_key = config.get("api_key", config.get("OPENAI_API_KEY"))
    base_url = config.get("base_url", config.get("OPENAI_BASE_URL"))
    if not os.environ.get("OPENAI_API_KEY") and isinstance(api_key, str) and api_key.strip():
        os.environ["OPENAI_API_KEY"] = api_key.strip()
    elif not os.environ.get("OPENAI_API_KEY"):
        raise ValueError(f"API config has no api_key: {api_config_path}")
    if not os.environ.get("OPENAI_BASE_URL") and isinstance(base_url, str) and base_url.strip():
        os.environ["OPENAI_BASE_URL"] = base_url.strip()

    # 本次 GPT 推理只使用上面设置的 OpenAI 兼容凭据。
    os.environ.pop("NVIDIA_API_KEY", None)
    os.environ.pop("NVHOST_OAI_CLIENT_ID", None)
    os.environ.pop("NVHOST_OAI_CLIENT_SECRET", None)

def _run_agent(
    model_name: str,
    timeout_sec: int,
    api_config_path: Path,
    data_loader_config: Any,
    data: dict[str, Any],
    labels_dir: Path,
) -> Path:
    """调用一次现有 CoC agent，并返回生成的 YAML 文件路径。

    例如，记录 ID 为 ``my_record``、事件时间戳为 ``123456`` 时，
    预期结果路径是 ``<labels_dir>/my_record/cot_123456.yaml``。
    """
    _configure_openai_environment(api_config_path)
    from coc_labeling.agents.vlm.coc_agent import VLMCoCAgent

    class LocalPortVLMCoCAgent(VLMCoCAgent):
        """只为本地港口数据在共享 CoC 提示后追加场景上下文。"""

        def _build_input_message_with_prompt(
            self, data: dict[str, Any], prompt_module: Any
        ) -> list[dict[str, Any]]:
            messages = super()._build_input_message_with_prompt(data, prompt_module)
            return _apply_local_prompt_overrides(messages, data, prompt_module)

    agent_config = SimpleNamespace(
        agent_name="VLMCoCAgent",
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        use_metaaction_check_pipeline=False,
    )
    agent = LocalPortVLMCoCAgent(
        model_name,
        agent_config,
        vector_config=None,
        data_loader_config=data_loader_config,
        timeout_sec=timeout_sec,
    )
    # 通过项目现有的 agent 工厂调用模型，确保输出仍是标准 CoC YAML。
    agent.run(data, save_root=str(labels_dir))
    yaml_path = labels_dir / str(data["clip_id"]) / f"cot_{data['event_start_timestamp']}.yaml"
    if not yaml_path.is_file():
        raise RuntimeError(f"CoC agent completed without writing expected YAML: {yaml_path}")
    return yaml_path
