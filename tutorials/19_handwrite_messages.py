# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""第 19 课：读取上一课证据，自己组织成 VLM 的 8 条消息。"""

import json
from pathlib import Path
from typing import Any

from coc_labeling.data_loader.motion_conventions import LATERAL_SPEED_SIGN_DESCRIPTION

BASE = Path(__file__).parent / "_outputs/rebuild"


def main() -> None:
    evidence: dict[str, Any] = json.loads(
        (BASE / "evidence.json").read_text(encoding="utf-8")
    )
    event_frame = int(evidence["event_start_frame_clipgt_10hz"])
    action_lines = [
        "(t={time:+.1f}s) longitudinal: {longitudinal}, lateral: {lateral}, lane: {lane}".format(
            time=item["relative_time_seconds"],
            longitudinal=item["longitudinal"],
            lateral=item["lateral"],
            lane=item["lane"],
        )
        for item in evidence["meta_actions"]
    ]
    ego_lines = [
        "(t={time:+.1f}s, v_long={v_long:.3f} m/s, v_lat={v_lat:.3f} m/s)".format(
            time=sample["relative_time_seconds"],
            v_long=sample["v_long_mps"],
            v_lat=sample["v_lat_mps"],
        )
        for sample in evidence["ego_speed_samples"]
    ]
    frame_names = [
        f"<ndarray for source frame {frame['frame_index']}>"
        for frame in evidence["video_frames"]
    ]
    messages = [
        {
            "role": "system",
            "m_type": "text",
            "content": (
                "You are a specialized autonomous-driving scene-understanding agent. "
                "Return structured JSON following the schema strictly."
            ),
        },
        {
            "role": "user",
            "m_type": "text",
            "content": (
                "These are front-view image frames at 2 Hz. The video is 8 seconds: "
                "2 seconds of history and 6 seconds of future."
            ),
        },
        {"role": "user", "m_type": "video", "content": frame_names},
        {
            "role": "user",
            "m_type": "text",
            "content": (
                "These meta-actions correspond to each frame. The lane field is a useful "
                "but fallible prior for lane changes and nudges; combine it with video "
                "and longitudinal motion."
            ),
        },
        {"role": "user", "m_type": "text", "content": "\n".join(action_lines)},
        {
            "role": "user",
            "m_type": "text",
            "content": f"These are ego body-frame motion states. {LATERAL_SPEED_SIGN_DESCRIPTION}",
        },
        {"role": "user", "m_type": "text", "content": "\n".join(ego_lines)},
        {
            "role": "user",
            "m_type": "text",
            "content": (
                "Use the closed decision vocabulary: Stop, Yield, Pass/Overtake, "
                "Gap-search, Adapt Speed, Resume Speed, Keep Distance, Keep Lane, "
                "Lane Change, Nudge, Turn, Merge, or Split. Interesting decisions "
                "have priority; lane change/nudge must not be downgraded to Keep Lane. "
                "Return one causal sentence in the required JSON schema.\n\n"
                "Examples: Change lanes to the right while maintaining safe distance.\n"
                "Nudge left around the parked vehicle while yielding to a pedestrian.\n"
                "Stop behind the lead vehicle at the yellow/red traffic light."
            ),
        },
    ]
    output = BASE / "messages.json"
    output.write_text(json.dumps(messages, indent=2), encoding="utf-8")

    print("事件: clip={clip}, frame={frame}, timestamp={timestamp} us".format(
        clip=evidence["clip_id"],
        frame=event_frame,
        timestamp=evidence["event_start_timestamp_micros"],
    ))
    print("消息条数:", len(messages))
    for index, message in enumerate(messages):
        print(index, message["role"], message["m_type"], type(message["content"]).__name__)
    print("saved:", output)
    print("\n这里的 ndarray 是文字占位；真实 video content 是 16 张图像数组，按 2 Hz 组成 8 秒输入。")
    print("最后一条只摘录了真实 output_prompts 的决策表、优先级和示例；生产文本约 50 KB。")
    print("完整规则包含 lane change/nudge/gap-search、cut-in、construction 等优先级与校验。")
    print("生产 VLMCoCAgent._build_input_message() 使用正式 CoC prompt 和同样的 8 段结构。")


if __name__ == "__main__":
    main()
