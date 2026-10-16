# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""第 06 章：用一个可预测的假模型模拟 VLM 接口。"""


messages = [
    {
        "role": "user",
        "m_type": "text",
        "content": "t=-0.1s action=Maintain Speed speed=8.0",
    },
    {
        "role": "user",
        "m_type": "video",
        "content": ["frame_1", "frame_2", "frame_3"],
    },
    {
        "role": "user",
        "m_type": "text",
        "content": "t=+0.0s action=Strong Deceleration speed=6.0",
    },
]


def get_text(messages: list[dict[str, object]]) -> str:
    """只提取文本消息，模拟 wrapper 对消息的整理。"""
    text_parts = [
        str(message["content"])
        for message in messages
        if message["m_type"] == "text"
    ]
    return "\n".join(text_parts)


def fake_vlm(messages: list[dict[str, object]]) -> dict[str, dict[str, str]]:
    """根据动作先验返回固定结构；它不具备真实视觉理解能力。"""
    text = get_text(messages)
    has_video = any(message["m_type"] == "video" for message in messages)
    if not has_video:
        raise ValueError("the VLM input must contain video evidence")
    if "Strong Deceleration" in text:
        return {
            "ego_behavior_schema": {
                "effect_on_ego_behavior": "The ego vehicle slows after the event anchor."
            }
        }
    return {
        "ego_behavior_schema": {
            "effect_on_ego_behavior": (
                "The ego vehicle continues steadily through the observed scene."
            )
        }
    }


print("发送给 fake_vlm 的消息数量:", len(messages))
print("fake_vlm 返回:")
print(fake_vlm(messages))

print("\n小实验：删除包含 Strong Deceleration 的文本消息，再运行一次。")
print("你会看到输出改变；真实 VLM 则还会综合视频像素和 ego 数值。")
print("真实默认生成也是一次 infer 调用；可选的横向修复调用不是默认三阶段生成。")
