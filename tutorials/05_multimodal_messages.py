# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""第 05 章：把一个事件窗口拆成 VLM 的多模态输入。"""

import json


samples = [
    {"t": -0.2, "frame": "frame_1", "action": "Maintain Speed", "speed": 10.0},
    {"t": -0.1, "frame": "frame_2", "action": "Maintain Speed", "speed": 8.0},
    {"t": 0.0, "frame": "frame_3", "action": "Strong Deceleration", "speed": 6.0},
    {"t": 0.1, "frame": "frame_4", "action": "Strong Deceleration", "speed": 4.0},
]

# 先分别构造每种信息；不要一开始就把它们拼成一个大字符串。
system_text = "You are a driving behavior analyst."
task_text = "Explain what happened around t=0."
video_frames = [sample["frame"] for sample in samples]
action_text = "\n".join(
    f"t={sample['t']:+.1f}s action={sample['action']}" for sample in samples
)
ego_text = "\n".join(
    f"t={sample['t']:+.1f}s speed={sample['speed']:.1f} m/s" for sample in samples
)

messages = [
    {"role": "system", "m_type": "text", "content": system_text},
    {"role": "user", "m_type": "text", "content": task_text},
    {"role": "user", "m_type": "video", "content": video_frames},
    {"role": "user", "m_type": "text", "content": action_text},
    {"role": "user", "m_type": "text", "content": ego_text},
]

print("VLM 将收到的消息:")
for index, message in enumerate(messages):
    print(f"\n[{index}] role={message['role']} m_type={message['m_type']}")
    print(json.dumps(message["content"], ensure_ascii=False, indent=2))

print("\nvideo 不是 prompt 文本；它是独立的帧列表。")
print("生产消息实际有 8 条：这里合并了几条说明，便于先理解数据类型。")
print("schema 也不是普通消息，生产代码会把它传给 infer(json_schema=...)。")
# 小实验：删除 video 消息，思考模型还缺少哪一种证据。
