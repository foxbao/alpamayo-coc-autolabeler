# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""第 08 章：把前面学过的最小步骤重新组合成完整 toy pipeline。"""

import json
from pathlib import Path
from typing import TypedDict


class Span(TypedDict):
    """一个连续 meta-action 区间；start/end 是帧索引。"""

    action: str
    start: int
    end: int


def classify(speed: float, delta: float) -> str:
    if speed < 0.5:
        return "Stop"
    if delta < -1.0:
        return "Strong Deceleration"
    if delta > 1.0:
        return "Acceleration"
    return "Maintain Speed"


def make_actions(speeds: list[float]) -> list[str]:
    actions = []
    for index, speed in enumerate(speeds):
        delta = 0.0 if index == 0 else speed - speeds[index - 1]
        actions.append(classify(speed, delta))
    return actions


def make_spans(actions: list[str]) -> list[Span]:
    spans = []
    start = 0
    current = actions[0]
    for index, action in enumerate(actions[1:], start=1):
        if action != current:
            spans.append({"action": current, "start": start, "end": index})
            start, current = index, action
    spans.append({"action": current, "start": start, "end": len(actions)})
    return spans


def make_messages(speeds: list[float], event_start: int) -> list[dict[str, object]]:
    """为一个事件创建极简的 system、视频、动作和 ego 消息。"""
    actions = make_actions(speeds)
    ticks = range(max(0, event_start - 2), min(len(speeds), event_start + 3))
    action_lines = []
    ego_lines = []
    frames = []
    for tick in ticks:
        relative_time = (tick - event_start) / 10
        frames.append(f"frame_{tick}")
        action_lines.append(f"t={relative_time:+.1f}s action={actions[tick]}")
        ego_lines.append(f"t={relative_time:+.1f}s speed={speeds[tick]:.1f} m/s")
    return [
        {"role": "system", "m_type": "text", "content": "You analyze driving."},
        {"role": "user", "m_type": "video", "content": frames},
        {"role": "user", "m_type": "text", "content": "\n".join(action_lines)},
        {"role": "user", "m_type": "text", "content": "\n".join(ego_lines)},
    ]


def fake_vlm(messages: list[dict[str, object]]) -> dict[str, dict[str, str]]:
    if not any(message["m_type"] == "video" for message in messages):
        raise ValueError("pipeline must provide video evidence")
    text = "\n".join(str(message["content"]) for message in messages)
    if "Strong Deceleration" in text:
        return {
            "ego_behavior_schema": {
                "effect_on_ego_behavior": "The ego vehicle slows around the selected event."
            }
        }
    return {
        "ego_behavior_schema": {
            "effect_on_ego_behavior": "The ego vehicle continues steadily."
        }
    }


def validate_output(output: dict[str, object]) -> None:
    """检查与真实 OutputSchema 相同的最小嵌套字段。"""
    behavior = output.get("ego_behavior_schema")
    if not isinstance(behavior, dict):
        raise TypeError("ego_behavior_schema must be an object")
    effect = behavior.get("effect_on_ego_behavior")
    if not isinstance(effect, str) or not effect.strip():
        raise ValueError("effect_on_ego_behavior must be a non-empty string")


speeds = [10.0, 10.0, 8.0, 6.0, 4.0, 2.0, 0.0, 0.0]
actions = make_actions(speeds)
spans = make_spans(actions)
event = None
for span in spans:
    duration = span["end"] - span["start"]
    if span["action"] == "Strong Deceleration" and duration >= 2:
        event = span
        break

if event is None:
    raise ValueError("Could not find a sufficiently long Strong Deceleration event.")

messages = make_messages(speeds, event["start"])
model_output = fake_vlm(messages)
validate_output(model_output)

result = {
    "clip_id": "toy_clip",
    "event": event,
    "messages": messages,
    "model_output": model_output,
}
output_path = Path(__file__).parent / "_outputs" / "full_pipeline_result.json"
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

print("model output validation: PASS")
print(json.dumps(result, indent=2))
print(f"\n完整结果已保存到: {output_path}")
print("\n这就是 toy autolabeler：数值 -> action -> event -> evidence -> output -> 保存。")
