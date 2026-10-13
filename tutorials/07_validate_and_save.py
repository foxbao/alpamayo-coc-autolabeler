# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""第 07 章：检查模型返回值，并保存可追溯的 JSON。"""

import json
from pathlib import Path


model_output = {
    "ego_behavior": "Strong Deceleration",
    "explanation": "The ego vehicle slows after the event anchor.",
}


def validate(output: dict[str, object]) -> None:
    """实现一个最小 schema：字段存在、类型正确、文本非空。"""
    required_fields = {"ego_behavior", "explanation"}
    missing_fields = required_fields - output.keys()
    if missing_fields:
        raise ValueError(f"missing fields: {sorted(missing_fields)}")
    if not isinstance(output["ego_behavior"], str):
        raise TypeError("ego_behavior must be a string")
    if not isinstance(output["explanation"], str) or not output["explanation"].strip():
        raise ValueError("explanation must be a non-empty string")


validate(model_output)

# 不只保存模型回答，也保存 clip 和事件位置，之后才能追查输入。
result = {
    "clip_id": "toy_clip",
    "event_start_frame": 3,
    "model_output": model_output,
}
output_path = Path("tutorials/_outputs/from_scratch_label.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

print("schema validation: PASS")
print("saved:", output_path)
print(
    "真实 CoC 输出是 "
    "{'ego_behavior_schema': {'effect_on_ego_behavior': '<causal sentence>'}}。"
)
print("真实 wrapper 主要在解码阶段用 structured output 约束格式；toy 才在生成后 validate。")
print("小实验：删除 explanation 字段，观察 validation 如何阻止保存。")
