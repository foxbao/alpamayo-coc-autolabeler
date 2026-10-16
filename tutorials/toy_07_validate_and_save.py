# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""第 07 章：检查模型返回值，并保存可追溯的 JSON。"""

import json
from pathlib import Path


model_output = {
    "ego_behavior_schema": {
        "effect_on_ego_behavior": "The ego vehicle slows after the event anchor."
    }
}


def validate(output: dict[str, object]) -> None:
    """验证最小嵌套结构，字段名与生产 OutputSchema 对齐。"""
    behavior = output.get("ego_behavior_schema")
    if not isinstance(behavior, dict):
        raise TypeError("ego_behavior_schema must be an object")
    effect = behavior.get("effect_on_ego_behavior")
    if not isinstance(effect, str) or not effect.strip():
        raise ValueError("effect_on_ego_behavior must be a non-empty string")


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
print("真实 schema 是 {'ego_behavior_schema': {'effect_on_ego_behavior': '<causal sentence>'}}。")
print("toy 在生成后检查结构；真实 Qwen wrapper 会把 OutputSchema 传给 vLLM 做结构化解码。")
print("生产结果写 YAML；本 toy 写 JSON，便于直接观察字段和层级。")
print("小实验：删除 effect_on_ego_behavior，观察 validation 如何阻止保存。")
