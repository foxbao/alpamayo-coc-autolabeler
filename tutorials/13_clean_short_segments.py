# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""第 13 课：动作先分类，再清理太短的段和相邻同类段。

分段器的边界不一定正好落在真实动作边界上。生产 longitudinal cleaner 会先合并
相邻同类段，再按 caption 的最短时长处理短段，最后再次合并同类段。本 toy 演示短暂
减速段被两侧同类长段吸收的情形，并展示合并时统计量需要一起更新。
"""

import json
from typing import TypedDict


TICK_SECONDS = 0.1
MIN_TICKS_BY_ACTION = {"MaintainSpeed": 12}
DEFAULT_MIN_TICKS = 8


class Segment(TypedDict):
    action: str
    start_tick: int
    duration_ticks: int
    delta_velocity: float
    average_speed: float


def average_acceleration(segment: Segment) -> float:
    """根据整段速度变化和持续时间重算平均加速度。"""
    duration = segment["duration_ticks"] * TICK_SECONDS
    return segment["delta_velocity"] / duration if duration > 0 else 0.0


def merge_segments(parts: list[Segment], action: str) -> Segment:
    """合并相邻片段，并保留速度变化的可加性及速度的时长加权平均。"""
    total_ticks = sum(part["duration_ticks"] for part in parts)
    delta_velocity = sum(part["delta_velocity"] for part in parts)
    weighted_speed = sum(
        part["average_speed"] * part["duration_ticks"] for part in parts
    )
    merged: Segment = {
        "action": action,
        "start_tick": parts[0]["start_tick"],
        "duration_ticks": total_ticks,
        "delta_velocity": delta_velocity,
        "average_speed": weighted_speed / total_ticks if total_ticks else 0.0,
    }
    return merged


def merge_neighboring_same_actions(segments: list[Segment]) -> list[Segment]:
    """把紧挨着、caption 相同的段预先合在一起。"""
    merged: list[Segment] = []
    for segment in segments:
        if merged and merged[-1]["action"] == segment["action"]:
            merged[-1] = merge_segments([merged[-1], segment], segment["action"])
        else:
            merged.append(segment)
    return merged


def clean_short_segments(segments: list[Segment]) -> list[Segment]:
    """用邻居吸收短段，演示 longitudinal cleaner 的核心规则。"""
    rows = merge_neighboring_same_actions(segments)
    index = 0
    while index < len(rows):
        segment = rows[index]
        minimum = MIN_TICKS_BY_ACTION.get(segment["action"], DEFAULT_MIN_TICKS)
        if segment["duration_ticks"] >= minimum:
            index += 1
            continue

        has_previous = index > 0
        has_next = index + 1 < len(rows)
        if has_previous and has_next and rows[index - 1]["action"] == rows[index + 1]["action"]:
            # 同类长段夹着一个短段：三段合起来，保留两侧共同的动作名。
            action = rows[index - 1]["action"]
            replacement = merge_segments(rows[index - 1 : index + 2], action)
            rows[index - 1 : index + 2] = [replacement]
            index = max(0, index - 1)
        elif has_previous:
            # 没有同类两侧邻居时，短段并入前一段，动作名沿用前一段。
            rows[index - 1] = merge_segments(
                [rows[index - 1], segment], rows[index - 1]["action"]
            )
            rows.pop(index)
            index = max(0, index - 1)
        elif has_next:
            # 如果是第一段，则并入后一段并沿用后一段动作名。
            rows[index + 1] = merge_segments(
                [segment, rows[index + 1]], rows[index + 1]["action"]
            )
            rows.pop(index)
        else:
            # 全片只有一个短段时，没有邻居可合并。
            index += 1

    return merge_neighboring_same_actions(rows)


def describe(segments: list[Segment]) -> list[dict[str, object]]:
    """构造便于打印的统计摘要。"""
    result = []
    for segment in segments:
        result.append(
            {
                **segment,
                "duration_seconds": round(segment["duration_ticks"] * TICK_SECONDS, 2),
                "average_acceleration": round(average_acceleration(segment), 3),
            }
        )
    return result


def main() -> None:
    noisy_segments: list[Segment] = [
        {
            "action": "GentleAcceleration",
            "start_tick": 0,
            "duration_ticks": 10,
            "delta_velocity": 0.5,
            "average_speed": 5.0,
        },
        {
            "action": "GentleDeceleration",
            "start_tick": 10,
            "duration_ticks": 2,
            "delta_velocity": -0.1,
            "average_speed": 5.2,
        },
        {
            "action": "GentleAcceleration",
            "start_tick": 12,
            "duration_ticks": 10,
            "delta_velocity": 0.5,
            "average_speed": 5.4,
        },
        {
            "action": "MaintainSpeed",
            "start_tick": 22,
            "duration_ticks": 15,
            "delta_velocity": 0.0,
            "average_speed": 5.9,
        },
    ]
    cleaned = clean_short_segments(noisy_segments)

    print("分类后、清理前:")
    print(json.dumps(describe(noisy_segments), indent=2, ensure_ascii=False))
    print("\n清理后:")
    print(json.dumps(describe(cleaned), indent=2, ensure_ascii=False))
    print(
        "\n这里 2 tick = 0.2 秒的 GentleDeceleration 被两段 GentleAcceleration 吸收；"
        "速度变化量相加，平均速度按时长加权，平均加速度重新计算。"
    )
    print(
        "生产默认纵向最短长度是 8 tick（0.8 秒），"
        "MaintainSpeed 是 12 tick（1.2 秒）；"
        "本 toy 只实现最容易观察的邻居合并规则。"
    )
    print("源码: meta_action/data_structures/ego_meta_action.py:clean_longitudinal_segments")


if __name__ == "__main__":
    main()
