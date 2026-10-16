# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""第 16 课：从动作区间筛候选，再按动作类别限制样本数量。

生产 keyframe_auto_select 按动作名、最短时长和起始帧范围筛选 ActionRecord，
再限制每种动作最多保留多少条；超量时优先保留较长区间。本 toy 用内存列表演示，
并指出 all/sampled 两种 JSON 缓存的用途。
"""

import json
from typing import TypedDict


class ActionRecord(TypedDict):
    meta_action: str
    clip_id: str
    event_start_frame: int
    event_end_frame: int
    duration: int


# 这里只取 3 个动作，便于教学。生产中的等价集合是
# ``RESERVE_ALL_ACTIONS = set(ALL_META_ACTIONS)``，覆盖全部约 23 个 canonical 动作。
BUILT_IN_ACTIONS = {
    "gentle_acceleration",
    "strong_deceleration",
    "stop",
}


def filter_records(
    records: list[ActionRecord],
    selected_actions: set[str],
    min_duration: int,
    min_start_frame: int,
    max_start_frame: int,
) -> list[ActionRecord]:
    """只留下目标动作、足够长且起始帧落在范围内的区间。"""
    return [
        record
        for record in records
        if record["meta_action"] in selected_actions
        and record["duration"] >= min_duration
        and min_start_frame <= record["event_start_frame"] <= max_start_frame
    ]


def balance_by_action(
    records: list[ActionRecord], target_count: int, selected_actions: set[str]
) -> dict[str, list[ActionRecord]]:
    """按动作分别设上限；数量超限时保留持续时间最长的候选。"""
    buckets: dict[str, list[ActionRecord]] = {action: [] for action in sorted(selected_actions)}
    for record in records:
        buckets.setdefault(record["meta_action"], []).append(record)

    balanced = {}
    for action, action_records in buckets.items():
        if len(action_records) < target_count and action not in BUILT_IN_ACTIONS:
            raise ValueError(f"Custom action {action!r} has fewer than {target_count} records.")
        ranked = sorted(action_records, key=lambda item: item["duration"], reverse=True)
        balanced[action] = ranked[:target_count]
    return balanced


def main() -> None:
    records: list[ActionRecord] = [
        {"meta_action": "gentle_acceleration", "clip_id": "clip_a", "event_start_frame": 14,
         "event_end_frame": 40, "duration": 26},  # 起始帧太早
        {"meta_action": "gentle_acceleration", "clip_id": "clip_b", "event_start_frame": 30,
         "event_end_frame": 38, "duration": 8},  # 区间太短
        {"meta_action": "gentle_acceleration", "clip_id": "clip_c", "event_start_frame": 30,
         "event_end_frame": 55, "duration": 25},
        {"meta_action": "gentle_acceleration", "clip_id": "clip_d", "event_start_frame": 65,
         "event_end_frame": 110, "duration": 45},
        {"meta_action": "gentle_acceleration", "clip_id": "clip_e", "event_start_frame": 90,
         "event_end_frame": 120, "duration": 30},
        {"meta_action": "strong_deceleration", "clip_id": "clip_f", "event_start_frame": 40,
         "event_end_frame": 72, "duration": 32},
        {"meta_action": "strong_deceleration", "clip_id": "clip_g", "event_start_frame": 80,
         "event_end_frame": 105, "duration": 25},
        {"meta_action": "stop", "clip_id": "clip_h", "event_start_frame": 140,
         "event_end_frame": 165, "duration": 25},  # 起始帧太晚
    ]
    selected = {"gentle_acceleration", "strong_deceleration", "stop"}
    filtered = filter_records(
        records,
        selected_actions=selected,
        min_duration=10,
        min_start_frame=20,
        max_start_frame=135,
    )
    balanced = balance_by_action(filtered, target_count=2, selected_actions=selected)

    print("筛选条件：动作在白名单内，duration >= 10 帧，起始帧在 [20, 135]。")
    print("筛选后:")
    print(json.dumps(filtered, indent=2, ensure_ascii=False))
    print("\n每类最多保留 2 条；超量时按 duration 从长到短:")
    print(json.dumps(balanced, indent=2, ensure_ascii=False))
    print(
        "\n生产 keyframe_auto_select.py:parse_txt_file 对每个 span 应用这些过滤条件；"
        "balance_action_records 再按动作分别限量。"
    )
    print(
        "本 toy 的 BUILT_IN_ACTIONS 只有 3 个；生产的 RESERVE_ALL_ACTIONS 覆盖全部 "
        "canonical longitudinal/lateral/lane 动作。内置动作不足 target_count 时会保留所有样本；"
        "自定义动作不足时会报错，"
        "以免拼错动作名却悄悄得到空数据。"
    )
    print(
        "生产输出 all.json 保存筛选后的候选，sampled.json 保存平衡后的候选。"
        "如果复用旧 all.json，改了筛选参数后应确认缓存是否需要重建。"
    )


if __name__ == "__main__":
    main()
