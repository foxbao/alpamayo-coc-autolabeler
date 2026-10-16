# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""第 14 课：纵向、横向、车道级动作是三条并行描述。

同一段时间可以同时有纵向加速、向左转向、向左换道等描述。三类动作使用不同证据，
不能把它们当作一个互斥分类器。本 toy 的输入是各通道已提取的区间特征；生产代码
还会从速度、heading、轨迹和 lane graph 计算这些特征。
"""

from typing import TypedDict


class LongitudinalEvidence(TypedDict):
    delta_velocity: float
    average_acceleration: float
    average_speed: float
    average_speed_along_heading: float


class LateralEvidence(TypedDict):
    average_turn_rate_rad_per_m: float
    total_heading_change_rad: float


class LaneEvidence(TypedDict):
    lane_decision: str
    lane_centerline_id_changed: bool


class Example(TypedDict):
    name: str
    longitudinal: LongitudinalEvidence
    lateral: LateralEvidence
    lane: LaneEvidence


def classify_longitudinal(evidence: LongitudinalEvidence) -> str:
    """按生产 longitudinal/classification.py 的优先级分类。"""
    if evidence["average_speed_along_heading"] <= -0.12:
        return "Reverse"
    if evidence["average_speed"] <= 0.12:
        return "Stop"
    if evidence["average_acceleration"] >= 1.4 and evidence["delta_velocity"] >= 0.8:
        return "StrongAcceleration"
    if (
        0.15 <= evidence["average_acceleration"] < 1.4
        and evidence["delta_velocity"] >= 0.8
    ):
        return "GentleAcceleration"
    if evidence["average_acceleration"] <= -1.4 and evidence["delta_velocity"] <= -0.8:
        return "StrongDeceleration"
    if (
        -1.4 < evidence["average_acceleration"] <= -0.15
        and evidence["delta_velocity"] <= -0.8
    ):
        return "GentleDeceleration"
    return "MaintainSpeed"


def classify_lateral(evidence: LateralEvidence) -> str:
    """横向转向看 heading 每米变化和整段 heading 变化。"""
    rate = evidence["average_turn_rate_rad_per_m"]
    heading = evidence["total_heading_change_rad"]
    if rate >= 0.0015 and heading >= 0.015:
        return "SteerLeft"
    if rate <= -0.0015 and heading <= -0.015:
        return "SteerRight"
    return "GoStraight"


def classify_lane(evidence: LaneEvidence) -> str:
    """车道级动作看车道决策及车道几何变化，不只看车头朝向。"""
    decision = evidence["lane_decision"]
    if decision == "left_lane_change" and evidence["lane_centerline_id_changed"]:
        return "LeftLaneChange"
    if decision == "right_lane_change" and evidence["lane_centerline_id_changed"]:
        return "RightLaneChange"
    if decision == "keep_lane":
        return "LaneKeep"
    return "Not Available"


def main() -> None:
    examples: list[Example] = [
        {
            "name": "弯道上轻微向左转向，但仍留在本车道",
            "longitudinal": {
                "delta_velocity": 0.1,
                "average_acceleration": 0.02,
                "average_speed": 8.0,
                "average_speed_along_heading": 8.0,
            },
            "lateral": {
                "average_turn_rate_rad_per_m": 0.004,
                "total_heading_change_rad": 0.08,
            },
            "lane": {"lane_decision": "keep_lane", "lane_centerline_id_changed": False},
        },
        {
            "name": "加速时向左换道",
            "longitudinal": {
                "delta_velocity": 1.0,
                "average_acceleration": 0.5,
                "average_speed": 8.5,
                "average_speed_along_heading": 8.5,
            },
            "lateral": {
                "average_turn_rate_rad_per_m": 0.004,
                "total_heading_change_rad": 0.08,
            },
            "lane": {"lane_decision": "left_lane_change", "lane_centerline_id_changed": True},
        },
        {
            "name": "强加速直行",
            "longitudinal": {
                "delta_velocity": 1.2,
                "average_acceleration": 1.8,
                "average_speed": 7.5,
                "average_speed_along_heading": 7.5,
            },
            "lateral": {
                "average_turn_rate_rad_per_m": 0.0,
                "total_heading_change_rad": 0.0,
            },
            "lane": {"lane_decision": "keep_lane", "lane_centerline_id_changed": False},
        },
        {
            "name": "轻微减速并向右换道",
            "longitudinal": {
                "delta_velocity": -1.0,
                "average_acceleration": -0.5,
                "average_speed": 6.0,
                "average_speed_along_heading": 6.0,
            },
            "lateral": {
                "average_turn_rate_rad_per_m": -0.004,
                "total_heading_change_rad": -0.08,
            },
            "lane": {"lane_decision": "right_lane_change", "lane_centerline_id_changed": True},
        },
        {
            "name": "强减速直行",
            "longitudinal": {
                "delta_velocity": -1.2,
                "average_acceleration": -1.8,
                "average_speed": 4.0,
                "average_speed_along_heading": 4.0,
            },
            "lateral": {
                "average_turn_rate_rad_per_m": 0.0,
                "total_heading_change_rad": 0.0,
            },
            "lane": {"lane_decision": "keep_lane", "lane_centerline_id_changed": False},
        },
        {
            "name": "停车",
            "longitudinal": {
                "delta_velocity": -0.1,
                "average_acceleration": -0.1,
                "average_speed": 0.08,
                "average_speed_along_heading": 0.08,
            },
            "lateral": {
                "average_turn_rate_rad_per_m": 0.0,
                "total_heading_change_rad": 0.0,
            },
            "lane": {"lane_decision": "keep_lane", "lane_centerline_id_changed": False},
        },
        {
            "name": "倒车",
            "longitudinal": {
                "delta_velocity": 0.0,
                "average_acceleration": 0.0,
                "average_speed": 0.3,
                "average_speed_along_heading": -0.3,
            },
            "lateral": {
                "average_turn_rate_rad_per_m": 0.0,
                "total_heading_change_rad": 0.0,
            },
            "lane": {"lane_decision": "keep_lane", "lane_centerline_id_changed": False},
        },
    ]

    for example in examples:
        labels = {
            "longitudinal": classify_longitudinal(example["longitudinal"]),
            "lateral": classify_lateral(example["lateral"]),
            "lane": classify_lane(example["lane"]),
        }
        print(example["name"])
        print("  同一时间段的三个标签:", labels)

    print(
        "\n关键区别：SteerLeft 描述车辆朝向/路径在转弯；"
        "LeftLaneChange 描述相对车道的变化。"
    )
    print(
        "生产纵向特征来自速度；横向特征来自 heading change per distance；"
        "lane 特征还使用 lane graph、车道中心线和 offset。"
    )
    print(
        "纵向 toy 已覆盖生产的 7 类：Reverse、Stop、Strong/GentleAcceleration、"
        "Strong/GentleDeceleration、MaintainSpeed。"
    )
    print(
        "生产横向还包括 SharpSteerLeft/Right、ReverseLeft/Right；生产 lane 还包括 "
        "SlightlyShiftLeft/Right、TurnLeft/Right、FollowCurveLeft/Right。"
    )
    print(
        "本 toy 的横向只演示 SteerLeft/Right/GoStraight，车道只演示 "
        "LaneKeep/LaneChange。显示标签叫 LaneKeep，但 loader 使用的 canonical key 是 keep_lane。"
    )
    print(
        "相关源码: processor/longitudinal、processor/lateral、processor/lane 的 "
        "segmentation.py 与 classification.py，以及 data_structures/ego_meta_action_wrapper.py。"
    )


if __name__ == "__main__":
    main()
