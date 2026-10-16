# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""第 17 课：Action span、keyframe candidate、SegmentInfo 是三个阶段的对象。

Action span 是 meta-action 算出的动作区间；keyframe candidate 带动作名、clip、区间帧。
ParsedSegmentGenerator 过滤并转换时间后，把 VLM 要处理的锚点装进 SegmentInfo。
10 Hz clipgt 索引、微秒 timestamp、源视频帧索引是不同单位和坐标系。
"""

import re
from dataclasses import asdict, dataclass


@dataclass
class ActionSpan:
    """动作算法产出的区间；帧号属于 10 Hz clipgt 时间轴。"""

    action: str
    clip_id: str
    start_frame: int
    end_frame: int


@dataclass
class ActionRecord:
    """keyframe 选择器保存的候选 span。"""

    meta_action: str
    clip_id: str
    event_start_frame: int
    event_end_frame: int
    duration: int


@dataclass
class SegmentInfo:
    """生产 SegmentInfo 为 loader 提供事件锚点和时间。"""

    clip_id: str
    event_start_timestamp: int
    event_start_clipgt_index: int


ACTION_ALIASES = {
    "LaneKeep": "keep_lane",
    "KeepLane": "keep_lane",
    "LeftLaneChange": "left_lane_change",
    "RightLaneChange": "right_lane_change",
}


def canonicalize_action(action: str) -> str:
    """把显示标签转换成 keyframe 筛选使用的 canonical action key。"""
    stripped = action.strip()
    if stripped in ACTION_ALIASES:
        return ACTION_ALIASES[stripped]
    return re.sub(r"(?<!^)(?=[A-Z])", "_", stripped).lower()


def make_candidate(span: ActionSpan) -> ActionRecord:
    """把动作区间整理成 keyframe JSON 的主要字段。"""
    action_key = canonicalize_action(span.action)
    return ActionRecord(
        meta_action=action_key,
        clip_id=span.clip_id,
        event_start_frame=span.start_frame,
        event_end_frame=span.end_frame,
        duration=span.end_frame - span.start_frame,
    )


def make_segment_info(
    candidate: ActionRecord, clip_start_micros: int, dt_micros: int
) -> SegmentInfo:
    """从 clip 起点和 10 Hz 帧间隔换算事件 timestamp，再组成 SegmentInfo。"""
    event_timestamp = clip_start_micros + candidate.event_start_frame * dt_micros
    clipgt_index = int((event_timestamp - clip_start_micros) / dt_micros)
    return SegmentInfo(
        clip_id=candidate.clip_id,
        event_start_timestamp=event_timestamp,
        event_start_clipgt_index=max(0, clipgt_index),
    )


def nearest_video_frame(event_timestamp: int, video_timestamps: list[int]) -> int:
    """toy sidecar 映射：找 timestamp 最近的源视频帧行号。"""
    return min(
        range(len(video_timestamps)),
        key=lambda index: abs(video_timestamps[index] - event_timestamp),
    )


def main() -> None:
    span = ActionSpan(
        action="GentleAcceleration",
        clip_id="example_clip",
        start_frame=114,
        end_frame=143,
    )
    candidate = make_candidate(span)
    lane_candidate = make_candidate(
        ActionSpan(
            action="LaneKeep",
            clip_id="example_clip",
            start_frame=200,
            end_frame=220,
        )
    )

    # 用较大的起点模拟真实微秒时间戳；10 Hz clipgt 的 dt=100,000 us。
    clip_start_micros = 1_700_000_000_000_000
    dt_micros = 100_000
    segment = make_segment_info(candidate, clip_start_micros, dt_micros)

    # toy 假设源视频约 30 FPS；生产用 sidecar 查找最近时间。
    video_timestamps = [clip_start_micros + index * 33_333 for index in range(600)]
    video_frame = nearest_video_frame(segment.event_start_timestamp, video_timestamps)

    print("1. meta-action 区间:")
    print(asdict(span))
    print("\n2. keyframe candidate (保存动作类别和区间):")
    print(asdict(candidate))
    print("LaneKeep 的 candidate canonical key:", asdict(lane_candidate))
    print("\n3. SegmentInfo (生产 loader 的事件锚点):")
    print(asdict(segment))
    print("\n4. toy timestamp sidecar 映射出的源视频帧:", video_frame)
    print(
        "\n114 是从 clip 起点数的 10 Hz clipgt index；"
        "event_start_timestamp 是绝对微秒；"
        f"toy 中源视频帧约为 {video_frame}。它们描述同一时刻，但不能互换。"
    )
    print(
        "SegmentInfo 不携带动作区间的 end/duration；"
        "VLM loader 以 event_start 为中心，"
        "另外取 history/future 窗口。"
    )
    print(
        "生产对应: ParsedSegmentGenerator.generate_segment_list/populate_timestamp、"
        "SegmentInfo，以及 LabelingAgent.run -> VideoLoader.get_index_from_timestamp。"
    )
    print(
        "真实 keyframe_auto_select.py 使用 meta_action_loader.normalize_meta_action 和 "
        "canonicalize_action；本课的显式别名加正则只是教学近似。"
    )
    print(
        "对象闭环：ActionRecord -> SegmentInfo -> LabelingAgent/VLMCoCAgent -> YAML，"
        "最后可回到第 09 课查看最终 YAML。"
    )


if __name__ == "__main__":
    main()
