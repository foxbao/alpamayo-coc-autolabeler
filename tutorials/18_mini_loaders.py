# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""第 18 课：自己写小 loader，从真实 PAI 文件里拼出事件证据摘要。

这一课真的读 meta-action 文本、视频时间戳、mp4 和 trajdata Feather cache。
这里的 final_outputs/<clip_id>.txt 是上一阶段 ``meta_action.data_labeling`` 的产物。
各动作通道的分段、分类和短段清理发生在 ``TemporalScenario`` 与
``EgoMetaAction`` 相关代码中。之后 ``post_processing.process_batch`` 把 raw tags
映射到 clip 时间范围、排序并格式化成文本。第 10、12、14 课用简化代码讲解前一部分；
第 11 课专门对照生产的平滑/滞回逻辑，第 13 课对应生产的短段清理规则（两课仍是教学实现）。
本课的 loader 仍是教学实现，只保存摘要。生产 VectorLoader/VideoLoader 还会处理
更多边界情况和坐标语义。
"""

import json
import math
import re
from pathlib import Path
from typing import Any

import cv2
import pandas as pd


BASE = Path(__file__).parent / "_outputs/rebuild"
RUN = Path("/home/baojiali/Downloads/qwen-drive/outputs/pai_reasoning_16_official_20261013")
FPS_CLIPGT = 10
VIDEO_FPS = 30
META_ACTION_LINE = re.compile(
    r"^\s*(?P<action>.*?)\s+-\s+Agent:.*?Start:(?P<start>-?\d+),\s*End:(?P<end>-?\d+)"
)

DISPLAY_NAMES = {
    "GentleAcceleration": "Gentle Acceleration",
    "StrongAcceleration": "Strong Acceleration",
    "GentleDeceleration": "Gentle Deceleration",
    "StrongDeceleration": "Strong Deceleration",
    "MaintainSpeed": "Maintain Speed",
    "GoStraight": "Go Straight",
    "SteerLeft": "Steer Left",
    "SteerRight": "Steer Right",
    "LaneKeep": "Keep Lane",
    "KeepLane": "Keep Lane",  # loader 接受的旧别名；canonical key 是 keep_lane。
    "LeftLaneChange": "Left Lane Change",
    "RightLaneChange": "Right Lane Change",
    "SlightlyShiftLeft": "Slightly Shift Left",
    "SlightlyShiftRight": "Slightly Shift Right",
    "Stop": "Stop",
}

LONGITUDINAL = {
    "GentleAcceleration",
    "StrongAcceleration",
    "GentleDeceleration",
    "StrongDeceleration",
    "MaintainSpeed",
    "Stop",
    "Reverse",
}
LANE = {
    "LaneKeep",
    "KeepLane",
    "LeftLaneChange",
    "RightLaneChange",
    "SlightlyShiftLeft",
    "SlightlyShiftRight",
    "TurnLeft",
    "TurnRight",
    "FollowCurveLeft",
    "FollowCurveRight",
}


def read_meta_action_ranges(clip_id: str) -> list[dict[str, Any]]:
    """读取原始区间；生产 loader 随后会把区间填充到逐帧数组。"""
    path = RUN / "meta_action/final_outputs" / f"{clip_id}.txt"
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = META_ACTION_LINE.match(line)
        if match is None:
            continue
        start = int(match.group("start"))
        end = int(match.group("end"))
        result.append(
            {
                "action": match.group("action").strip(),
                "start_frame": start,
                "end_frame_exclusive": end,
            }
        )
    return result


def sample_meta_actions(clip_id: str, event_frame: int) -> list[dict[str, Any]]:
    """生成与生产 prompt 相同语义的 2 Hz、三字段采样表。

    生产代码暂时不把 lateral 通道送入 VLM（源码中明确注释了这一点），
    因此这里也保留该字段但固定为 ``Not Available``；lane 是高价值横向先验。
    """
    ranges = read_meta_action_ranges(clip_id)
    samples = []
    # 与生产 MetaActionLoader 一样，未来区间是 [t=0, t=+6s)，最后一点为 +5.5s。
    for offset in range(-2 * FPS_CLIPGT, 6 * FPS_CLIPGT, 5):
        frame = event_frame + offset
        longitudinal = "Not Available"
        lane = "Not Available"
        for item in ranges:
            if not (item["start_frame"] <= frame < item["end_frame_exclusive"]):
                continue
            raw = item["action"]
            display = DISPLAY_NAMES.get(raw, raw)
            if raw in LONGITUDINAL:
                longitudinal = display
            elif raw in LANE:
                lane = display
        samples.append(
            {
                "relative_time_seconds": offset / FPS_CLIPGT,
                "longitudinal": longitudinal,
                "lateral": "Not Available",
                "lane": lane,
            }
        )
    return samples


def event_timestamp_micros(event_frame: int) -> int:
    """把这个 PAI clip 的 10 Hz clipgt 帧换成相对起点的微秒时间。"""
    # 本例 clipgt 起点为 0，dt=0.1 秒；第 20 课会用 trajdata timing 通用计算。
    return event_frame * 100_000


def read_video_frames(clip_id: str, event_timestamp: int) -> tuple[int, list[dict[str, int]]]:
    """从 timestamp sidecar 找事件源视频帧，再抽取 2s/6s 窗口的 0.5s 样本。"""
    video_path = RUN / "video/camera" / f"{clip_id}.camera_front_wide_120fov.mp4"
    timestamp_path = video_path.with_suffix(".timestamps.parquet")
    table = pd.read_parquet(timestamp_path).reset_index()
    if "frame_index" not in table.columns:
        table = table.rename(columns={table.columns[0]: "frame_index"})
    nearest_row = (table["timestamp"] - event_timestamp).abs().idxmin()
    source_frame = int(table.loc[nearest_row, "frame_index"])

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"无法打开视频：{video_path}")

    # 视频采样是 -2 到 +5.5 秒的 16 帧；生产 ego 文本还会列出 +6.0 秒状态。
    sample_indices = range(
        max(0, source_frame - 2 * VIDEO_FPS),
        source_frame + 6 * VIDEO_FPS,
        VIDEO_FPS // 2,
    )
    frames = []
    for frame_index in sample_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()
        if success:
            height, width = frame.shape[:2]
            frames.append({"frame_index": frame_index, "width": width, "height": height})
    capture.release()
    if not frames:
        raise RuntimeError("事件窗口没有成功读取任何视频帧。")
    return source_frame, frames


def read_ego_speed_samples(clip_id: str, event_frame: int) -> list[dict[str, Any]]:
    """按每帧 heading 把 feather 的 vx/vy 投影到车体坐标系，得到 v_long/v_lat。

    做法与生产 ``data_loader/trajdata.py`` 中的速度投影一致。
    """
    path = RUN / "cache/trajdata/pai" / clip_id / "agent_data_dt0.10.feather"
    table = pd.read_feather(path)
    ego = table[table["agent_id"] == "ego"].copy()
    first_tick = event_frame - 2 * FPS_CLIPGT
    end_tick = event_frame + 6 * FPS_CLIPGT
    ego = ego[
        (ego["scene_ts"] >= first_tick)
        & (ego["scene_ts"] <= end_tick)
        & ((ego["scene_ts"] - event_frame) % 5 == 0)
    ]

    samples = []
    for row in ego.itertuples(index=False):
        tick = int(row.scene_ts)
        vx = float(row.vx)
        vy = float(row.vy)
        samples.append(
            {
                "relative_time_seconds": (tick - event_frame) / FPS_CLIPGT,
                # 生产 trajdata.py 会按每帧 heading 把全局速度旋转到车体坐标系。
                # 这样 v_long/v_lat 才分别表示车头方向和横向方向的速度。
                "v_long_mps": vx * math.cos(float(row.heading))
                + vy * math.sin(float(row.heading)),
                "v_lat_mps": -vx * math.sin(float(row.heading))
                + vy * math.cos(float(row.heading)),
            }
        )
    return samples


def main() -> None:
    selected = json.loads((BASE / "one_event.json").read_text(encoding="utf-8"))
    record = selected["record"]
    clip_id = str(record["clip_id"])
    event_frame = int(record["event_start_frame"])
    timestamp = event_timestamp_micros(event_frame)
    source_frame, frames = read_video_frames(clip_id, timestamp)
    evidence = {
        "action_bucket": selected["bucket"],
        "clip_id": clip_id,
        "event_start_frame_clipgt_10hz": event_frame,
        "event_start_timestamp_micros": timestamp,
        "event_start_frame_source_video": source_frame,
        "meta_actions": sample_meta_actions(clip_id, event_frame),
        "meta_action_ranges": read_meta_action_ranges(clip_id),
        "video_frames": frames,
        "ego_speed_samples": read_ego_speed_samples(clip_id, event_frame),
    }
    output = BASE / "evidence.json"
    output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence, indent=2))
    print("\nmeta_actions 现在是 16 个逐采样点的 longitudinal/lateral/lane 三字段。")
    print("lateral 按生产实现保留为 Not Available；")
    print("源码明确暂时不把该通道喂给 VLM。")
    print("v_long/v_lat 经过 heading 投影；")
    print("正 lateral 表示向右，负 lateral 表示向左。")
    print("当前 PAI 子集没有 lane action，因此 lane 通道会全部是 Not Available。")
    print("第 19 课读取这个 evidence.json，自己构造 8 条消息。")
    print("生产 LabelingAgent.run() 会调用 VectorLoader、VideoLoader、MetaActionLoader。")
    print("SegmentInfo 同时携带 clipgt 10 Hz 的 event_start_clipgt_index")
    print("和微秒 timestamp；它们都不是 30 FPS 视频源帧。")
    print("VideoLoader 还要通过 timestamp sidecar 转换。")


if __name__ == "__main__":
    main()
