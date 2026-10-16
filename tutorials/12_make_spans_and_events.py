# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""第 12 课：把逐帧动作合并成区间，认识生产代码里的 chunk。"""

import json
from pathlib import Path
from typing import TypedDict


BASE = Path(__file__).parent / "_outputs/rebuild"
INTERESTING_ACTIONS = {
    "GentleAcceleration",
    "StrongAcceleration",
    "GentleDeceleration",
    "StrongDeceleration",
    "Stop",
}


class Span(TypedDict):
    action: str
    start: int
    end: int
    duration: int


class Chunk(TypedDict):
    """一个供 classify_chunk 判断的区间摘要。"""

    start: int
    end: int
    duration_seconds: float
    delta_velocity: float
    average_acceleration: float
    average_speed: float
    toy_frame_action: str


def make_spans(rows: list[dict[str, object]]) -> list[Span]:
    if not rows:
        return []
    spans: list[Span] = []
    start = 0
    current = str(rows[0]["action"])
    for index in range(1, len(rows)):
        action = str(rows[index]["action"])
        if action != current:
            spans.append(
                {"action": current, "start": start, "end": index, "duration": index - start}
            )
            start, current = index, action
    spans.append(
        {"action": current, "start": start, "end": len(rows), "duration": len(rows) - start}
    )
    return spans


def summarize_chunk(rows: list[dict[str, object]], span: Span) -> Chunk:
    """把 span 覆盖的逐帧样本汇总成 chunk 特征。"""
    samples = rows[int(span["start"]) : int(span["end"])]
    first_speed = float(samples[0]["speed"])
    last_speed = float(samples[-1]["speed"])
    # toy rows 的相邻样本相隔 0.1 秒；首尾样本间隔数比样本数少一。
    duration_seconds = max(len(samples) - 1, 0) * 0.1
    delta_velocity = last_speed - first_speed
    average_acceleration = (
        delta_velocity / duration_seconds if duration_seconds > 0 else 0.0
    )
    average_speed = sum(float(sample["speed"]) for sample in samples) / len(samples)
    return {
        "start": int(span["start"]),
        "end": int(span["end"]),
        "duration_seconds": duration_seconds,
        "delta_velocity": delta_velocity,
        "average_acceleration": average_acceleration,
        "average_speed": average_speed,
        "toy_frame_action": str(span["action"]),
    }


def classify_chunk(chunk: Chunk) -> str:
    """教学版区间分类：规则读 chunk 的统计量，而不是某一帧。

    本课的输入没有沿 heading 投影的有符号速度，所以暂不判断 Reverse；第 14 课再补齐。
    """
    if chunk["average_speed"] <= 0.12:
        return "Stop"
    if chunk["average_acceleration"] >= 1.4 and chunk["delta_velocity"] >= 0.8:
        return "StrongAcceleration"
    if (
        0.15 <= chunk["average_acceleration"] < 1.4
        and chunk["delta_velocity"] >= 0.8
    ):
        return "GentleAcceleration"
    if chunk["average_acceleration"] <= -1.4 and chunk["delta_velocity"] <= -0.8:
        return "StrongDeceleration"
    if chunk["average_acceleration"] <= -0.15 and chunk["delta_velocity"] <= -0.8:
        return "GentleDeceleration"
    return "MaintainSpeed"


def main() -> None:
    rows = json.loads((BASE / "action_rows.json").read_text(encoding="utf-8"))
    spans = make_spans(rows)
    chunks = [summarize_chunk(rows, span) for span in spans]
    # end 是右开边界；事件锚点用 start。平稳驾驶保留在 spans，但不作为本例候选。
    events = []
    for span in spans:
        if span["action"] in INTERESTING_ACTIONS and span["duration"] >= 2:
            events.append(span)
    (BASE / "spans.json").write_text(json.dumps(spans, indent=2), encoding="utf-8")
    (BASE / "chunks.json").write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    (BASE / "events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
    print("spans:")
    print(json.dumps(spans, indent=2))
    print("\nchunk summaries (the input shape expected by classify_chunk):")
    for chunk in chunks:
        print(
            "  frames [{start}, {end}): dv={dv:+.2f} m/s, "
            "avg_a={acc:+.2f} m/s², avg_speed={speed:.2f} m/s -> {action}".format(
                start=chunk["start"],
                end=chunk["end"],
                dv=chunk["delta_velocity"],
                acc=chunk["average_acceleration"],
                speed=chunk["average_speed"],
                action=classify_chunk(chunk),
            )
        )
    print("\nselected events:")
    print(json.dumps(events, indent=2))
    print("\n真实 classify_chunk(chunk) 接收的就是一段的统计行（生产中是 pandas.Series）。")
    print("真实顺序是先 segment_longitudinal() 生成 chunk，再由 classify_longitudinal()")
    print("逐行调用 classify_chunk()；本 toy 为了先看懂规则，先逐帧分类再合并。")
    print("生产代码还会按动作类型/时长/起始帧筛选 keyframe，再平衡采样。")
    print("对应 keyframe_auto_select.py:parse_txt_file / write_action_json。")
    print("产物：spans.json、chunks.json 和 events.json；第 15 课会读取并使用 events.json。")


if __name__ == "__main__":
    main()
