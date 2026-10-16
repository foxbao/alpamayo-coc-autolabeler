# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""第 15 课：用 toy 事件的动作类别，去真实 PAI keyframe 清单中挑一条事件。

toy event 的时间来自合成序列，不能直接拿去读取真实视频。本课只把它的动作类别
映射到真实数据的 bucket，再从那个 bucket 选择一个真实 clip/frame。
"""

import json
from pathlib import Path
from typing import Any


BASE = Path(__file__).parent / "_outputs/rebuild"
SOURCE = Path(
    "/home/baojiali/Downloads/qwen-drive/outputs/"
    "pai_reasoning_16_official_20261013/keyframes/segments_relative_timestamp_sampled.json"
)
PREFERRED_CLIP = "05ee29be-af7b-4fd6-853f-96a49f0d6bc7"
PREFERRED_FRAME = 114
ACTION_TO_BUCKET = {
    "GentleAcceleration": "gentle_acceleration",
    "StrongAcceleration": "strong_acceleration",
    "GentleDeceleration": "gentle_deceleration",
    "StrongDeceleration": "strong_deceleration",
    "Stop": "stop",
}


def main() -> None:
    events: list[dict[str, Any]] = json.loads(
        (BASE / "events.json").read_text(encoding="utf-8")
    )
    real: dict[str, list[dict[str, Any]]] = json.loads(SOURCE.read_text(encoding="utf-8"))

    toy_event = next(
        (event for event in events if event["action"] in ACTION_TO_BUCKET),
        None,
    )
    if toy_event is None:
        raise RuntimeError("toy 事件里没有可以映射到真实 meta-action 的动作。")

    bucket = ACTION_TO_BUCKET[toy_event["action"]]
    candidates = real.get(bucket, [])
    if not candidates:
        raise RuntimeError(f"真实 keyframe 清单没有 {bucket!r} bucket。")

    # 固定选择之前对照过的事件，便于重现教程和使用本机缓存；若清单变动就取第一条。
    record = next(
        (
            item
            for item in candidates
            if item.get("clip_id") == PREFERRED_CLIP
            and int(item.get("event_start_frame", -1)) == PREFERRED_FRAME
        ),
        candidates[0],
    )
    payload = {"toy_event": toy_event, "bucket": bucket, "record": record}
    output = BASE / "one_event.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("toy event:", json.dumps(toy_event, indent=2))
    print("mapped real bucket:", bucket)
    print("selected real record:", json.dumps(record, indent=2))
    print("saved:", output)
    print("\n这里选的是另一条真实数据；toy 的 frame 不能当成真实 clip 的 frame。")
    print("生产代码会在 keyframe_auto_select.py 生成 bucket，再由 ParsedSegmentGenerator 读取。")
    print(
        "\n真实清单同时包含 lateral bucket（如 steer_left/steer_right），"
        "但 lane bucket（如 left_lane_change/right_lane_change/keep_lane）在本批 PAI 数据中为空。"
    )
    print("这解释了为什么本教程的 lane 字段是 Not Available，而 lateral 也按生产设计不喂给 VLM。")


if __name__ == "__main__":
    main()
