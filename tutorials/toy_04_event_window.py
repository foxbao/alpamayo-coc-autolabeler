# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""第 04 章：围绕事件锚点取一个前后时间窗口。"""


# 先构造一条足够长的 toy 时间线：前面正常行驶，随后进入强减速。
speeds = [10.0] * 30 + [8.0, 6.0, 4.0, 2.0] + [0.0] * 70
actions = ["Maintain Speed"] * len(speeds)
for index in range(30, 34):
    actions[index] = "Strong Deceleration"
for index in range(34, len(actions)):
    actions[index] = "Stop"

# 事件锚点来自 meta-action 的起始帧，而不是随便挑一个数字。
event_start = None
for index in range(1, len(actions)):
    action_changed = actions[index] != actions[index - 1]
    is_target_action = actions[index] == "Strong Deceleration"
    if action_changed and is_target_action:
        event_start = index
        break

if event_start is None:
    raise ValueError("Could not find a Strong Deceleration event.")

# 真实默认配置：历史 2 秒、未来 6 秒、每 0.5 秒取一个点、原始数据 10 Hz。
fps = 10
history_seconds = 2
future_seconds = 6
sample_interval_seconds = 0.5
history_frames = int(history_seconds * fps)
future_frames = int(future_seconds * fps)
step_frames = int(sample_interval_seconds * fps)
ticks = range(
    event_start - history_frames,
    event_start + future_frames,
    step_frames,
)
samples = []
for tick in ticks:
    if tick < 0 or tick >= len(speeds):
        # 真实 loader 在数据边界处会截断；这里显式跳过不存在的帧。
        continue
    samples.append(
        {
            "relative_time": (tick - event_start) / fps,
            "frame": f"frame_{tick}",
            "action": actions[tick],
            "speed": speeds[tick],
        }
    )

print("event_start =", event_start, "(Strong Deceleration 的起始帧，对应 t=0)")
print("\n事件窗口中的样本:")
for sample in samples:
    print(sample)

print("\n同一行的 frame、action、speed 必须对应同一个 relative_time。")
print("真实窗口默认是 4 个历史点 + 12 个未来点，共 16 个采样点。")
print("小实验：把 sample_interval_seconds 改成 1.0，观察样本数量如何减少。")
