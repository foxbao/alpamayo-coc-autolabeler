# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""第 02 章：把速度和变化量通过有优先级的规则变成 action。"""


speeds = [10.0, 10.0, 9.5, 9.0, 8.5, 2.0, 0.0, 0.0, 2.0, 4.0]
delta_speeds = [0.0]
for index in range(1, len(speeds)):
    delta_speeds.append(speeds[index] - speeds[index - 1])


def classify(speed: float, delta_speed: float) -> str:
    """按顺序应用规则；顺序就是冲突时的优先级。"""
    if speed < 0.5:
        return "Stop"
    if delta_speed < -1.0:
        return "Strong Deceleration"
    if delta_speed > 1.0:
        return "Acceleration"
    return "Maintain Speed"


actions = []
for speed, delta_speed in zip(speeds, delta_speeds):
    actions.append(classify(speed, delta_speed))

print("frame | speed | delta | action")
print("--------------------------------")
for index, (speed, delta_speed, action) in enumerate(
    zip(speeds, delta_speeds, actions)
):
    print(f"{index:5} | {speed:5.1f} | {delta_speed:+5.1f} | {action}")

print("\n规则顺序实验：")
print("classify(speed=0.2, delta_speed=2.0) ->", classify(0.2, 2.0))
print("虽然速度变化量为正，但 Stop 规则先执行。")
print("真实动作还包括 Gentle/Strong、Reverse、横向和车道级类别；这里先只学纵向骨架。")
print("真实 meta-action 判定以 5 Hz 处理，toy 为了直观按每个输入元素判断。")

# 小实验：把 -1.0 改成 -0.3，观察 -0.5 的变化何时变成 Strong Deceleration。
