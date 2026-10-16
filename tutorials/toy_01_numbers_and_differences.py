# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""第 01 章：只用列表和循环，从速度得到逐帧变化量。"""


# 先不要定义函数。我们直接写出一小段 10 Hz 速度序列，每个元素代表 0.1 秒。
speeds = [10.0, 10.0, 9.5, 9.0, 8.5, 8.0, 8.0]
print("原始速度 speeds:", speeds)

# 相邻两帧的速度差。严格说它是 Δv，不是带单位 m/s² 的物理加速度。
delta_speeds = []
for index, speed in enumerate(speeds):
    if index == 0:
        # 第一帧没有上一帧，所以先用 0.0 占位。
        delta_speed = 0.0
    else:
        delta_speed = speed - speeds[index - 1]
    delta_speeds.append(delta_speed)

print("逐帧速度变化 delta_speeds:", delta_speeds)
print("\n逐行观察：")
print("frame | speed | delta speed")
print("---------------------------")
for index, (speed, delta_speed) in enumerate(zip(speeds, delta_speeds)):
    print(f"{index:5} | {speed:5.1f} | {delta_speed:+11.1f}")

print("\n每个元素仍然只是一帧的数值，还没有 action。下一章再写规则。")
print("真实代码会计算 np.diff(speed) / dt；本 toy 省略除以 0.1 秒，只观察变化方向。")

# 小实验：把 speeds[2] 改成 7.0，先预测第 2、3 帧的变化量，再重新运行。
