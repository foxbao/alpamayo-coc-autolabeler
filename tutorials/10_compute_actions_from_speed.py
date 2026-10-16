# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""第 10 课：从逐帧速度算加速度，再按简单规则标注动作。

本课使用 10 Hz toy 数据，先用逐帧规则建立直觉；生产在 10 Hz 数据上按 STEP=2、即 5 Hz 评定动作。
真实流程先把连续轨迹切成 chunk（区间），再根据每个 chunk 的平均加速度、总速度变化量、
平均速度分类。第 12 课会把单帧样本汇总成 chunk，展示真实 ``classify_chunk(chunk)`` 的输入形状。
"""

import json
from pathlib import Path


OUT = Path(__file__).parent / "_outputs/rebuild"
DT = 0.1  # 生产 clipgt 时间步长：10 Hz


def classify(speed: float, acceleration: float) -> str:
    """教学用的逐帧分类规则；生产分类看整个动作区间的统计量。"""
    if speed < 0.12:
        return "Stop"
    if acceleration >= 1.4:
        return "StrongAcceleration"
    if acceleration >= 0.15:
        return "GentleAcceleration"
    if acceleration <= -1.4:
        return "StrongDeceleration"
    if acceleration <= -0.15:
        return "GentleDeceleration"
    return "MaintainSpeed"


def main() -> None:
    # 先以 +0.05 m/s/帧缓慢加速，使整个 chunk 的 Δv 超过生产门槛 0.8 m/s。
    # 然后以 -0.2 m/s/帧强减速，最后停住。
    speeds = [5.0 + 0.05 * index for index in range(21)]
    speed = speeds[-1]
    for _ in range(30):
        speed = max(0.0, speed - 0.2)
        speeds.append(speed)
    rows = []
    for index, speed in enumerate(speeds):
        delta_velocity = 0.0 if index == 0 else speed - speeds[index - 1]
        acceleration = delta_velocity / DT
        rows.append(
            {
                "frame": index,
                "speed": speed,
                "delta_velocity": delta_velocity,
                "acceleration": acceleration,
                "action": classify(speed, acceleration),
            }
        )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "action_rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\n关键边界帧:")
    for index in (0, 1, 20, 21, 49, 50):
        print(rows[index])
    action_counts = {}
    for row in rows:
        action = str(row["action"])
        action_counts[action] = action_counts.get(action, 0) + 1
    print("\n逐帧动作计数:", action_counts)
    print("\n请观察：逐帧 Δv=0.05 m/s 看起来很小，但除以 0.1 秒后是 +0.5 m/s²。")
    print("生产流程会平滑加速度、使用滞回分段，再检查整个区间的平均加速度和 Δv。")
    print("生产动作还包括 Reverse、横向动作和车道级动作；这里先只做纵向分类。")
    print("产物：tutorials/_outputs/rebuild/action_rows.json，第 12 课会读取它。")


if __name__ == "__main__":
    main()
