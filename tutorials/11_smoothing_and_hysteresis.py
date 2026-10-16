# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""第 11 课：平滑和滞回，避免边界附近的动作区间抖动。

生产 longitudinal segmentation 先对加速度做 centered rolling mean，再用带滞回的
阈值生成状态序列。这里用 toy 加速度展示为什么需要这两步。
生产代码还比较速度和多个加速度阈值，并根据任一状态变化来切段。
"""

from collections.abc import Sequence


def centered_rolling_mean(values: Sequence[float], window: int) -> list[float]:
    """用居中窗口求均值，边缘处只使用实际存在的样本。"""
    radius = max(0, int(window) // 2)
    smoothed = []
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        smoothed.append(sum(values[start:end]) / (end - start))
    return smoothed


def simple_threshold(values: Sequence[float], threshold: float) -> list[bool]:
    """每个点独立比较阈值；噪声跨线时状态也会来回跳。"""
    return [value >= threshold for value in values]


def hysteresis_threshold(
    values: Sequence[float], threshold: float, fraction: float
) -> list[bool]:
    """进入和退出使用不同门槛，避免数值在同一个门槛附近反复切换。"""
    if not values:
        return []

    if threshold >= 0:
        enter_threshold = threshold * (1 + fraction)
        exit_threshold = threshold * (1 - fraction)
    else:
        enter_threshold = threshold * (1 - fraction)
        exit_threshold = threshold * (1 + fraction)

    # 初始状态用 threshold 判断，之后使用 enter/exit 门槛，与生产实现一致。
    state = values[0] >= threshold
    states = [state]
    for value in values[1:]:
        if not state and value >= enter_threshold:
            state = True
        elif state and value < exit_threshold:
            state = False
        states.append(state)
    return states


def count_transitions(states: Sequence[bool]) -> int:
    """统计状态发生变化的次数。"""
    return sum(left != right for left, right in zip(states, states[1:]))


def cut_with_production_guard(states: Sequence[bool]) -> list[tuple[int, int]]:
    """演示生产分段器的切段冷却和尾部保护；返回 tick 边界对。"""
    if not states:
        return []

    segments = []
    start = 0
    current_state = states[0]
    index = 1
    while index < len(states):
        if states[index] != current_state:
            # 快到序列末尾时不再切一小段，让当前段延伸到结尾。
            if len(states) - index <= 3:
                break
            segments.append((start, index))
            start = index
            current_state = states[index]
            # 生产实现把下一次状态检查推迟到 index + 3。
            index += 3
            continue
        index += 1

    segments.append((start, len(states) - 1))
    return segments


def main() -> None:
    dt_seconds = 0.1
    threshold = 0.1
    hysteresis_fraction = 0.15
    raw_acceleration = [
        0.02, 0.12, 0.13, 0.09, 0.13, 0.10, 0.14,
        0.12, 0.10, 0.14, 0.13, 0.11, 0.07, 0.03,
    ]
    smoothed = centered_rolling_mean(raw_acceleration, window=5)
    raw_states = simple_threshold(raw_acceleration, threshold)
    smoothed_states = hysteresis_threshold(smoothed, threshold, hysteresis_fraction)

    print("每一行是相邻 0.1 秒的加速度样本；阈值为 +0.1 m/s²。")
    print(" idx | raw acc | smooth acc | raw >= threshold | smooth + hysteresis")
    for index, (raw, smooth, raw_state, final_state) in enumerate(
        zip(raw_acceleration, smoothed, raw_states, smoothed_states)
    ):
        print(
            f" {index:>3} | {raw:>7.3f} | {smooth:>10.3f} | "
            f"{str(raw_state):>16} | {final_state}"
        )

    print("\n直接阈值切换次数:", count_transitions(raw_states))
    print("平滑 + 滞回切换次数:", count_transitions(smoothed_states))
    noisy_states = [False, False, True, False, True, True, True, True, True, False, False]
    print("\n带短促抖动的状态序列:", noisy_states)
    print("应用生产切段保护后的边界 tick:", cut_with_production_guard(noisy_states))
    print(
        "\n滞回 15% 时，进入门槛是 0.115，退出门槛是 0.085 m/s²；"
        "状态打开后，不会因为轻微跌回 0.1 就马上关闭。"
    )
    print(
        "生产对应: longitudinal/segmentation.py 的 _smooth_series 和 _hysteresis_boolean。"
    )
    print(
        f"生产加速度样本间隔 DELTA_TIMESTAMP={dt_seconds:.1f}s；"
        "实际分段还会同时比较 agent_speed_along_heading 的阈值状态。"
    )
    print(
        "生产切段后把下一次检查推迟 3 个 tick；若剩余不超过 3 个 tick，"
        "就把尾部留在当前段。"
    )


if __name__ == "__main__":
    main()
