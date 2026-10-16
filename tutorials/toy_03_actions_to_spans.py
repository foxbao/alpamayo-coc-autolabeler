# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""第 03 章：把逐帧 action 合并成连续区间。"""


actions = [
    "Maintain Speed",
    "Maintain Speed",
    "Strong Deceleration",
    "Strong Deceleration",
    "Strong Deceleration",
    "Stop",
    "Stop",
    "Acceleration",
]
print("逐帧 actions:", actions)

# 先用肉眼写出答案，理解区间的左闭右开表示法。
print("\n手工观察：")
print("Maintain Speed       [0, 2)")
print("Strong Deceleration   [2, 5)")
print("Stop                  [5, 7)")
print("Acceleration          [7, 8)")

# 再把刚才的观察写成循环。
spans = []
start = 0
current_action = actions[0]
for index in range(1, len(actions)):
    if actions[index] != current_action:
        spans.append(
            {
                "action": current_action,
                "start": start,
                "end": index,
                "duration": index - start,
            }
        )
        start = index
        current_action = actions[index]

spans.append(
    {
        "action": current_action,
        "start": start,
        "end": len(actions),
        "duration": len(actions) - start,
    }
)

print("\n循环生成的 spans:")
for span in spans:
    print(span)

# 只选择持续至少两帧且值得解释的 meta-action；Maintain Speed 暂时作为背景。
# toy_10 使用生产源码中的细分名称；这里也列出相同类别的 V2 写法。
interesting = {
    "Strong Deceleration",
    "Stop",
    "Acceleration",
    "GentleAcceleration",
    "StrongAcceleration",
    "GentleDeceleration",
    "StrongDeceleration",
}
selected = [
    span
    for span in spans
    if span["action"] in interesting and span["duration"] >= 2
]
print("\n筛选后的事件:", selected)

# 小实验：把 minimum duration 从 2 改成 3，观察哪些事件消失。
