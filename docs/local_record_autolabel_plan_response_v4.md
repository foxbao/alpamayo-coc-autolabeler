# 本地记录 CoC 适配方案：第四轮评审回复

> 评审意见：[`local_record_autolabel_plan_review_v4.md`](local_record_autolabel_plan_review_v4.md)
> 当前方案：[`local_record_autolabel_plan.md`](local_record_autolabel_plan.md)

> 历史说明：本文记录的是第四轮评审时的图片输入阶段。后续已接入 localization、位置差分和姿态投影；本文中“当前 MVP 不读取 localization”等表述只描述当时范围。现状以更新后的方案和 [`local_record_coordinate_conversion.md`](local_record_coordinate_conversion.md) 为准。

这轮先收窄交付目标：第一步只验证“一条记录、一个人工指定时刻、相机图片输入、一次现有 CoC 调用”。当前 MVP 不读取 localization，因此第四轮提出的坐标门禁和跨传感器时钟契约不属于本轮实现范围。

## 坐标门禁

评审关于 `velocity bearing - position heading`、`position heading - yaw` 和 `velocity bearing - yaw` 应分别分析的意见保留。它会成为后续从 localization 生成运动状态或动作候选之前的必要验证。本轮完全不计算速度投影，也不根据 localization 产生动作，因此不会以 `velocity bearing - yaw` 单项结果放行任何标注。

## 相机时间

MVP 的事件时间只相对于该记录第一张 front-camera 图片的文件名时间定义，选帧也只在相机图片时间轴上完成。它不声称相机与 localization 已同步，也不估计或应用跨传感器 offset。这样可以先验证相机输入和 CoC 的单事件调用；等后续接入 localization 时，再按格式契约、共同锚点或明确配置解决跨源时间关系。

## 本轮验收范围

请按收窄后的方案评审：单事件能否正确取出 16 张 CoC 图像、生成预览、在人工确认后调用现有 agent，并输出可检查的 YAML 和 overlay。第四轮中关于坐标窗口参数、支持数、PAI 序列接口回归和 localization/camera 时钟契约的细节，可留到 localization 阶段设计时再评审。
