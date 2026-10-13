# 从零理解 CoC autolabeler

这套教程重新从最小的 Python 变量开始构造一个 toy autolabeler。每个文件都可以
直接运行，不依赖仓库内部模块、视频、trajdata、GPU 或模型服务。

学习方式是逐步展开：先打印原始数据和中间变量，再解释为什么需要下一步，最后才把
重复代码封装成函数。这样可以先理解算法，再阅读生产实现。

## 课程顺序

```text
速度列表
  -> 加速度列表
  -> 每帧动作
  -> 连续动作区间
  -> 事件时间窗
  -> 多模态消息
  -> 假 VLM 输出
  -> schema 验证和 JSON 文件
```

| 文件 | 本章问题 |
| --- | --- |
| [01_numbers_and_differences.py](01_numbers_and_differences.py) | 如何从速度得到变化量？ |
| [02_rules_to_actions.py](02_rules_to_actions.py) | 如何用阈值把数值变成动作？ |
| [03_actions_to_spans.py](03_actions_to_spans.py) | 为什么要把逐帧动作合并成区间？ |
| [04_event_window.py](04_event_window.py) | 如何围绕事件取前后上下文？ |
| [05_multimodal_messages.py](05_multimodal_messages.py) | VLM 除了 prompt 还收到什么？ |
| [06_fake_vlm.py](06_fake_vlm.py) | 模型如何根据证据返回解释？ |
| [07_validate_and_save.py](07_validate_and_save.py) | 如何验收并保存模型结果？ |
| [08_full_pipeline.py](08_full_pipeline.py) | 如何把前面步骤重新组合？ |

## Toy 概念和生产代码

Toy 版本故意更短，但每一步在生产仓库中都有对应职责：

| Toy 概念 | 生产实现 |
| --- | --- |
| 速度和时间差 | `src/meta_action/processor/longitudinal/segmentation.py` |
| 规则动作 | `src/meta_action/processor/longitudinal/classification.py` |
| 连续动作区间 | `src/meta_action/post_processing.py` 和 meta-action spans |
| 候选事件筛选 | `src/coc_labeling/keyframe_auto_select.py` |
| 视频时间窗 | `src/coc_labeling/data_loader/video_loader.py` |
| 动作时间窗 | `src/coc_labeling/data_loader/meta_action_loader.py` |
| ego 状态 | `src/coc_labeling/data_loader/vector_loader.py` |
| 内部模型消息 | `VLMCoCAgent._build_input_message()` |
| 结构化输出约束 | `src/coc_labeling/prompts/common.py` + VLM wrapper |

生产实现还包含横向、车道级动作，以及不同的采样频率。Toy 章节先只引入少量纵向
动作，避免把工程细节和核心算法混在一起。

从仓库根目录运行：

```bash
python tutorials/01_numbers_and_differences.py
python tutorials/08_full_pipeline.py
```

每一章末尾都有一个小实验。先预测输出，再运行代码；修改输入后，观察哪一个中间
变量最先发生变化。这里的规则和数值是教学用的简化版本，不代表生产系统的阈值。
