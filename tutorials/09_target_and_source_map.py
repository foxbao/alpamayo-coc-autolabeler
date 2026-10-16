# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""第 09 课：先看最终标签，再给重构版搭一张源码地图。

运行后会写出一个小计划文件。后续课程从零实现一个可运行的简化流程，
并在每课末尾对照表里指出生产实现的位置。
"""

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).parent / "_outputs/rebuild"
REAL = Path("/home/baojiali/Downloads/qwen-drive/outputs/pai_reasoning_16_official_20261013")


def main() -> None:
    sample = REAL / "experiments/coc/05ee29be-af7b-4fd6-853f-96a49f0d6bc7/cot_11400000.yaml"
    print("最终产物:", sample)
    if sample.exists():
        saved = yaml.safe_load(sample.read_text(encoding="utf-8"))
        print("event_start_frame:", saved["event_start_frame"])
        print("event_start_timestamp:", saved["event_start_timestamp"])
        print("coc_prompt_version:", saved["coc_prompt_version"])
        print("final_content:", json.dumps(saved["final_content"], ensure_ascii=False))
        print("prompt messages:")
        for index, message in enumerate(saved["prompt"]):
            print(
                f"  [{index}] role={message['role']} m_type={message['m_type']} "
                f"chars={len(str(message['content']))}"
            )
    else:
        print("示例 YAML 不存在；后续第 20 课会生成同样结构的文件。")
    print("\ntoy -> production 函数对照：")
    mapping = {
        "速度 -> 加速度": (
            "src/meta_action/processor/longitudinal/segmentation.py:prepare_longitudinal_data"
        ),
        "meta-action CLI": "src/meta_action/data_labeling.py:main/process_clip",
        "meta-action 后处理": "src/meta_action/post_processing.py:process_batch",
        "轨迹 -> 动作区间": "src/meta_action/processor/longitudinal/segmentation.py:segment_longitudinal",
        "区间 -> 动作名": "src/meta_action/processor/longitudinal/classification.py:classify_chunk",
        "横向/车道动作": "src/meta_action/processor/lateral/ + src/meta_action/processor/lane/",
        "动作 -> keyframe": "src/coc_labeling/keyframe_auto_select.py:parse_txt_file",
        "keyframe -> 事件时间": (
            "src/coc_labeling/data_loader/segment_generators.py:ParsedSegmentGenerator"
        ),
        "事件 -> 证据": "src/coc_labeling/agents/labeling_agent.py:run",
        "事件对象": "src/coc_labeling/data_loader/segment_generators.py:SegmentInfo",
        "顶层协调器": "src/coc_labeling/agents/labeling_agent.py:__init__/parse_dataset/run",
        "VLM agent 工厂": "src/coc_labeling/agents/vlm/factory.py:VLMFactory",
        "时间戳 -> 视频帧": "src/coc_labeling/data_loader/video_loader.py:get_index_from_timestamp",
        "证据 -> 消息": "src/coc_labeling/agents/vlm/coc_agent.py:_build_input_message",
        "消息 -> 结构化输出": "src/coc_labeling/model_clients/vlm_wrappers/qwen.py:infer",
        "输出 schema": "src/coc_labeling/prompts/common.py:OutputSchema",
        "输出 -> YAML": "src/coc_labeling/agents/vlm/coc_agent.py:process_results",
    }
    for idea, location in mapping.items():
        print(f"  {idea:<18} -> {location}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plan.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print("\n第 10、12、15、18、19 课组成 JSON 数据主线。")
    print("第 11、13、14、16、17 课就近加深主线中的源码细节。")
    print("第 20 课手动展示单事件内部调用，第 21 课再换成真正的 LabelingAgent 入口。")


if __name__ == "__main__":
    main()
