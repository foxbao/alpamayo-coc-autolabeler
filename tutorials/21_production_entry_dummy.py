# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""第 21 课：用 dummy 验证生产 LabelingAgent 的单事件协调流程。

第 20 课为了看清内部顺序，手写了 VectorLoader、VideoLoader、MetaActionLoader 和
VLMCoCAgent。这里把这份协调工作交给生产 ``LabelingAgent``：它会创建
``ParsedSegmentGenerator``、三个 loader 和 ``VLMFactory``，然后遍历 segment list。

本教程直接创建 ``LabelingAgent``，没有运行 ``data_labeling.py`` 的完整 Hydra CLI。
dummy 会走数据加载、消息构造和结果落盘，但不会运行真实模型推理。
它忽略输入消息和 ``OutputSchema``，固定返回 ``Dummy response.``，
因此不能验证模型理解或输出 schema。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
import yaml

from coc_labeling.agents.labeling_agent import LabelingAgent


ROOT = Path(__file__).resolve().parents[1]
BASE = Path(__file__).parent / "_outputs/rebuild"
DATA = Path("/home/baojiali/Downloads/qwen-drive/datasets/pai_reasoning_16")
RUN = Path("/home/baojiali/Downloads/qwen-drive/outputs/pai_reasoning_16_official_20261013")


def make_config(segment_path: Path) -> DictConfig:
    """Compose the config family used by the labeling CLI for this one-event run."""
    config_dir = ROOT / "src/coc_labeling/config"
    overrides = [
        "model_name=dummy",
        f"data.data_dir={DATA}",
        f"data.cache_dir={RUN / 'cache/trajdata'}",
        f"data.video_dir={RUN / 'video'}",
        f"data.meta_action_dir={RUN / 'meta_action/final_outputs'}",
        f"data.segment_config_path={segment_path}",
        "data_loader.keyframe.segment_generator_type=json",
        "data_loader.video.event_start_from_source_fps=true",
        "data_loader.video.save_segment_videos=false",
        "data_loader.vector.trajdata.num_workers=4",
    ]
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name="base_config_vlm_rel_ts", overrides=overrides)


def make_single_segment_json() -> Path:
    """把第 15 课选中的记录改成 ParsedSegmentGenerator 需要的输入格式。"""
    selected: dict[str, Any] = json.loads(
        (BASE / "one_event.json").read_text(encoding="utf-8")
    )
    record = dict(selected["record"])
    path = BASE / "production_single_segment.json"
    path.write_text(
        json.dumps({selected["bucket"]: [record]}, indent=2),
        encoding="utf-8",
    )
    return path


def main() -> None:
    segment_path = make_single_segment_json()
    cfg = make_config(segment_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    save_root = BASE / "production_dummy" / f"run_{stamp}"

    # 直接创建生产协调器处理一个事件；本教程不运行完整 Hydra CLI。
    agent = LabelingAgent(
        cfg=cfg,
        save_root=str(save_root),
        verbose=cfg.verbose,
        mode="vlm",
    )
    agent.parse_dataset(cfg, str(save_root))
    agent.run()

    selected: dict[str, Any] = json.loads(
        (BASE / "one_event.json").read_text(encoding="utf-8")
    )
    record = selected["record"]
    output_paths = sorted(save_root.rglob("cot_*.yaml"))
    if len(output_paths) == 1:
        yaml_path = output_paths[0]
        saved = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        message_lengths = [len(str(item["content"])) for item in saved["prompt"]]
        print("输入事件:", record["clip_id"], "clipgt frame", record["event_start_frame"])
        print("实际输出 YAML:", yaml_path)
        print("读回生产 YAML 的 prompt 字符数:", message_lengths)
        print("这应与第 09 课看到的生产消息长度一致。")
    else:
        print(f"预期为这个单事件找到 1 个 YAML，实际找到 {len(output_paths)} 个:")
        for output_path in output_paths:
            print(" ", output_path)

    print(
        "生产协调器运行完成：LabelingAgent -> ParsedSegmentGenerator -> "
        "loaders -> VLMFactory"
    )
    print(
        "model_name=dummy；DummyWrapper 忽略 messages 和 OutputSchema，"
        "只返回占位文本。"
    )
    print(
        "本课验证单事件协调、时间转换、消息构造和 YAML 落盘，"
        "不验证模型推理或 schema。"
    )
    print("输出目录:", save_root)
    print(
        "\n生产批处理会遍历全部 SegmentInfo；特定视频加载错误和 "
        "agent.run 异常会记录后跳过。"
    )
    print("向量或 meta-action loader 等其它阶段的异常不一定会在这里被捕获。")
    print("segment_generator_type 也可以切换 parquet；本教程为了可读性使用 json。")
    print(
        "若要真实推理，请回到第 20 课使用 qwen3.5_35b；"
        "不要把 dummy response 当作标签。"
    )


if __name__ == "__main__":
    main()
