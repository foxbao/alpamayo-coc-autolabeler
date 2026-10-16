# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""第 20 课：把前课选出的真实事件交给生产 loaders 和本地 Qwen 标注。"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from coc_labeling.agents.vlm.coc_agent import VLMCoCAgent
from coc_labeling.data_loader.meta_action_loader import MetaActionLoader
from coc_labeling.data_loader.vector_loader import VectorLoader
from coc_labeling.data_loader.video_loader import VideoLoader
from coc_labeling.utils.data import get_clip_timing_info
from coc_labeling.utils.general_helpers import get_vlm_yaml_path


ROOT = Path(__file__).resolve().parents[1]
BASE = Path(__file__).parent / "_outputs/rebuild"
DATA = Path("/home/baojiali/Downloads/qwen-drive/datasets/pai_reasoning_16")
RUN = Path("/home/baojiali/Downloads/qwen-drive/outputs/pai_reasoning_16_official_20261013")


def make_config() -> DictConfig:
    """Compose the same base config used by the repository's CoC pipeline."""
    config_dir = ROOT / "src/coc_labeling/config"
    overrides = [
        "model_name=qwen3.5_35b",
        f"data.data_dir={DATA}",
        f"data.cache_dir={RUN / 'cache/trajdata'}",
        f"data.video_dir={RUN / 'video'}",
        f"data.meta_action_dir={RUN / 'meta_action/final_outputs'}",
        "data_loader.video.event_start_from_source_fps=true",
        "data_loader.video.save_segment_videos=false",
        "data_loader.vector.trajdata.num_workers=4",
    ]
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name="base_config_vlm_rel_ts", overrides=overrides)


def load_real_event(cfg: DictConfig, clip_id: str, event_frame: int) -> dict[str, Any]:
    """Mirror LabelingAgent.run(): load vector, source-video frames, then meta-actions."""
    vector_loader = VectorLoader(cfg.data_loader.vector, cfg.data)
    if vector_loader.dataset is None:
        raise RuntimeError("VectorLoader 没有初始化 trajdata dataset。")
    clip_start_micros, dt_micros = get_clip_timing_info(vector_loader.dataset, clip_id)
    if clip_start_micros is None or dt_micros is None:
        raise RuntimeError(f"trajdata 中找不到 clip timing：{clip_id}")
    event_timestamp = int(clip_start_micros + event_frame * dt_micros)

    video_loader = VideoLoader(cfg.data_loader.video, cfg.data)
    meta_action_loader = MetaActionLoader(cfg.data_loader.meta_action, cfg.data)
    data: dict[str, Any] = {
        "clip_id": clip_id,
        "event_start_frame": event_frame,
        "event_start_timestamp": event_timestamp,
    }

    data.update(vector_loader.load(clip_id, event_frame))
    source_frame, _ = video_loader.get_index_from_timestamp(clip_id, event_timestamp)
    video = video_loader.load(
        clip_id,
        source_frame,
        event_start_timestamp=event_timestamp,
    )
    data.update(
        {
            "hist_fpv_frames": video.hist_fpv_frames,
            "fut_fpv_frames": video.fut_fpv_frames,
            "all_fpv_frames": video.all_fpv_frames,
            "all_fpv_frames_info": video.all_fpv_frames_info,
            "hist_bev_frames": video.hist_bev_frames,
            "fut_bev_frames": video.fut_bev_frames,
            "all_bev_frames": video.all_bev_frames,
            "all_bev_frames_info": video.all_bev_frames_info,
        }
    )
    data.update(meta_action_loader.load(clip_id, event_frame))
    return data


def main() -> None:
    selected = json.loads((BASE / "one_event.json").read_text(encoding="utf-8"))
    record = selected["record"]
    clip_id = str(record["clip_id"])
    event_frame = int(record["event_start_frame"])

    # Set vLLM cache/runtime options before constructing the real agent.
    os.environ.setdefault("MODEL_CACHE_DIR", "/home/baojiali/.cache/alpamayo-coc-models")
    os.environ.setdefault("VLLM_MAX_MODEL_LEN", "32768")
    os.environ.setdefault("VLLM_GPU_MEMORY_UTILIZATION", "0.90")

    cfg = make_config()
    data = load_real_event(cfg, clip_id, event_frame)
    agent = VLMCoCAgent(
        model_name=cfg.model_name,
        agent_config=cfg.vlm_agent,
        vector_config=cfg.data_loader.vector,
        data_loader_config=cfg.data_loader,
    )

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    save_root = BASE / "real_result" / f"run_{run_stamp}"
    save_root.mkdir(parents=True, exist_ok=True)
    messages, result = agent.run(data, save_root=str(save_root))
    yaml_path = Path(get_vlm_yaml_path(str(save_root), data))

    print("真实模型调用完成。")
    print("event:", selected["bucket"], clip_id, "clipgt frame", event_frame)
    print("timestamp:", data["event_start_timestamp"], "us")
    print("消息数:", len(messages))
    print("final_content:", json.dumps(result["final_content"], ensure_ascii=False, indent=2))
    print("YAML:", yaml_path)
    saved_prompt = result["prompt"]
    handwrite_path = BASE / "messages.json"
    if handwrite_path.exists():
        handwrite_messages = json.loads(handwrite_path.read_text(encoding="utf-8"))
        print("与第 19 课消息对账：")
        print("  message count:", len(handwrite_messages), "->", len(saved_prompt))
        print(
            "  role/m_type skeleton same:",
            [(m["role"], m["m_type"]) for m in handwrite_messages]
            == [(m["role"], m["m_type"]) for m in saved_prompt],
        )
        print("  生产 process_results 已把 video content 替换成 frame metadata 占位信息。")
    print("\n生产 schema: {\"ego_behavior_schema\": {\"effect_on_ego_behavior\": \"因果句\"}}")
    print("OutputSchema 在解码时约束结构；process_results() 保存 prompt、事件信息和 YAML。")
    print("模型生成是候选标签，仍要对照视频审核。")


if __name__ == "__main__":
    main()
