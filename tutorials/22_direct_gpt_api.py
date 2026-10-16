# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""第 22 课：从本地配置文件读取凭证，把同一个真实事件交给 GPT API。

本课只改变模型后端。事件选择、时间对齐、loader、messages、OutputSchema
和 YAML 保存仍然沿用第 20 课；OpenAIWrapper 会把 numpy 视频帧转换成
base64 JPEG image_url。

配置文件示例见 ``tutorials/openai_api.example.yaml``。不要把真实 key 写入仓库。
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

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


def load_api_config(path: Path) -> dict[str, str]:
    """Read API credentials without printing or putting them on the command line."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("API config must be a YAML mapping")

    api_key = payload.get("api_key", payload.get("OPENAI_API_KEY"))
    base_url = payload.get("base_url", payload.get("OPENAI_BASE_URL"))
    model_name = str(payload.get("model", "gpt5.5"))
    model_name = {
        "gpt-5": "gpt5",
        "gpt-5.5": "gpt5.5",
        # The compatible endpoint exposes the configured gpt6.1 alias as this model ID.
        "gpt6.1": "gpt-6.1-sol",
    }.get(model_name, model_name)
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("API config must contain a non-empty api_key")
    if base_url is not None and not isinstance(base_url, str):
        raise ValueError("base_url must be a string when provided")
    supported_models = {"gpt5", "gpt5.5", "gpt-6.1-sol"}
    if model_name not in supported_models:
        raise ValueError(f"This tutorial supports models: {', '.join(sorted(supported_models))}")
    config = {"api_key": api_key, "base_url": base_url or "", "model": model_name}
    for field in ("organization", "project"):
        value = payload.get(field)
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty string when provided")
            config[field] = value
    return config


def configure_openai(api_config: dict[str, str]) -> None:
    """Configure the repository's standard OpenAI backend in this process only."""
    # Prevent the repository's NVIDIA credential precedence from selecting another backend.
    for variable in ("NVHOST_OAI_CLIENT_ID", "NVHOST_OAI_CLIENT_SECRET", "NVIDIA_API_KEY"):
        os.environ.pop(variable, None)
    os.environ["OPENAI_API_KEY"] = api_config["api_key"]
    if api_config["base_url"]:
        os.environ["OPENAI_BASE_URL"] = api_config["base_url"]
    else:
        os.environ.pop("OPENAI_BASE_URL", None)
    for field, environment_name in (
        ("organization", "OPENAI_ORG_ID"),
        ("project", "OPENAI_PROJECT"),
    ):
        if api_config.get(field):
            os.environ[environment_name] = api_config[field]
        else:
            os.environ.pop(environment_name, None)


def make_config(model_name: str) -> DictConfig:
    """Compose the same base config used by the repository's CoC pipeline."""
    config_dir = ROOT / "src/coc_labeling/config"
    overrides = [
        f"model_name={model_name}",
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path.home() / ".config/alpamayo-coc/openai_api.yaml",
        help="YAML file containing api_key, base_url (optional), and model.",
    )
    args = parser.parse_args()
    api_config = load_api_config(args.config.expanduser())
    configure_openai(api_config)

    selected = json.loads((BASE / "one_event.json").read_text(encoding="utf-8"))
    record = selected["record"]
    clip_id = str(record["clip_id"])
    event_frame = int(record["event_start_frame"])

    cfg = make_config(api_config["model"])
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

    print("GPT API 真实模型调用完成。")
    print("backend: standard OpenAI-compatible API")
    print("model:", api_config["model"])
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
