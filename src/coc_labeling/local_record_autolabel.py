# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""为本地记录提供 CoC 自动标注命令行入口。"""

import argparse
import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from .local_recording.agent import (
    _configure_openai_environment,
    _format_position_ground_speed,
    _run_agent,
    build_agent_data,
)
from .local_recording.artifacts import (
    _write_agent_input_preview,
    _write_frame_manifest,
)
from .local_recording.camera import (
    _event_timestamp,
    _load_rgb_frames,
    _parse_event_time,
    _parse_filename_timestamp,
    _sample_offsets,
    _timestamp_micros,
    load_timed_images,
    select_frames,
)
from .local_recording.localization import (
    _ensure_continuous_position_interval,
    _position_motion_offsets,
    _position_trajectory_offsets,
    build_position_motion_samples,
    derive_position_motion,
    load_localization_positions,
    resample_position_trajectory,
    rotate_map_velocity_to_body_flu,
    sample_orientation_at,
    sample_position_at,
    sample_recorded_linear_velocity_at,
)
from .local_recording.position_artifacts import (
    write_position_motion_audit,
    write_resampled_position_trajectory,
)
from .local_recording.types import (
    LOCAL_POSITION_EGO_STATE_PROMPT,
    LOCALIZATION_RESAMPLE_INTERVAL_SEC,
    MAX_FRAME_ERROR_SEC,
    MAX_LOCALIZATION_GAP_SEC,
    POSITION_DERIVATIVE_WINDOW_SEC,
    PREVIEW_COLUMNS,
    THUMBNAIL_WIDTH,
    PositionMotionSample,
    ResampledPosition,
    SelectedFrame,
    TimedImage,
    TimedPosition,
)
from .local_recording.visualization import (
    write_contact_sheet_with_coc,
    write_previews,
)

__all__ = [
    "LOCALIZATION_RESAMPLE_INTERVAL_SEC",
    "LOCAL_POSITION_EGO_STATE_PROMPT",
    "MAX_FRAME_ERROR_SEC",
    "MAX_LOCALIZATION_GAP_SEC",
    "POSITION_DERIVATIVE_WINDOW_SEC",
    "PREVIEW_COLUMNS",
    "THUMBNAIL_WIDTH",
    "PositionMotionSample",
    "ResampledPosition",
    "SelectedFrame",
    "TimedImage",
    "TimedPosition",
    "_configure_openai_environment",
    "_ensure_continuous_position_interval",
    "_event_timestamp",
    "_format_position_ground_speed",
    "_load_rgb_frames",
    "_load_temporal_config",
    "_make_parser",
    "_parse_event_time",
    "_parse_filename_timestamp",
    "_position_motion_offsets",
    "_position_trajectory_offsets",
    "_run_agent",
    "_sample_offsets",
    "_timestamp_micros",
    "_write_agent_input_preview",
    "_write_frame_manifest",
    "build_agent_data",
    "build_position_motion_samples",
    "derive_position_motion",
    "load_localization_positions",
    "load_timed_images",
    "main",
    "resample_position_trajectory",
    "rotate_map_velocity_to_body_flu",
    "sample_orientation_at",
    "sample_position_at",
    "sample_recorded_linear_velocity_at",
    "select_frames",
    "write_contact_sheet_with_coc",
    "write_position_motion_audit",
    "write_previews",
    "write_resampled_position_trajectory",
]

DEFAULT_INPUT_ROOT = Path("/home/baojiali/Downloads/disk3/20261019_lin_007_out")
DEFAULT_OUTPUT_ROOT = (
    Path(__file__).resolve().parents[2] / "outputs" / "local_record_autolabel_mvp"
)
DEFAULT_API_CONFIG = Path.home() / ".config/alpamayo-coc/openai_api.yaml"
DEFAULT_MODEL = "gpt-6.1-sol"


def _load_temporal_config() -> tuple[Decimal, Decimal, Decimal, Any]:
    """读取 CoC 提示词使用的默认时间窗口和采样间隔。

    当前默认值对应历史 2 秒、未来 6 秒、每 0.5 秒采样一次。
    """
    config_path = Path(__file__).parent / "config" / "data_loader" / "default.yaml"
    config = OmegaConf.load(config_path)
    # 复用项目标准 CoC 历史窗口、未来窗口和采样频率。
    video_config = config.video
    history = Decimal(str(video_config.hist_length_sec))
    future = Decimal(str(video_config.fut_length_sec))
    interval = Decimal(str(video_config.time_interval))
    return history, future, interval, config

def _make_parser() -> argparse.ArgumentParser:
    """定义命令行参数；记录名和事件相对秒数必须由调用者提供。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--record", required=True, help="Record directory name")
    parser.add_argument("--event-time-sec", required=True, type=_parse_event_time)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    # 推理是显式选择的：普通运行不联网，只生成预览。
    parser.add_argument("--run-inference", action="store_true", help="Send one CoC model request")
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    parser.add_argument("--timeout-sec", type=int, default=600)
    return parser

def main() -> None:
    """执行选帧和预览；只有传入 ``--run-inference`` 才调用模型。"""
    args = _make_parser().parse_args()
    if args.timeout_sec <= 0:
        raise SystemExit("--timeout-sec must be positive")
    if Path(args.record).name != args.record or args.record in {".", ".."}:
        raise SystemExit("--record must be a single record directory name")

    record_dir = args.input_root / args.record
    camera_dir = record_dir / "camera" / "front"
    localization_dir = record_dir / "localization"
    if not record_dir.is_dir():
        raise FileNotFoundError(f"Record directory not found: {record_dir}")

    run_dir = args.output_dir / args.record / f"event_{args.event_time_sec!s}s"

    # 第一阶段：读取配置的时间窗口，并为每个采样时刻匹配实际相机图片。
    history_sec, future_sec, interval_sec, data_loader_config = _load_temporal_config()
    offsets = _sample_offsets(history_sec, future_sec, interval_sec)
    images = load_timed_images(camera_dir)
    selected = select_frames(images, args.event_time_sec, offsets)
    decoded_frames = _load_rgb_frames(selected)
    localization_positions = load_localization_positions(localization_dir)
    event_timestamp = _event_timestamp(selected)
    position_trajectory_offsets = _position_trajectory_offsets(
        history_sec,
        future_sec,
        derivative_window_sec=POSITION_DERIVATIVE_WINDOW_SEC,
    )
    resampled_positions = resample_position_trajectory(
        localization_positions,
        event_timestamp,
        position_trajectory_offsets,
    )
    position_offsets = _position_motion_offsets(history_sec, future_sec, interval_sec)
    position_motion = build_position_motion_samples(
        localization_positions,
        event_timestamp,
        position_offsets,
        resampled_positions=resampled_positions,
    )
    data = build_agent_data(
        args.record,
        args.event_time_sec,
        selected,
        decoded_frames,
        position_motion=position_motion,
    )

    # 无论是否推理，都先生成供人工检查的预览和数据清单。
    run_dir.mkdir(parents=True, exist_ok=True)
    contact_sheet, preview_video = write_previews(selected, decoded_frames, run_dir)
    position_audit_path, position_plot_path = write_position_motion_audit(
        position_motion,
        run_dir,
        args.record,
        event_timestamp,
        trajectory_samples=resampled_positions,
    )
    position_trajectory_path = write_resampled_position_trajectory(
        resampled_positions,
        localization_positions,
        run_dir,
        args.record,
        event_timestamp,
    )
    manifest_path = run_dir / "frame_manifest.json"
    _write_frame_manifest(manifest_path, args.record, args.event_time_sec, selected)
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload["position_motion_audit"] = str(position_audit_path)
    manifest_payload["position_trajectory_plot"] = str(position_plot_path)
    manifest_payload["position_trajectory_10hz"] = str(position_trajectory_path)
    agent_input_preview_path = _write_agent_input_preview(
        data,
        run_dir / "agent_input_preview.json",
    )
    manifest_payload["agent_input_preview"] = str(agent_input_preview_path)
    manifest_payload["position_motion_valid_sample_count"] = sum(
        sample.position_valid and sample.motion_valid for sample in position_motion
    )
    manifest_payload["position_motion_invalid_sample_count"] = sum(
        not (sample.position_valid and sample.motion_valid) for sample in position_motion
    )
    manifest_payload["body_velocity_valid_sample_count"] = sum(
        sample.body_velocity_forward is not None
        and sample.body_velocity_right is not None
        for sample in position_motion
    )
    manifest_payload["body_velocity_invalid_sample_count"] = sum(
        sample.body_velocity_forward is None or sample.body_velocity_right is None
        for sample in position_motion
    )
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Contact sheet: {contact_sheet}")
    print(f"Preview video: {preview_video}")
    print(f"Frame list: {manifest_path}")
    print(f"Position motion audit: {position_audit_path}")
    print(
        "Body-frame velocity valid samples: "
        f"{manifest_payload['body_velocity_valid_sample_count']}/"
        f"{len(position_motion)}"
    )
    print(f"Position trajectory: {position_plot_path}")
    print(f"10 Hz position trajectory: {position_trajectory_path}")
    print(f"Agent input preview: {agent_input_preview_path}")

    if not args.run_inference:
        print("Prepared only; no model or API credentials were loaded.")
        return

    # 第二阶段只有在用户明确要求时才执行：调用一次 VLM，
    # 再把 CoC 结果绘制到同一段预览视频，并在清单中记录各阶段状态。
    labels_dir = run_dir / "labels"
    try:
        yaml_path = _run_agent(
            DEFAULT_MODEL,
            args.timeout_sec,
            args.api_config.expanduser(),
            data_loader_config,
            data,
            labels_dir,
        )
    except Exception as exc:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["inference_status"] = "failed"
        payload["error_type"] = type(exc).__name__
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        raise

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["inference_status"] = "label_written_pending_visualization"
    payload["label_yaml"] = str(yaml_path)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    overlay_path = run_dir / "preview_with_coc.mp4"
    contact_sheet_with_coc_path = run_dir / "contact_sheet_with_coc.jpg"
    overlay_script = Path(__file__).resolve().parents[2] / "scripts" / "overlay_coc_result.py"
    try:
        subprocess.run(
            [
                sys.executable,
                str(overlay_script),
                "--video",
                str(preview_video),
                "--result",
                str(yaml_path),
                "--output",
                str(overlay_path),
                "--duration-seconds",
                str(history_sec + future_sec),
                "--event-offset-seconds",
                str(history_sec),
            ],
            check=True,
        )
        write_contact_sheet_with_coc(contact_sheet, yaml_path, contact_sheet_with_coc_path)
    except Exception as exc:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["inference_status"] = "visualization_failed"
        payload["error_type"] = type(exc).__name__
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        raise
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["inference_status"] = "completed_pending_review"
    payload["label_yaml"] = str(yaml_path)
    payload["overlay_video"] = str(overlay_path)
    payload["contact_sheet_with_coc"] = str(contact_sheet_with_coc_path)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"CoC YAML: {yaml_path}")
    print(f"Overlay video: {overlay_path}")
    print(f"CoC contact sheet: {contact_sheet_with_coc_path}")


if __name__ == "__main__":
    main()
