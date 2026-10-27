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

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from .recording_autolabel.actions import write_longitudinal_action_audit
from .recording_autolabel.agent import (
    _run_agent,
    build_agent_data,
)
from .recording_autolabel.artifacts import (
    _write_agent_input_preview,
    _write_frame_manifest,
)
from .recording_autolabel.camera import (
    ImgLoader,
    LoadedImages,
    _parse_event_time,
)
from .recording_autolabel.motion_loader import LocalMotionData, LocalMotionLoader
from .recording_autolabel.position_artifacts import (
    write_position_motion_audit,
    write_resampled_position_trajectory,
)
from .recording_autolabel.types import TimedImage, TimedPosition
from .recording_autolabel.visualization import (
    write_contact_sheet_with_coc,
    write_previews,
)

__all__ = [
    "main",
]

DEFAULT_INPUT_ROOT = Path("/home/baojiali/Downloads/disk3/20261019_lin_007_out")
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "outputs" / "local_record_autolabel_mvp"
DEFAULT_API_CONFIG = Path.home() / ".config/alpamayo-coc/openai_api.yaml"
DEFAULT_MODEL = "gpt-6.1-sol"


@dataclass(frozen=True)
class _PreparedLocalRecord:
    """准备阶段留在内存中的结果，供后续推理和可视化继续使用。

    ``images`` 和 ``motion`` 分别归拢相机帧数据与定位运动数据；这里不保存输出文件路径清单。
    """

    images: LoadedImages
    motion: LocalMotionData
    agent_data: dict[str, Any]


@dataclass(frozen=True)
class _PreparationArtifacts:
    """准备阶段写入磁盘的文件路径和统计，不保存文件内容本身。

    与 ``_PreparedLocalRecord`` 分开，是为了区分后续步骤要消费的内存数据和已生成的文件。
    """

    contact_sheet: Path
    preview_video: Path
    position_audit: Path
    position_plot: Path
    position_trajectory: Path
    longitudinal_actions: Path
    agent_input_preview: Path
    manifest: Path
    position_motion_sample_count: int
    body_velocity_valid_sample_count: int


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


def _resolve_run_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """验证记录参数并解析相机、定位和本次运行的目录。"""
    if args.timeout_sec <= 0:
        raise SystemExit("--timeout-sec must be positive")
    if Path(args.record).name != args.record or args.record in {".", ".."}:
        raise SystemExit("--record must be a single record directory name")

    record_dir = args.input_root / args.record
    if not record_dir.is_dir():
        raise FileNotFoundError(f"Record directory not found: {record_dir}")

    camera_dir = record_dir / "camera" / "front"
    localization_dir = record_dir / "localization"
    run_dir = args.output_dir / args.record / f"event_{args.event_time_sec!s}s"
    return camera_dir, localization_dir, run_dir


def _prepare_local_record(
    args: argparse.Namespace,
    camera_dir: Path,
    localization_dir: Path,
    history_sec: Decimal,
    future_sec: Decimal,
    interval_sec: Decimal,
    *,
    camera_images: Sequence[TimedImage] | None = None,
    localization_positions: list[TimedPosition] | None = None,
) -> _PreparedLocalRecord:
    """读取图片和定位，并构造运动样本及 Agent 输入。"""
    image_loader = ImgLoader(history_sec, future_sec, interval_sec)
    images = (
        image_loader.load_from_images(camera_images, args.event_time_sec)
        if camera_images is not None
        else image_loader.load(camera_dir, args.event_time_sec)
    )
    motion_loader = LocalMotionLoader(history_sec, future_sec, interval_sec)
    motion = (
        motion_loader.load_from_positions(localization_positions, images.event_timestamp)
        if localization_positions is not None
        else motion_loader.load(localization_dir, images.event_timestamp)
    )
    agent_data = build_agent_data(
        args.record,
        args.event_time_sec,
        images.selected_frames,
        images.decoded_frames,
        position_motion=motion.coc_motion_samples,
        longitudinal_actions=motion.longitudinal_actions,
    )

    return _PreparedLocalRecord(
        images=images,
        motion=motion,
        agent_data=agent_data,
    )


def _write_preparation_artifacts(
    args: argparse.Namespace,
    run_dir: Path,
    prepared: _PreparedLocalRecord,
) -> _PreparationArtifacts:
    """写预览、运动审计、Agent 输入快照和准备阶段 manifest。"""
    run_dir.mkdir(parents=True, exist_ok=True)
    contact_sheet, preview_video = write_previews(
        prepared.images.selected_frames,
        prepared.images.decoded_frames,
        run_dir,
    )
    position_audit_path, position_plot_path = write_position_motion_audit(
        prepared.motion.coc_motion_samples,
        run_dir,
        args.record,
        prepared.images.event_timestamp,
        trajectory_samples=prepared.motion.resampled_positions,
    )
    position_trajectory_path = write_resampled_position_trajectory(
        prepared.motion.resampled_positions,
        prepared.motion.localization_positions,
        run_dir,
        args.record,
        prepared.images.event_timestamp,
    )
    longitudinal_action_path = write_longitudinal_action_audit(
        prepared.motion.longitudinal_actions,
        run_dir,
        args.record,
        prepared.images.event_timestamp,
    )

    agent_input_preview_path = _write_agent_input_preview(
        prepared.agent_data,
        run_dir / "agent_input_preview.json",
    )

    motion_sample_count = len(prepared.motion.coc_motion_samples)
    valid_motion_count = sum(
        sample.position_valid and sample.motion_valid
        for sample in prepared.motion.coc_motion_samples
    )
    valid_body_velocity_count = sum(
        sample.body_velocity_forward is not None and sample.body_velocity_right is not None
        for sample in prepared.motion.coc_motion_samples
    )
    manifest_path = run_dir / "frame_manifest.json"
    _write_frame_manifest(
        manifest_path,
        args.record,
        args.event_time_sec,
        prepared.images.selected_frames,
        extra_fields={
            "position_motion_audit": str(position_audit_path),
            "position_trajectory_plot": str(position_plot_path),
            "position_trajectory_10hz": str(position_trajectory_path),
            "agent_input_preview": str(agent_input_preview_path),
            "longitudinal_meta_action": str(longitudinal_action_path),
            "longitudinal_meta_action_span_count": len(prepared.motion.longitudinal_actions.spans),
            "longitudinal_meta_action_valid_tick_count": sum(
                point.valid for point in prepared.motion.longitudinal_actions.points
            ),
            "position_motion_valid_sample_count": valid_motion_count,
            "position_motion_invalid_sample_count": (motion_sample_count - valid_motion_count),
            "body_velocity_valid_sample_count": valid_body_velocity_count,
            "body_velocity_invalid_sample_count": (motion_sample_count - valid_body_velocity_count),
        },
    )

    return _PreparationArtifacts(
        contact_sheet=contact_sheet,
        preview_video=preview_video,
        position_audit=position_audit_path,
        position_plot=position_plot_path,
        position_trajectory=position_trajectory_path,
        longitudinal_actions=longitudinal_action_path,
        agent_input_preview=agent_input_preview_path,
        manifest=manifest_path,
        position_motion_sample_count=motion_sample_count,
        body_velocity_valid_sample_count=valid_body_velocity_count,
    )


def _print_preparation_summary(
    artifacts: _PreparationArtifacts,
) -> None:
    """打印准备阶段的文件位置和速度样本统计。"""
    print(f"Contact sheet: {artifacts.contact_sheet}")
    print(f"Preview video: {artifacts.preview_video}")
    print(f"Frame list: {artifacts.manifest}")
    print(f"Position motion audit: {artifacts.position_audit}")
    print(
        "Body-frame velocity valid samples: "
        f"{artifacts.body_velocity_valid_sample_count}/"
        f"{artifacts.position_motion_sample_count}"
    )
    print(f"Position trajectory: {artifacts.position_plot}")
    print(f"10 Hz position trajectory: {artifacts.position_trajectory}")
    print(f"Longitudinal meta-actions: {artifacts.longitudinal_actions}")
    print(f"Agent input preview: {artifacts.agent_input_preview}")


def _update_manifest(manifest_path: Path, **updates: Any) -> None:
    """更新 manifest 中的运行状态或推理产物路径。"""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.update(updates)
    manifest_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _run_inference_and_visualize(
    args: argparse.Namespace,
    prepared: _PreparedLocalRecord,
    artifacts: _PreparationArtifacts,
    history_sec: Decimal,
    future_sec: Decimal,
    data_loader_config: Any,
) -> None:
    """调用一次 CoC Agent，并生成带 CoC 的视频和联系表。"""
    labels_dir = artifacts.manifest.parent / "labels"
    try:
        yaml_path = _run_agent(
            DEFAULT_MODEL,
            args.timeout_sec,
            args.api_config.expanduser(),
            data_loader_config,
            prepared.agent_data,
            labels_dir,
        )
    except Exception as exc:
        _update_manifest(
            artifacts.manifest,
            inference_status="failed",
            error_type=type(exc).__name__,
        )
        raise

    _update_manifest(
        artifacts.manifest,
        inference_status="label_written_pending_visualization",
        label_yaml=str(yaml_path),
    )
    overlay_path = artifacts.manifest.parent / "preview_with_coc.mp4"
    contact_sheet_with_coc_path = artifacts.manifest.parent / "contact_sheet_with_coc.jpg"
    overlay_script = Path(__file__).resolve().parents[2] / "scripts" / "overlay_coc_result.py"
    try:
        subprocess.run(
            [
                sys.executable,
                str(overlay_script),
                "--video",
                str(artifacts.preview_video),
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
        write_contact_sheet_with_coc(
            artifacts.contact_sheet,
            yaml_path,
            contact_sheet_with_coc_path,
        )
    except Exception as exc:
        _update_manifest(
            artifacts.manifest,
            inference_status="visualization_failed",
            error_type=type(exc).__name__,
        )
        raise

    _update_manifest(
        artifacts.manifest,
        inference_status="completed_pending_review",
        label_yaml=str(yaml_path),
        overlay_video=str(overlay_path),
        contact_sheet_with_coc=str(contact_sheet_with_coc_path),
    )
    print(f"CoC YAML: {yaml_path}")
    print(f"Overlay video: {overlay_path}")
    print(f"CoC contact sheet: {contact_sheet_with_coc_path}")


def main() -> None:
    """串联本地数据准备；只有传入 ``--run-inference`` 才调用模型。"""
    args = _make_parser().parse_args()
    camera_dir, localization_dir, run_dir = _resolve_run_paths(args)
    history_sec, future_sec, interval_sec, data_loader_config = _load_temporal_config()
    prepared = _prepare_local_record(
        args,
        camera_dir,
        localization_dir,
        history_sec,
        future_sec,
        interval_sec,
    )
    # prepared 提供后续推理需要的数据；artifacts 记录准备阶段写出的文件，供查看和继续可视化。
    artifacts = _write_preparation_artifacts(args, run_dir, prepared)
    _print_preparation_summary(artifacts)

    if not args.run_inference:
        print("Prepared only; no model or API credentials were loaded.")
        return

    _run_inference_and_visualize(
        args,
        prepared,
        artifacts,
        history_sec,
        future_sec,
        data_loader_config,
    )


if __name__ == "__main__":
    main()
