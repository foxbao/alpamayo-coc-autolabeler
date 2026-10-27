# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""从已生成的纵向 meta-action 和相机图片中筛选 CoC 候选片段。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from omegaconf import OmegaConf

from .recording_autolabel.camera import (
    load_rgb_frames,
    load_timed_images,
    sample_offsets,
    select_frames,
)
from .recording_autolabel.meta_action_artifacts import load_full_record_meta_action
from .recording_autolabel.scenario import LOCAL_LONGITUDINAL_ACTION_TYPES
from .recording_autolabel.segment_candidates import (
    DEFAULT_ACTION_TARGET_COUNT,
    DEFAULT_MINIMUM_ACTION_DURATION_SEC,
    DEFAULT_MINIMUM_CANDIDATE_SEPARATION_SEC,
    DEFAULT_PERIODIC_CANDIDATE_INTERVAL_SEC,
    SegmentCandidate,
    balance_and_select_candidates,
    candidate_manifest_payload,
    select_segment_candidates,
    write_candidate_manifest,
)
from .recording_autolabel.types import LocalMotionChunk
from .recording_autolabel.visualization import write_contact_sheet

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "outputs" / "local_record_candidates"


@dataclass
class _PreparedCandidateRecord:
    """一条记录的候选计算结果，等待跨记录类别平衡后写出。"""

    record_id: str
    record_output: Path
    camera_dir: Path
    meta_action_path: Path
    localization_start: Decimal
    camera_start: Decimal
    camera_end: Decimal
    action_tick_count: int
    action_span_count: int
    candidates: list[SegmentCandidate]


def _parse_decimal(value: str) -> Decimal:
    """解析命令行中的正十进制秒数。"""
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError("value must be a decimal number of seconds") from exc
    if not number.is_finite() or number <= 0:
        raise argparse.ArgumentTypeError("value must be finite and greater than zero")
    return number


def _parse_positive_int(value: str) -> int:
    """解析至少为 1 的候选数量上限。"""
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return number


def _parser() -> argparse.ArgumentParser:
    """定义输入记录、已有 meta-action 和候选筛选参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root directory containing local recording folders.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--meta-action-root",
        type=Path,
        help="Directory containing full_record_meta_action.json; defaults to --output-dir.",
    )
    parser.add_argument(
        "--record",
        nargs="+",
        help=(
            "record directory names; omit to find generated meta-action files under "
            "--meta-action-root (or --output-dir)"
        ),
    )
    parser.add_argument(
        "--periodic-interval-sec",
        type=_parse_decimal,
        default=DEFAULT_PERIODIC_CANDIDATE_INTERVAL_SEC,
    )
    parser.add_argument(
        "--minimum-separation-sec",
        type=_parse_decimal,
        default=DEFAULT_MINIMUM_CANDIDATE_SEPARATION_SEC,
    )
    parser.add_argument(
        "--minimum-action-duration-sec",
        type=_parse_decimal,
        default=DEFAULT_MINIMUM_ACTION_DURATION_SEC,
        help="Minimum duration for action-span candidates; periodic coverage is unaffected.",
    )
    parser.add_argument(
        "--actions",
        nargs="+",
        choices=LOCAL_LONGITUDINAL_ACTION_TYPES,
        help="Only include these action types; omit to use all local longitudinal actions.",
    )
    parser.add_argument(
        "--target-count",
        type=_parse_positive_int,
        default=DEFAULT_ACTION_TARGET_COUNT,
        help=(
            "Maximum selected action-span candidates per action type across this run; "
            "longer spans are kept first. Coverage candidates are unaffected."
        ),
    )
    return parser


def _discover_records(meta_action_root: Path, requested: list[str] | None) -> list[str]:
    """按显式列表或已有的 meta-action 文件确定候选筛选对象。"""
    if requested:
        for record_id in requested:
            if Path(record_id).name != record_id or record_id in {".", ".."}:
                raise SystemExit(f"--record must be a directory name: {record_id}")
        if len(requested) != len(set(requested)):
            raise SystemExit("--record must not contain duplicate record names")
        return requested
    if not meta_action_root.is_dir():
        raise FileNotFoundError(f"Meta-action root not found: {meta_action_root}")
    return sorted(
        path.name
        for path in meta_action_root.iterdir()
        if path.is_dir() and (path / "full_record_meta_action.json").is_file()
    )


def _load_window_config() -> tuple[Decimal, Decimal, Decimal]:
    """复用 CoC 默认配置中的历史、未来时长和抽帧间隔。"""
    config_path = Path(__file__).parent / "config" / "data_loader" / "default.yaml"
    video = OmegaConf.load(config_path).video
    return (
        Decimal(str(video.hist_length_sec)),
        Decimal(str(video.fut_length_sec)),
        Decimal(str(video.time_interval)),
    )


def _contact_sheet_header(candidate: SegmentCandidate) -> str:
    """生成写在候选联系表顶部的时刻和动作说明。"""
    seed = candidate.seed
    action = seed.action_label or "no longitudinal action label"
    source = seed.source.replace("_", " ")
    merged_count = len(candidate.merged_candidates)
    merged_text = f" | merged nearby candidates: {merged_count}" if merged_count else ""
    return (
        f"{seed.candidate_id} | camera t={seed.camera_time_sec:+.1f}s | "
        f"longitudinal: {action} | source: {source}{merged_text}"
    )


def _prepare_clip_candidates(
    input_root: Path,
    meta_action_root: Path,
    output_root: Path,
    record_id: str,
    history_sec: Decimal,
    future_sec: Decimal,
    interval_sec: Decimal,
    periodic_interval_sec: Decimal,
    minimum_separation_sec: Decimal,
    minimum_action_duration_sec: Decimal = DEFAULT_MINIMUM_ACTION_DURATION_SEC,
    action_types: set[str] | None = None,
) -> _PreparedCandidateRecord:
    """读取一条记录并准备候选；联系表和清单稍后统一写出。"""
    record_dir = input_root / record_id
    camera_dir = record_dir / "camera" / "front"
    if not record_dir.is_dir():
        raise FileNotFoundError(f"Record directory not found: {record_dir}")

    meta_action_path = meta_action_root / record_id / "full_record_meta_action.json"
    meta_action = load_full_record_meta_action(meta_action_path)
    if meta_action.record_id != record_id:
        raise ValueError(
            f"Meta-action record_id {meta_action.record_id!r} does not match "
            f"requested record {record_id!r}: {meta_action_path}"
        )

    camera_images = load_timed_images(camera_dir)
    action_motions: list[LocalMotionChunk] = list(meta_action.action_motions)
    candidates = select_segment_candidates(
        action_motions,
        meta_action.localization_start,
        camera_images,
        history_sec,
        future_sec,
        interval_sec,
        periodic_interval_sec=periodic_interval_sec,
        minimum_separation_sec=minimum_separation_sec,
        minimum_action_duration_sec=minimum_action_duration_sec,
        action_types=action_types,
    )

    return _PreparedCandidateRecord(
        record_id=record_id,
        record_output=output_root / record_id,
        camera_dir=camera_dir,
        meta_action_path=meta_action_path,
        localization_start=meta_action.localization_start,
        camera_start=camera_images[0].timestamp,
        camera_end=camera_images[-1].timestamp,
        action_tick_count=meta_action.point_count,
        action_span_count=len(action_motions),
        candidates=candidates,
    )


def _write_candidate_outputs(
    prepared: _PreparedCandidateRecord,
    history_sec: Decimal,
    future_sec: Decimal,
    interval_sec: Decimal,
    periodic_interval_sec: Decimal,
    minimum_separation_sec: Decimal,
    minimum_action_duration_sec: Decimal,
    action_types: set[str] | None,
    action_target_count: int,
) -> Path:
    """为最终选中的候选写联系表和清单，并在重跑后清除旧候选图片。"""
    record_id = prepared.record_id
    record_output = prepared.record_output
    contact_sheet_dir = record_output / "contact_sheets"
    selected_candidates = [
        candidate for candidate in prepared.candidates if candidate.status == "selected"
    ]
    candidates_needing_frames = [
        candidate for candidate in selected_candidates if candidate.selected_frames is None
    ]
    if candidates_needing_frames:
        camera_images = load_timed_images(prepared.camera_dir)
        offsets = sample_offsets(history_sec, future_sec, interval_sec)
        for candidate in candidates_needing_frames:
            candidate.selected_frames = select_frames(
                camera_images,
                candidate.seed.camera_time_sec,
                offsets,
            )

    selected_ids = {
        candidate.seed.candidate_id
        for candidate in selected_candidates
        if candidate.selected_frames is not None
    }
    for candidate in selected_candidates:
        if candidate.status != "selected" or candidate.selected_frames is None:
            continue
        decoded_frames = load_rgb_frames(candidate.selected_frames)
        write_contact_sheet(
            candidate.selected_frames,
            decoded_frames,
            contact_sheet_dir / f"{candidate.seed.candidate_id}.jpg",
            header_text=_contact_sheet_header(candidate),
        )

    manifest = candidate_manifest_payload(
        record_id,
        prepared.candidates,
        prepared.camera_start,
        prepared.camera_end,
        prepared.localization_start,
        history_sec,
        future_sec,
        interval_sec,
        periodic_interval_sec,
        minimum_separation_sec,
        minimum_action_duration_sec=minimum_action_duration_sec,
        action_types=action_types,
        action_target_count=action_target_count,
    )
    manifest_path = write_candidate_manifest(
        manifest,
        record_output / "segment_candidates.json",
    )
    if contact_sheet_dir.is_dir():
        for old_contact_sheet in contact_sheet_dir.glob("candidate_*.jpg"):
            if old_contact_sheet.stem not in selected_ids:
                old_contact_sheet.unlink()
    selected_action_count = sum(
        candidate.status == "selected" and candidate.seed.source == "action_span_start"
        for candidate in prepared.candidates
    )
    selected_coverage_count = sum(
        candidate.status == "selected" and candidate.seed.source == "periodic_coverage"
        for candidate in prepared.candidates
    )
    print(
        f"{record_id}: {prepared.action_tick_count} action ticks, "
        f"{prepared.action_span_count} action spans, {selected_action_count} selected action candidates, "
        f"{selected_coverage_count} selected coverage samples"
    )
    print(f"  Meta-action: {prepared.meta_action_path}")
    print(f"  Candidate list: {manifest_path}")
    return manifest_path


def main() -> None:
    args = _parser().parse_args()
    meta_action_root = args.meta_action_root or args.output_dir
    records = _discover_records(meta_action_root, args.record)
    if not records:
        raise SystemExit(f"No generated meta-action files found in {meta_action_root}")
    history_sec, future_sec, interval_sec = _load_window_config()

    action_types = set(args.actions) if args.actions is not None else None
    prepared_records = [
        _prepare_clip_candidates(
            args.input_root,
            meta_action_root,
            args.output_dir,
            record_id,
            history_sec,
            future_sec,
            interval_sec,
            args.periodic_interval_sec,
            args.minimum_separation_sec,
            args.minimum_action_duration_sec,
            action_types,
        )
        for record_id in records
    ]
    balance_summary = balance_and_select_candidates(
        [prepared.candidates for prepared in prepared_records],
        args.target_count,
        args.minimum_separation_sec,
    )
    for prepared in prepared_records:
        _write_candidate_outputs(
            prepared,
            history_sec,
            future_sec,
            interval_sec,
            args.periodic_interval_sec,
            args.minimum_separation_sec,
            args.minimum_action_duration_sec,
            action_types,
            args.target_count,
        )
    if balance_summary:
        print("Action candidate selection across this run:")
        for action, (available, selected) in balance_summary.items():
            print(
                f"  {action}: available {available} -> selected {selected} "
                f"(cap {args.target_count}; separation applied)"
            )


if __name__ == "__main__":
    main()
