# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""从候选清单准备本地 CoC 输入，并可选择调用 GPT 推理。"""

from __future__ import annotations

import argparse
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .local_record_autolabel import (
    DEFAULT_API_CONFIG,
    DEFAULT_INPUT_ROOT,
    _load_temporal_config,
    _prepare_local_record,
    _print_preparation_summary,
    _run_inference_and_visualize,
    _update_manifest,
    _write_preparation_artifacts,
)
from .recording_autolabel.camera import load_timed_images
from .recording_autolabel.localization import load_localization_positions
from .recording_autolabel.types import TimedImage, TimedPosition

DEFAULT_CANDIDATE_ROOT = Path(__file__).resolve().parents[2] / "outputs" / "local_record_candidates"
DEFAULT_LABEL_OUTPUT_ROOT = (
    Path(__file__).resolve().parents[2] / "outputs" / "local_record_autolabel_candidates"
)
CANDIDATE_TYPES = ("all", "action_span", "coverage")


def _parser() -> argparse.ArgumentParser:
    """定义候选输入、输出目录和可选推理参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_LABEL_OUTPUT_ROOT)
    parser.add_argument("--record", nargs="+", help="只处理指定的记录目录名")
    parser.add_argument("--candidate-id", nargs="+", help="只处理指定候选 ID")
    parser.add_argument(
        "--candidate-type",
        choices=CANDIDATE_TYPES,
        default="all",
        help="只处理动作区间候选、定时覆盖候选，或全部候选（默认：all）",
    )
    parser.add_argument(
        "--run-inference",
        action="store_true",
        help="对每个选中的候选调用 GPT 并写出 CoC 标签",
    )
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    parser.add_argument("--timeout-sec", type=int, default=600)
    return parser


def _record_directories(candidate_root: Path, requested: list[str] | None) -> list[Path]:
    """按显式记录名或候选清单发现记录。"""
    if requested:
        paths = []
        for record_id in requested:
            if Path(record_id).name != record_id or record_id in {".", ".."}:
                raise SystemExit(f"--record must be a directory name: {record_id}")
            paths.append(candidate_root / record_id)
        return paths
    if not candidate_root.is_dir():
        raise FileNotFoundError(f"Candidate root not found: {candidate_root}")
    return sorted(
        path
        for path in candidate_root.iterdir()
        if path.is_dir() and (path / "segment_candidates.json").is_file()
    )


def _load_selected_candidates(
    manifest_path: Path,
    requested_ids: set[str] | None,
    candidate_type: str = "all",
) -> tuple[str, list[tuple[str, Decimal]]]:
    """读取选中候选的 ID 和相机相对时刻，可按候选池筛选。"""
    if candidate_type not in CANDIDATE_TYPES:
        raise ValueError(f"Unsupported candidate type: {candidate_type}")
    try:
        payload: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取候选清单：{manifest_path}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"候选清单顶层必须是 JSON object：{manifest_path}")
    record_id = payload.get("record_id")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError(f"候选清单缺少有效 record_id：{manifest_path}")

    # v2 将动作区间样本与无标签覆盖样本分开放；仍读取旧版扁平清单。
    if "action_candidates" in payload or "coverage_candidates" in payload:
        raw_candidates: list[tuple[str, Any, str | None]] = []
        candidate_groups = (
            ("action_candidates", "action_span"),
            ("coverage_candidates", "coverage"),
        )
        for group_name, group_type in candidate_groups:
            group = payload.get(group_name, [])
            if not isinstance(group, list):
                raise TypeError(f"候选清单 {group_name} 必须是 JSON array：{manifest_path}")
            raw_candidates.extend(
                (f"{group_name}[{index}]", candidate, group_type)
                for index, candidate in enumerate(group)
            )
    else:
        legacy_candidates = payload.get("candidates")
        if not isinstance(legacy_candidates, list):
            raise TypeError(f"候选清单 candidates 必须是 JSON array：{manifest_path}")
        raw_candidates = [
            (
                f"candidates[{index}]",
                candidate,
                _legacy_candidate_type(candidate),
            )
            for index, candidate in enumerate(legacy_candidates)
        ]

    selected: list[tuple[str, Decimal]] = []
    found_ids: set[str] = set()
    matching_type_ids: set[str] = set()
    for candidate_path, candidate, inferred_type in raw_candidates:
        if not isinstance(candidate, dict) or candidate.get("status") != "selected":
            continue
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError(f"{candidate_path} 缺少有效 candidate_id：{manifest_path}")
        found_ids.add(candidate_id)
        if candidate_type != "all" and inferred_type != candidate_type:
            continue
        matching_type_ids.add(candidate_id)
        if requested_ids is not None and candidate_id not in requested_ids:
            continue
        try:
            event_time = Decimal(str(candidate["camera_relative_time_sec"]))
        except (KeyError, InvalidOperation) as exc:
            raise ValueError(
                f"{candidate_path} 缺少有效 camera_relative_time_sec：{manifest_path}"
            ) from exc
        if not event_time.is_finite() or event_time < 0:
            raise ValueError(f"{candidate_path} 的事件时间无效：{manifest_path}")
        selected.append((candidate_id, event_time))

    missing_ids = requested_ids - found_ids if requested_ids is not None else set()
    if missing_ids:
        raise ValueError(
            f"候选 ID 不存在或不是 selected 状态：{', '.join(sorted(missing_ids))} "
            f"({manifest_path})"
        )
    wrong_type_ids = requested_ids - matching_type_ids if requested_ids is not None else set()
    if wrong_type_ids:
        raise ValueError(
            f"候选 ID 不属于 --candidate-type {candidate_type}："
            f"{', '.join(sorted(wrong_type_ids))} ({manifest_path})"
        )
    selected.sort(key=lambda item: (item[1], item[0]))
    return record_id, selected


def _legacy_candidate_type(candidate: Any) -> str | None:
    """根据旧版扁平候选中的类型或来源推断候选池。"""
    if not isinstance(candidate, dict):
        return None
    explicit_type = candidate.get("candidate_type")
    if explicit_type in ("action_span", "coverage"):
        return explicit_type
    source = candidate.get("source")
    if source == "action_span_start":
        return "action_span"
    if source == "periodic_coverage":
        return "coverage"
    if candidate.get("longitudinal_action") is not None:
        return "action_span"
    return None


class LocalLabelingAgent:
    """协调候选读取、图片/定位 Loader、CoC Agent 和结果产物。"""

    def __init__(
        self,
        input_root: Path,
        output_root: Path,
        *,
        run_inference: bool,
        api_config: Path,
        timeout_sec: int,
        candidate_type: str = "all",
    ) -> None:
        self.input_root = input_root
        self.output_root = output_root
        self.run_inference = run_inference
        self.api_config = api_config
        self.timeout_sec = timeout_sec
        self.candidate_type = candidate_type
        (
            self.history_sec,
            self.future_sec,
            self.interval_sec,
            self.data_loader_config,
        ) = _load_temporal_config()

    def process_record(
        self,
        record_dir: Path,
        requested_ids: set[str] | None,
    ) -> int:
        """处理一条记录；图片时间表和 localization 在候选循环外各读取一次。"""
        manifest_path = record_dir / "segment_candidates.json"
        record_id, candidates = _load_selected_candidates(
            manifest_path,
            requested_ids,
            candidate_type=self.candidate_type,
        )
        if record_id != record_dir.name:
            raise ValueError(
                f"Manifest record_id {record_id!r} does not match directory "
                f"{record_dir.name!r}: {manifest_path}"
            )
        if not candidates:
            return 0
        data_record_dir = self.input_root / record_id
        if not data_record_dir.is_dir():
            raise FileNotFoundError(f"Record directory not found: {data_record_dir}")

        camera_dir = data_record_dir / "camera" / "front"
        localization_dir = data_record_dir / "localization"
        camera_images = load_timed_images(camera_dir)
        localization_positions = load_localization_positions(localization_dir)
        for candidate_id, event_time_sec in candidates:
            self._process_candidate(
                record_id,
                candidate_id,
                event_time_sec,
                camera_dir,
                localization_dir,
                camera_images,
                localization_positions,
            )
        return len(candidates)

    def _process_candidate(
        self,
        record_id: str,
        candidate_id: str,
        event_time_sec: Decimal,
        camera_dir: Path,
        localization_dir: Path,
        camera_images: list[TimedImage],
        localization_positions: list[TimedPosition],
    ) -> None:
        """复用单事件入口准备候选输入，并按需调用模型。"""
        args = SimpleNamespace(
            input_root=self.input_root,
            record=record_id,
            event_time_sec=event_time_sec,
            output_dir=self.output_root,
            run_inference=self.run_inference,
            api_config=self.api_config,
            timeout_sec=self.timeout_sec,
        )
        run_dir = self.output_root / record_id / f"event_{event_time_sec!s}s"
        prepared = _prepare_local_record(
            args,
            camera_dir,
            localization_dir,
            self.history_sec,
            self.future_sec,
            self.interval_sec,
            camera_images=camera_images,
            localization_positions=localization_positions,
        )
        artifacts = _write_preparation_artifacts(args, run_dir, prepared)
        _update_manifest(artifacts.manifest, candidate_id=candidate_id)
        print(f"{record_id}/{candidate_id} at camera t={event_time_sec}s")
        _print_preparation_summary(artifacts)
        if self.run_inference:
            _run_inference_and_visualize(
                args,
                prepared,
                artifacts,
                self.history_sec,
                self.future_sec,
                self.data_loader_config,
            )


def main() -> None:
    args = _parser().parse_args()
    if args.timeout_sec <= 0:
        raise SystemExit("--timeout-sec must be positive")
    requested_ids = set(args.candidate_id) if args.candidate_id else None
    if requested_ids is not None and args.record is None:
        raise SystemExit("Use --record when selecting by --candidate-id")
    if requested_ids is not None and len(args.record) != 1:
        raise SystemExit("Select exactly one --record when using --candidate-id")

    record_dirs = _record_directories(args.candidate_root, args.record)
    if not record_dirs:
        raise SystemExit(f"No candidate manifests found in {args.candidate_root}")
    labeler = LocalLabelingAgent(
        args.input_root,
        args.output_dir,
        run_inference=args.run_inference,
        api_config=args.api_config,
        timeout_sec=args.timeout_sec,
        candidate_type=args.candidate_type,
    )
    processed = 0
    for record_dir in record_dirs:
        processed += labeler.process_record(record_dir, requested_ids)
    if processed == 0:
        raise SystemExit("No selected candidates matched the requested filters")


if __name__ == "__main__":
    main()
