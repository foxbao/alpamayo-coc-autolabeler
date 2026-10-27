# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""从本地 temporal scenario 的动作区间筛选 CoC 候选片段。

函数名按 PAI-AV 的处理阶段对应，便于对照：

- ``select_segment_candidates`` 对应 ``keyframe_auto_select`` 的候选筛选阶段。

本地候选阶段另加定时覆盖点和相机窗口检查，并按类别设置动作候选上限；
当前没有对应 PAI 的横向或 lane 标签。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from .camera import sample_offsets, select_frames
from .types import LocalMotionChunk, SelectedFrame, TimedImage

DEFAULT_PERIODIC_CANDIDATE_INTERVAL_SEC = Decimal(5)
DEFAULT_MINIMUM_CANDIDATE_SEPARATION_SEC = Decimal(2)
# PAI keyframe_auto_select defaults to 10 frames; at its 10 Hz timeline this is 1 second.
DEFAULT_MINIMUM_ACTION_DURATION_SEC = Decimal("1.0")
DEFAULT_ACTION_TARGET_COUNT = 500_000


@dataclass(frozen=True)
class CandidateSeed:
    """一个动作区间样本或无动作标签的定时覆盖候选。"""

    candidate_id: str
    camera_time_sec: Decimal
    absolute_timestamp: Decimal
    localization_time_sec: Decimal
    source: str
    action_label: str | None
    action_end_localization_time_sec: Decimal | None = None

    @property
    def action_duration_sec(self) -> Decimal | None:
        """返回动作区间时长；覆盖候选没有动作区间。"""
        if self.action_end_localization_time_sec is None:
            return None
        return self.action_end_localization_time_sec - self.localization_time_sec


@dataclass
class SegmentCandidate:
    """候选事件的窗口验证、去重状态和可选图片帧。"""

    seed: CandidateSeed
    status: str
    reason: str | None = None
    merged_into: str | None = None
    merged_candidates: list[CandidateSeed] = field(default_factory=list)
    selected_frames: list[SelectedFrame] | None = None


def _candidate_seeds(
    action_motions: list[LocalMotionChunk],
    localization_start: Decimal,
    camera_images: list[TimedImage],
    history_sec: Decimal,
    future_sec: Decimal,
    periodic_interval_sec: Decimal,
) -> list[CandidateSeed]:
    """把不同来源的候选时刻统一到同一份时间信息中。

    动作区间的时间相对 localization 首个样本，因此先换成 Unix 时间，
    再减去第一张相机图片的时间戳，得到选帧函数使用的相机相对秒数：
    ``camera_time = localization_start + motion.start_time - camera_start``。

    定时覆盖点直接从相机时间轴的 ``history_sec`` 开始，每隔
    ``periodic_interval_sec`` 产生一个点；这样最早的候选也能容纳完整历史窗口。
    所有来源最后都保存相机相对时间、定位相对时间和绝对时间戳，便于核对转换。
    """
    if periodic_interval_sec <= 0:
        raise ValueError("Periodic candidate interval must be positive")
    if not camera_images:
        raise ValueError("No camera images are available")

    camera_start = camera_images[0].timestamp
    camera_end = camera_images[-1].timestamp
    seeds: list[tuple[Decimal, Decimal, str, str | None, Decimal | None]] = []
    for motion in action_motions:
        if motion.end_time <= motion.start_time:
            raise ValueError(f"Invalid action span: {motion}")
        absolute_timestamp = localization_start + motion.start_time
        seeds.append(
            (
                absolute_timestamp - camera_start,
                absolute_timestamp,
                "action_span_start",
                motion.action_type,
                motion.end_time,
            )
        )

    # 定时点按相机时间轴对齐，确保早段/晚段也有稳定覆盖。
    camera_duration = camera_end - camera_start
    camera_time = history_sec
    while camera_time + future_sec <= camera_duration:
        absolute_timestamp = camera_start + camera_time
        seeds.append(
            (
                camera_time,
                absolute_timestamp,
                "periodic_coverage",
                None,
                None,
            )
        )
        camera_time += periodic_interval_sec

    seeds.sort(key=lambda item: (item[0], item[2], item[3] or ""))
    return [
        CandidateSeed(
            candidate_id=f"candidate_{index:03d}",
            camera_time_sec=camera_time_sec,
            absolute_timestamp=absolute_timestamp,
            localization_time_sec=absolute_timestamp - localization_start,
            source=source,
            action_label=action_label,
            action_end_localization_time_sec=action_end_time_sec,
        )
        for index, (
            camera_time_sec,
            absolute_timestamp,
            source,
            action_label,
            action_end_time_sec,
        ) in enumerate(seeds, start=1)
    ]


def select_segment_candidates(
    action_motions: list[LocalMotionChunk],
    localization_start: Decimal,
    camera_images: list[TimedImage],
    history_sec: Decimal,
    future_sec: Decimal,
    interval_sec: Decimal,
    *,
    periodic_interval_sec: Decimal = DEFAULT_PERIODIC_CANDIDATE_INTERVAL_SEC,
    minimum_separation_sec: Decimal = DEFAULT_MINIMUM_CANDIDATE_SEPARATION_SEC,
    minimum_action_duration_sec: Decimal = DEFAULT_MINIMUM_ACTION_DURATION_SEC,
    action_types: set[str] | None = None,
) -> list[SegmentCandidate]:
    """从整段纵向动作和定时覆盖点中挑出可供 CoC 检查的片段。

    这一阶段对应 PAI 的 ``keyframe_auto_select``：读取动作区间、应用筛选条件，
    产生下游使用的片段候选。本地版本会直接验证图片窗口并附上 16 张选中帧；
    返回候选后还可用 ``balance_and_select_candidates`` 按动作类别限额和最小间距
    重新选择。

    参数中的时间原点各不相同：

    - ``action_motions`` 的动作区间时间相对 localization 首个样本；
    - ``localization_start`` 是该首样本的绝对 Unix 秒时间戳；
    - ``camera_images`` 中的时间戳也为绝对 Unix 秒，但 CoC 事件时间要相对首张图片。

    动作候选按下面的关系换到相机时间轴：
    ``event_unix = localization_start + motion.start_time``，
    ``event_camera_sec = event_unix - camera_images[0].timestamp``。
    随后会把这些动作区间起点与相机轴上的定时覆盖点合并处理。

    处理步骤：

    1. ``_candidate_seeds`` 生成动作区间起点和定时点。所有动作标签都会参与，
       包括 ``Maintain Speed``；它们是规则提示，不代表人工真值。
    2. 检查候选时刻能否容纳完整的历史和未来窗口。超出相机覆盖范围的候选标为
       ``rejected``，并保留具体原因，不参与后续去重。
    3. 调用 ``select_frames`` 按 CoC 的采样间隔选图。帧缺失、误差超限或同一图片
       被重复使用时，也会把候选标为 ``rejected``。
    4. 对有效候选去重：动作区间起点优先于定时覆盖点；同一来源内按时间先后处理。
       与已选候选相距小于 ``minimum_separation_sec`` 的候选会标为 ``suppressed``，
       并记录它合并到了哪个候选。间距刚好等于阈值时会保留。

    返回值包含所有候选，按相机相对时间排序。状态为 ``selected`` 的项目带有
    ``selected_frames``，可以直接用于生成联系表；被拒绝或抑制的项目也保留在清单中，
    方便理解为什么它们没有单独生成预览。
    """
    if minimum_separation_sec < 0:
        raise ValueError("Minimum candidate separation cannot be negative")
    if minimum_action_duration_sec < 0:
        raise ValueError("Minimum action duration cannot be negative")
    camera_start = camera_images[0].timestamp if camera_images else None
    camera_end = camera_images[-1].timestamp if camera_images else None
    if camera_start is None or camera_end is None:
        raise ValueError("No camera images are available")
    offsets = sample_offsets(history_sec, future_sec, interval_sec)

    # 先验证每个候选是否真的有完整窗口和可用图片；无效项不应挤掉有效候选。
    candidates: list[SegmentCandidate] = []
    for seed in _candidate_seeds(
        action_motions,
        localization_start,
        camera_images,
        history_sec,
        future_sec,
        periodic_interval_sec,
    ):
        if seed.source == "action_span_start":
            if action_types is not None and seed.action_label not in action_types:
                candidates.append(SegmentCandidate(seed, "filtered", "action_type_not_selected"))
                continue
            duration = seed.action_duration_sec
            if duration is None:
                raise ValueError(f"Action candidate has no action interval: {seed.candidate_id}")
            if duration < minimum_action_duration_sec:
                candidates.append(
                    SegmentCandidate(seed, "filtered", "action_span_below_minimum_duration")
                )
                continue
        if seed.camera_time_sec - history_sec < 0:
            candidates.append(SegmentCandidate(seed, "rejected", "insufficient_camera_history"))
            continue
        if seed.camera_time_sec + future_sec > camera_end - camera_start:
            candidates.append(SegmentCandidate(seed, "rejected", "insufficient_camera_future"))
            continue
        try:
            selected_frames = select_frames(camera_images, seed.camera_time_sec, offsets)
        except ValueError as exc:
            candidates.append(SegmentCandidate(seed, "rejected", str(exc)))
            continue
        candidates.append(SegmentCandidate(seed, "pending", selected_frames=selected_frames))

    return apply_candidate_separation(candidates, minimum_separation_sec)


def apply_candidate_separation(
    candidates: list[SegmentCandidate], minimum_separation_sec: Decimal
) -> list[SegmentCandidate]:
    """按动作优先和时间间距重新选择候选，覆盖初次去重或限额后的重算。

    只重新处理相机窗口有效的 ``selected``/``suppressed`` 项。被过滤或拒绝的
    候选不会恢复；定时覆盖点可以在动作类别限额移除邻近动作后重新入选。
    """
    if minimum_separation_sec < 0:
        raise ValueError("Minimum candidate separation cannot be negative")
    for candidate in candidates:
        if candidate.status in {"selected", "suppressed"}:
            candidate.status = "pending"
            candidate.reason = None
            candidate.merged_into = None
            candidate.merged_candidates.clear()

    pending = [candidate for candidate in candidates if candidate.status == "pending"]
    pending.sort(
        key=lambda candidate: (
            candidate.seed.source != "action_span_start",
            candidate.seed.camera_time_sec,
        )
    )
    selected: list[SegmentCandidate] = []
    for candidate in pending:
        nearby = [
            item
            for item in selected
            if abs(item.seed.camera_time_sec - candidate.seed.camera_time_sec)
            < minimum_separation_sec
        ]
        if nearby:
            nearest = min(
                nearby,
                key=lambda item: abs(item.seed.camera_time_sec - candidate.seed.camera_time_sec),
            )
            candidate.status = "suppressed"
            candidate.reason = "within_minimum_candidate_separation"
            candidate.merged_into = nearest.seed.candidate_id
            nearest.merged_candidates.append(candidate.seed)
        else:
            candidate.status = "selected"
            selected.append(candidate)

    return sorted(candidates, key=lambda candidate: candidate.seed.camera_time_sec)


def balance_action_candidates(
    candidates: list[SegmentCandidate], target_count: int = DEFAULT_ACTION_TARGET_COUNT
) -> dict[str, tuple[int, int]]:
    """按动作类别保留时长最长的候选；定时覆盖项不参与平衡。

    与 PAI 的 ``balance_action_records`` 一样，每类最多保留 ``target_count`` 项，
    按动作区间时长降序选择。长度相同时，按绝对发生时间和候选 ID 稳定排序。
    这里只处理相机窗口有效、尚未过滤或拒绝的动作候选；包括第一次近邻去重中
    被压制的候选。限额过滤时释放其选帧对象；若之后回填，输出阶段会重新匹配图片。

    返回各类别的 ``(平衡前数量, 平衡后数量)``，便于命令行汇报。
    """
    if target_count < 1:
        raise ValueError("Action candidate target count must be at least one")

    by_action: dict[str, list[SegmentCandidate]] = {}
    for candidate in candidates:
        if candidate.status not in {"selected", "suppressed"}:
            continue
        if candidate.seed.source != "action_span_start":
            continue
        action_label = candidate.seed.action_label
        if action_label is None or candidate.seed.action_duration_sec is None:
            raise ValueError(f"Selected action candidate has no action interval: {candidate.seed}")
        by_action.setdefault(action_label, []).append(candidate)

    summary: dict[str, tuple[int, int]] = {}
    for candidate in candidates:
        if candidate.status in {"selected", "suppressed"}:
            candidate.status = "pending"
            candidate.reason = None
            candidate.merged_into = None
            candidate.merged_candidates.clear()

    for action_label, action_candidates in sorted(by_action.items()):
        ranked_with_duration: list[tuple[Decimal, SegmentCandidate]] = []
        for candidate in action_candidates:
            duration = candidate.seed.action_duration_sec
            if duration is None:
                raise ValueError(f"Selected action candidate has no duration: {candidate.seed}")
            ranked_with_duration.append((duration, candidate))
        ranked = [
            candidate
            for _, candidate in sorted(
                ranked_with_duration,
                key=lambda item: (
                    -item[0],
                    item[1].seed.absolute_timestamp,
                    item[1].seed.candidate_id,
                ),
            )
        ]
        keep_count = min(len(ranked), target_count)
        for candidate in ranked[keep_count:]:
            candidate.status = "filtered"
            candidate.reason = "action_type_target_count_exceeded"
            candidate.selected_frames = None
        summary[action_label] = (len(action_candidates), keep_count)
    return summary


def balance_and_select_candidates(
    candidate_groups: list[list[SegmentCandidate]],
    target_count: int,
    minimum_separation_sec: Decimal,
) -> dict[str, tuple[int, int]]:
    """按类别限额和记录内最小间距选择候选，并回填可行的动作候选。

    ``target_count`` 是每类上限。先依 PAI 语义优先保留持续时间较长的动作；
    若这些候选因记录内的时间间距而相互抑制，就从该类别剩余候选中按时长
    递减回填。回填只接受与本记录已选动作候选保持最小间距的项目。最后重新
    应用间距规则，让动作候选继续优先于定时覆盖候选。

    类别平衡跨所有记录统计，间距约束只在同一记录内生效。返回每个类别的
    ``(可用候选数, 最终选中数)``。

    输入应来自 ``select_segment_candidates``：动作候选必须已经经过相机窗口检查
    和首次间距筛选，状态为 ``selected``、``suppressed``、``filtered`` 或
    ``rejected``。内部回填时才会把候选重置为 ``pending``。
    """
    if minimum_separation_sec < 0:
        raise ValueError("Minimum candidate separation cannot be negative")

    all_candidates = [candidate for group in candidate_groups for candidate in group]
    pending_actions = [
        candidate.seed.candidate_id
        for candidate in all_candidates
        if candidate.status == "pending" and candidate.seed.source == "action_span_start"
    ]
    if pending_actions:
        candidate_ids = ", ".join(pending_actions)
        raise ValueError(
            "Action candidates must pass initial window and separation checks before balancing; "
            f"pending: {candidate_ids}"
        )

    balance_summary = balance_action_candidates(all_candidates, target_count)
    available_by_action = {
        action: available for action, (available, _) in balance_summary.items()
    }
    for group in candidate_groups:
        apply_candidate_separation(group, minimum_separation_sec)

    selected_actions_by_group = [
        [
            candidate
            for candidate in group
            if candidate.status == "selected"
            and candidate.seed.source == "action_span_start"
        ]
        for group in candidate_groups
    ]
    selected_by_action: dict[str, int] = {}
    for selected_actions in selected_actions_by_group:
        for candidate in selected_actions:
            action = candidate.seed.action_label
            if action is not None:
                selected_by_action[action] = selected_by_action.get(action, 0) + 1

    overflow = sorted(
        (
            (group_index, candidate)
            for group_index, group in enumerate(candidate_groups)
            for candidate in group
            if candidate.status == "filtered"
            and candidate.reason == "action_type_target_count_exceeded"
        ),
        key=lambda item: (
            -(item[1].seed.action_duration_sec or Decimal(0)),
            item[1].seed.absolute_timestamp,
            item[1].seed.candidate_id,
        ),
    )
    for group_index, candidate in overflow:
        action = candidate.seed.action_label
        if action is None or selected_by_action.get(action, 0) >= target_count:
            continue
        selected_actions = selected_actions_by_group[group_index]
        if any(
            abs(item.seed.camera_time_sec - candidate.seed.camera_time_sec)
            < minimum_separation_sec
            for item in selected_actions
        ):
            continue

        # 输出阶段会为回填候选重新匹配图片，避免平衡阶段长期保留多余帧对象。
        candidate.status = "pending"
        candidate.reason = None
        candidate.merged_into = None
        candidate.merged_candidates.clear()
        selected_actions.append(candidate)
        selected_by_action[action] = selected_by_action.get(action, 0) + 1

    for group in candidate_groups:
        apply_candidate_separation(group, minimum_separation_sec)

    final_by_action: dict[str, int] = {}
    for candidate in all_candidates:
        if candidate.status == "selected" and candidate.seed.source == "action_span_start":
            action = candidate.seed.action_label
            if action is not None:
                final_by_action[action] = final_by_action.get(action, 0) + 1
    return {
        action: (available, final_by_action.get(action, 0))
        for action, available in sorted(available_by_action.items())
    }


def candidate_manifest_payload(
    record_id: str,
    candidates: list[SegmentCandidate],
    camera_start: Decimal,
    camera_end: Decimal,
    localization_start: Decimal,
    history_sec: Decimal,
    future_sec: Decimal,
    interval_sec: Decimal,
    periodic_interval_sec: Decimal,
    minimum_separation_sec: Decimal,
    *,
    minimum_action_duration_sec: Decimal = DEFAULT_MINIMUM_ACTION_DURATION_SEC,
    action_types: set[str] | None = None,
    action_target_count: int = DEFAULT_ACTION_TARGET_COUNT,
) -> dict[str, Any]:
    """构造按动作区间和定时覆盖分组的候选清单。"""

    def seed_payload(seed: CandidateSeed) -> dict[str, Any]:
        is_action_candidate = seed.source == "action_span_start"
        payload: dict[str, Any] = {
            "candidate_id": seed.candidate_id,
            "candidate_type": "action_span" if is_action_candidate else "coverage",
            "camera_relative_time_sec": str(seed.camera_time_sec),
            "localization_relative_time_sec": str(seed.localization_time_sec),
            "absolute_unix_timestamp_sec": str(seed.absolute_timestamp),
            "source": seed.source,
            "longitudinal_action": seed.action_label,
        }
        if is_action_candidate:
            end_time = seed.action_end_localization_time_sec
            if end_time is None:
                raise ValueError(f"Action candidate has no interval: {seed.candidate_id}")
            payload["action_span"] = {
                "start_localization_time_sec": str(seed.localization_time_sec),
                "end_localization_time_sec": str(end_time),
                "duration_sec": str(end_time - seed.localization_time_sec),
                "start_unix_timestamp_sec": str(localization_start + seed.localization_time_sec),
                "end_unix_timestamp_sec": str(localization_start + end_time),
                "end_is_exclusive": True,
            }
        else:
            payload["action_span"] = None
        return payload

    action_entries: list[dict[str, Any]] = []
    coverage_entries: list[dict[str, Any]] = []
    for candidate in candidates:
        seed = candidate.seed
        entry: dict[str, Any] = {
            **seed_payload(seed),
            "status": candidate.status,
            "reason": candidate.reason,
            "merged_into": candidate.merged_into,
            "merged_candidates": [seed_payload(merged) for merged in candidate.merged_candidates],
        }
        if candidate.status == "selected" and candidate.selected_frames is not None:
            entry["contact_sheet"] = f"contact_sheets/{seed.candidate_id}.jpg"
            entry["selected_frames"] = [
                {
                    "relative_time_sec": str(frame.relative_time),
                    "target_unix_timestamp_sec": str(frame.target_timestamp),
                    "source_image": frame.image.path.name,
                    "time_error_sec": str(frame.error_sec),
                }
                for frame in candidate.selected_frames
            ]
        target_entries = action_entries if seed.source == "action_span_start" else coverage_entries
        target_entries.append(entry)

    return {
        "schema_version": 2,
        "record_id": record_id,
        "time_origins": {
            "camera_start_unix_timestamp_sec": str(camera_start),
            "camera_end_unix_timestamp_sec": str(camera_end),
            "localization_start_unix_timestamp_sec": str(localization_start),
            "camera_relative_time_definition": (
                "absolute event timestamp minus first front-camera filename timestamp"
            ),
        },
        "window": {
            "history_sec": str(history_sec),
            "future_sec": str(future_sec),
            "sample_interval_sec": str(interval_sec),
            "periodic_coverage_interval_sec": str(periodic_interval_sec),
            "minimum_candidate_separation_sec": str(minimum_separation_sec),
            "minimum_action_duration_sec": str(minimum_action_duration_sec),
            "action_types": sorted(action_types) if action_types is not None else None,
            "action_target_count_per_type": action_target_count,
            "action_target_selection": "longest_action_span_first",
        },
        "meta_action_source": "full_record_meta_action.json",
        "candidate_note": (
            "Action-span candidates carry rule-based longitudinal hints and full action intervals; "
            "these labels require review and are subject to the per-action target count, keeping "
            "longer intervals first. Coverage candidates have no action label and are not subject "
            "to this limit."
        ),
        "action_candidates": action_entries,
        "coverage_candidates": coverage_entries,
    }


def write_candidate_manifest(payload: dict[str, Any], output_path: Path) -> Path:
    """写入候选清单 JSON。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path
