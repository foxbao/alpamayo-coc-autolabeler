# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""从本地记录的 localization 独立生成整段纵向 meta-action。"""

from __future__ import annotations

import argparse
from pathlib import Path

from .recording_autolabel.localization import load_localization_positions
from .recording_autolabel.meta_action_artifacts import write_full_record_meta_action
from .recording_autolabel.scenario import LocalTemporalScenario

DEFAULT_INPUT_ROOT = Path("/home/baojiali/Downloads/disk3/20261019_lin_007_out")
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "outputs" / "local_record_candidates"


def _parser() -> argparse.ArgumentParser:
    """定义输入记录及 meta-action 输出目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--record",
        nargs="+",
        help="record directory names; omit to process localization directories under input-root",
    )
    return parser


def _validate_record_names(record_names: list[str]) -> None:
    """阻止记录参数通过路径片段访问 input-root 之外的目录。"""
    for record_name in record_names:
        if Path(record_name).name != record_name or record_name in {".", ".."}:
            raise SystemExit(f"--record must be a directory name: {record_name}")


def _discover_records(input_root: Path, requested: list[str] | None) -> list[str]:
    """使用显式记录列表，或发现根目录下含 localization 的记录。"""
    if requested:
        _validate_record_names(requested)
        return requested
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    return sorted(
        path.name
        for path in input_root.iterdir()
        if path.is_dir() and (path / "localization").is_dir()
    )


def process_clip(input_root: Path, output_root: Path, record_id: str) -> Path:
    """从一条记录的 localization 生成并写出完整动作结果。"""
    localization_dir = input_root / record_id / "localization"
    positions = load_localization_positions(localization_dir)
    scenario = LocalTemporalScenario(record_id, positions)
    result = scenario.longitudinal_action_result
    output_path = write_full_record_meta_action(
        result,
        output_root / record_id / "full_record_meta_action.json",
        record_id,
        scenario.localization_start,
    )
    print(f"{record_id}: {len(result.points)} action ticks, {len(result.spans)} action spans")
    print(f"  Meta-action: {output_path}")
    return output_path


def main() -> None:
    args = _parser().parse_args()
    records = _discover_records(args.input_root, args.record)
    if not records:
        raise SystemExit(f"No localization record directories found in {args.input_root}")

    for record_id in records:
        process_clip(args.input_root, args.output_dir, record_id)


if __name__ == "__main__":
    main()
