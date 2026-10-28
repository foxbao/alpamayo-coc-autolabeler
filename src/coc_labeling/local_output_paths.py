# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build output paths grouped by the local run date."""

from __future__ import annotations

from datetime import date
from pathlib import Path


def dated_output_dir(stage_name: str, run_date: date | None = None) -> Path:
    """Return ``outputs/YYYYMMDD/<stage_name>`` for a run's local date."""
    if Path(stage_name).name != stage_name or stage_name in {".", ".."}:
        raise ValueError(f"Output stage must be a directory name: {stage_name}")
    run_date = run_date or date.today()
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "outputs" / run_date.strftime("%Y%m%d") / stage_name
