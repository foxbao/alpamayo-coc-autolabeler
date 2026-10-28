# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discover local datasets that use ``sample/<batch>/<record>`` layout."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from coc_labeling.recording_autolabel.camera import (
    front_camera_directory,
    load_timed_images,
)
from coc_labeling.recording_autolabel.localization import (
    load_localization_positions,
)
from coc_labeling.recording_autolabel.types import TimedImage, TimedPosition

DATASET_ROOTS = {
    "vla_kl": "data/vla_kl",
    "kl_8": "data/kl_8",
}


@dataclass(frozen=True)
class SampleRecord:
    """One record found below ``sample/<batch>/<record>``."""

    dataset_name: str
    batch_id: str
    record_id: str
    record_dir: Path

    @property
    def qualified_id(self) -> str:
        """Return a stable ID that remains unique across batches."""
        return f"{self.batch_id}/{self.record_id}"

    @property
    def front_camera_dir(self) -> Path:
        return front_camera_directory(self.record_dir)

    @property
    def localization_dir(self) -> Path:
        return self.record_dir / "localization"

    def load_front_images(self) -> list[TimedImage]:
        """Load timestamped front-camera JPEGs with the shared image reader."""
        return load_timed_images(self.front_camera_dir)

    def load_localization(self) -> list[TimedPosition]:
        """Load localization samples with the shared localization reader."""
        return load_localization_positions(self.localization_dir)


class SampleDataset:
    """Select and discover datasets sharing the same sample directory layout."""

    def __init__(self, dataset_name: str, root: Path | None = None) -> None:
        if dataset_name not in DATASET_ROOTS:
            choices = ", ".join(sorted(DATASET_ROOTS))
            raise ValueError(f"Unknown dataset {dataset_name!r}; choose from: {choices}")
        self.dataset_name = dataset_name
        self.root = root or Path(__file__).resolve().parents[3] / DATASET_ROOTS[dataset_name]
        self.sample_dir = self.root / "sample"

    def discover_records(self) -> list[SampleRecord]:
        """Return complete records in stable batch/record order.

        Directories missing either front images or localization are skipped, so partial
        exports do not enter downstream processing as valid records.
        """
        if not self.sample_dir.is_dir():
            raise FileNotFoundError(f"Dataset sample directory not found: {self.sample_dir}")

        records = []
        for batch_dir in sorted(path for path in self.sample_dir.iterdir() if path.is_dir()):
            for record_dir in sorted(path for path in batch_dir.iterdir() if path.is_dir()):
                record = SampleRecord(
                    dataset_name=self.dataset_name,
                    batch_id=batch_dir.name,
                    record_id=record_dir.name,
                    record_dir=record_dir,
                )
                if record.front_camera_dir.is_dir() and record.localization_dir.is_dir():
                    records.append(record)
        return records

    def get_record(self, qualified_id: str) -> SampleRecord:
        """Find one record by its ``<batch>/<record>`` ID."""
        parts = Path(qualified_id).parts
        if len(parts) != 2 or any(part in {".", ".."} for part in parts):
            raise ValueError("Record ID must have the form '<batch>/<record>'")
        for record in self.discover_records():
            if record.qualified_id == qualified_id:
                return record
        raise KeyError(f"Record {qualified_id!r} not found in dataset {self.dataset_name!r}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_ROOTS), required=True)
    parser.add_argument("--dataset-root", type=Path, help="Override the repository data path")
    parser.add_argument(
        "--verify-record",
        help="Load images and localization for one '<batch>/<record>' record",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    dataset = SampleDataset(args.dataset, args.dataset_root)
    records = dataset.discover_records()
    print(f"Dataset: {dataset.dataset_name}")
    print(f"Root: {dataset.root}")
    print(f"Readable records: {len(records)}")
    for record in records[:10]:
        print(f"  {record.qualified_id}")
    if len(records) > 10:
        print(f"  ... and {len(records) - 10} more")
    if args.verify_record:
        record = dataset.get_record(args.verify_record)
        images = record.load_front_images()
        positions = record.load_localization()
        print(f"Verified {record.qualified_id}")
        print(f"  Front images: {len(images)}")
        print(f"  Localization samples: {len(positions)}")


if __name__ == "__main__":
    main()
