# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Overlay a saved CoC ``final_content`` sentence on a video for review."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Any

import cv2
import yaml


def read_effect(result_path: Path) -> str:
    """Read the structured effect sentence from a VLM result YAML file."""
    payload: dict[str, Any] = yaml.safe_load(result_path.read_text(encoding="utf-8"))
    final_content = payload.get("final_content", {})
    if not isinstance(final_content, dict):
        raise ValueError("final_content must be a mapping")
    schema = final_content.get("ego_behavior_schema", {})
    if not isinstance(schema, dict):
        raise ValueError("final_content.ego_behavior_schema must be a mapping")
    effect = schema.get("effect_on_ego_behavior")
    if not isinstance(effect, str) or not effect.strip():
        raise ValueError("effect_on_ego_behavior is missing or empty")
    return " ".join(effect.split())


def overlay_text(
    video_path: Path,
    output_path: Path,
    text: str,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
    event_offset_seconds: float | None = None,
    caption_position: str = "top",
) -> None:
    """Write a video excerpt with a readable caption."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = max(0, int(round(start_seconds * fps)))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_limit = (
        int(round(duration_seconds * fps)) if duration_seconds is not None else None
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot create output video: {output_path}")

    if caption_position not in {"top", "bottom"}:
        raise ValueError("caption_position must be 'top' or 'bottom'")
    lines = textwrap.wrap(text, width=90 if caption_position == "top" else 68) or [text]
    line_height = 38
    title = "VLM final_content"
    if event_offset_seconds is not None:
        title += f" | event at t=+{event_offset_seconds:.1f}s"
    box_height = 54 + line_height * len(lines)
    if caption_position == "top":
        x0, y0, box_width = 0, 0, width
    else:
        box_width = int(width * 0.64)
        x0 = (width - box_width) // 2
        y0 = height - box_height - 32
    written = 0
    while frame_limit is None or written < frame_limit:
        ok, frame = capture.read()
        if not ok:
            break
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x0, y0),
            (x0 + box_width, y0 + box_height),
            (0, 0, 0),
            -1,
        )
        frame = cv2.addWeighted(overlay, 0.68, frame, 0.32, 0)
        cv2.putText(
            frame,
            title,
            (x0 + 24, y0 + 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
        for line_index, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (x0 + 24, y0 + 72 + line_index * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.82,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        writer.write(frame)
        written += 1
    capture.release()
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--duration-seconds", type=float, default=None)
    parser.add_argument("--event-offset-seconds", type=float, default=None)
    parser.add_argument(
        "--caption-position",
        choices=("top", "bottom"),
        default="top",
        help="Place the caption at the top or at the bottom center of the frame.",
    )
    args = parser.parse_args()
    effect = read_effect(args.result)
    overlay_text(
        args.video,
        args.output,
        effect,
        start_seconds=args.start_seconds,
        duration_seconds=args.duration_seconds,
        event_offset_seconds=args.event_offset_seconds,
        caption_position=args.caption_position,
    )
    print(f"Wrote review video: {args.output}")
    print(f"Caption: {effect}")


if __name__ == "__main__":
    main()
