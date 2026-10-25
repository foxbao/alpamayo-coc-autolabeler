# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import math
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

import coc_labeling.local_recording.agent as local_record_agent
from coc_labeling.agents.vlm.coc_agent import VLMCoCAgent
from coc_labeling.local_record_autolabel import (
    LOCAL_POSITION_EGO_STATE_PROMPT,
    PositionMotionSample,
    ResampledPosition,
    SelectedFrame,
    TimedImage,
    TimedPosition,
    _load_temporal_config,
    _parse_filename_timestamp,
    _position_motion_offsets,
    _position_trajectory_offsets,
    _run_agent,
    _sample_offsets,
    build_agent_data,
    build_position_motion_samples,
    derive_position_motion,
    load_localization_positions,
    resample_position_trajectory,
    rotate_map_velocity_to_body_flu,
    sample_orientation_at,
    sample_position_at,
    sample_recorded_linear_velocity_at,
    select_frames,
    write_contact_sheet_with_coc,
    write_position_motion_audit,
    write_resampled_position_trajectory,
)


def test_parse_filename_timestamp_uses_decimal_seconds() -> None:
    assert _parse_filename_timestamp(Path("1792395581.796917231.jpg")) == Decimal(
        "1792395581.796917231"
    )


def test_sample_frames_selects_unique_nearest_images_within_tolerance() -> None:
    start = Decimal(1000)
    offsets = _sample_offsets(Decimal(2), Decimal(6), Decimal("0.5"))
    images = [
        TimedImage(
            path=Path(f"{start + Decimal(index) / 10:.1f}.jpg"),
            timestamp=start + Decimal(index) / 10,
        )
        for index in range(300)
    ]

    selected = select_frames(images, Decimal("21.6"), offsets)

    assert len(selected) == 16
    assert selected[0].relative_time == Decimal(-2)
    assert selected[3].relative_time == Decimal("-0.5")
    assert selected[4].relative_time == Decimal("0.0")
    assert selected[-1].relative_time == Decimal("5.5")
    assert len({frame.image.path for frame in selected}) == 16
    assert all(frame.error_sec <= Decimal("0.050") for frame in selected)


def test_select_frames_rejects_reused_source_image() -> None:
    images = [TimedImage(Path("1002.0.jpg"), Decimal("1002.0"))]

    with pytest.raises(ValueError, match="reused"):
        select_frames(images, Decimal(0), [Decimal(0), Decimal("0.01")])


def test_load_localization_positions_sorts_files_and_validates_coordinates(
    tmp_path: Path,
) -> None:
    localization_dir = tmp_path / "localization"
    localization_dir.mkdir()
    (localization_dir / "10.02.json").write_text(
        '{"position": {"x": 2, "y": -1}}', encoding="utf-8"
    )
    (localization_dir / "10.01.json").write_text(
        '{"position": {"x": 1, "y": -0.5}}', encoding="utf-8"
    )

    samples = load_localization_positions(localization_dir)

    assert [sample.timestamp for sample in samples] == [Decimal("10.01"), Decimal("10.02")]
    assert [(sample.x, sample.y) for sample in samples] == [(1.0, -0.5), (2.0, -1.0)]


def test_orientation_slerp_and_inverse_rotation_use_flu_axes() -> None:
    q_identity = (1.0, 0.0, 0.0, 0.0)
    half = 0.5**0.5
    q_yaw_90 = (half, 0.0, 0.0, half)
    samples = [
        TimedPosition(
            Decimal("1.0"), 0.0, 0.0, Path("1.0.json"), orientation_wxyz=q_identity
        ),
        TimedPosition(
            Decimal("1.01"), 0.0, 0.0, Path("1.01.json"), orientation_wxyz=q_yaw_90
        ),
    ]

    interpolated = sample_orientation_at(samples, Decimal("1.005"))
    assert interpolated == pytest.approx(
        (math.cos(math.pi / 8), 0.0, 0.0, math.sin(math.pi / 8))
    )
    # yaw=+90° 时，地图北向速度逆旋转后应为车体前向正速度。
    body_velocity = rotate_map_velocity_to_body_flu((0.0, 2.0, 0.0), q_yaw_90)
    assert body_velocity == pytest.approx((2.0, 0.0, 0.0))


def test_position_motion_projects_10hz_map_velocity_with_same_time_orientation() -> None:
    q_yaw_90 = (0.5**0.5, 0.0, 0.0, 0.5**0.5)
    raw = [
        TimedPosition(
            timestamp=Decimal(index) / 100,
            x=0.0,
            y=2.0 * index / 100,
            source_path=Path(f"{index}.json"),
            orientation_wxyz=q_yaw_90,
            linear_velocity_xyz=(2.0, 0.0, 0.0),
        )
        for index in range(101)
    ]
    resampled = [
        ResampledPosition(
            relative_time=Decimal(index) / 10 - Decimal("0.5"),
            target_timestamp=Decimal(index) / 10,
            position_valid=True,
            x=0.0,
            y=2.0 * index / 10,
            position_error=None,
            connected_to_previous=index > 0,
        )
        for index in range(11)
    ]

    motion = build_position_motion_samples(
        raw,
        Decimal("0.5"),
        [Decimal("0.0")],
        resampled_positions=resampled,
    )[0]

    assert motion.velocity_x == pytest.approx(0.0)
    assert motion.velocity_y == pytest.approx(2.0)
    assert motion.body_velocity_right == pytest.approx(0.0, abs=1e-7)
    assert motion.body_velocity_forward == pytest.approx(2.0)
    assert motion.body_velocity_residual_mps == pytest.approx(0.0)
    assert sample_recorded_linear_velocity_at(raw, Decimal("0.505")) == pytest.approx(
        (2.0, 0.0, 0.0)
    )


def test_flu_leftward_motion_becomes_negative_right_positive_lateral_speed() -> None:
    q_yaw_90 = (0.5**0.5, 0.0, 0.0, 0.5**0.5)
    raw = [
        TimedPosition(
            timestamp=Decimal(index) / 100,
            x=-2.0 * index / 100,
            y=0.0,
            source_path=Path(f"{index}.json"),
            orientation_wxyz=q_yaw_90,
            linear_velocity_xyz=(0.0, 2.0, 0.0),
        )
        for index in range(101)
    ]
    resampled = [
        ResampledPosition(
            relative_time=Decimal(index) / 10 - Decimal("0.5"),
            target_timestamp=Decimal(index) / 10,
            position_valid=True,
            x=-2.0 * index / 10,
            y=0.0,
            position_error=None,
            connected_to_previous=index > 0,
        )
        for index in range(11)
    ]

    motion = build_position_motion_samples(
        raw,
        Decimal("0.5"),
        [Decimal("0.0")],
        resampled_positions=resampled,
    )[0]

    assert motion.body_velocity_forward == pytest.approx(0.0, abs=1e-7)
    assert motion.body_velocity_right == pytest.approx(-2.0)
    assert motion.body_velocity_residual_mps == pytest.approx(0.0)


def test_position_motion_keeps_ground_speed_when_orientation_is_missing() -> None:
    raw = [
        TimedPosition(
            timestamp=Decimal(index) / 100,
            x=float(index) / 100,
            y=0.0,
            source_path=Path(f"{index}.json"),
        )
        for index in range(101)
    ]

    motion = build_position_motion_samples(raw, Decimal("0.5"), [Decimal("0.0")])[0]

    assert motion.motion_valid
    assert motion.ground_speed == pytest.approx(1.0)
    assert motion.body_velocity_forward is None
    assert motion.body_velocity_right is None
    assert "orientation is unavailable" in (motion.body_motion_error or "")


def test_load_localization_positions_rejects_non_finite_position(tmp_path: Path) -> None:
    localization_dir = tmp_path / "localization"
    localization_dir.mkdir()
    (localization_dir / "10.0.json").write_text(
        '{"position": {"x": NaN, "y": 0}}', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="must be finite"):
        load_localization_positions(localization_dir)


def test_sample_position_interpolates_short_gaps_and_rejects_long_gaps_and_extrapolation() -> None:
    samples = [
        TimedPosition(Decimal("0.0"), 0.0, 0.0, Path("0.0.json")),
        TimedPosition(Decimal("0.01"), 2.0, -1.0, Path("0.01.json")),
        TimedPosition(Decimal("0.434593"), 3.0, -1.5, Path("0.434593.json")),
    ]

    assert sample_position_at(samples, Decimal("0.005")) == (1.0, -0.5)
    with pytest.raises(ValueError, match="exceeds"):
        sample_position_at(samples, Decimal("0.2"))
    with pytest.raises(ValueError, match="outside localization coverage"):
        sample_position_at(samples, Decimal("-0.001"))


def test_derive_position_motion_uses_centered_position_difference() -> None:
    samples = [
        TimedPosition(
            timestamp=Decimal(index) / 100,
            x=2.0 * index / 100 + 5.0,
            y=-index / 100 + 4.0,
            source_path=Path(f"{index}.json"),
        )
        for index in range(101)
    ]

    x, y, velocity_x, velocity_y, ground_speed, bearing = derive_position_motion(
        samples, Decimal("0.5")
    )

    assert x == pytest.approx(6.0)
    assert y == pytest.approx(3.5)
    assert velocity_x == pytest.approx(2.0)
    assert velocity_y == pytest.approx(-1.0)
    assert ground_speed == pytest.approx(5**0.5)
    assert bearing == pytest.approx(-26.565051, abs=1e-5)


def test_motion_uses_valid_query_positions_despite_an_unrelated_internal_gap() -> None:
    times = [
        Decimal(index) / 100
        for index in range(101)
        if index < 55 or index > 65
    ]
    samples = [
        TimedPosition(timestamp, float(timestamp), -float(timestamp), Path(f"{timestamp}.json"))
        for timestamp in times
    ]

    x, y, velocity_x, velocity_y, speed, _ = derive_position_motion(
        samples, Decimal("0.5")
    )

    assert x == pytest.approx(0.5)
    assert y == pytest.approx(-0.5)
    assert velocity_x == pytest.approx(1.0)
    assert velocity_y == pytest.approx(-1.0)
    assert speed == pytest.approx(2**0.5)


def test_position_trajectory_offsets_align_to_10hz_and_cover_derivative_edges() -> None:
    offsets = _position_trajectory_offsets(Decimal(2), Decimal(6))

    assert len(offsets) == 87
    assert offsets[0] == Decimal("-2.3")
    assert offsets[23] == Decimal(0)
    assert offsets[-1] == Decimal("6.3")
    assert all(
        later - earlier == Decimal("0.1")
        for earlier, later in zip(offsets, offsets[1:])
    )


def test_resample_position_trajectory_marks_grid_points_inside_long_raw_gaps() -> None:
    times = [Decimal(index) / 100 for index in range(101) if index < 55 or index > 65]
    samples = [
        TimedPosition(
            timestamp,
            2.0 * float(timestamp),
            -float(timestamp),
            Path(f"{timestamp}.json"),
        )
        for timestamp in times
    ]
    offsets = _position_trajectory_offsets(
        Decimal("0.2"), Decimal("0.2"), derivative_window_sec=Decimal("0.2")
    )

    trajectory = resample_position_trajectory(samples, Decimal("0.5"), offsets)

    missing = next(sample for sample in trajectory if sample.relative_time == Decimal("0.1"))
    assert missing.target_timestamp == Decimal("0.6")
    assert not missing.position_valid
    assert "exceeds" in (missing.position_error or "")
    assert trajectory[3].relative_time == Decimal("0.0")
    assert trajectory[3].position_valid


def test_motion_can_use_10hz_positions_around_a_missing_internal_grid_point() -> None:
    times = [Decimal(index) / 100 for index in range(101) if index < 60 or index > 66]
    raw = [
        TimedPosition(
            timestamp,
            2.0 * float(timestamp),
            -float(timestamp),
            Path(f"{timestamp}.json"),
        )
        for timestamp in times
    ]
    grid_offsets = _position_trajectory_offsets(
        Decimal("0.2"), Decimal("0.2"), derivative_window_sec=Decimal("0.2")
    )
    grid = resample_position_trajectory(raw, Decimal("0.5"), grid_offsets)
    # Use the default 0.5-second difference window to make the missing +0.1s grid point
    # an unrelated interior gap for the t=0 estimate.
    wide_grid_offsets = _position_trajectory_offsets(Decimal("0.3"), Decimal("0.3"))
    wide_grid = resample_position_trajectory(raw, Decimal("0.5"), wide_grid_offsets)
    motion = build_position_motion_samples(
        raw,
        Decimal("0.5"),
        [Decimal("0.0")],
        resampled_positions=wide_grid,
    )

    assert any(not sample.position_valid for sample in grid)
    assert len(motion) == 1
    assert motion[0].position_valid
    assert motion[0].motion_valid
    assert motion[0].ground_speed == pytest.approx(5**0.5)


def test_position_motion_offsets_match_coc_vector_time_grid() -> None:
    offsets = _position_motion_offsets(Decimal(2), Decimal(6), Decimal("0.5"))

    assert len(offsets) == 17
    assert offsets[0] == Decimal("-2.0")
    assert offsets[4] == Decimal("0.0")
    assert offsets[5] == Decimal("0.5")
    assert offsets[-1] == Decimal("6.0")


def test_write_position_motion_audit_writes_json_and_equal_scale_plot(tmp_path: Path) -> None:
    sample = PositionMotionSample(
        relative_time=Decimal("0.0"),
        target_timestamp=Decimal("100.0"),
        position_valid=True,
        relative_x=0.0,
        relative_y=0.0,
        position_error=None,
        motion_valid=True,
        velocity_x=2.0,
        velocity_y=-1.0,
        ground_speed=5**0.5,
        trajectory_bearing_deg=-26.565051,
        motion_error=None,
        connected_to_previous=False,
        body_velocity_right=1.0,
        body_velocity_forward=2.0,
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        recorded_linear_velocity_xyz=(2.0, -1.0, 0.0),
        body_velocity_residual_mps=0.0,
    )

    audit_path, plot_path = write_position_motion_audit(
        [sample], tmp_path, "sample_record", Decimal("100.0")
    )

    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    rendered = cv2.imread(str(plot_path), cv2.IMREAD_COLOR)
    assert payload["samples"][0]["velocity_map_xy_mps"] == {
        "x": 2.0,
        "y": -1.0,
    }
    assert payload["samples"][0]["ground_speed_mps"] == pytest.approx(5**0.5)
    assert payload["samples"][0]["v_long_forward_positive_mps"] == pytest.approx(2.0)
    assert payload["samples"][0]["v_lat_right_positive_mps"] == pytest.approx(1.0)
    assert payload["samples"][0]["velocity_body_flu_mps"] == {
        "forward": 2.0,
        "left": -1.0,
    }
    assert payload["samples"][0]["recorded_linear_velocity_xyz_candidate_flu_mps"] == {
        "x": 2.0,
        "y": -1.0,
        "z": 0.0,
    }
    assert payload["samples"][0]["recorded_linear_velocity_as_map_xy_residual_mps"] == pytest.approx(0.0)
    assert "strongly supported" in payload["recorded_linear_velocity_frame_hypothesis"]
    assert "not sent to the agent" in payload["recorded_velocity_audit"]
    assert "front-camera" in payload["coordinate_warning"]
    assert "not producer metadata" in payload["coordinate_warning"]
    assert rendered is not None


def test_write_resampled_position_trajectory_keeps_raw_gap_details(tmp_path: Path) -> None:
    raw = [
        TimedPosition(Decimal("0.0"), 0.0, 0.0, Path("0.0.json")),
        TimedPosition(Decimal("0.01"), 0.01, 0.0, Path("0.01.json")),
        TimedPosition(Decimal("0.05"), 0.05, 0.0, Path("0.05.json")),
        TimedPosition(Decimal("0.06"), 0.06, 0.0, Path("0.06.json")),
    ]
    trajectory = [
        ResampledPosition(
            relative_time=Decimal("0.0"),
            target_timestamp=Decimal("0.01"),
            position_valid=True,
            x=0.01,
            y=0.0,
            position_error=None,
            connected_to_previous=False,
        ),
        ResampledPosition(
            relative_time=Decimal("0.1"),
            target_timestamp=Decimal("0.11"),
            position_valid=True,
            x=0.11,
            y=0.0,
            position_error=None,
            connected_to_previous=False,
        ),
    ]

    audit_path = write_resampled_position_trajectory(
        trajectory, raw, tmp_path, "sample_record", Decimal("0.01")
    )

    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    assert payload["grid_interval_sec"] == pytest.approx(0.1)
    assert len(payload["raw_sampling_gaps_over_limit"]) == 1
    assert payload["raw_sampling_gaps_over_limit"][0]["gap_ms"] == pytest.approx(40.0)


def test_write_contact_sheet_with_coc_adds_caption_and_preserves_original(
    tmp_path: Path,
) -> None:
    contact_sheet_path = tmp_path / "contact_sheet.jpg"
    result_yaml_path = tmp_path / "cot_result.yaml"
    output_path = tmp_path / "contact_sheet_with_coc.jpg"
    assert cv2.imwrite(str(contact_sheet_path), np.full((120, 320, 3), 80, dtype=np.uint8))
    original_bytes = contact_sheet_path.read_bytes()
    result_yaml_path.write_text(
        yaml.safe_dump(
            {
                "final_content": {
                    "ego_behavior_schema": {
                        "effect_on_ego_behavior": "Proceed straight while maintaining a safe speed."
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result = write_contact_sheet_with_coc(contact_sheet_path, result_yaml_path, output_path)

    original = cv2.imread(str(contact_sheet_path), cv2.IMREAD_COLOR)
    rendered = cv2.imread(str(result), cv2.IMREAD_COLOR)
    assert output_path.is_file()
    assert contact_sheet_path.read_bytes() == original_bytes
    assert rendered.shape[1] == original.shape[1]
    assert rendered.shape[0] > original.shape[0]


def test_build_agent_data_matches_single_pass_coc_fields() -> None:
    selected = [
        SelectedFrame(
            relative_time=Decimal(index) / 2 - 2,
            target_timestamp=Decimal(2000) + Decimal("10.25") + Decimal(index) / 2 - 2,
            image=TimedImage(
                Path(f"{index}.jpg"),
                Decimal(2000) + Decimal("10.25") + Decimal(index) / 2 - 2,
            ),
            error_sec=Decimal(0),
        )
        for index in range(16)
    ]
    decoded = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in selected]

    data = build_agent_data("sample_record", Decimal("10.25"), selected, decoded)

    assert data["clip_id"] == "sample_record"
    assert data["event_start_timestamp"] == 2_010_250_000
    assert data["event_start_frame"] == 102
    assert len(data["hist_fpv_frames"]) == 4
    assert len(data["fut_fpv_frames"]) == 12
    assert len(data["all_fpv_frames_info"]) == 16
    assert data["all_meta_action_str"].count("Not Available") == 48
    assert "v_long (forward-positive): Not Available" in data["ego_text"]


def test_build_agent_data_adds_position_ground_speed_and_local_prompt() -> None:
    selected = [
        SelectedFrame(
            relative_time=Decimal(index) / 2 - 2,
            target_timestamp=Decimal(2000) + Decimal("10.25") + Decimal(index) / 2 - 2,
            image=TimedImage(
                Path(f"{index}.jpg"),
                Decimal(2000) + Decimal("10.25") + Decimal(index) / 2 - 2,
            ),
            error_sec=Decimal(0),
        )
        for index in range(16)
    ]
    decoded = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in selected]
    offsets = _position_motion_offsets(Decimal(2), Decimal(6), Decimal("0.5"))
    position_motion = [
        PositionMotionSample(
            relative_time=offset,
            target_timestamp=Decimal("2010.25") + offset,
            position_valid=True,
            relative_x=float(index),
            relative_y=0.0,
            position_error=None,
            motion_valid=index != 4,
            velocity_x=2.0 if index != 4 else None,
            velocity_y=0.0 if index != 4 else None,
            ground_speed=2.0 if index != 4 else None,
            trajectory_bearing_deg=0.0 if index != 4 else None,
            motion_error="sample gap" if index == 4 else None,
            connected_to_previous=index > 0,
            body_velocity_right=0.25 if index != 4 else None,
            body_velocity_forward=2.0 if index != 4 else None,
            body_motion_error="sample gap" if index == 4 else None,
        )
        for index, offset in enumerate(offsets)
    ]

    data = build_agent_data(
        "sample_record",
        Decimal("10.25"),
        selected,
        decoded,
        position_motion=position_motion,
    )

    assert data["ego_text"].count("v_ground=") == 17
    assert data["ego_text"].count("v_long=") == 17
    assert data["ego_text"].count("v_lat=") == 17
    assert "(t=+0.0s, v_long=Not Available, v_lat=Not Available" in data["ego_text"]
    assert "2.000 m/s" in data["ego_text"]
    assert "v_lat is right-positive" in data["ego_text"]
    assert data["ego_state_prompt"] == LOCAL_POSITION_EGO_STATE_PROMPT
    assert data["all_meta_action_str"].count("Not Available") == 48


def test_coc_agent_uses_optional_local_ego_prompt_without_changing_default() -> None:
    agent = object.__new__(VLMCoCAgent)
    agent.data_loader_config = None
    prompt_module = SimpleNamespace(
        system="system",
        images_prompt="images",
        meta_action_prompt="meta",
        ego_state_prompt="default ego prompt",
        output_prompts="output",
    )
    data = {
        "hist_fpv_frames": ["history"],
        "fut_fpv_frames": ["future"],
        "all_meta_action_str": "meta values",
        "ego_text": "ground speed values",
        "ego_state_prompt": LOCAL_POSITION_EGO_STATE_PROMPT,
    }

    messages = agent._build_input_message_with_prompt(data, prompt_module)
    ego_prompt_message = next(
        message for message in messages if message["content"] == LOCAL_POSITION_EGO_STATE_PROMPT
    )
    assert ego_prompt_message["role"] == "user"
    assert prompt_module.ego_state_prompt == "default ego prompt"

    default_messages = agent._build_input_message_with_prompt(
        {key: value for key, value in data.items() if key != "ego_state_prompt"},
        prompt_module,
    )
    assert any(message["content"] == "default ego prompt" for message in default_messages)


def test_dummy_agent_writes_standard_coc_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(local_record_agent, "_configure_openai_environment", lambda _: None)
    selected = [
        SelectedFrame(
            relative_time=Decimal(index) / 2 - 2,
            target_timestamp=Decimal(2000) + Decimal("10.25") + Decimal(index) / 2 - 2,
            image=TimedImage(
                Path(f"{index}.jpg"),
                Decimal(2000) + Decimal("10.25") + Decimal(index) / 2 - 2,
            ),
            error_sec=Decimal(0),
        )
        for index in range(16)
    ]
    decoded = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in selected]
    data = build_agent_data("sample_record", Decimal("10.25"), selected, decoded)
    _, _, _, data_loader_config = _load_temporal_config()
    expected_output = tmp_path / "labels" / "sample_record" / "cot_2010250000.yaml"
    expected_output.parent.mkdir(parents=True)
    expected_output.write_text("final_content: old result\n", encoding="utf-8")

    output_path = _run_agent(
        "dummy",
        30,
        tmp_path / "unused-api-config.yaml",
        data_loader_config,
        data,
        tmp_path / "labels",
    )

    result = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert result["event_start_frame"] == 102
    assert result["event_start_timestamp"] == 2_010_250_000
    assert result["final_content"] == "Dummy response."
