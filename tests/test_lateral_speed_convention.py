# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from coc_labeling.data_loader.motion_conventions import (
    LATERAL_SPEED_SIGN_DESCRIPTION,
    project_velocity_to_longitudinal_lateral,
)
from coc_labeling.data_loader.vector_loader import VectorLoader
from coc_labeling.prompts.vlm.coc import ego_state_prompt


def test_right_positive_projection_in_x_right_y_front_axes() -> None:
    # In the transformed frame, straight ahead is +y, so h=pi/2.
    velocity = np.array([[0.0, 1.0], [-1.0, 0.0], [1.0, 0.0]])
    heading = np.full(3, np.pi / 2)

    longitudinal, lateral = project_velocity_to_longitudinal_lateral(velocity, heading)

    assert longitudinal == pytest.approx([1.0, 0.0, 0.0])
    assert lateral == pytest.approx([0.0, -1.0, 1.0])


def test_projection_uses_right_axis_for_arbitrary_heading() -> None:
    heading = np.array([-0.7, 0.2, 1.1])
    speed = np.array([2.0, 3.0, 4.0])
    forward = np.stack((np.cos(heading), np.sin(heading)), axis=-1)
    right = np.stack((np.sin(heading), -np.cos(heading)), axis=-1)

    forward_long, forward_lat = project_velocity_to_longitudinal_lateral(
        speed[:, None] * forward, heading
    )
    right_long, right_lat = project_velocity_to_longitudinal_lateral(speed[:, None] * right, heading)

    assert forward_long == pytest.approx(speed)
    assert forward_lat == pytest.approx(np.zeros_like(speed))
    assert right_long == pytest.approx(np.zeros_like(speed))
    assert right_lat == pytest.approx(speed)


def test_projection_is_consistent_across_repo_axis_rotation() -> None:
    # trajdata axes are x-forward/y-left; production rotates them to x-right/y-front.
    source_velocity = np.array([[2.0, 0.4], [1.2, -0.8], [-0.5, 1.4]])
    source_heading = np.array([0.3, -0.9, 1.2])
    transformed_velocity = np.stack(
        (-source_velocity[:, 1], source_velocity[:, 0]), axis=-1
    )
    transformed_heading = source_heading + np.pi / 2

    source_long, source_lat = project_velocity_to_longitudinal_lateral(
        source_velocity, source_heading
    )
    transformed_long, transformed_lat = project_velocity_to_longitudinal_lateral(
        transformed_velocity, transformed_heading
    )

    assert transformed_long == pytest.approx(source_long)
    assert transformed_lat == pytest.approx(source_lat)


def test_projection_preserves_speed_norm() -> None:
    velocity = np.array([[1.0, 2.0], [-3.0, 0.5], [0.2, -4.0]])
    heading = np.array([0.0, 0.6, -1.2])

    longitudinal, lateral = project_velocity_to_longitudinal_lateral(velocity, heading)

    assert longitudinal**2 + lateral**2 == pytest.approx(np.sum(velocity**2, axis=-1))


def test_prompt_and_speed_text_share_right_positive_semantics() -> None:
    loader = object.__new__(VectorLoader)
    loader.vector_config = type("Config", (), {"time_interval": 0.5})()
    speed_data = {
        "ego_hist_longitudinal_speed": np.array([1.0]),
        "ego_hist_lateral_speed": np.array([0.2]),
        "ego_fut_longitudinal_speed": np.array([1.0]),
        "ego_fut_lateral_speed": np.array([-0.2]),
    }

    _, _, ego_text = loader.convert_speed_to_text(speed_data)

    assert LATERAL_SPEED_SIGN_DESCRIPTION in ego_text
    assert LATERAL_SPEED_SIGN_DESCRIPTION in ego_state_prompt
