# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coordinate and motion conventions shared by loaders and VLM prompts."""

import numpy as np

LATERAL_SPEED_SIGN_DESCRIPTION = (
    "Body-frame lateral velocity is the instantaneous sideways velocity relative to "
    "the vehicle heading: positive means rightward and negative means leftward. "
    "It is not a lane-change direction."
)


def project_velocity_to_longitudinal_lateral(
    velocity: np.ndarray, heading: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project velocity and heading from the same planar frame onto body axes.

    ``heading`` is measured counter-clockwise from the positive x axis, and the
    vehicle's forward unit vector is ``(cos(h), sin(h))``. The returned lateral
    component is right-positive. This works both before and after the repo's
    ``x-forward/y-left`` to ``x-right/y-forward`` axis rotation, provided velocity
    and heading are represented in the same frame:

    ``v_long = vx*cos(h) + vy*sin(h)``
    ``v_lat  = vx*sin(h) - vy*cos(h)``

    The lateral sign is deliberately right-positive for the VLM contract;
    ``trajdata``'s own ``lat_component`` uses the opposite, left-positive sign.

    The function accepts arrays with a final dimension of length two and supports
    scalar or broadcastable heading arrays.
    """
    cos_heading = np.cos(heading)
    sin_heading = np.sin(heading)
    longitudinal_speed = velocity[..., 0] * cos_heading + velocity[..., 1] * sin_heading
    lateral_speed = velocity[..., 0] * sin_heading - velocity[..., 1] * cos_heading
    return longitudinal_speed, lateral_speed
