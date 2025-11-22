from math import ceil

import numpy as np

from uav_service.logic.models import Coordinates, Coordinates3D, Drone
from uav_service.logic.utils import dh_transform


def compute_drone_positions(
    user_coordinates: Coordinates,
    base_coordinates: Coordinates3D,
    drones: list[Drone],
    num_to_use: int,
    step_size: float = 5.0,
) -> dict[str, list[Coordinates3D]]:
    """
    Computes DH-based drone steps.
    Returns:
        {
            "UAV_1": [Coordinates3D, ...],
            "UAV_2": [...],
            ...
        }
    """

    if num_to_use > len(drones):
        raise ValueError("num_to_use cannot exceed available drone count")

    selected = drones[:num_to_use]

    # Destination: x, y from user; z from base height
    target = np.array(
        [
            user_coordinates.x,
            user_coordinates.y,
            base_coordinates.z,
        ],
        dtype=float,
    )

    result: dict[str, list[Coordinates3D]] = {}

    for drone in selected:
        steps: list[Coordinates3D] = [drone.coordinates]

        start = np.array(
            [
                drone.coordinates.x,
                drone.coordinates.y,
                drone.coordinates.z,
            ],
            dtype=float,
        )

        # Compute distance and independent number of steps
        distance = np.linalg.norm(target - start)
        steps_per_drone = max(2, ceil(distance / step_size))

        for step in range(steps_per_drone + 1):
            t = step / steps_per_drone

            # Linear interpolation between start and target
            pos = start * (1 - t) + target * t

            # Map interpolated XYZ into DH parameters
            a = pos[0]
            theta = np.deg2rad(pos[1]) / 5  # scaled rotation
            d = pos[2]
            alpha = np.deg2rad(5)

            T = dh_transform(theta, d, a, alpha)

            # Extract transformed coordinates
            x, y, z = T[0, 3], T[1, 3], T[2, 3]

            steps.append(Coordinates3D(x=x, y=y, z=z))

        result[drone.label] = steps

    return result
