from math import ceil

import numpy as np

from uav_service.logic.models import Coordinates, Coordinates3D, Drone
from uav_service.logic.utils import dh_transform


def compute_drone_positions(
    user_coordinates: Coordinates,
    base_coordinates: Coordinates3D,
    drones: list[Drone],
    max_drone_spacing: float = 10.0,  # maximum distance between drones
    step_size: float = 1.0,
) -> dict[str, list[Coordinates3D]]:
    """
    Computes DH-based drone steps forming a bridge between base and user.
    Number of drones is automatically calculated based on distance and max_drone_spacing.
    Returns:
        {
            "UAV_1": [Coordinates3D, ...],
            "UAV_2": [...],
            ...
        }
    """

    base = np.array(
        [base_coordinates.x, base_coordinates.y, base_coordinates.z], dtype=float
    )
    user = np.array(
        [user_coordinates.x, user_coordinates.y, base_coordinates.z], dtype=float
    )

    # Calculate total distance
    total_distance = np.linalg.norm(user - base)

    # Calculate number of drones needed
    num_to_use = ceil(total_distance / max_drone_spacing)
    num_to_use = min(num_to_use, len(drones))
    num_to_use = max(1, num_to_use)

    selected = drones[:num_to_use]

    # Compute evenly spaced target positions along the line base → user
    targets = []
    for i in range(1, num_to_use + 1):
        t = i / (num_to_use + 1)
        pos = base * (1 - t) + user * t
        targets.append(pos)

    result: dict[str, list[Coordinates3D]] = {}

    for drone, target in zip(selected, targets):
        start = np.array(
            [drone.coordinates.x, drone.coordinates.y, drone.coordinates.z], dtype=float
        )

        # Compute distance and independent number of steps
        distance = np.linalg.norm(target - start)
        steps_per_drone = max(2, ceil(distance / step_size))

        steps: list[Coordinates3D] = [drone.coordinates]  # step 0 = start

        for step in range(1, steps_per_drone + 1):
            t = step / steps_per_drone

            # Linear interpolation toward the drone's target
            pos = start * (1 - t) + target * t

            # Map XYZ into DH parameters
            a = pos[0]
            theta = np.deg2rad(pos[1]) / 5
            d = pos[2]
            alpha = np.deg2rad(5)

            T = dh_transform(theta, d, a, alpha)
            x, y, z = T[0, 3], T[1, 3], T[2, 3]

            steps.append(Coordinates3D(x=float(x), y=float(y), z=float(z)))

        result[drone.label] = steps

    return result
