from math import ceil

import numpy as np

from uav_service.logic.models import Coordinates, Coordinates3D, Drone
from uav_service.logic.utils import dh_transform


import numpy as np
from math import ceil


def compute_drone_positions(
    user_coordinates: Coordinates,
    base_coordinates: Coordinates3D,
    drones: list[Drone],
    max_drone_spacing: float = 10.0,
    step_size: float = 1.0,
) -> dict[str, list[Coordinates3D]]:
    """
    Computes DH-based drone steps forming a bridge between base and user.
    Drones are assigned to target positions optimally based on distance.
    Number of drones is calculated automatically.
    Returns:
        {
            "UAV_label": [Coordinates3D, ...],
            ...
        }
    """

    base = np.array(
        [base_coordinates.x, base_coordinates.y, base_coordinates.z], dtype=float
    )
    user = np.array(
        [user_coordinates.x, user_coordinates.y, base_coordinates.z], dtype=float
    )

    # Calculate total distance and number of drones needed
    total_distance = np.linalg.norm(user - base)
    num_to_use = ceil(total_distance / max_drone_spacing)
    num_to_use = min(num_to_use, len(drones))
    num_to_use = max(1, num_to_use)

    # Compute evenly spaced target positions along the line
    targets = [
        base * (1 - (i / (num_to_use + 1))) + user * (i / (num_to_use + 1))
        for i in range(1, num_to_use + 1)
    ]

    # Greedy assignment: assign closest drone to each target
    remaining_drones = drones.copy()
    assignments = []

    for target in targets:
        # Find the closest drone to this target
        distances = [
            np.linalg.norm(
                np.array(
                    [d.coordinates.x, d.coordinates.y, d.coordinates.z], dtype=float
                )
                - target
            )
            for d in remaining_drones
        ]
        idx = int(np.argmin(distances))
        drone = remaining_drones.pop(idx)
        assignments.append((drone, target))

    # Compute steps for each assigned drone
    result: dict[str, list[Coordinates3D]] = {}

    for drone, target in assignments:
        start = np.array(
            [drone.coordinates.x, drone.coordinates.y, drone.coordinates.z], dtype=float
        )
        distance = np.linalg.norm(target - start)
        steps_per_drone = max(2, ceil(distance / step_size))

        steps: list[Coordinates3D] = [drone.coordinates]  # step 0 = start

        for step in range(1, steps_per_drone + 1):
            t = step / steps_per_drone
            pos = start * (1 - t) + target * t

            # DH transform
            a = pos[0]
            theta = np.deg2rad(pos[1]) / 5
            d = pos[2]
            alpha = np.deg2rad(5)
            T = dh_transform(theta, d, a, alpha)
            x, y, z = T[0, 3], T[1, 3], T[2, 3]

            steps.append(Coordinates3D(x=float(x), y=float(y), z=float(z)))

        result[drone.label] = steps

    return result
