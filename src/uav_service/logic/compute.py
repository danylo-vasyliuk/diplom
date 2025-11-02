import numpy as np

from uav_service.logic.models import Coordinates, Coordinates3D, Drone
from uav_service.logic.utils import dh_transform


def compute_drone_positions(
    user_coordinates: Coordinates,
    base_coordinates: Coordinates3D,
    drones: list[Drone],
    num_to_use: int = 3,
) -> list[Drone]:
    """
    Use DH parameters to compute drone positions forming a Wi-Fi chain.
    """
    if num_to_use > len(drones):
        raise ValueError(
            "num_drones_to_use cannot exceed total number of drones available"
        )

    # Use the first N drones
    selected = drones[:num_to_use]

    # Compute direction from base to user
    dx, dy = (
        user_coordinates.x - base_coordinates.x,
        user_coordinates.y - base_coordinates.y,
    )
    total_dist = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)

    # Parameters for DH model
    link_length = total_dist / (num_to_use + 1)
    alpha = np.deg2rad(5)  # small upward angle
    d = 10  # vertical offset

    # Start at base
    T = np.eye(4)
    T[:3, 3] = np.array([base_coordinates.x, base_coordinates.y, base_coordinates.z])

    result = []

    for i, drone in enumerate(selected, start=1):
        T_i = dh_transform(theta, d, link_length, alpha)
        T = T @ T_i

        # Combine with small adjustment based on initial drone position
        offset = (
            np.array([drone.coordinates.x, drone.coordinates.y, drone.coordinates.z])
            * 0.03
        )
        pos = T[:3, 3] + offset

        result.append(
            Drone(
                label=drone.label,
                coordinates=Coordinates3D(x=pos[0], y=pos[1], z=pos[2]),
            )
        )

    return result
