from math import ceil

import numpy as np

from uav_service.logic.models import Coordinates, Coordinates3D, Drone
from uav_service.logic.utils import dh_transform


import numpy as np
from math import ceil


import numpy as np
from math import ceil


import numpy as np
from math import ceil


def compute_drone_positions(
    user_coordinates: Coordinates,
    base_coordinates: Coordinates3D,
    drones: list[Drone],
    max_drone_spacing: float = 7.0,
    step_size: float = 1.0,
) -> dict[str, list[Coordinates3D]]:
    """
    Build a drone bridge between base and user.
    Each drone flies ONLY to its assigned bridge point, not to the user.
    DH transform affects orientation only, not translation.
    """

    base = np.array([base_coordinates.x, base_coordinates.y, base_coordinates.z], float)
    user = np.array([user_coordinates.x, user_coordinates.y, base_coordinates.z], float)

    # --- 1) Compute required number of drones for spacing limit ---
    total_distance = np.linalg.norm(user - base)
    num_required = ceil(total_distance / max_drone_spacing)

    num_to_use = min(max(1, num_required), len(drones))

    # --- 2) Compute bridge points (not including base and user) ---
    targets = []
    for i in range(1, num_to_use + 1):
        t = i / (num_to_use + 1)
        pos = base * (1 - t) + user * t
        targets.append(pos)

    # --- 3) Assign closest drones to closest targets ---
    remaining = drones.copy()
    assignments = []

    for target in targets:
        dists = [
            np.linalg.norm(
                np.array([d.coordinates.x, d.coordinates.y, d.coordinates.z]) - target
            )
            for d in remaining
        ]
        idx = int(np.argmin(dists))
        assignments.append((remaining.pop(idx), target))

    # --- 4) Fly each drone independently to its target ---
    result: dict[str, list[Coordinates3D]] = {}

    for drone, target in assignments:
        start = np.array(
            [drone.coordinates.x, drone.coordinates.y, drone.coordinates.z], float
        )

        distance = np.linalg.norm(target - start)
        steps_per_drone = max(2, ceil(distance / step_size))

        steps: list[Coordinates3D] = [drone.coordinates]

        for step in range(1, steps_per_drone + 1):
            t = step / steps_per_drone
            pos = start * (1 - t) + target * t  # <-- real bridge position

            # DH should NOT move the drone, only rotate
            theta = np.deg2rad(3)  # small rotation
            alpha = np.deg2rad(0)
            a = 0
            d = 0

            T = dh_transform(theta, d, a, alpha)

            # apply small orientation to the position vector
            pos_h = np.array([pos[0], pos[1], pos[2], 1])
            oriented = T @ pos_h

            x = float(oriented[0])
            y = float(oriented[1])
            z = float(pos[2])  # maintain correct height

            steps.append(Coordinates3D(x=x, y=y, z=z))

        result[drone.label] = steps

    return result
