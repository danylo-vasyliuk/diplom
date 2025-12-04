import math
from math import ceil
from typing import Dict, List, Tuple

import numpy as np

from uav_service.logic.models import Coordinates, Coordinates3D, Drone
from uav_service.logic.utils import dh_transform


def calculate_bridge_targets(
    base: np.ndarray,
    user: np.ndarray,
    max_drone_spacing: float,
    num_available_drones: int,
) -> List[np.ndarray]:
    """
    Calculate optimal target positions along the line from base to user.
    """
    direction_vector = user - base
    total_distance = np.linalg.norm(direction_vector)

    if total_distance < 0.1:
        return []

    num_drones_needed = max(1, ceil(total_distance / max_drone_spacing) - 1)
    num_drones_to_use = min(num_drones_needed, num_available_drones)

    bridge_targets = []
    for i in range(1, num_drones_to_use + 1):
        t = i / (num_drones_to_use + 1)
        x = base[0] + (user[0] - base[0]) * t
        y = base[1] + (user[1] - base[1]) * t
        z = base[2] * (1 - t)  # Ladder from base.z to 0

        bridge_targets.append(np.array([x, y, z], float))

    return bridge_targets


def assign_drones_to_targets(
    drones: List[Drone],
    bridge_targets: List[np.ndarray],
    base: np.ndarray,
) -> List[Tuple[Drone, np.ndarray]]:
    """Assign drones to bridge positions in a reasonable, monotonic way.

    - Sort drones by distance from base.
    - Sort targets by distance from base.
    - Pair i-th drone with i-th target (no crossing, no 'closest to base' flying far).
    """
    if not bridge_targets:
        return []

    # Distance-from-base helpers
    def dist_from_base_pos(pos: np.ndarray) -> float:
        return float(np.linalg.norm(pos - base))

    def dist_from_base_drone(d: Drone) -> float:
        return float(
            np.linalg.norm(
                np.array([d.coordinates.x, d.coordinates.y, d.coordinates.z], float)
                - base
            )
        )

    # Sort drones and targets along base→user direction
    drones_sorted = sorted(drones, key=dist_from_base_drone)
    targets_sorted = sorted(bridge_targets, key=dist_from_base_pos)

    assignments: List[Tuple[Drone, np.ndarray]] = []

    for drone, target in zip(drones_sorted, targets_sorted):
        assignments.append((drone, target))

    return assignments


def calculate_yaw_to_user(position: np.ndarray, user: np.ndarray) -> float:
    """Calculate yaw (in radians) to face the user in the XY plane."""
    dx = user[0] - position[0]
    dy = user[1] - position[1]

    if abs(dx) < 0.001 and abs(dy) < 0.001:
        return 0.0

    return math.atan2(dy, dx)


def generate_simple_trajectory(
    start: np.ndarray, target: np.ndarray, user: np.ndarray, step_size: float
) -> List[Coordinates3D]:
    """
    Straight-line movement in XYZ.
    Yaw is recomputed at each step to face the user, so you can see the rotation.
    """
    movement_vector = target - start
    total_distance = np.linalg.norm(movement_vector)

    if total_distance < 0.001:
        final_yaw = calculate_yaw_to_user(target, user)
        return [
            Coordinates3D(
                x=float(target[0]),
                y=float(target[1]),
                z=float(target[2]),
                yaw=math.degrees(final_yaw),
            )
        ]

    steps_count = max(2, ceil(total_distance / step_size))
    step_distance = total_distance / (steps_count - 1)

    direction = movement_vector / total_distance

    trajectory: List[Coordinates3D] = []

    for step_idx in range(steps_count):
        distance_along = step_idx * step_distance
        position = start + direction * distance_along

        yaw = calculate_yaw_to_user(position, user)

        trajectory.append(
            Coordinates3D(
                x=float(position[0]),
                y=float(position[1]),
                z=float(position[2]),
                yaw=math.degrees(yaw),
            )
        )

    # Ensure exact target on the last step
    trajectory[-1] = Coordinates3D(
        x=float(target[0]),
        y=float(target[1]),
        z=float(target[2]),
        yaw=math.degrees(calculate_yaw_to_user(target, user)),
    )

    return trajectory


def generate_dh_trajectory_simple(
    start: np.ndarray, target: np.ndarray, user: np.ndarray, step_size: float
) -> List[Coordinates3D]:
    """
    DH-based trajectory along a straight line from start to target.

    We model the motion as a virtual 'arm' growing from the start point:
      - theta = yaw of the movement in XY
      - a     = horizontal projection of distance (along X')
      - d     = vertical offset (along Z)

    For step k we use a = k * step_forward, d = k * step_vertical
    and compute T_k = T_start @ DH(theta, d, a, alpha=0).

    Yaw is computed at each step toward the user to show rotation of the 'signal'.
    """
    movement_vector = target - start
    total_distance = np.linalg.norm(movement_vector)

    if total_distance < 0.001:
        final_yaw = calculate_yaw_to_user(target, user)
        return [
            Coordinates3D(
                x=float(target[0]),
                y=float(target[1]),
                z=float(target[2]),
                yaw=math.degrees(final_yaw),
            )
        ]

    steps_count = max(2, ceil(total_distance / step_size))

    dx, dy, dz = movement_vector
    xy_distance = math.sqrt(dx * dx + dy * dy)

    # Movement direction in XY (yaw) and vertical pitch
    move_yaw = math.atan2(dy, dx) if xy_distance > 0.001 else 0.0
    pitch = math.atan2(dz, xy_distance) if xy_distance > 0.001 else 0.0

    step_distance = total_distance / (steps_count - 1)
    step_forward = step_distance * math.cos(pitch)
    step_vertical = step_distance * math.sin(pitch)

    trajectory: List[Coordinates3D] = []

    # Start transform
    T_start = np.eye(4)
    T_start[:3, 3] = start.copy()

    for k in range(steps_count):
        # Distance from start at this step
        a_k = step_forward * k
        d_k = step_vertical * k

        # DH transform from local frame to step k
        T_rel = dh_transform(theta=move_yaw, d=d_k, a=a_k, alpha=0.0)

        # Absolute transform
        T = T_start @ T_rel
        x, y, z = T[:3, 3]

        yaw = calculate_yaw_to_user(np.array([x, y, z], float), user)

        trajectory.append(
            Coordinates3D(
                x=float(x),
                y=float(y),
                z=float(z),
                yaw=math.degrees(yaw),
            )
        )

    # Ensure exact target on the last step
    trajectory[-1] = Coordinates3D(
        x=float(target[0]),
        y=float(target[1]),
        z=float(target[2]),
        yaw=math.degrees(calculate_yaw_to_user(target, user)),
    )

    return trajectory


def compute_drone_positions(
    user_coordinates: Coordinates,
    base_coordinates: Coordinates3D,
    drones: List[Drone],
    max_drone_spacing: float = 7.0,
    step_size: float = 1.0,
    use_dh_transform: bool = True,
) -> Dict[str, List[Coordinates3D]]:
    """
    Main function to compute drone bridge positions.
    """
    if not drones:
        return {}

    base = np.array([base_coordinates.x, base_coordinates.y, base_coordinates.z], float)
    user = np.array([user_coordinates.x, user_coordinates.y, 0.0], float)

    bridge_targets = calculate_bridge_targets(
        base=base,
        user=user,
        max_drone_spacing=max_drone_spacing,
        num_available_drones=len(drones),
    )

    # No bridge needed – just one relay in the middle
    if not bridge_targets:
        midpoint = (base + user) / 2
        closest_drone = min(
            drones,
            key=lambda d: np.linalg.norm(
                np.array([d.coordinates.x, d.coordinates.y, d.coordinates.z], float)
                - midpoint
            ),
        )

        final_yaw = calculate_yaw_to_user(midpoint, user)
        return {
            closest_drone.label: [
                Coordinates3D(
                    x=float(midpoint[0]),
                    y=float(midpoint[1]),
                    z=float(midpoint[2]),
                    yaw=math.degrees(final_yaw),
                )
            ]
        }

    # Assign drones to bridge positions (ordered from base to user)
    assignments = assign_drones_to_targets(drones, bridge_targets, base)

    result: Dict[str, List[Coordinates3D]] = {}

    for drone, target in assignments:
        start = np.array(
            [drone.coordinates.x, drone.coordinates.y, drone.coordinates.z], float
        )

        if use_dh_transform:
            trajectory = generate_dh_trajectory_simple(
                start=start, target=target, user=user, step_size=step_size
            )
        else:
            trajectory = generate_simple_trajectory(
                start=start, target=target, user=user, step_size=step_size
            )

        result[drone.label] = trajectory

    # Unused drones: stay in place, face user
    assigned_labels = {drone.label for drone, _ in assignments}
    for drone in drones:
        if drone.label not in assigned_labels:
            drone_pos = np.array(
                [drone.coordinates.x, drone.coordinates.y, drone.coordinates.z], float
            )
            yaw = calculate_yaw_to_user(drone_pos, user)
            result[drone.label] = [
                Coordinates3D(
                    x=drone.coordinates.x,
                    y=drone.coordinates.y,
                    z=drone.coordinates.z,
                    yaw=math.degrees(yaw),
                )
            ]

    return result


def compute_drone_bridge_positions(
    user_coordinates: Coordinates,
    base_coordinates: Coordinates3D,
    drones: List[Drone],
    max_drone_spacing: float = 7.0,
    step_size: float = 1.0,
) -> Dict[str, List[Coordinates3D]]:
    """
    Version that uses DH internally for movement, with per-step yaw
    from each drone to the user, so you see the 'signal' from base to user.
    """
    return compute_drone_positions(
        user_coordinates=user_coordinates,
        base_coordinates=base_coordinates,
        drones=drones,
        max_drone_spacing=max_drone_spacing,
        step_size=step_size,
        use_dh_transform=True,
    )
