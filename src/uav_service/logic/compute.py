import math
from math import ceil
from typing import Dict, List, Tuple

import numpy as np

from uav_service.logic.models import Coordinates, Coordinates3D, Drone
from uav_service.logic.utils import dh_transform


def drone_distance_to_bridge_line(drone, base, user):
    p = np.array([drone.coordinates.x, drone.coordinates.y, drone.coordinates.z])
    return np.linalg.norm(np.cross(p - base, p - user)) / np.linalg.norm(user - base)


def angle_deg(a, b):
    """Return yaw angle in DEGREES from point a -> b."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]

    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0

    return math.degrees(math.atan2(dy, dx))


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


def assign_drones_to_targets(drones, bridge_targets, base, user):
    """Choose closest drones to line, sort by distance along trajectory, assign 1-to-1."""

    # Sort by closeness to trajectory
    drones_sorted = sorted(
        drones, key=lambda d: drone_distance_to_bridge_line(d, base, user)
    )

    needed = len(bridge_targets)
    selected = drones_sorted[:needed]

    # Sort drones along the base→user direction
    drones_ordered = sorted(
        selected,
        key=lambda d: np.linalg.norm(
            np.array([d.coordinates.x, d.coordinates.y, d.coordinates.z]) - base
        ),
    )

    # Sort targets along the same direction
    targets_ordered = sorted(bridge_targets, key=lambda t: np.linalg.norm(t - base))

    # Assign i-th drone → i-th target
    return list(zip(drones_ordered, targets_ordered))


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


def generate_dh_trajectory_simple(start, target, user, step_size, initial_yaw_deg):
    movement = target - start
    dist = np.linalg.norm(movement)

    final_yaw_deg = angle_deg(target, user)

    if dist < 0.001:
        return [
            Coordinates3D(
                x=float(start[0]),
                y=float(start[1]),
                z=float(start[2]),
                yaw=final_yaw_deg,
            )
        ]

    steps = max(2, math.ceil(dist / step_size))

    dx, dy, dz = movement
    xy = math.sqrt(dx * dx + dy * dy)

    move_yaw_rad = math.atan2(dy, dx)
    pitch = math.atan2(dz, xy)

    step_dist = dist / (steps - 1)
    step_forward = step_dist * math.cos(pitch)
    step_vertical = step_dist * math.sin(pitch)

    T = np.eye(4)
    T[:3, 3] = start.copy()

    result = []

    for k in range(steps):
        a_k = k * step_forward
        d_k = k * step_vertical

        T_rel = dh_transform(theta=move_yaw_rad, d=d_k, a=a_k, alpha=0.0)
        T_step = T @ T_rel
        x, y, z = T_step[:3, 3]

        yaw = initial_yaw_deg if k == 0 else final_yaw_deg

        result.append(Coordinates3D(x=float(x), y=float(y), z=float(z), yaw=float(yaw)))

    # Ensure exact target
    result[-1].x = float(target[0])
    result[-1].y = float(target[1])
    result[-1].z = float(target[2])
    result[-1].yaw = float(final_yaw_deg)

    return result


def compute_drone_positions(
    user_coordinates,
    base_coordinates,
    drones,
    max_drone_spacing=7.0,
    step_size=5.0,
    use_dh_transform=True,
):

    base = np.array([base_coordinates.x, base_coordinates.y, base_coordinates.z], float)
    user = np.array([user_coordinates.x, user_coordinates.y, 0.0], float)

    # Generate bridge targets
    bridge_targets = calculate_bridge_targets(
        base, user, max_drone_spacing, len(drones)
    )

    # Assign drones
    assignments = assign_drones_to_targets(drones, bridge_targets, base, user)

    result = {}

    for drone, target in assignments:
        start = np.array(
            [drone.coordinates.x, drone.coordinates.y, drone.coordinates.z], float
        )

        initial_yaw_deg = float(drone.coordinates.yaw)

        traj = generate_dh_trajectory_simple(
            start=start,
            target=target,
            user=user,
            step_size=step_size,
            initial_yaw_deg=initial_yaw_deg,
        )

        result[drone.label] = traj

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
