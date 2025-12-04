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
    drones: List[Drone], bridge_targets: List[np.ndarray]
) -> List[Tuple[Drone, np.ndarray]]:
    """Assign drones to nearest bridge positions."""
    if not bridge_targets:
        return []

    # Sort drones by distance from base (so drones closer to base get assigned first)
    drones_sorted = sorted(
        drones,
        key=lambda d: np.linalg.norm(
            np.array([d.coordinates.x, d.coordinates.y, d.coordinates.z])
        ),
    )

    # Sort targets by distance from base
    bridge_targets.sort(key=lambda p: np.linalg.norm(p))

    assignments = []
    available_targets = bridge_targets.copy()

    for drone in drones_sorted:
        if not available_targets:
            break

        drone_pos = np.array(
            [drone.coordinates.x, drone.coordinates.y, drone.coordinates.z]
        )

        # Find closest available target
        min_distance = float("inf")
        best_target_idx = -1

        for target_idx, target in enumerate(available_targets):
            distance = np.linalg.norm(drone_pos - target)
            if distance < min_distance:
                min_distance = distance
                best_target_idx = target_idx

        if best_target_idx >= 0:
            assignments.append((drone, available_targets.pop(best_target_idx)))

    return assignments


def calculate_yaw_to_user(position: np.ndarray, user: np.ndarray) -> float:
    """Calculate yaw to face the user."""
    dx = user[0] - position[0]
    dy = user[1] - position[1]

    if abs(dx) < 0.001 and abs(dy) < 0.001:
        return 0.0

    return math.atan2(dy, dx)


def generate_simple_trajectory(
    start: np.ndarray, target: np.ndarray, user: np.ndarray, step_size: float
) -> List[Coordinates3D]:
    """
    SIMPLE trajectory: Straight line movement, constant yaw facing user from final position.
    This prevents spinning and teleportation.
    """
    movement_vector = target - start
    total_distance = np.linalg.norm(movement_vector)

    if total_distance < 0.001:
        # Already at target
        final_yaw = calculate_yaw_to_user(target, user)
        return [
            Coordinates3D(
                x=float(target[0]),
                y=float(target[1]),
                z=float(target[2]),
                yaw=math.degrees(final_yaw),
            )
        ]

    # Calculate steps - ensure smooth movement
    steps_count = max(2, ceil(total_distance / step_size))

    # Calculate final yaw (from target position to user)
    # ALL steps use this same yaw to prevent spinning
    final_yaw = calculate_yaw_to_user(target, user)

    trajectory = []

    for step_idx in range(steps_count):
        # Linear interpolation
        t = step_idx / (steps_count - 1) if steps_count > 1 else 0
        position = start + movement_vector * t

        # Use constant yaw (facing user from target position)
        trajectory.append(
            Coordinates3D(
                x=float(position[0]),
                y=float(position[1]),
                z=float(position[2]),
                yaw=math.degrees(final_yaw),
            )
        )

    # Ensure exact target at the end
    if trajectory:
        trajectory[-1] = Coordinates3D(
            x=float(target[0]),
            y=float(target[1]),
            z=float(target[2]),
            yaw=math.degrees(final_yaw),
        )

    return trajectory


def generate_dh_trajectory_simple(
    start: np.ndarray, target: np.ndarray, user: np.ndarray, step_size: float
) -> List[Coordinates3D]:
    """
    Simple DH trajectory that doesn't cause spinning.
    Uses DH only for movement, yaw is calculated separately.
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

    # Calculate movement direction once
    dx, dy, dz = movement_vector
    xy_distance = math.sqrt(dx * dx + dy * dy)

    move_yaw = math.atan2(dy, dx) if xy_distance > 0.001 else 0.0
    pitch = math.atan2(dz, xy_distance) if xy_distance > 0.001 else 0.0

    # Calculate step components
    step_distance = total_distance / steps_count
    step_forward = step_distance * math.cos(pitch)
    step_vertical = step_distance * math.sin(pitch)

    # Final yaw (facing user from target)
    final_yaw = calculate_yaw_to_user(target, user)

    # Initialize at start
    T = np.eye(4)
    T[:3, 3] = start.copy()

    trajectory = []

    # Start point with final yaw
    trajectory.append(
        Coordinates3D(
            x=float(start[0]),
            y=float(start[1]),
            z=float(start[2]),
            yaw=math.degrees(final_yaw),
        )
    )

    # Apply DH steps
    for step_idx in range(1, steps_count):
        T_step = dh_transform(
            theta=move_yaw, d=step_vertical, a=step_forward, alpha=0.0
        )

        T = T @ T_step

        x, y, z = T[:3, 3]

        # Use constant final yaw for all points
        trajectory.append(
            Coordinates3D(
                x=float(x), y=float(y), z=float(z), yaw=math.degrees(final_yaw)
            )
        )

    # Ensure exact target
    if trajectory:
        trajectory[-1] = Coordinates3D(
            x=float(target[0]),
            y=float(target[1]),
            z=float(target[2]),
            yaw=math.degrees(final_yaw),
        )

    return trajectory


def compute_drone_positions(
    user_coordinates: Coordinates,
    base_coordinates: Coordinates3D,
    drones: List[Drone],
    max_drone_spacing: float = 7.0,
    step_size: float = 1.0,
    use_dh_transform: bool = False,  # CHANGED TO FALSE BY DEFAULT
) -> Dict[str, List[Coordinates3D]]:
    """
    Main function to compute drone bridge positions.
    """
    if not drones:
        return {}

    # Convert to numpy arrays
    base = np.array([base_coordinates.x, base_coordinates.y, base_coordinates.z], float)
    user = np.array([user_coordinates.x, user_coordinates.y, 0.0], float)

    # Calculate bridge positions
    bridge_targets = calculate_bridge_targets(
        base=base,
        user=user,
        max_drone_spacing=max_drone_spacing,
        num_available_drones=len(drones),
    )

    # If no bridge needed (too close)
    if not bridge_targets:
        # Just use first drone at midpoint
        midpoint = (base + user) / 2
        closest_drone = min(
            drones,
            key=lambda d: np.linalg.norm(
                np.array([d.coordinates.x, d.coordinates.y, d.coordinates.z]) - midpoint
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

    # Assign drones to bridge positions
    assignments = assign_drones_to_targets(drones, bridge_targets)

    result = {}

    # Generate trajectories for assigned drones
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

    # Handle unassigned drones (stay in place, face user)
    assigned_labels = {drone.label for drone, _ in assignments}
    for drone in drones:
        if drone.label not in assigned_labels:
            final_yaw = calculate_yaw_to_user(
                np.array(
                    [drone.coordinates.x, drone.coordinates.y, drone.coordinates.z]
                ),
                user,
            )
            result[drone.label] = [
                Coordinates3D(
                    x=drone.coordinates.x,
                    y=drone.coordinates.y,
                    z=drone.coordinates.z,
                    yaw=math.degrees(final_yaw),
                )
            ]

    return result


# Recommended: Simple version that works
def compute_drone_bridge_positions(
    user_coordinates: Coordinates,
    base_coordinates: Coordinates3D,
    drones: List[Drone],
    max_drone_spacing: float = 7.0,
    step_size: float = 1.0,
) -> Dict[str, List[Coordinates3D]]:
    """
    SIMPLE VERSION - Use this for stable bridge formation.
    Drones move in straight lines to bridge positions, all face user with same yaw.
    """
    return compute_drone_positions(
        user_coordinates=user_coordinates,
        base_coordinates=base_coordinates,
        drones=drones,
        max_drone_spacing=max_drone_spacing,
        step_size=step_size,
        use_dh_transform=True,  # Simple straight lines
    )
