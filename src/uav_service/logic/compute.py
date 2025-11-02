import math
from math import ceil
from typing import Dict, List, Tuple

import numpy as np

from uav_service.logic.models import Coordinates, Coordinates3D, Drone
from uav_service.logic.utils import dh_transform


# ---------- HELPERS ----------
def drone_distance_to_bridge_segment(drone, base, user):
    # Convert everything to proper vectors
    p = np.array([drone.coordinates.x,
                  drone.coordinates.y,
                  drone.coordinates.z], dtype=float)

    a = np.array([base[0], base[1], base[2]], dtype=float)
    b = np.array([user[0], user[1], user[2]], dtype=float)

    ab = b - a
    ab_len_sq = np.dot(ab, ab)

    # Base and user are (almost) the same point
    if ab_len_sq < 1e-12:
        return float(np.linalg.norm(p - a))

    # Projection factor onto segment
    t = np.dot(p - a, ab) / ab_len_sq

    # Clamp strictly to the segment
    t = max(0.0, min(1.0, t))

    closest_point = a + t * ab
    return float(np.linalg.norm(p - closest_point))

# def drone_distance_to_bridge_line(drone, base, user):
#     p = np.array([drone.coordinates.x, drone.coordinates.y, drone.coordinates.z])
#     line = user - base
#     if np.linalg.norm(line) < 1e-6:
#         return 0.0
#     return np.linalg.norm(np.cross(p - base, line)) / np.linalg.norm(line)


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Yaw in DEGREES from point a -> b."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]

    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0

    return math.degrees(math.atan2(dy, dx))


# ---------- TARGET GENERATION ----------

def calculate_bridge_targets(base, user, max_drone_spacing, num_available_drones):
    direction = user - base
    dist = np.linalg.norm(direction)

    if dist < 0.1:
        return []

    num_needed = max(1, ceil(dist / max_drone_spacing) - 1)
    num_use = min(num_needed, num_available_drones)

    targets = []
    for i in range(1, num_use + 1):
        t = i / (num_use + 1)

        x = base[0] + (user[0] - base[0]) * t
        y = base[1] + (user[1] - base[1]) * t
        z = base[2] * (1 - t)

        targets.append(np.array([x, y, z], float))

    return targets


# ---------- DRONE ASSIGNMENT ----------

def assign_drones_to_targets(drones, bridge_targets, base, user):
    """Pick drones closest to line, then sort along trajectory."""
    if not bridge_targets:
        return []

    drones_sorted = sorted(
        drones, key=lambda d: drone_distance_to_bridge_segment(d, base, user)
    )

    selected = drones_sorted[: len(bridge_targets)]

    drones_ordered = sorted(
        selected,
        key=lambda d: np.linalg.norm(
            np.array([d.coordinates.x, d.coordinates.y, d.coordinates.z]) - base
        ),
    )

    targets_ordered = sorted(bridge_targets, key=lambda t: np.linalg.norm(t - base))

    return list(zip(drones_ordered, targets_ordered))


# ---------- DH TRAJECTORY ----------

def generate_dh_trajectory_simple(
        start: np.ndarray,
        target: np.ndarray,
        user: np.ndarray,
        step_size: float,
        initial_yaw_deg: float
):
    movement = target - start
    dist = np.linalg.norm(movement)

    # if not moving
    if dist < 1e-6:
        return [Coordinates3D(
            x=start[0], y=start[1], z=start[2], yaw=initial_yaw_deg
        )]

    steps = max(3, math.ceil(dist / step_size) + 1)

    dx, dy, dz = movement
    xy = math.sqrt(dx*dx + dy*dy)

    move_yaw_rad = math.atan2(dy, dx) if xy > 1e-6 else 0.0
    pitch = math.atan2(dz, xy) if xy > 1e-6 else 0.0

    step_dist = dist / (steps - 1)
    step_forward = step_dist * math.cos(pitch)
    step_vertical = step_dist * math.sin(pitch)

    T0 = np.eye(4)
    T0[:3, 3] = start

    trajectory = []

    for k in range(steps):

        if k == 0:
            pos = start.copy()
            yaw = initial_yaw_deg   # Use payload yaw
        else:
            a = step_forward * k
            d = step_vertical * k

            T_rel = dh_transform(theta=move_yaw_rad, d=d, a=a, alpha=0.0)
            T_step = T0 @ T_rel
            pos = T_step[:3, 3]

            # From step 1 → face user
            yaw = angle_deg(pos, user)

        trajectory.append(Coordinates3D(
            x=float(pos[0]),
            y=float(pos[1]),
            z=float(pos[2]),
            yaw=float(yaw)
        ))

    # Final correction
    trajectory[-1].x = target[0]
    trajectory[-1].y = target[1]
    trajectory[-1].z = target[2]
    trajectory[-1].yaw = angle_deg(target, user)

    return trajectory

# ---------- MAIN PIPELINE ----------

def compute_drone_positions(
    user_coordinates,
    base_coordinates,
    drones,
    max_drone_spacing=7.0,
    step_size=5.0,
    use_dh_transform=True,
):

    if not drones:
        return {}

    base = np.array([base_coordinates.x, base_coordinates.y, base_coordinates.z], float)
    user = np.array([user_coordinates.x, user_coordinates.y, 0.0], float)

    bridge_targets = calculate_bridge_targets(base, user, max_drone_spacing, len(drones))
    assignments = assign_drones_to_targets(drones, bridge_targets, base, user)

    result = {}

    for drone, target in assignments:
        start = np.array(
            [drone.coordinates.x, drone.coordinates.y, drone.coordinates.z], float
        )

        # yaw is inside coordinates, correct
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
):
    return compute_drone_positions(
        user_coordinates=user_coordinates,
        base_coordinates=base_coordinates,
        drones=drones,
        max_drone_spacing=max_drone_spacing,
        step_size=step_size,
        use_dh_transform=True,
    )
