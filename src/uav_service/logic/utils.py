import math

import numpy as np


def dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """
    Create a Denavit-Hartenberg transformation matrix.

    Args:
        theta: Rotation about Z axis (radians)
        d: Translation along Z axis
        a: Translation along X axis
        alpha: Rotation about X axis (radians)

    Returns:
        4x4 transformation matrix
    """
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    cos_alpha = math.cos(alpha)
    sin_alpha = math.sin(alpha)

    T = np.array(
        [
            [cos_theta, -sin_theta * cos_alpha, sin_theta * sin_alpha, a * cos_theta],
            [sin_theta, cos_theta * cos_alpha, -cos_theta * sin_alpha, a * sin_theta],
            [0, sin_alpha, cos_alpha, d],
            [0, 0, 0, 1],
        ]
    )

    return T
