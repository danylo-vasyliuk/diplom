import numpy as np


def dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """Compute individual Denavit–Hartenberg transform."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)

    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0, sa, ca, d],
            [0, 0, 0, 1],
        ]
    )
