"""
Control Barrier Function safety filter (part of Algorithm 3)
"""

import numpy as np
from scipy.optimize import minimize


def safety_filter(
    desired_action: np.ndarray,
    state: np.ndarray,
    alpha_h: float = 0.5,
    bounds=None,
) -> np.ndarray:
    """
    Project an actor action onto the safe set defined by a CBF.
    This is a simplified quadratic program placeholder.
    """
    def objective(u):
        return np.sum((u - desired_action) ** 2)

    # Simplified linear constraint for demonstration
    constraints = []
    if bounds is None:
        bounds = [(0.0, 1.0)] * len(desired_action)

    res = minimize(
        objective,
        desired_action,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
    )
    return res.x if res.success else desired_action
