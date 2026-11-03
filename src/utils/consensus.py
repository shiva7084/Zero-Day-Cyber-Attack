"""
Distributed consensus protocol (Algorithm 2)
"""

import numpy as np
from typing import List


def distributed_consensus(
    local_estimates: np.ndarray,
    adjacency: np.ndarray,
    alpha: float = 0.1,
    max_iters: int = 50,
    tol: float = 1e-4,
) -> np.ndarray:
    """
    Run distributed averaging consensus.
    local_estimates: (N, R) array of local resource demand estimates
    adjacency: (N, N) weighted adjacency matrix
    """
    z = local_estimates.copy()
    N = z.shape[0]
    for k in range(max_iters):
        z_new = z.copy()
        for i in range(N):
            neighbors = np.where(adjacency[i] > 0)[0]
            if len(neighbors) == 0:
                continue
            diff = np.sum(
                [adjacency[i, j] * (z[j] - z[i]) for j in neighbors],
                axis=0,
            )
            z_new[i] = z[i] + alpha * diff
        z = z_new
        # Check consensus error
        mean_z = z.mean(axis=0, keepdims=True)
        error = np.mean(np.linalg.norm(z - mean_z, axis=1))
        if error < tol:
            break
    return z
