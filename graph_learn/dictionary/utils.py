"""Utility function for dictionary learning"""

import numpy as np
from numpy.random import RandomState
from numpy.typing import NDArray


def mc_coefficients(
    coefficients: NDArray[np.float64], mc_samples: int, random_state: RandomState
) -> NDArray[np.float64]:
    """Sample combinations of coefficients using Monte-Carlo"""
    n_atoms, n_samples = coefficients.shape
    combination_map = 2 ** np.arange(n_atoms)
    arange = np.arange(n_samples)

    out = np.zeros((2**n_atoms, n_samples), dtype=float)
    for _ in range(mc_samples):
        sampled = combination_map @ (random_state.uniform(size=coefficients.shape) <= coefficients)

        out[sampled, arange] += 1

    return out / mc_samples


def powerset_matrix(n_atoms: int) -> NDArray[np.int64]:
    """Return matrix of all combinations of n_atoms binary variables

    Args:
        n_atoms (int): Number of atoms

    Returns:
        NDArray[np.int64]: Combination matrix of shape (n_atoms, 2**n_atoms)
    """
    return np.array(
        [[(j >> i) & 1 for j in range(2**n_atoms)] for i in range(n_atoms)]
    )  # shape (n_atoms, 2**n_atoms)


def combinations_prob(
    coefficients: NDArray[np.float64], pwset_mat: NDArray[np.int64] = None
) -> NDArray[np.float64]:
    """Compute exact probability of each combination of coefficients"""
    if pwset_mat is None:
        pwset_mat = powerset_matrix(coefficients.shape[0])

    # This is faster that broadcasting
    return np.prod(
        np.where(
            pwset_mat[:, :, np.newaxis],
            coefficients[:, np.newaxis, :],
            1 - coefficients[:, np.newaxis, :],
        ),
        axis=0,
    )
