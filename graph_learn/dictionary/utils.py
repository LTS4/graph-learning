"""Utility function for dictionary learning"""
import numpy as np
from numpy.random import RandomState
from numpy.typing import NDArray


def mc_activations(
    activations: NDArray[np.float_], mc_samples: int, random_state: RandomState
) -> NDArray[np.float_]:
    """Sample combinations of activations using Monte-Carlo"""
    n_atoms, n_samples = activations.shape
    combination_map = 2 ** np.arange(n_atoms)
    arange = np.arange(n_samples)

    out = np.zeros((2**n_atoms, n_samples), dtype=float)
    for _ in range(mc_samples):
        sampled = combination_map @ (random_state.uniform(size=activations.shape) <= activations)

        out[sampled, arange] += 1

    return out / mc_samples


def powerset_matrix(n_atoms: int) -> NDArray[np.int_]:
    """Return matrix of all combinations of n_atoms binary variables

    Args:
        n_atoms (int): Number of components

    Returns:
        NDArray[np.int_]: Combination matrix of shape (n_atoms, 2**n_atoms)
    """
    return np.array(
        [[(j >> i) & 1 for j in range(2**n_atoms)] for i in range(n_atoms)]
    )  # shape (n_atoms, 2**n_atoms)


def combinations_prob(
    activations: NDArray[np.float_], pwset_mat: NDArray[np.int_] = None
) -> NDArray[np.float_]:
    """Compute exact probability of each combination of activations"""
    if pwset_mat is None:
        pwset_mat = powerset_matrix(activations.shape[0])

    return np.stack(
        [
            np.prod(activations[combi], axis=0) * np.prod(1 - activations[~combi], axis=0)
            for combi in pwset_mat.astype(bool).T
        ]
    )
