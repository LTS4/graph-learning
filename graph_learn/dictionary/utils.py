"""Utility function for dictionary learning"""
import numpy as np
from numpy.random import RandomState
from numpy.typing import NDArray


def mc_activations(
    activations: NDArray[np.float_], mc_samples: int, random_state: RandomState
) -> NDArray[np.float_]:
    """Sample combinations of activations using Monte-Carlo"""
    n_components, n_samples = activations.shape
    combination_map = 2 ** np.arange(n_components)
    arange = np.arange(n_samples)

    out = np.zeros((2**n_components, n_samples), dtype=float)
    for _ in range(mc_samples):
        sampled = combination_map @ (random_state.uniform(size=activations.shape) <= activations)

        out[sampled, arange] += 1

    return out / mc_samples


def powerset_matrix(n_components: int) -> NDArray[np.int_]:
    """Return matrix of all combinations of n_components binary variables

    Args:
        n_components (int): Number of components

    Returns:
        NDArray[np.int_]: Combination matrix of shape (n_components, 2**n_components)
    """
    return np.array(
        [[(j >> i) & 1 for j in range(2**n_components)] for i in range(n_components)]
    )  # shape (n_components, 2**n_components)


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
