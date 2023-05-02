"""Utility functions for Laplacians manipulation"""
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import squareform


def laplacian_squareform(x: NDArray[np.float_]) -> NDArray[np.float_]:
    """Get Laplacians from vetorized edge weights

    Args:
        x (NDArray[np.float]): Array of vectorized edge weights of shape (n_edges,)

    Returns:
        NDArray[np.float_]: Laplacian array of shape (n_nodes, n_nodes)
    """
    lapl = -squareform(x)
    np.fill_diagonal(lapl, -lapl.sum(axis=-1))
    return lapl
