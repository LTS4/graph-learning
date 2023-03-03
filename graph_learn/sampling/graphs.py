"""Function to sample random graph laplacians"""
import numpy as np
from numpy.random import RandomState
from numpy.typing import NDArray
from scipy.spatial.distance import squareform


def laplacian_squareform(x: NDArray[np.float_]) -> NDArray[np.float_]:
    out = squareform(x)
    np.fill_diagonal(out, -out.sum(axis=-1))
    return -out


def sample_uniform_laplacian(n_nodes: int, random_state: int | RandomState = None) -> NDArray:
    """Create a Laplacian matrix with uniform weights in [0,1]"""
    random_state = RandomState(random_state)

    return laplacian_squareform(random_state.uniform(size=((n_nodes**2 - n_nodes) // 2)))


def sample_er_laplacian(
    n_nodes: int,
    edge_p: float,
    edge_w_min: float,
    edge_w_max: float,
    random_state: int | RandomState = None,
) -> NDArray[np.float_]:
    """Create an  Laplacian matrix with uniform weights in [0,1]"""
    random_state = RandomState(random_state)

    out = random_state.uniform(size=((n_nodes**2 - n_nodes) // 2))
    edges = out < edge_p

    out[edges] = (out[edges] / edge_p) * (edge_w_max - edge_w_min) + edge_w_min
    out[~edges] = 0

    return laplacian_squareform(out)
