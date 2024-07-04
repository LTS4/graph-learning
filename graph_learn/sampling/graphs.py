"""Function to sample random graph laplacians"""
import numpy as np
from numpy.random import Generator, default_rng
from numpy.typing import NDArray

from graph_learn.operators import laplacian_squareform


def sample_uniform_laplacian(n_nodes: int, seed: int | Generator = None) -> NDArray:
    """Create a Laplacian matrix with uniform weights in [0,1]"""
    rng = default_rng(seed)

    return laplacian_squareform(rng.uniform(size=(n_nodes**2 - n_nodes) // 2))


def sample_er_laplacian(
    n_nodes: int,
    edge_p: float,
    edge_w_min: float,
    edge_w_max: float,
    seed: int | Generator = None,
) -> NDArray[np.float64]:
    """Create an  Laplacian matrix with uniform weights in [0,1]"""
    rng = default_rng(seed)

    out = rng.uniform(size=(n_nodes**2 - n_nodes) // 2)
    edges = out < edge_p

    out[edges] = (out[edges] / edge_p) * (edge_w_max - edge_w_min) + edge_w_min
    out[~edges] = 0

    return laplacian_squareform(out)
