"""Function to sample random graph laplacians"""

import numpy as np
from numpy.random import Generator, default_rng
from numpy.typing import NDArray

from graph_learn.operators import laplacian_squareform, laplacian_squareform_vec


def sample_uniform_laplacian(n_nodes: int, seed: int | Generator = None) -> NDArray:
    """Create a Laplacian matrix with uniform weights in [0,1]"""
    rng = default_rng(seed)

    return laplacian_squareform(rng.uniform(size=(n_nodes**2 - n_nodes) // 2))


def sample_er_laplacian(
    n_nodes: int,
    edge_p: float,
    edge_w_min: float,
    edge_w_max: float,
    n_graphs: int = 1,
    *,
    seed: int | Generator = None,
) -> NDArray[np.float64]:
    """Create Laplacian matrices for an Erdos-Renyi(n_nodes, edge_p) model, with uniform weights in
    the given interval.

    Args:
        n_nodes (int): Number of nodes
        edge_p (float): Probability of edge existence
        edge_w_min (float): Minimum edge weight
        edge_w_max (float): Maximum edge weight
        n_graphs (int, optional): Number of graphs to generate. Defaults to 1.
        seed (int | Generator, optional): Random number generator or seed. Defaults to None.`

    Returns:
        NDArray[np.float64]: Laplacian matrix of shape (n_nodes, n_nodes),
            or (n_graphs, n_nodes, n_nodes) if n_graphs > 1
    """
    rng = default_rng(seed)

    w_size = (n_graphs, (n_nodes**2 - n_nodes) // 2)

    return laplacian_squareform_vec(
        (rng.uniform(size=w_size) < edge_p) * rng.uniform(edge_w_min, edge_w_max, size=w_size)
    ).squeeze()
