"Functions to sample signals on graphs"
import numpy as np
from numpy.random import Generator, default_rng
from numpy.typing import NDArray


def sample_lgmrf(
    laplacian: NDArray[np.float64], n_samples: int, seed: int | Generator = None
) -> NDArray[np.float64]:
    """Sample signals from a Laplacian-constrained Gaussian Markov random field.

    Args:
        laplacian (NDArray[np.float64]): Graph Laplacian
        n_samples (int): Number of samples
        random_state (int | Generator, optional): Random number genrator or seed. Defaults to None.

    Returns:
        NDArray[np.float64]: Sample matrix of shape (n_samples, n_nodes)
    """
    rng = default_rng(seed)

    n_nodes, *_ = laplacian.shape

    return rng.multivariate_normal(
        mean=np.zeros(n_nodes),
        cov=np.linalg.pinv(laplacian),
        size=n_samples,
    )
