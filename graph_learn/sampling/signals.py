import numpy as np
from numpy.random import RandomState
from numpy.typing import NDArray


def sample_lgmrf(
    laplacian: NDArray[np.float_], n_samples: int, random_state: int | RandomState = None
) -> NDArray[np.float_]:
    """Sample signals from a Laplacian-constrained Gaussian Markov random field.

    Args:
        laplacian (NDArray[np.float_]): Graph Laplacian
        n_samples (int): Number of samples
        random_state (int | RandomState, optional): Random state or seed. Defaults to None.

    Returns:
        NDArray[np.float_]: Sample matrix of shape (n_samples, n_nodes)
    """
    random_state = RandomState(random_state)

    evals, evecs = np.linalg.eigh(laplacian)
    assert np.allclose(evals[evals < 0], 0)
    evals[evals < 0] = 0

    A = evecs.T @ np.diag(np.sqrt(evals)) @ evecs
    # A = np.linalg.cholesky(np.linalg.pinv(laplacian))
    return random_state.standard_normal((n_samples, laplacian.shape[0])) @ A.T
