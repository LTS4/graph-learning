"""Optimization functions for PDS in Compponets model"""
import numpy as np
from numpy.linalg import eigh
from numpy.typing import NDArray
from scipy.spatial.distance import squareform

from graph_learn.utils import laplacian_squareform


def relaxed_update(
    *pairs: list[tuple[NDArray[np.float_], NDArray[np.float_]]],
    relaxation: float,
) -> tuple[list[NDArray], NDArray]:
    """Compute relaxed update for each `(var, var1)` in `pairs`:
    `new_var = relaxation * var1 + (1 - relaxation) * var`

    Args:
        pairs (list[tuple[NDArray[np.float_], NDArray[np.float_]]]): Pairs of variables
        relaxation (float): Parameter in (0,2).

    Returns:
        tuple[list[NDArray], NDArray]: Update variables
    """
    # Denominators are previous iteration ones
    out = []
    rel_norms = np.empty(len(pairs), dtype=float)
    for i, (var, var1) in enumerate(pairs):
        out.append(relaxation * var1 + (1 - relaxation) * var)

        var_norm = np.linalg.norm(var.ravel())
        if var_norm > 0:
            rel_norms[i] = relaxation * np.linalg.norm((var1 - var).ravel()) / var_norm
        else:
            rel_norms[i] = np.inf

    return out, rel_norms


laplacian_squareform_vec = np.vectorize(
    laplacian_squareform, otypes=[float], signature="(e)->(n,n)"
)


def laplacian_squareform_adj(laplacian: NDArray[np.float_]) -> NDArray[np.float_]:
    """Adjoint of laplacian squareform"""
    out = -2 * laplacian
    neg_degs = np.sum(laplacian - np.diag(np.diag(laplacian)), axis=0)
    return squareform(out - neg_degs[:, np.newaxis] - neg_degs[np.newaxis, :], checks=False)


laplacian_squareform_adj_vec = np.vectorize(
    laplacian_squareform_adj, otypes=[float], signature="(n,n)->(e)"
)


def prox_gdet_star(dvar: NDArray[np.float_], sigma: float) -> NDArray[np.float_]:
    """Proximal operator of the Moreau's transpose of negative generalized log-determinant

    Args:
        dvar (NDArray[np.float_]): Stack of SPD matrices, of shape (k, n, n)
        sigma (float): Proximal scale

    Returns:
        NDArray[np.float_]: Proximal point
    """
    _shape = dvar.shape
    # I have to identify the Laplacians which are unique, to speed-up computations
    eigvals, eigvecs = eigh(dvar)

    # Input shall be SPD, so negative values come from numerical erros
    eigvals[eigvals < 0] = 0

    # Proximal step
    eigvals = (eigvals - np.sqrt(eigvals**2 + 4 * sigma)) / 2
    dvar = np.stack(
        [eigvec @ np.diag(eigval) @ eigvec.T for eigval, eigvec in zip(eigvals, eigvecs)]
    )
    assert dvar.shape == _shape

    # Remove constant eignevector. Initial eigval was 0, with prox step is  -np.sqrt(sigma)
    # Note that the norm of the eigenvector is sqrt(n_nodes)
    return dvar + np.sqrt(sigma) / _shape[1]