"""Operators for Laplacians and coefficients manipulation"""

import numpy as np
from numpy.linalg import eigh
from numpy.typing import NDArray
from scipy.sparse.linalg import svds
from scipy.spatial.distance import squareform


def relaxed_update(
    *pairs: list[tuple[NDArray[np.float64], NDArray[np.float64]]],
    relaxation: float,
) -> tuple[list[NDArray], NDArray]:
    """Compute relaxed update for each `(var, var1)` in `pairs`:
    `new_var = relaxation * var1 + (1 - relaxation) * var`

    Args:
        pairs (list[tuple[NDArray[np.float64], NDArray[np.float64]]]): Pairs of variables
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


def square_to_vec(x: NDArray[np.float64]) -> NDArray[np.float64]:
    triu = np.triu_indices(x.shape[-1], k=1)
    return x[:, triu[0], triu[1]]


def laplacian_squareform(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Get Laplacians from vectorized edge weights

    Args:
        x (NDArray[np.float]): Array of vectorized edge weights of shape (n_edges,)

    Returns:
        NDArray[np.float64]: Laplacian array of shape (n_nodes, n_nodes)
    """
    lapl = -squareform(x)
    np.fill_diagonal(lapl, -lapl.sum(axis=-1))
    return lapl


def laplacian_squareform_vec(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Tensor form of laplacian squareform"""
    n_nodes = int(np.sqrt(2 * x.shape[-1] + 0.25) + 0.5)
    triu = np.triu_indices(n_nodes, k=1)

    laplacians = np.zeros((x.shape[0], n_nodes, n_nodes), dtype=x.dtype)
    laplacians[:, triu[0], triu[1]] = -x
    laplacians += np.transpose(laplacians, (0, 2, 1))

    arange = np.arange(n_nodes)
    laplacians[:, arange, arange] = -laplacians.sum(axis=-1)

    return laplacians


def laplacian_squareform_adj(laplacian: NDArray[np.float64]) -> NDArray[np.float64]:
    """Adjoint of laplacian squareform"""
    # out = -2 * laplacian
    # neg_degs = np.sum(laplacian - np.diag(np.diag(laplacian)), axis=0)
    # return squareform(out - neg_degs[:, np.newaxis] - neg_degs[np.newaxis, :], checks=False)
    diag = np.diag(laplacian)
    return squareform(
        -laplacian - laplacian.T + diag[:, np.newaxis] + diag[np.newaxis, :],
        checks=False,
    )


def laplacian_squareform_adj_vec(laplacians: NDArray[np.float64]) -> NDArray[np.float64]:
    """Tensor form of adjoint of laplacian squareform"""
    diags = np.diagonal(laplacians, axis1=-2, axis2=-1)
    laplacians = (
        -laplacians
        - np.transpose(laplacians, (0, 2, 1))
        + diags[:, :, np.newaxis]
        + diags[:, np.newaxis, :]
    )
    return square_to_vec(laplacians)


def prox_gdet_star_spectral_update(
    sigma: float, eigvals: NDArray, eigvecs: NDArray = None
) -> NDArray | tuple[NDArray, NDArray]:
    """
    Compute the proximal step for updating eigenvalues in the dual operator of pseudo-determinant.

    Args:
        sigma (float): The regularization parameter.
        eigvals (NDArray): The eigenvalues of the matrix.
        eigvecs (NDArray, optional): The eigenvectors of the matrix. Defaults to None.

    Returns:
        NDArray or tuple[NDArray, NDArray]: The updated eigenvalues and eigenvectors (if provided).

    """
    dim = eigvals.shape[1]

    # Input shall be SPD, so negative values come from numerical errors
    # FIXME: I don't think the input is always SPD
    # eigvals[eigvals < 0] = 0
    zeros = np.isclose(eigvals, 0)

    # degenerate_idx = np.isclose(eigvals.sum(axis=1), 0)
    degenerate_idx = np.all(zeros, axis=1)

    # Proximal step

    # Generalized update
    eigvals = ~zeros * (eigvals - np.sqrt(eigvals**2 + 4 * sigma)) / 2

    if degenerate_idx.any():
        try:
            eigvals[degenerate_idx, :] = -np.sqrt(sigma[degenerate_idx])
        except (TypeError, IndexError):
            eigvals[degenerate_idx, :] = -np.sqrt(sigma)

        eigvals[degenerate_idx, 0] = 0

        if eigvecs is not None:
            # # identify const eigenvector and set its eigval to zero again
            eigvecs[degenerate_idx] -= np.diag(np.ones(dim - 1), 1)[np.newaxis, ...]
            eigvecs[degenerate_idx] /= np.sqrt(2)

            eigvecs[degenerate_idx, :, 0] = np.ones(dim) / np.sqrt(dim)

    if eigvecs is None:
        return eigvals
    else:
        return eigvals, eigvecs


def prox_gdet_star(
    dvar: NDArray[np.float64], sigma: float, return_eigvals: bool = False
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Proximal operator of the Moreau's transpose of negative generalized log-determinant

    Args:
        dvar (NDArray[np.float64]): Stack of SPD matrices, of shape (k, n, n)
        sigma (float): Proximal scale

    Returns:
        NDArray[np.float64]: Proximal point
    """
    _shape = dvar.shape
    # I should identify unique Laplacians, to speed-up computations
    eigvals, eigvecs = eigh(dvar)
    # _eigvals = eigvals.copy()

    eigvals, eigvecs = prox_gdet_star_spectral_update(sigma, eigvals, eigvecs)

    dvar = np.matmul(eigvecs * eigvals[:, np.newaxis, :], np.transpose(eigvecs, (0, 2, 1)))
    assert dvar.shape == _shape

    if return_eigvals:
        return dvar, eigvals

    return dvar


def op_adj_dual(dualv: NDArray) -> NDArray:
    diags = np.diagonal(dualv, axis1=-1, axis2=-2)
    return square_to_vec(
        diags[:, :, np.newaxis] + diags[:, np.newaxis, :] - dualv - dualv.transpose((0, 2, 1))
    )


def op_adj_weights(coefficients: NDArray, dualv: NDArray) -> NDArray:
    """Compute the adjoint of the bilinear inst-Laplacian operator wrt weights

    Args:
        coefficients (NDArray): Array of coefficients of shape (n_components, n_samples)
        dualv (NDArray): Instantaneous Laplacians, of shape (n_samples, n_nodes, n_nodes)

    Returns:
        NDArray: Dual weights of shape (n_components, n_edges)
    """
    return coefficients @ op_adj_dual(dualv)


def op_weights_norm(coefficients: NDArray, n_nodes: int) -> float:
    """Compute the norm of the inst-Laplacian operator restricted to weights"""
    if 1 in coefficients.shape:
        return np.sqrt(2 * n_nodes * np.sum(coefficients**2))
    return np.sqrt(2 * n_nodes) * svds(coefficients, k=1, return_singular_vectors=False)[0]


def op_adj_coefficients(weights: NDArray, dualv: NDArray) -> NDArray:
    """Compute the adjoint of the bilinear inst-Laplacian operator wrt coefficients

    Args:
        weights (NDArray): Array of weights of shape (n_components, n_edges)
        dualv (NDArray): Instantaneous Laplacians, of shape (n_samples, n_nodes, n_nodes)

    Returns:
        NDArray: Adjoint coefficients of shape (n_components, n_samples)
    """
    # return np.tensordot(laplacian_squareform_vec(weights), dualv, axes=((-2, -1), (-2, -1)))
    return weights @ op_adj_dual(dualv).T


def op_coefficients_norm(lapl: NDArray) -> float:
    """Compute the norm of the inst-Laplacian operator restricted to coefficients"""
    if lapl.shape[0] == 1:
        return np.sqrt(np.sum(lapl**2))
    return svds(lapl.reshape(lapl.shape[0], -1), k=1, return_singular_vectors=False)[0]


def squared_pdiffs(x: NDArray) -> NDArray:
    """Compute scaled nodewise differences, as used in graph ditionary smoothness"""
    return square_to_vec(x[:, :, np.newaxis] - x[:, np.newaxis, :]) ** 2


def autocorr(x: NDArray) -> NDArray:
    """Compute the autocorrelation (n_dim, n_dim) of a sample matrix of shape (n_samples, n_dim)"""
    n = x.shape[0]
    x = x - x.mean(0, keepdims=True)
    return x.T @ x / (n - 1)


def dictionary_smoothness(coeffs: NDArray, weights: NDArray, signals: NDArray):
    return np.sum((coeffs.T @ weights) * squared_pdiffs(signals))


def simplex_projection(x: NDArray):
    """Project rows of x onto the unitary simplex.

    Algorithm from Condat, 2014.

    Args:
        x (NDArray): Array of shape (n_samples, n_dim)
    """
    out = -np.sort(-x, axis=1)
    partials = (np.cumsum(out, axis=1) - 1) / np.arange(1, x.shape[1] + 1)[np.newaxis, :]
    tokeep = np.sum(partials < out, axis=1) - 1
    out = x - partials[np.arange(x.shape[0]), tokeep, np.newaxis]
    out[out < 0] = 0
    return out
