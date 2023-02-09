"""Graph components learning original method"""
from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.random import RandomState
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator


def primal_dual_splitting(
    p_var: NDArray[np.float_],
    d_var: NDArray[np.float_],
    tau: float,
    sigma: float,
    rho: float,
    lin_op: Callable[[NDArray], NDArray],
    dual_op: Callable[[NDArray], NDArray],
    prox_g: Callable[[NDArray, float], NDArray],
    prox_h: Callable[[NDArray, float], NDArray],
    max_iter: int = 100,
    tol: int = 1e-3,
) -> tuple[NDArray, NDArray]:
    r"""PDS algorithm for problems of the form
    .. math::
        argmin g(\mathbf{x}) + h(\mathbf{K x})

    Args:
        p_var (NDArray[np.float_]): _description_
        d_var (NDArray[np.float_]): _description_
        tau (float): _description_
        sigma (float): _description_
        rho (float): _description_
        lin_op (Callable[[NDArray], NDArray]): _description_
        dual_op (Callable[[NDArray], NDArray]): _description_
        prox_prim (Callable[[NDArray, float], NDArray]): _description_
        prox_dual (Callable[[NDArray, float], NDArray]): _description_
        max_iter (int, optional): _description_. Defaults to 100.
        tol (int, optional): _description_. Defaults to 1e-3.

    Returns:
        tuple[NDArray, NDArray]: _description_
    """
    for i in range(max_iter):
        p_var1 = prox_g(p_var - tau * dual_op(d_var), tau)
        d_var1 = prox_h(d_var + sigma * lin_op(2 * p_var1 - p_var))

        if i > 0:
            # Denominators are previous iteration ones
            rel_norm_primal = (
                (1 - rho) * np.linalg.norm((p_var1 - p_var).ravel()) / np.linalg.norm(p_var.ravel())
            )
            rel_norm_dual = (
                (1 - rho) * np.linalg.norm((d_var1 - d_var).ravel()) / np.linalg.norm(d_var.ravel())
            )
        else:
            rel_norm_primal = rel_norm_dual = np.inf

        p_var = rho * p_var + (1 - rho) * p_var1
        d_var = rho * d_var + (1 - rho) * d_var1

        if rel_norm_primal < tol and rel_norm_dual < tol:
            break

    return p_var, d_var


def prox_gdet(dvar: NDArray[np.float_], sigma: float) -> NDArray[np.float_]:
    """Proximal operator of the generalized determinant

    Args:
        dvar (NDArray[np.float_]): Stack of SPD matrices, of shape (k, n, n)
        sigma (float): Proximal scale

    Returns:
        NDArray[np.float_]: Proximal point
    """
    # I have to identify the Laplacians which are unique, to speed-up computations
    eigvals, eigvecs = np.linalg.eigh(dvar)
    eigvals = (eigvals + np.sqrt(eigvals + 4 * sigma)) / 2
    return np.stack(
        [eigvec @ np.diag(eigval) @ eigvec.T for eigval, eigvec in zip(eigvals, eigvecs)]
    )


def laplacian_squareform(weights: NDArray[np.float]) -> NDArray[np.float_]:
    """Get Laplacians from batch of vetorized edge weights

    Args:
        weights (NDArray[np.float]): Array of vectorized edge weights of shape (n_graphs, n_edges)

    Returns:
        NDArray[np.float_]: Laplacians stacked in array of shape (n_graphs, n_nodes, n_nodes)
    """

    return np.stack([np.diag(np.sum((lapl := squareform(w)), axis=1)) - lapl for w in weights])


def laplacian_squareform_dual(laplacian: NDArray[np.float_]) -> NDArray[np.float_]:
    laplacian = laplacian.copy()
    np.fill_diagonal(laplacian, 0)
    s = np.sum(laplacian, axis=1)
    L1 = 2 * laplacian + s[:, np.newaxis] + s[np.newaxis, :]
    # np.fill_diagonal(L1, 0)
    return -squareform(L1, checks=False)


class GraphComponents(BaseEstimator):
    def __init__(
        self,
        n_components=1,
        alpha: float = 0.5,  # TODO: find best val
        *,
        max_iter: int = 100,
        max_iter_pds: int = 100,
        tol_pds: float = 1e-5,
        m_sigma: float = None,
        m_rho: float = None,
        e_sigma: float = None,
        e_rho: float = None,
        random_state: RandomState = None,
    ) -> None:
        super().__init__()

        self.n_components = n_components
        self.alpha = alpha

        self.max_iter = max_iter
        self.max_iter_pds = max_iter_pds
        self.tol_pds = tol_pds
        self.m_sigma = m_sigma
        self.m_rho = m_rho
        self.e_sigma = e_sigma
        self.e_rho = e_rho

        self.random_state = RandomState(random_state)

        self.activations_: NDArray[np.float_]  # shape (n_components, n_samples)
        self.weights_: NDArray[np.float_]  # shape (n_components, n_edges )

        self._discretize = False

    def _initialize(self, x: NDArray):
        n_nodes, n_samples = x.shape

        self.activations_ = self.random_state.rand(self.n_components, n_samples)
        self.weights_ = self.random_state.rand(self.n_components, n_nodes * (n_nodes - 1) // 2)

    def _e_step(
        self,
        # activations: NDArray[np.float_],
        # dual_var: NDArray[np.float_],
        x: NDArray[np.float_],
        laplacians: NDArray[np.float_],
    ):
        r"""Expectation step: compute activations

        Args:
            # activations (NDArray[np.float_]): Initial activations
            #     :math:`\Delta`, of shape (n_components, n_samples)
            # dual_var (NDArray[np.float_]): Initial dual variable :math:`V =
            #     L_W \Delta`, of shape (n_samples, n_nodes, n_nodes)
            x (NDArray[np.float_]): Signals matrix, of shape (n_samples, n_nodes)
            laplacians (NDArray[np.float_]): Laplacians estimates, of shape
                (n_components, n_nodes, n_nodes)
        """

        def lin_op(delta):
            return np.einsum("knm,kt->tnm", laplacians, delta)

        def dual_op(dvar):
            return np.einsum("knm,tnm->kt", laplacians, dvar)

        smoothness = np.einsum("ktn,tn->kt", x @ laplacians, x)  # shape: n_components, n_samples

        def prox_g(delta, tau):
            out = delta - tau * smoothness
            out = np.where(out < 1, out, 1)
            return np.where(out > 0, out, 0)

        self.activations_, _ = primal_dual_splitting(
            self.activations_,
            lin_op(self.activations_),
            tau=self.e_sigma,
            sigma=self.e_sigma,
            rho=self.e_rho,
            lin_op=lin_op,
            dual_op=dual_op,
            prox_g=prox_g,
            prox_h=prox_gdet,
            max_iter=self.max_iter_pds,
            tol=self.tol_pds,
        )

    def _m_step(self, pdists: NDArray[np.float_]):
        """Maximization step: compute weight matrices

        Args:
            pdists (NDArray[np.float_]): pairwise distances of sample-vectors,
                shape (n_edges, n_components)
        """

        def lin_op(weights):
            laplacians = laplacian_squareform(weights)
            return np.einsum("knm,kt->tnm", laplacians, self.activations_)

        def dual_op(dvar):
            dvar = np.einsum("tnm,tk->knm", dvar, self.activations_)
            return np.stack([laplacian_squareform_dual(laplacian) for laplacian in dvar])

        def prox_g(weights, tau):
            out = weights - tau * (pdists / 2 + self.alpha)
            return np.where(out > 0, out, 0)

        self.weights_, _ = primal_dual_splitting(
            self.weights_,
            lin_op(self.weights_),
            tau=self.m_sigma,
            sigma=self.m_sigma,
            rho=self.m_rho,
            lin_op=lin_op,
            dual_op=dual_op,
            prox_g=prox_g,
            prox_h=prox_gdet,
            max_iter=self.max_iter_pds,
            tol=self.tol_pds,
        )

    def _component_pdist(self, x):
        if self._discretize:
            # This discretize the activations
            return np.stack([pdist(x[mask > 0.5].T) ** 2 for mask in self.activations_])

        # This works with continuous activations
        pdiffs = x[:, np.newaxis, :] - x[:, :, np.newaxis]
        np.stack(
            [
                squareform(kdiff)
                for kdiff in np.einsum("kt,tnm->knm", self.activations_, pdiffs**2)
            ]
        )

    def fit(self, x: NDArray[np.float_], _y=None) -> GraphComponents:
        pass
