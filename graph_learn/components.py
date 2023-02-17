"""Graph components learning original method"""
from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.linalg import eigh
from numpy.random import RandomState
from numpy.typing import NDArray
from scipy.linalg import svdvals
from scipy.sparse.linalg import LinearOperator
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator


def _relaxed_update(
    p_var: NDArray[np.float_],
    p_var1: NDArray[np.float_],
    d_var: NDArray[np.float_],
    d_var1: NDArray[np.float_],
    relaxation: float,
    tol: int = 1e-3,
) -> tuple[NDArray, NDArray, bool]:
    # Denominators are previous iteration ones
    rel_norm_primal = (
        relaxation * np.linalg.norm((p_var1 - p_var).ravel()) / np.linalg.norm(p_var.ravel())
    )
    rel_norm_dual = (
        relaxation * np.linalg.norm((d_var1 - d_var).ravel()) / np.linalg.norm(d_var.ravel())
    )

    return (
        relaxation * p_var1 + (1 - relaxation) * p_var,
        relaxation * d_var1 + (1 - relaxation) * d_var,
        rel_norm_primal < tol and rel_norm_dual < tol,
    )


def _pds_step_primal(
    p_var: NDArray[np.float_],
    d_var: NDArray[np.float_],
    tau: float,
    sigma: float,
    relaxation: float,
    lin_op: LinearOperator,
    prox_g: Callable[[NDArray, float], NDArray],
    prox_h: Callable[[NDArray, float], NDArray],
    tol: int = 1e-3,
) -> tuple[NDArray, NDArray, bool]:
    p_var1 = prox_g(p_var - tau * lin_op.rmatvec(d_var), tau)

    # TODO: verify wheter (2 * p_var1 - p_var) can be enforced to be positive
    # Recall that prox_hs(x, sigma) = x - sigma * prox_h(x/sigma, 1/sigma)
    d_var1 = d_var + sigma * lin_op.matvec(2 * p_var1 - p_var)
    d_var1 -= sigma * prox_h(d_var1 / sigma, 1 / sigma)

    return _relaxed_update(
        p_var=p_var, p_var1=p_var1, d_var=d_var, d_var1=d_var1, relaxation=relaxation, tol=tol
    )


def _pds_step_dual(
    p_var: NDArray[np.float_],
    d_var: NDArray[np.float_],
    tau: float,
    sigma: float,
    relaxation: float,
    lin_op: LinearOperator,
    prox_g: Callable[[NDArray, float], NDArray],
    prox_h: Callable[[NDArray, float], NDArray],
    tol: int = 1e-3,
) -> tuple[NDArray, NDArray, bool]:
    d_var1 = d_var - sigma * prox_h(d_var / sigma + lin_op.matvec(p_var), 1 / sigma)

    p_var1 = prox_g(p_var - tau * lin_op.rmatvec(2 * d_var1 - d_var), tau)

    return _relaxed_update(
        p_var=p_var, p_var1=p_var1, d_var=d_var, d_var1=d_var1, relaxation=relaxation, tol=tol
    )


def primal_dual_splitting(
    p_var: NDArray[np.float_],
    d_var: NDArray[np.float_],
    tau: float,
    sigma: float,
    relaxation: float,
    lin_op: LinearOperator,
    prox_g: Callable[[NDArray, float], NDArray],
    prox_h: Callable[[NDArray, float], NDArray],
    max_iter: int = 100,
    tol: int = 1e-3,
    update: str = "dual",
) -> tuple[NDArray, NDArray]:
    r"""PDS algorithm for problems of the form
    .. math::
        argmin g(\mathbf{x}) + h(\mathbf{K x})

    Args:
        p_var (NDArray[np.float_]): _description_
        d_var (NDArray[np.float_]): _description_
        tau (float): _description_
        sigma (float): _description_
        relaxation (float): _description_
        lin_op (LinearOperator): _description_
        prox_prim (Callable[[NDArray, float], NDArray]): _description_
        prox_dual (Callable[[NDArray, float], NDArray]): _description_
        max_iter (int, optional): _description_. Defaults to 100.
        tol (int, optional): _description_. Defaults to 1e-3.

    Returns:
        tuple[NDArray, NDArray]: _description_
    """
    for i in range(max_iter):
        if update == "primal":
            p_var, d_var, converged = _pds_step_primal(
                p_var=p_var,
                d_var=d_var,
                tau=tau,
                sigma=sigma,
                relaxation=relaxation,
                lin_op=lin_op,
                prox_g=prox_g,
                prox_h=prox_h,
                tol=tol,
            )
        elif update == "dual":
            p_var, d_var, converged = _pds_step_dual(
                p_var=p_var,
                d_var=d_var,
                tau=tau,
                sigma=sigma,
                relaxation=relaxation,
                lin_op=lin_op,
                prox_g=prox_g,
                prox_h=prox_h,
                tol=tol,
            )
        else:
            raise ValueError(f"Invald PDS update, got '{update}'")

        if i > 0 and converged:
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
    # TODO: verify why this is wrong

    # Regularize dvar
    # dvar += (eps * np.eye(dvar.shape[-1]))[np.newaxis, ...]
    # I have to identify the Laplacians which are unique, to speed-up computations
    eigvals, eigvecs = eigh(dvar)

    # Input shall be SPD, so negative values come from numerical erros
    eigvals[eigvals < 0] = 0

    # Proximal step
    eigvals = (eigvals + np.sqrt(eigvals**2 + 4 * sigma)) / 2
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
    """Dual operator of :func:`laplacian_squareform`

    Args:
        laplacian (NDArray[np.float_]): Laplacian matrix of shape (n_nodes, n_nodes)

    Returns:
        NDArray[np.float_]: Vector
    """
    laplacian = laplacian.copy()
    np.fill_diagonal(laplacian, 0)
    s = np.sum(laplacian, axis=1)
    L1 = 2 * laplacian + s[:, np.newaxis] + s[np.newaxis, :]
    # np.fill_diagonal(L1, 0)
    return -squareform(L1, checks=False)


class _ExpectationLinOp:
    def __init__(self, laplacians: NDArray[np.float_]):
        self.laplacians = laplacians
        self._norm: float = None

    def matvec(self, x):
        return np.einsum("knm,kt->tnm", self.laplacians, x)

    def rmatvec(self, x):
        return np.einsum("knm,tnm->kt", self.laplacians, x)

    def norm(self) -> float:
        if self._norm is None:
            self._norm = svdvals(self.laplacians.reshape(self.laplacians.shape[0], -1))[0]

        return self._norm


class _MaximizationLinOp:
    def __init__(self, activations: NDArray[np.float_], n_nodes: int):
        self.activations = activations
        self.n_nodes = n_nodes

        self._norm: float = None

    def matvec(self, x):
        laplacians = laplacian_squareform(x)
        if laplacians.shape[-1] != self.n_nodes:
            raise ValueError("Invalid number of nodes")

        return np.einsum("knm,kt->tnm", laplacians, self.activations)

    def rmatvec(self, x):
        x = np.einsum("tnm,kt->knm", x, self.activations)
        return np.stack([laplacian_squareform_dual(laplacian) for laplacian in x])

    def norm(self):
        if self._norm is None:
            self._norm = np.sqrt(2 * self.n_nodes) * svdvals(self.activations)[0]

        return self._norm


class GraphComponents(BaseEstimator):
    def __init__(
        self,
        n_components=1,
        alpha: float = 0.5,  # TODO: find best val
        *,
        max_iter: int = 100,
        max_iter_pds: int = 100,
        tol_pds: float = 1e-3,
        pds_relaxation: float = None,
        random_state: RandomState = None,
        init_startegy: str = "uniform",
        weigth_scale: float = None,
        discretize: bool = False,
    ) -> None:
        super().__init__()

        self.n_components = n_components
        self.alpha = alpha

        self.max_iter = max_iter
        self.max_iter_pds = max_iter_pds
        self.tol_pds = tol_pds
        self.pds_relaxation = pds_relaxation

        self.random_state = RandomState(random_state)
        self.init_strategy = init_startegy
        self.weigth_scale = weigth_scale

        self.activations_: NDArray[np.float_]  # shape (n_components, n_samples)
        self.weights_: NDArray[np.float_]  # shape (n_components, n_edges )
        self.n_nodes_: int

        self.discretize = discretize

    def _initialize(self, x: NDArray):
        n_samples, self.n_nodes_ = x.shape

        if self.init_strategy == "uniform":
            self.activations_ = self.random_state.rand(self.n_components, n_samples)
            self.weights_ = self.weigth_scale * self.random_state.rand(
                self.n_components, self.n_nodes_ * (self.n_nodes_ - 1) // 2
            )
        else:
            raise ValueError(f"Invalid initialization, got {self.init_strategy}")

    def _e_step(self, x: NDArray[np.float_], laplacians: NDArray[np.float_]):
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

        lin_op = _ExpectationLinOp(laplacians)
        smoothness = np.einsum("ktn,tn->kt", x @ laplacians, x)  # shape: n_components, n_samples
        # TODO: Yamada&Tanaka use sctrictly smaller
        tau = 0.9 / lin_op.norm()

        def prox_g(delta, tau):
            out = delta - tau * smoothness
            out = np.where(out < 1, out, 1)
            return np.where(out > 0, out, 0)

        self.activations_, _ = primal_dual_splitting(
            self.activations_,
            tau * lin_op.matvec(self.activations_),
            tau=tau,
            sigma=tau,
            relaxation=self.pds_relaxation,
            lin_op=lin_op,
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
        lin_op = _MaximizationLinOp(self.activations_, self.n_nodes_)
        tau = 0.9 / lin_op.norm()

        pdists = pdists / 2 + self.alpha

        def prox_g(weights, tau):
            out = weights - tau * pdists
            return np.where(out > 0, out, 0)

        self.weights_, _ = primal_dual_splitting(
            self.weights_,
            tau * lin_op.matvec(self.weights_),
            tau=tau,
            sigma=tau,
            relaxation=self.pds_relaxation,
            lin_op=lin_op,
            prox_g=prox_g,
            prox_h=prox_gdet,
            max_iter=self.max_iter_pds,
            tol=self.tol_pds,
        )

    def _component_pdist(self, x: NDArray[np.float_]) -> NDArray[np.float_]:
        """Compute pairwise distances on each componend, based on activations

        Args:
            x (NDArray[np.float_]): Design matrix of shape (n_samples, n_nodes)

        Returns:
            NDArray[np.float_]: Pairwise node distances of shape (n_components, n_nodes, n_nodes)
        """
        if self.discretize:
            # This discretize the activations
            return np.stack([pdist(x[mask > 0.5].T) ** 2 for mask in self.activations_])

        # This works with continuous activations
        pdiffs = x[:, np.newaxis, :] - x[:, :, np.newaxis]
        return np.stack(
            [
                squareform(kdiff)
                for kdiff in np.einsum("kt,tnm->knm", self.activations_, pdiffs**2)
            ]
        )

    def fit(self, x: NDArray[np.float_], _y=None) -> GraphComponents:
        self._initialize(x)

        for _it in range(self.max_iter):
            self._m_step(self._component_pdist(x))

            self._e_step(x, laplacian_squareform(self.weights_))

        return self
