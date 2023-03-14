"""Graph components learning original method"""
from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.linalg import eigh
from numpy.random import RandomState
from numpy.typing import NDArray
from scipy.linalg import svdvals
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator

from graph_learn.evaluation import relative_error


def _relaxed_update(
    *pairs: list[tuple[NDArray[np.float_], NDArray[np.float_]]],
    relaxation: float,
) -> tuple[list[NDArray], NDArray]:
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


def laplacian_squareform(weights: NDArray[np.float]) -> NDArray[np.float_]:
    """Get Laplacians from batch of vetorized edge weights

    Args:
        weights (NDArray[np.float]): Array of vectorized edge weights of shape (n_graphs, n_edges)

    Returns:
        NDArray[np.float_]: Laplacians stacked in array of shape (n_graphs, n_nodes, n_nodes)
    """

    return np.stack([np.diag(np.sum((lapl := squareform(w)), axis=1)) - lapl for w in weights])


def laplacian_squareform_adj(laplacian: NDArray[np.float]) -> NDArray[np.float_]:
    """Adjoint of laplacian squareform"""
    out = -2 * laplacian
    neg_degs = np.sum(laplacian - np.diag(np.diag(laplacian)), axis=0)
    return squareform(out - neg_degs[:, np.newaxis] - neg_degs[np.newaxis, :], checks=False)


def prox_gdet_star(dvar: NDArray[np.float_], sigma: float) -> NDArray[np.float_]:
    """Proximal operator of the Moreau's transpose of negative generalized log-determinant

    Args:
        dvar (NDArray[np.float_]): Stack of SPD matrices, of shape (k, n, n)
        sigma (float): Proximal scale

    Returns:
        NDArray[np.float_]: Proximal point
    """
    # I have to identify the Laplacians which are unique, to speed-up computations
    eigvals, eigvecs = eigh(dvar)

    # Input shall be SPD, so negative values come from numerical erros
    eigvals[eigvals < 0] = 0

    # Proximal step
    eigvals = (eigvals - np.sqrt(eigvals**2 + 4 * sigma)) / 2
    return np.stack(
        [eigvec @ np.diag(eigval) @ eigvec.T for eigval, eigvec in zip(eigvals, eigvecs)]
    )


class GraphComponents(BaseEstimator):
    def __init__(
        self,
        n_components=1,
        alpha: float = 0.5,  # TODO: find best val
        *,
        max_iter: int = 50,
        tol: float = 1e-2,
        max_iter_pds: int = 100,
        tol_pds: float = 1e-3,
        pds_relaxation: float = None,
        random_state: RandomState = None,
        init_strategy: str = "uniform",
        weight_prior=None,
        discretize: bool = False,
        verbose: int = 0,
    ) -> None:
        super().__init__()

        self.n_components = n_components
        self.alpha = alpha

        self.max_iter = max_iter
        self.tol = tol
        self.max_iter_pds = max_iter_pds
        self.tol_pds = tol_pds
        self.pds_relaxation = pds_relaxation

        self.random_state = RandomState(random_state)
        self.init_strategy = init_strategy
        self.weight_prior = weight_prior

        self.verbose = verbose

        self.activations_: NDArray[np.float_]  # shape (n_components, n_samples)
        self.weights_: NDArray[np.float_]  # shape (n_components, n_edges )
        self.n_nodes_: int
        self.converged_: int

        self.discretize = discretize

    def _initialize(self, x: NDArray):
        n_samples, self.n_nodes_ = x.shape

        if self.init_strategy == "uniform":
            self.weight_prior = self.weight_prior or 1

            self.activations_ = self.random_state.rand(self.n_components, n_samples)
            self.weights_ = self.weight_prior * self.random_state.rand(
                self.n_components, self.n_nodes_ * (self.n_nodes_ - 1) // 2
            )
        elif self.init_strategy == "exact":
            if self.weight_prior is None:
                raise ValueError("Must provide a weight prior if init_stategy is 'exact'")

            self.weights_ = self.weight_prior
            self.activations_ = self.random_state.rand(self.n_components, n_samples)
        else:
            raise ValueError(f"Invalid initialization, got {self.init_strategy}")

        self.converged_ = -1

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

        # Optimal if =1/norm(linop)
        sigma = 1 / svdvals(laplacians.reshape(laplacians.shape[0], -1))[0]
        smoothness = np.einsum("ktn,tn->kt", x @ laplacians, x)  # shape: n_components, n_samples

        # dual = np.einsum("knm,kt->tnm", laplacians, self.activations_)
        dual = np.zeros((self.activations_.shape[1], self.n_nodes_, self.n_nodes_))

        converged = -1
        for pds_it in range(self.max_iter_pds):
            # Dual update
            dualp = prox_gdet_star(
                dual + sigma * np.einsum("knm,kt->tnm", laplacians, self.activations_), sigma
            )

            activationsp = self.activations_ - sigma * np.einsum(
                "knm,tnm->kt", laplacians, 2 * dualp - dual
            )
            # Proximal primal
            activationsp -= sigma * smoothness
            activationsp[activationsp < 0] = 0
            activationsp[activationsp > 1] = 1

            (self.activations_, dual), rel_norms = _relaxed_update(
                (self.activations_, activationsp),
                (dual, dualp),
                relaxation=self.pds_relaxation,
            )

            if np.all(rel_norms < self.tol_pds):
                converged = pds_it
                break

        if self.verbose > 1:
            if converged > 0:
                print(f"\tE-step converged in {converged} steps")
            else:
                print("\tE-step did not converge")

            if self.verbose > 2:
                print(self.activations_)

    def _m_step(self, pdists: NDArray[np.float_]):
        """Maximization step: compute weight matrices

        Args:
            pdists (NDArray[np.float_]): pairwise distances of sample-vectors,
                shape (n_edges, n_components)
        """

        sigma = 1 / np.sqrt(2 * self.n_nodes_) / svdvals(self.activations_)[0]
        dual = np.zeros((self.activations_.shape[1], self.n_nodes_, self.n_nodes_))

        converged = -1
        for pds_it in range(self.max_iter_pds):
            # Dual update
            dualp = prox_gdet_star(
                dual
                + sigma
                * np.einsum("knm,kt->tnm", laplacian_squareform(self.weights_), self.activations_),
                sigma,
            )

            # Primal update
            weightsp = self.weights_ - sigma * laplacian_squareform_adj(
                np.einsum("tnm,kt->nm", (2 * dualp - dual), self.activations_)
            )
            weightsp -= sigma * (pdists / 2 + self.alpha)
            weightsp[weightsp < 0] = 0

            (self.weights_, dual), rel_norms = _relaxed_update(
                (self.weights_, weightsp),
                (dual, dualp),
                relaxation=self.pds_relaxation,
            )

            if np.all(rel_norms < self.tol_pds):
                converged = pds_it
                break

        if self.verbose > 1:
            if converged > 0:
                print(f"\tM-step converged in {converged} steps")
            else:
                print("\tM-step did not converge")
            if self.verbose > 2:
                print(*(squareform(weight) for weight in self.weights_), sep="\n")

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

        for cycle in range(self.max_iter):
            if self.verbose > 0:
                print(f"Iteration {cycle}")

            weights_pre = self.weights_.copy()

            pdists = self._component_pdist(x)
            self._m_step(pdists / pdists.mean())

            self._e_step(x, laplacian_squareform(self.weights_))

            rel_err = relative_error(weights_pre, self.weights_)

            if self.verbose > 0:
                print(f"\tRelative weight change: {rel_err}")

            if rel_err < self.tol:
                self.converged_ = cycle
                break

        return self
