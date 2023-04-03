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
        [
            eigvec @ np.diag(eigval) @ eigvec.T
            for eigval, eigvec in zip(eigvals[..., 1:], eigvecs[..., 1:])
        ]
    )


class GraphComponents(BaseEstimator):
    def __init__(
        self,
        n_components=1,
        l1_weights: float = 1,
        boost_activations: float = 1,
        l1_activations: float = 1,
        *,
        max_iter: int = 50,
        tol: float = 1e-3,
        max_iter_pds: int = 100,
        tol_pds: float = 1e-3,
        pds_relaxation: float = None,
        random_state: RandomState = None,
        init_strategy: str = "uniform",
        weight_prior=None,
        activation_prior=None,
        discretize: bool = False,
        verbose: int = 0,
    ) -> None:
        super().__init__()

        self.n_components = n_components
        self.l1_weights = l1_weights
        self.boost_activations = boost_activations
        self.l1_activations = l1_activations

        self.max_iter = max_iter
        self.tol = tol
        self.max_iter_pds = max_iter_pds
        self.tol_pds = tol_pds
        self.pds_relaxation = pds_relaxation

        self.random_state = RandomState(random_state)
        self.init_strategy = init_strategy
        self.weight_prior = weight_prior
        self.activation_prior = activation_prior

        self.verbose = verbose

        self.activations_: NDArray[np.float_]  # shape (n_components, n_samples)
        self.weights_: NDArray[np.float_]  # shape (n_components, n_edges )
        self.dual_m_: NDArray[np.float_]  # shape (n_samples, n_nodes, n_nodes)
        self.dual_e_: NDArray[np.float_]  # shape (n_samples, n_nodes, n_nodes)
        self.n_nodes_: int
        self.n_samples_: int
        self.converged_: int
        self.history_: dict[int, dict[str, int]]

        self.discretize = discretize

    def _initialize(self, x: NDArray):
        self.n_samples_, self.n_nodes_ = x.shape

        match self.init_strategy:
            case "uniform":
                self.weight_prior = self.weight_prior or 1

                self.activations_ = self.random_state.rand(self.n_components, self.n_samples_)
                self.weights_ = self.weight_prior * self.random_state.rand(
                    self.n_components, self.n_nodes_ * (self.n_nodes_ - 1) // 2
                )
            case "exact":
                if self.weight_prior is None and self.activation_prior is None:
                    raise ValueError(
                        "Must provide a prior for at least one of weigths or activations if init_stategy is 'exact'"
                    )

                if self.weight_prior is None:
                    self.weights_ = self.random_state.rand(
                        self.n_components, self.n_nodes_ * (self.n_nodes_ - 1) // 2
                    )
                else:
                    self.weights_ = self.weight_prior
                if self.activation_prior is None:
                    self.activations_ = self.random_state.rand(self.n_components, self.n_samples_)
                else:
                    self.activations_ = self.activation_prior
            case _:
                raise ValueError(f"Invalid initialization, got {self.init_strategy}")

        self.dual_m_ = np.zeros((self.n_samples_, self.n_nodes_, self.n_nodes_))
        self.dual_e_ = np.zeros_like(self.dual_m_)

        self.history_ = {}
        self.converged_ = -1

    def _e_step(self, x: NDArray[np.float_]) -> int:
        r"""Expectation step: compute activations

        Args:
            x (NDArray[np.float_]): Signals matrix of shape (n_samples, n_nodes)

        Returns:
            int: number of PDS iteation for convergence (-1 if not converged)
        """

        # shape: (n_components, n_nodes, n_nodes)
        laplacians = laplacian_squareform(self.weights_)

        # Optimal if =1/norm(linop)
        sigma = 1 / svdvals(laplacians.reshape(laplacians.shape[0], -1))[0]

        # shape: (n_components, n_samples)
        smoothness = np.einsum("ktn,tn->kt", x @ laplacians, x)
        # TODO: maybe is better to divide by self.activations_.sum(0)[np.newaxis, :]
        # smoothness *= (sigma / smoothness.mean(0))[np.newaxis, :]
        smoothness *= sigma / self.n_samples_

        # prox step
        smoothness += sigma * self.l1_activations

        # shape: (n_samples, n_nodes, n_nodes)
        # dual = np.einsum("knm,kt->tnm", laplacians, self.activations_)
        # self.dual_e_ = np.zeros((self.n_samples_, self.n_nodes_, self.n_nodes_))

        converged = -1
        for pds_it in range(self.max_iter_pds):
            # Primal update
            activationsp = self.activations_ - sigma * np.einsum(
                "knm,tnm->kt", laplacians, self.dual_e_
            )
            # Gradient step
            # activationsp += (sigma / self.activations_.sum(0))[np.newaxis, :]
            # Proximal step
            activationsp -= smoothness
            activationsp[activationsp < 0] = 0
            activationsp[activationsp > 1] = 1

            # Dual update
            dualp = prox_gdet_star(
                self.dual_e_
                + sigma
                * np.einsum("knm,kt->tnm", laplacians, 2 * activationsp - self.activations_),
                sigma / self.n_samples_,
            )

            (self.activations_, self.dual_e_), rel_norms = _relaxed_update(
                (self.activations_, activationsp),
                (self.dual_e_, dualp),
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

        return converged

    def _m_step(self, x: NDArray[np.float_]) -> int:
        """Maximization step: compute weight matrices

        Args:
            x (NDArray[np.float_]): Signal matrix of shape (n_edges, n_components)

        Returns:
            int: number of PDS iteation for convergence (-1 if not converged)
        """

        # TODO: what is the best way to rescale this?
        # Options:
        # - No rescaling: all weight are zero in one step
        # x self.n_samples_: Theoretically correct one
        # - self.activations_.sum(1)[:, np.newaxis]: seems to work better
        #   than n_samples, as low-activity components are quickly learned
        # - pdists.mean(): similar to activations, brought down by components with small activations
        # - pdists.mean(1)[:, np.newaxis]: similar to activations, almost same vals

        # Try to use sqrt(nb components) as in yamada2021temporal
        op_norm = np.sqrt(2 * self.n_nodes_) * svdvals(self.activations_)[0]
        tau = 1 / op_norm
        sigma = 1 / tau / op_norm**2
        # tau = sigma = 0.1
        # self.dual_m_ = np.zeros((self.n_samples_, self.n_nodes_, self.n_nodes_))

        #  pdist.shape: (n_edges, n_components) = self.weights_.shape
        pdists = self._component_pdist(x)
        pdists /= (self.activations_.mean(1) ** self.boost_activations * self.n_samples_)[
            :, np.newaxis
        ]

        # prox step
        pdists += self.l1_weights
        pdists *= tau

        converged = -1
        for pds_it in range(self.max_iter_pds):
            # Primal update
            weightsp = self.weights_ - tau * laplacian_squareform_adj(
                np.einsum("tnm,kt->nm", self.dual_m_, self.activations_)
            )
            weightsp -= pdists
            weightsp[weightsp < 0] = 0

            # Dual update
            dualp = prox_gdet_star(
                self.dual_m_
                + sigma
                * np.einsum(
                    "knm,kt->tnm",
                    laplacian_squareform(2 * weightsp - self.weights_),
                    self.activations_,
                ),
                sigma / self.n_samples_,
            )

            (self.weights_, self.dual_m_), rel_norms = _relaxed_update(
                (self.weights_, weightsp),
                (self.dual_m_, dualp),
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

        return converged

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
            activations_pre = self.activations_.copy()

            self.history_[cycle] = {
                "m_step": self._m_step(x),
                "e_step": self._e_step(x),
            }

            w_rel_change = relative_error(weights_pre, self.weights_)
            a_rel_change = relative_error(activations_pre, self.activations_)

            if self.verbose > 0:
                print(f"\tRelative weight change: {w_rel_change}")
                print(f"\tRelative activation change: {a_rel_change}")

            if w_rel_change < self.tol and a_rel_change < self.tol:
                self.converged_ = cycle
                return self

        return self
