"""Graph components learning original method"""
from __future__ import annotations

from typing import Callable
from warnings import warn

import numpy as np
from numpy.linalg import eigvalsh
from numpy.random import RandomState
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator

from graph_learn import OptimizationError
from graph_learn.components.utils import (
    laplacian_squareform_adj_vec,
    laplacian_squareform_vec,
    prox_gdet_star,
    relaxed_update,
)
from graph_learn.evaluation import relative_error


class GraphComponents(BaseEstimator):
    def __init__(
        self,
        n_components=1,
        l1_weights: float = 1,
        ortho_weights: float = 0,
        boost_activations: float = 0,
        l1_activations: float = 0,
        *,
        max_iter: int = 50,
        tol: float = 1e-3,
        max_iter_pds: int = 100,
        tol_pds: float = 1e-3,
        pds_relaxation: float = 1,
        random_state: RandomState = None,
        init_strategy: str = "uniform",
        weight_prior: float | NDArray[np.float_] = None,
        activation_prior: float | NDArray[np.float_] = None,
        discretize: bool = False,
        normalize: bool = False,
        verbose: int = 0,
    ) -> None:
        super().__init__()

        self.n_components = n_components
        self.l1_weights = l1_weights
        self.ortho_weights = ortho_weights
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
        self.normalize = normalize

    def _initialize(self, x: NDArray) -> None:
        self.n_samples_, self.n_nodes_ = x.shape

        match self.init_strategy:
            case "constant":
                if self.activation_prior is None or isinstance(self.activation_prior, float):
                    self.activation_prior = self.activation_prior or 1
                    self.activation_prior = self.activation_prior * np.ones(
                        (self.n_components, self.n_samples_)
                    )

                self.activations_ = self.activation_prior

                if self.weight_prior is None or isinstance(self.weight_prior, float):
                    self.weight_prior = self.weight_prior or 1
                    self.weight_prior = self.weight_prior * np.ones(
                        (self.n_components, self.n_nodes_ * (self.n_nodes_ - 1) // 2)
                    )

                self.weights_ = self.weight_prior
            case "uniform":
                self.weight_prior = self.weight_prior or 1

                self.activations_ = self.random_state.rand(self.n_components, self.n_samples_)
                self.weights_ = self.weight_prior * self.random_state.rand(
                    self.n_components, self.n_nodes_ * (self.n_nodes_ - 1) // 2
                )
            case "exact":
                if self.weight_prior is None and self.activation_prior is None:
                    raise ValueError(
                        "Must provide a prior for at least one of weigths or"
                        " activations if init_stategy is 'exact'"
                    )

                if self.weight_prior is None:
                    self.weights_ = self.random_state.rand(
                        self.n_components, self.n_nodes_ * (self.n_nodes_ - 1) // 2
                    )
                else:
                    if isinstance(self.weight_prior, (float, int)):
                        self.weight_prior = self.weight_prior * np.ones(
                            (self.n_components, self.n_nodes_ * (self.n_nodes_ - 1) // 2)
                        )
                    self.weights_ = self.weight_prior

                if self.activation_prior is None:
                    self.activations_ = self.random_state.rand(self.n_components, self.n_samples_)
                else:
                    if isinstance(self.activation_prior, (float, int)):
                        self.activation_prior = self.activation_prior * np.ones(
                            (self.n_components, self.n_samples_)
                        )

                    self.activations_ = self.activation_prior
            case _:
                raise ValueError(f"Invalid initialization, got {self.init_strategy}")

        if self.discretize:
            self.activations_ = (self.activations_ > 0.5).astype(int)

        self.dual_m_ = np.zeros((self.n_samples_, self.n_nodes_, self.n_nodes_))
        self.dual_e_ = np.zeros_like(self.dual_m_)

        self.history_ = {}
        self.converged_ = -1

    def _component_pdist_sq(self, x: NDArray[np.float_]) -> NDArray[np.float_]:
        """Compute pairwise square distances on each componend, based on activations

        Args:
            x (NDArray[np.float_]): Design matrix of shape (n_samples, n_nodes)

        Returns:
            NDArray[np.float_]: Pairwise node squared distances of shape
                (n_components, n_nodes, n_nodes)
        """
        if False and self.discretize:
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

    def fit(
        self,
        x: NDArray[np.float_],
        _y=None,
        callback: Callable[[GraphComponents, int]] = None,
    ) -> GraphComponents:
        self._initialize(x)
        if callback is not None:
            callback(self, -1)

        for cycle in range(self.max_iter):
            if self.verbose > 0:
                print(f"Iteration {cycle}")

            weights_pre = self.weights_.copy()
            activations_pre = self.activations_.copy()

            # TODO: try to change order
            self.history_[cycle] = {
                "m_step": self._m_step(x),
                "e_step": self._e_step(x),
            }

            w_rel_change = relative_error(weights_pre, self.weights_)
            a_rel_change = relative_error(activations_pre, self.activations_)

            if self.verbose > 0:
                print(f"\tRelative weight change: {w_rel_change}")
                print(f"\tRelative activation change: {a_rel_change}")

            if callback is not None:
                callback(self, cycle)

            if w_rel_change < self.tol and a_rel_change < self.tol:
                self.converged_ = cycle
                return self

        return self

    # EXPECTATION - activations ################################################

    def _e_step(self, x: NDArray[np.float_]) -> int:
        r"""Expectation step: compute activations

        Args:
            x (NDArray[np.float_]): Signals matrix of shape (n_samples, n_nodes)

        Returns:
            int: number of PDS iteation for convergence (-1 if not converged)
        """
        # shape: (n_components, n_nodes, n_nodes)
        laplacians = laplacian_squareform_vec(self.weights_)
        assert laplacians.shape == (self.n_components, self.n_nodes_, self.n_nodes_)

        # Optimal if =1/norm(linop)
        op_norm = np.sqrt(np.max(eigvalsh(laplacians)))
        sigma = 1 / op_norm

        # shape: (n_components, n_samples)
        smoothness = np.einsum("ktn,tn->kt", x @ laplacians, x)

        # FIXME: random hack
        smoothness /= smoothness.mean() * 2  # Center the distribution to ~0.5
        # smoothness /= self.n_components

        # prox step
        smoothness += self.l1_activations
        smoothness *= sigma
        # TODO: consider boost

        # shape: (n_samples, n_nodes, n_nodes)
        # dual = np.einsum("knm,kt->tnm", laplacians, self.activations_)
        # self.dual_e_ = np.zeros((self.n_samples_, self.n_nodes_, self.n_nodes_))

        converged = -1
        for pds_it in range(self.max_iter_pds):
            if self.discretize:
                rel_norms = self._e_pds_discrete(
                    sigma=sigma, smoothness=smoothness, laplacians=laplacians
                )
            else:
                rel_norms = self._e_pds_continuous(
                    sigma=sigma, smoothness=smoothness, laplacians=laplacians
                )

            if np.all(rel_norms < self.tol_pds):
                converged = pds_it
                break

            if np.allclose(self.activations_, 0):
                raise OptimizationError("Activations dropped to zero")

        if self.verbose > 1:
            if converged > 0:
                print(f"\tE-step converged in {converged} steps")
            else:
                print("\tE-step did not converge")

            if self.verbose > 2:
                print(self.activations_)

        return converged

    def _e_pds_continuous(
        self, sigma: float, smoothness: NDArray[np.float_], laplacians: NDArray[np.float_]
    ) -> int:
        """Expactation PDS step with continuous activations"""
        # Primal update
        _dual_step = sigma * np.einsum("knm,tnm->kt", laplacians, self.dual_e_)
        activationsp = self.activations_ - _dual_step - smoothness
        # Gradient step
        # activationsp += (sigma / self.activations_.sum(0))[np.newaxis, :]

        # Projection
        activationsp[activationsp < 0] = 0
        activationsp[activationsp > 1] = 1

        # Dual update
        dualp = prox_gdet_star(
            self.dual_e_
            + sigma * np.einsum("knm,kt->tnm", laplacians, 2 * activationsp - self.activations_),
            sigma / self.n_components,  # FIXME: random hack
        )

        (self.activations_, self.dual_e_), rel_norms = relaxed_update(
            (self.activations_, activationsp),
            (self.dual_e_, dualp),
            relaxation=self.pds_relaxation,
        )

        return rel_norms

    def _e_pds_discrete(
        self, sigma: float, smoothness: NDArray[np.float_], laplacians: NDArray[np.float_]
    ) -> int:
        """Expactation PDS step with discrete activations"""
        # Primal update
        _dual_step = sigma * np.einsum("knm,tnm->kt", laplacians, self.dual_e_)
        # Proximal step
        activationsp = self.activations_ - _dual_step - smoothness
        # Gradient step
        # TODO: I have to activate this again and look into parameterization
        # activationsp += (sigma / self.activations_.sum(0))[np.newaxis, :]
        # TODO: test whether centering smoothness around 0.5 is useful.
        # TODO: verify whether this is moving activations enough

        # FIXME: I am brutally turning off activations for which the dual is smaller than smoothness
        # activationsp = (-_dual_step > smoothness).astype(int)

        if False:
            # TODO: consider wether this is relevant
            _thresh = activationsp < 0.5
            activationsp[_thresh] = 0
            activationsp[~_thresh] = 1
        else:
            # TODO: since I am discretizing the overshoot it might be
            # interesting to let activationsp unbounded
            activationsp[activationsp < 0] = 0
            activationsp[activationsp > 1] = 1

        # Dual update
        # TODO: Removing line 340 this is the only actual place where I discretize
        _overshoot = ((2 * activationsp - self.activations_) > 0.5).astype(int)
        hashtable = [x.data.tobytes() for x in self.activations_.T]
        activations_keys = set(hashtable)
        unique_activations_indices = np.array([hashtable.index(key) for key in activations_keys])
        unique_activations = _overshoot[:, unique_activations_indices]

        dual_dict = dict(
            zip(
                activations_keys,
                prox_gdet_star(
                    self.dual_e_[unique_activations_indices]
                    + sigma
                    * np.einsum(
                        "knm,kt->tnm",
                        laplacians,
                        unique_activations,
                    ),
                    sigma / self.n_components,  # FIXME: random hack
                ),
            )
        )

        dualp = np.stack([dual_dict[x] for x in hashtable])
        assert dualp.shape == self.dual_e_.shape

        (self.activations_, self.dual_e_), rel_norms = relaxed_update(
            (self.activations_, activationsp),
            (self.dual_e_, dualp),
            relaxation=self.pds_relaxation,
        )

        # if np.allclose(self.activations_, 0):
        #     raise OptimizationError("Activations collapsed")

        # # TODO: I am removing this to allow for continuous optimization.
        # This requires me to discretize in M-step
        # self.activations_ = (self.activations_ > 0.5).astype(int)

        return rel_norms

    # MAXIMIZATION - weights ###################################################

    def _m_step(self, x: NDArray[np.float_]) -> int:
        if self.discretize:
            return self._m_step_discrete(x)

        return self._m_step_continuous(x)

    def _m_step_continuous(self, x: NDArray[np.float_]) -> int:
        """Maximization step: compute weight matrices

        Args:
            x (NDArray[np.float_]): Signal matrix of shape (n_edges, n_components)

        Returns:
            int: number of PDS iteation for convergence (-1 if not converged)
        """
        # NOTE: This is an upper bound, might get better convergence with tailored steps
        op_norm = np.sqrt(2 * self.n_nodes_ * self.activations_.sum(1).max())
        if self.ortho_weights > 0:
            warn("Lipschitz constant is wrong")
            _beta2 = self.ortho_weights * (self.n_components - 1)
            sigma = tau = (-_beta2 + np.sqrt(_beta2**2 + 4 * op_norm**2)) / (2 * op_norm**2)
        else:
            sigma = tau = 1 / op_norm

        #  pdist.shape: (n_edges, n_components) = self.weights_.shape
        sq_pdists = self._component_pdist_sq(x) / self.n_samples_

        # prox step
        sq_pdists += self.l1_weights
        sq_pdists *= tau
        # TODO: consider boost

        converged = -1
        for pds_it in range(self.max_iter_pds):
            # Primal update
            _dual_step = tau * laplacian_squareform_adj_vec(
                np.einsum("tnm,kt->knm", self.dual_m_, self.activations_)
            )
            assert _dual_step.shape == self.weights_.shape

            if self.ortho_weights > 0:
                # # Dot product regularization
                # _grad_step = tau * 2 * self.ortho_weights * self.weights_.sum(0, keepdims=True)

                # Dot product externals
                _grad_step = (
                    tau
                    * 2
                    * self.ortho_weights
                    * (np.ones((self.n_components, self.n_components)) - np.eye(self.n_components))
                    @ self.weights_
                )

                # # Normalized orthogonality
                # _inv_norms = 1 / np.linalg.norm(self.weights_, axis=1)
                # _grad_step = (
                #     tau
                #     * 2
                #     * self.ortho_weights
                #     * _inv_norms[:, np.newaxis]
                #     * (
                #         (
                #             np.eye(self.weights_.shape[1])[np.newaxis, ...]
                #             - (_inv_norms**2)[:, np.newaxis, np.newaxis]
                #             * np.stack([np.outer(w, w) for w in self.weights_])
                #         )
                #         @ (self.weights_.T @ _inv_norms)
                #     )
                # )
            else:
                _grad_step = 0

            # Proximal step
            weightsp = self.weights_ - _dual_step - _grad_step
            weightsp -= sq_pdists
            weightsp[weightsp < 0] = 0

            # Normalization
            if self.normalize:
                weightsp /= np.linalg.norm(weightsp, axis=1, keepdims=True)

            # Dual update
            dualp = prox_gdet_star(
                self.dual_m_
                + sigma
                * np.einsum(
                    "knm,kt->tnm",
                    laplacian_squareform_vec(2 * weightsp - self.weights_),
                    self.activations_,
                ),
                sigma / self.n_samples_,
            )

            assert dualp.shape == self.dual_m_.shape

            (self.weights_, self.dual_m_), rel_norms = relaxed_update(
                (self.weights_, weightsp),
                (self.dual_m_, dualp),
                relaxation=self.pds_relaxation,
            )

            if np.all(rel_norms < self.tol_pds):
                converged = pds_it
                break

            if np.allclose(self.weights_, 0):
                raise OptimizationError("Weights dropped to zero")

        if self.verbose > 1:
            if converged > 0:
                print(f"\tM-step converged in {converged} steps")
            else:
                print("\tM-step did not converge")
            if self.verbose > 2:
                print(*(squareform(weight) for weight in self.weights_), sep="\n")

        return converged

    def _m_step_discrete(self, x: NDArray[np.float_]) -> int:
        """Maximization step: compute weight matrices

        Args:
            x (NDArray[np.float_]): Signal matrix of shape (n_edges, n_components)

        Returns:
            int: number of PDS iteation for convergence (-1 if not converged)
        """
        # Discrete preparation
        discrete_activ = np.where(self.activations_ > 0.5, 1, 0)
        hashtable = [x.data.tobytes() for x in discrete_activ.T]
        activations_keys = set(hashtable)
        unique_activations_indices = np.array([hashtable.index(key) for key in activations_keys])
        unique_activations = discrete_activ[:, unique_activations_indices]
        # counts = {key: hashtable.count(key) for key in activations_keys}

        # shape: (n_components, n_unique)
        assert unique_activations.shape[0] == self.n_components

        # Common part
        # NOTE: This is an upper bound, might get better convergence with tailored steps
        op_norm = np.sqrt(2 * self.n_nodes_ * discrete_activ.sum(1).max())
        if self.ortho_weights > 0:
            # warn("Lipschitz constant is wrong")
            _beta2 = self.ortho_weights * (self.n_components - 1)
            sigma = tau = (-_beta2 + np.sqrt(_beta2**2 + 4 * op_norm**2)) / (2 * op_norm**2)
        else:
            sigma = tau = 1 / op_norm

        #  pdist.shape: (n_edges, n_components) = self.weights_.shape
        sq_pdists = self._component_pdist_sq(x) / self.n_samples_

        # prox step
        sq_pdists += self.l1_weights
        sq_pdists *= tau
        # TODO: consider boost

        converged = -1
        for pds_it in range(self.max_iter_pds):
            # Primal update
            # TODO: I should change to discrete:
            # self.dual_m_[unique_activations_indices], np.array(counts.values()) * discrete_activ
            # TODO: having a full dual_m is memory inefficient, but is needed for the first iteration,
            # as activations might have changed from prev step
            _dual_step = tau * laplacian_squareform_adj_vec(
                np.einsum("tnm,kt->knm", self.dual_m_, discrete_activ)
            )
            assert _dual_step.shape == self.weights_.shape

            if self.ortho_weights > 0:
                # # Dot product regularization
                # _grad_step = tau * 2 * self.ortho_weights * self.weights_.sum(0, keepdims=True)

                # Dot product externals
                _grad_step = (
                    tau
                    * 2
                    * self.ortho_weights
                    * (np.ones((self.n_components, self.n_components)) - np.eye(self.n_components))
                    @ self.weights_
                )

                # # Normalized orthogonality
                # _inv_norms = 1 / np.linalg.norm(self.weights_, axis=1)
                # _grad_step = (
                #     tau
                #     * 2
                #     * self.ortho_weights
                #     * _inv_norms[:, np.newaxis]
                #     * (
                #         (
                #             np.eye(self.weights_.shape[1])[np.newaxis, ...]
                #             - (_inv_norms**2)[:, np.newaxis, np.newaxis]
                #             * np.stack([np.outer(w, w) for w in self.weights_])
                #         )
                #         @ (self.weights_.T @ _inv_norms)
                #     )
                # )
            else:
                _grad_step = 0

            weightsp = self.weights_ - _dual_step - _grad_step
            weightsp -= sq_pdists
            weightsp[weightsp < 0] = 0

            # Normalization
            if self.normalize:
                weightsp /= np.linalg.norm(weightsp, axis=1, keepdims=True)

            # Dual update
            # Discrete specific
            dual_dict = dict(
                zip(
                    activations_keys,
                    prox_gdet_star(
                        self.dual_m_[unique_activations_indices]
                        + sigma
                        * np.einsum(
                            "knm,kt->tnm",
                            laplacian_squareform_vec(2 * weightsp - self.weights_),
                            unique_activations,
                        ),
                        sigma / self.n_samples_,
                    ),
                )
            )

            dualp = np.stack([dual_dict[x] for x in hashtable])

            # common again
            assert dualp.shape == self.dual_m_.shape

            (self.weights_, self.dual_m_), rel_norms = relaxed_update(
                (self.weights_, weightsp),
                (self.dual_m_, dualp),
                relaxation=self.pds_relaxation,
            )

            if np.allclose(self.weights_, 0):
                raise OptimizationError("Weights dropped to zero")

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
