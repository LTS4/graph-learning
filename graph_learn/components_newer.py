"""Graph components learning original method"""
from __future__ import annotations

from warnings import warn

import numpy as np
from numpy.linalg import eigh, eigvalsh
from numpy.random import RandomState
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator

from graph_learn import OptimizationError
from graph_learn.evaluation import relative_error
from graph_learn.utils import laplacian_squareform


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


laplacian_squareform_vec = np.vectorize(
    laplacian_squareform, otypes=[float], signature="(e)->(n,n)"
)


def laplacian_squareform_adj(laplacian: NDArray[np.float]) -> NDArray[np.float_]:
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
        laplacians = laplacian_squareform_vec(self.weights_)
        assert laplacians.shape == (self.n_components, self.n_nodes_, self.n_nodes_)

        # Optimal if =1/norm(linop)
        op_norm = np.sqrt(np.max(eigvalsh(laplacians)))
        sigma = 1 / op_norm

        # shape: (n_components, n_samples)
        smoothness = np.einsum("ktn,tn->kt", x @ laplacians, x)
        smoothness /= self.n_samples_

        # prox step
        smoothness += self.l1_activations
        smoothness *= sigma
        # TODO: consider boost

        # shape: (n_samples, n_nodes, n_nodes)
        # dual = np.einsum("knm,kt->tnm", laplacians, self.activations_)
        # self.dual_e_ = np.zeros((self.n_samples_, self.n_nodes_, self.n_nodes_))

        converged = -1
        for pds_it in range(self.max_iter_pds):
            # Primal update
            _dual_step = sigma * np.einsum("knm,tnm->kt", laplacians, self.dual_e_)
            activationsp = self.activations_ - _dual_step
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

    def _m_step(self, x: NDArray[np.float_]) -> int:
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
            _beta2 = self.ortho_weights * np.sqrt(self.n_components)
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

            # # Dot product regularization
            # _grad_step = tau * 2 * self.ortho_weights * self.weights_.sum(0, keepdims=True)

            # Normalized orthogonality
            _inv_norms = 1 / np.linalg.norm(self.weights_, axis=1)
            if self.ortho_weights > 0:
                _grad_step = (
                    tau
                    * 2
                    * self.ortho_weights
                    * _inv_norms[:, np.newaxis]
                    * (
                        (
                            np.eye(self.weights_.shape[1])[np.newaxis, ...]
                            - (_inv_norms**2)[:, np.newaxis, np.newaxis]
                            * np.stack([np.outer(w, w) for w in self.weights_])
                        )
                        @ (self.weights_.T @ _inv_norms)
                    )
                )
            else:
                _grad_step = 0

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

            (self.weights_, self.dual_m_), rel_norms = _relaxed_update(
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

    def _component_pdist_sq(self, x: NDArray[np.float_]) -> NDArray[np.float_]:
        """Compute pairwise square distances on each componend, based on activations

        Args:
            x (NDArray[np.float_]): Design matrix of shape (n_samples, n_nodes)

        Returns:
            NDArray[np.float_]: Pairwise node squared distances of shape
                (n_components, n_nodes, n_nodes)
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


### PARTIAL CLASSES ############################################################


class FixedWeights(GraphComponents):
    """Subclass of :class:`GraphComponents` with fixed weights.

    Only Activations are optimized.
    """

    def _initialize(self, x: NDArray):
        super()._initialize(x)

        if not isinstance(self.weight_prior, np.ndarray):
            raise TypeError(
                f"Weight prior must be a numpy array, got {type(self.activation_prior)}"
            )
        if self.weights_.shape != self.weight_prior.shape:
            raise ValueError(
                f"Invalid weight prior shape, expected {self.activations_.shape},"
                f" got {self.weight_prior.shape}"
            )

        self.weights_ = self.weight_prior

    def _m_step(self, x: NDArray[np.float_]) -> int:
        return 0


class FixedActivations(GraphComponents):
    """Subclass of :class:`GraphComponents` with fixed activations.

    Only weights are optimized.
    """

    def _initialize(self, x: NDArray):
        super()._initialize(x)

        if not isinstance(self.activation_prior, np.ndarray):
            raise TypeError(
                f"Activation prior must be a numpy array, got {type(self.activation_prior)}"
            )
        if self.activations_.shape != self.activation_prior.shape:
            raise ValueError(
                f"Invalid activation prior shape, expected {self.activations_.shape},"
                f" got {self.activation_prior.shape}"
            )

        self.activations_ = self.activation_prior

    def _e_step(self, x: NDArray[np.float_]) -> int:
        return 0
