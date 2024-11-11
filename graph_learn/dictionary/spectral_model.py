"""Implementing the parametric graph dictionary learning model from Thanou et al. 2013"""

import numpy as np
from numpy.typing import NDArray

from graph_learn.operators import autocorr, prox_gdet_star_spectral_update

from .exact_model import GraphDictExact


class GraphDictSpectral(GraphDictExact):
    r"""Graph dictionary learning model from Thanou et al. 2013, within our framework.

    Specific properties:
    - Graph filter eigenvalues are fixed (prior or correlation) :math:`U`
    - Atoms are described by eigenvalues of filters :math:`\Lambda_k` so that
    atoms laplacians are :math:`L_k = U^\top \Lambda_k U`

    Recalling that x has shape (n_samples, n_nodes), the new smoothness term is
    .. math::

        \sum_{t} x_t^\top L_t x_t
            = \sum_t\sum_k \delta_{kt} x_t^\top L_k x_t
            = \sum_t\sum_k \delta_{kt} x_t^\top U^\top \Lambda_k U x_t
            = \sum_{tkn} \delta_{kt} \lambda_{kn} [X U^\top]_{tn}^2

    I can magically reuse :method:`GraphDictBase._update_coefficients` since the only difference is
    in the smoothness term, which now uses eigenvalues instead of weights, and squared GFT signals
    instead of sq_pdiffs.


    """

    def __init__(self, *args, eigenvecs_prior: NDArray = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.eigenvecs_prior_ = eigenvecs_prior
        self.eigenvecs_: NDArray  # shape: (n_nodes, n_nodes)

        # Instead of weights and dual I keep eigenvalues only
        # self.weights_ <- self.atom_eigenvals_: NDArray  # shape: (n_atoms, n_nodes)
        # self.dual <- self.dual_eigenvals_: NDArray  # shape: (n_samples, n_nodes)

    # INITIALIZATION ###############################################################################

    def _init_weigths(self, x: NDArray = None) -> NDArray:
        _, n_nodes = x.shape
        atom_eigenvals = np.ones((self.n_atoms, n_nodes), dtype=float)

        match self.init_strategy_w:
            case "uniform":
                atom_eigenvals = self.random_state.uniform(size=(self.n_atoms, n_nodes))

            case _:
                raise ValueError(f"Invalid init_strategy for weights: {self.init_strategy_w}")

        return atom_eigenvals

    def _init_dual(self, x: NDArray) -> NDArray:
        return np.zeros_like(x)

    def _initialize(self, x: NDArray) -> None:
        if self.window_size > 1:
            raise NotImplementedError

        if self.eigenvecs_prior_ is None:
            _eigvals, self.eigenvecs_ = np.linalg.eigh(
                autocorr(x)
            )  # ((n_nodes,), (n_nodes, n_nodes))
        else:
            self.eigenvecs_ = self.eigenvecs_prior_

        super()._initialize(x)

    # UTILITY FUNCTIONS ############################################################################

    def _squared_pdiffs(self, x: NDArray) -> NDArray:
        return (x @ self.eigenvecs_) ** 2  # shape (n_samples, n_nodes)

    # UPDATE FUNCTIONS #############################################################################

    def _op_adj_coefficients(self, weights: NDArray, dualv: NDArray) -> NDArray:
        return weights @ dualv.T

    def _op_adj_weights(self, coefficients: NDArray, dualv: NDArray) -> NDArray:
        return coefficients @ dualv

    # FIXME: Should I leave weights update in spectral space (as it is now?)

    def _update_dual(self, weights: NDArray, coefficients: NDArray, dual: NDArray):
        # I work directly in eigenvalue space

        # z1 = dualv + step * bilinear_op(x_overshoot, y_overshoot)
        # z1 -= step * prox_h(z1 / step, 1 / step)

        n_atoms, _n_samples = coefficients.shape
        sigma = self.step_dual / n_atoms

        step = coefficients.T @ weights
        return prox_gdet_star_spectral_update(sigma=sigma, eigvals=dual + sigma * step)

    def score(self, x: NDArray[np.float64], _y=None) -> float:
        return np.nan
