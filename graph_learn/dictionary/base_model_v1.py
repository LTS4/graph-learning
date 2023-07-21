"""Graph components learning original method"""
from __future__ import annotations

from typing import Callable
from warnings import warn

import numpy as np
from numpy.random import RandomState
from numpy.typing import NDArray
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator

# from graph_learn import OptimizationError
from graph_learn.components.utils import laplacian_squareform_vec, prox_gdet_star
from graph_learn.evaluation import relative_error


class GraphDictionary(BaseEstimator):
    """Graph components learning original method"""

    def __init__(
        self,
        n_components=1,
        l1_weights: float = 1,
        l1_activations: float = 0,
        *,
        max_iter: int = 50,
        alpha: float = 1e-2,
        mc_samples: int = 100,
        tol: float = 1e-3,
        random_state: RandomState = None,
        init_strategy: str = "uniform",
        weight_prior: float | NDArray[np.float_] = None,
        activation_prior: float | NDArray[np.float_] = None,
        verbose: int = 0,
    ) -> None:
        super().__init__()

        self.n_components = n_components
        self.l1_weights = l1_weights
        # self.ortho_weights = ortho_weights
        self.l1_activations = l1_activations

        self.max_iter = max_iter
        self.alpha = alpha
        self.mc_samples = mc_samples
        self.tol = tol

        self.random_state = RandomState(random_state)
        self.init_strategy = init_strategy
        self.weight_prior = weight_prior
        self.activation_prior = activation_prior

        self.verbose = verbose

        self._combination_map = np.array([2**k for k in range(self.n_components)])
        # Combinations are binary representation of their column index
        self._combinations = np.array(
            [
                [(j >> i) & 1 for j in range(2**self.n_components)]
                for i in range(self.n_components)
            ]
        )  # shape (n_components, 2**n_components)

        self.activations_: NDArray[np.float_]  # shape (n_components, n_samples)
        self.weights_: NDArray[np.float_]  # shape (n_components, n_edges )
        self.dual_: NDArray[np.float_]  # shape (2**n_components, n_nodes, n_nodes)
        self.n_nodes_: int
        self.n_samples_: int
        self.scale_: float
        self.converged_: int
        self.history_: dict[int, dict[str, int]]

    def _initialize(self, x: NDArray) -> None:
        self.n_samples_, self.n_nodes_ = x.shape

        self.weights_ = np.ones((self.n_components, (self.n_nodes_**2 - self.n_nodes_) // 2))
        self.activations_ = np.ones((self.n_components, self.n_samples_))
        self.dual_ = np.zeros((2**self.n_components, self.n_nodes_, self.n_nodes_))

        self.converged_ = -1

    def _component_pdist_sq(self, x: NDArray[np.float_]) -> NDArray[np.float_]:
        """Compute pairwise square distances on each componend, based on activations

        Args:
            x (NDArray[np.float_]): Design matrix of shape (n_samples, n_nodes)

        Returns:
            NDArray[np.float_]: Pairwise node squared distances of shape
                (n_components, n_nodes, n_nodes)
        """
        # This works with continuous activations
        pdiffs = x[:, np.newaxis, :] - x[:, :, np.newaxis]
        return np.stack(
            [
                squareform(kdiff)
                for kdiff in np.einsum("kt,tnm->knm", self.activations_, pdiffs**2)
            ]
        )

    def _mc_activations(self):
        arange = np.arange(self.n_samples_)

        out = np.zeros((2**self.n_components, self.n_samples_), dtype=np.float_)
        for _ in range(self.mc_samples):
            sampled = self._combination_map @ (
                self.random_state.uniform(size=self.activations_.shape) <= self.activations_
            )

            out[sampled, arange] += 1

        return out / self.mc_samples

    def op_adj_activ(self, weights: NDArray, dualv: NDArray) -> NDArray:
        return np.einsum("tnm,knm->kt", dualv, laplacian_squareform_vec(weights))

    def _prox_l1_activations(self, activations: NDArray, alpha: float) -> NDArray:
        out = activations - alpha * self.l1_activations
        out[out < 0] = 0
        return out

    def _grad_smoothness_activations(self, x: NDArray, activation_mc: NDArray) -> NDArray:
        # TODO: check
        return np.einsum(
            "ct,tn,tm,knm,kc->kt",
            activation_mc,
            x,
            x,
            laplacian_squareform_vec(self.weights_),
            self._combinations,
        )

    def op_adj_weights(self, activations: NDArray, dualv: NDArray) -> NDArray:
        y = np.stack([np.diag(y) - y for y in dualv])
        y += np.transpose(y, (0, 2, 1))

        return np.stack([squareform(lapl) for lapl in np.einsum("kt,tnm->knm", activations, y)])

    def _grad_smoothness_weights(self, x: NDArray, activation_mc: NDArray) -> NDArray:
        laplacian_grad = np.einsum(
            "ct,tn,tm,kc->knm",
            activation_mc,
            x,
            x,
            self._combinations,
        )

        return np.stack(
            [
                squareform(np.repeat(np.diag(gg), gg.shape[0]).reshape(gg.shape), checks=False)
                - 2 * squareform(gg, checks=False)
                for gg in laplacian_grad
            ]
        )

    def _prox_l1_weights(self, weights: NDArray, alpha: float) -> NDArray:
        out = weights - alpha * self.l1_weights * self.n_samples_

        out[out < 0] = 0
        return out

    def _update_activations(
        self, x: NDArray[np.float_], mc_activations: NDArray[np.float_], dual: NDArray[np.float_]
    ) -> NDArray[np.float_]:
        return self._prox_l1_activations(
            self.activations_
            - self.alpha
            * (
                self.op_adj_activ(self.weights_, dual)
                + self._grad_smoothness_activations(x, mc_activations)
            ),
            alpha=self.alpha,
        )

    def _update_weights(
        self, x: NDArray[np.float_], mc_activations: NDArray[np.float_], dual: NDArray[np.float_]
    ) -> NDArray[np.float_]:
        return self._prox_l1_weights(
            self.weights_
            - self.alpha
            * (
                self.op_adj_weights(self.activations_, dual)
                + self._grad_smoothness_weights(x, mc_activations)
            ),
            self.alpha,
        )

    def fit(
        self,
        x: NDArray[np.float_],
        _y=None,
        *,
        callback: Callable[[GraphDictionary, int]] = None,
    ) -> GraphDictionary:
        self._initialize(x)

        warn("The dual objective is too weak to keep weights up", RuntimeWarning)

        for i in range(self.max_iter):
            mc_activations = self._mc_activations()
            dual = np.einsum("ct,cnm->tnm", mc_activations, self.dual_)

            # primal update
            # x1 = prox_gx(x - alpha * (op_adjx(x, dualv) + gradf_x(x, y, gz)), alpha)
            activations = self._update_activations(x, mc_activations, dual)

            # y1 = prox_gy(y - alpha * (op_adjy(y, dualv) + gradf_y(x, y, gz)), alpha)
            weights = self._update_weights(x, mc_activations, dual)

            # # dual update
            # x_overshoot = 2 * activations - self.activations_
            y_overshoot = 2 * weights - self.weights_

            # z1 = dualv + alpha * bilinear_op(x_overshoot, y_overshoot)
            # z1 -= alpha * prox_h(z1 / alpha, 1 / alpha)
            self.dual_ = self.dual_ + self.alpha * np.einsum(
                "kc,knm->cnm", self._combinations, laplacian_squareform_vec(y_overshoot)
            )
            self.dual_ = prox_gdet_star(self.dual_, sigma=self.alpha)
            # return x1, y1, z1

            self.activations_ = activations
            self.weights_ = weights

            if callback is not None:
                callback(self, i)

            if (
                False
                and (relative_error(self.activations_, activations) < self.tol)
                and (relative_error(self.weights_, weights) < self.tol)
            ):
                self.converged_ = i
                return self

        return self
