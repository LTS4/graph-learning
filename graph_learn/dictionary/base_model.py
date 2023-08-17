"""Graph components learning original method"""
from __future__ import annotations

from time import time
from typing import Callable
from warnings import warn

import numpy as np
import pandas as pd
from numpy.random import RandomState
from numpy.typing import NDArray
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator

from graph_learn.evaluation import relative_error

# from graph_learn import OptimizationError
from graph_learn.operators import (
    laplacian_squareform_vec,
    op_activations_norm,
    op_adj_activations,
    op_adj_weights,
    op_weights_norm,
    prox_gdet_star,
)


def mc_activations(
    activations: NDArray[np.float], mc_samples: int, random_state: RandomState
) -> NDArray[np.float]:
    """Sample combinations of activations using Monte-Carlo"""
    n_components, n_samples = activations.shape
    combination_map = 2 ** np.arange(n_components)
    arange = np.arange(n_samples)

    out = np.zeros((2**n_components, n_samples), dtype=float)
    for _ in range(mc_samples):
        sampled = combination_map @ (random_state.uniform(size=activations.shape) <= activations)

        out[sampled, arange] += 1

    return out / mc_samples


class GraphDictionary(BaseEstimator):
    """Graph components learning original method"""

    def __init__(
        self,
        n_components=1,
        *,
        window_size: int = 1,
        l1_w: float = 0,
        ortho_w: float = 0,
        smooth_a: float = 1,
        l1_a: float = 0,
        log_a: float = 0,
        l1_diff_a: float = 0,
        max_iter: int = 1000,
        step: float = 1,
        step_a: float = None,
        step_w: float = None,
        step_dual: float = None,
        mc_samples: int = 1000,
        tol: float = 1e-3,
        reduce_step_on_plateau: bool = False,
        random_state: RandomState = None,
        init_strategy: str = "uniform",
        weight_prior: float | NDArray[np.float_] = None,
        activation_prior: float | NDArray[np.float_] = None,
        verbose: int = 0,
    ) -> None:
        super().__init__()

        self.n_components = n_components
        self.window_size = window_size
        self.ortho_w = ortho_w
        self.l1_w = l1_w
        self.smooth_a = smooth_a
        self.l1_a = l1_a
        self.log_a = log_a
        self.l1_diff_a = l1_diff_a

        self.max_iter = max_iter

        self.step_a = step_a or step
        self.step_w = step_w or step
        self.step_dual = step_dual or step

        self.mc_samples = mc_samples
        self.tol = tol
        self.reduce_step_on_plateau = reduce_step_on_plateau

        self.random_state = RandomState(random_state)
        self.init_strategy = init_strategy
        self.weight_prior = weight_prior
        self.activation_prior = activation_prior

        self.verbose = verbose

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
        self.history_: pd.DataFrame
        self.fit_time_: float

    def _initialize(self, x: NDArray) -> None:
        self.n_samples_, self.n_nodes_ = x.shape

        self.weights_ = np.ones((self.n_components, (self.n_nodes_**2 - self.n_nodes_) // 2))
        self.activations_ = np.ones((self.n_components, self.n_samples_))

        match self.init_strategy:
            case "uniform":
                if self.weight_prior is not None and self.activation_prior is not None:
                    raise ValueError("Need one free parameter for uniform initialization")

                if self.weight_prior is None:
                    self.weights_ = self.random_state.uniform(
                        size=(self.n_components, (self.n_nodes_**2 - self.n_nodes_) // 2)
                    )
                else:
                    self.weights_ *= self.weight_prior

                if self.activation_prior is None:
                    self.activations_ = self.random_state.uniform(
                        size=(self.n_components, self.n_samples_)
                    )
                else:
                    self.activations_ *= self.activation_prior

            case "exp":
                self.weights_ = self.random_state.exponential(
                    scale=self.weight_prior or 1, size=self.weights_.shape
                )
                self.activations_ = self.random_state.exponential(
                    scale=self.activation_prior or 1, size=self.activations_.shape
                )

            case "exact":
                if self.weight_prior is not None:
                    self.weights_ *= self.weight_prior

                if self.activation_prior is not None:
                    self.activations_ *= self.activation_prior
            case _:
                raise ValueError(f"Invalid init strategy {self.init_strategy}")

        if self.window_size > 1:
            self.activations_ = np.repeat(
                self.activations_[:, :: self.window_size], self.window_size, axis=1
            )[:, : self.n_samples_]

        self.dual_ = np.zeros((2**self.n_components, self.n_nodes_, self.n_nodes_))

        self.converged_ = -1
        self.fit_time_ = -1
        self.history_ = pd.DataFrame(
            data=-np.ones((self.max_iter, 2), dtype=float),
            columns=["activ_change", "weight_change"],
            index=np.arange(self.max_iter),
        )

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

    def _update_activations(
        self,
        x: NDArray[np.float_],
        activations: NDArray[np.float_],
        mc_a: NDArray[np.float_],
        dual: NDArray[np.float_],
    ) -> NDArray[np.float_]:
        """Update activations

        Args:
            x (NDArray[np.float_]): Signal matrix of shape (n_samples, n_nodes)
            activations (NDArray[np.float_]): Current activations of shape (n_components, n_samples)
            mc_a (NDArray[np.float_]): Monte-Carlo probabilities of combinations
                for each sample. Array of shape (2**n_components, n_samples)
            dual (NDArray[np.float_]): Dual variable (instantaneous Laplacians)
                of shape (2**n_components, n_nodes, n_nodes)

        Returns:
            NDArray[np.float_]: Updated activations of shape (n_components, n_samples)
        """
        laplacians = laplacian_squareform_vec(self.weights_)

        # Dual step
        # We apply the MC after the adjoint as n_samples >> n_combinantions
        step = self.n_samples_ * op_adj_activations(self.weights_, dual) @ mc_a

        if self.log_a > 0:
            step -= self.log_a / activations.sum(axis=0, keepdims=True)

        if self.l1_diff_a > 0:
            step -= self.l1_diff_a * (
                np.diff(np.sign(np.diff(activations, axis=1)), axis=1, prepend=0, append=0)
            )

        smoothness = np.einsum("ktn,tn->kt", x @ laplacians, x) * self.smooth_a
        if self.window_size > 1:
            smoothness = np.repeat(
                # average non-overlapping windows
                smoothness.reshape(self.n_components, -1, self.window_size).mean(2),
                self.window_size,
                axis=1,
            )

        # Note: the step might be divided by the operator norm
        step += smoothness + self.l1_a

        # Proximal and gradient step
        activations = activations - self.step_a / self.n_samples_ * step

        # Projection
        activations[activations < 0] = 0
        activations[activations > 1] = 1

        if np.allclose(activations, 0):
            warn("All activations dropped to 0", UserWarning)
            activations.fill(1)

        return activations

    def _update_weights(
        self,
        x: NDArray[np.float_],
        weights: NDArray[np.float_],
        mc_a: NDArray[np.float_],
        dual: NDArray[np.float_],
    ) -> NDArray[np.float_]:
        """Update the weights of the model

        Args:
            x (NDArray[np.float_]): Signal matrix of shape (n_samples, n_nodes)
            mc_a (NDArray[np.float_]): Monte-Carlo probabilities of combinations
                for each sample. Array of shape (2**n_components, n_samples)
            dual (NDArray[np.float_]): Dual variable (instantaneous Laplacians)
                of shape (2**n_components, n_nodes, n_nodes)

        Returns:
            NDArray[np.float_]: updated weights of shape (n_components, (n_nodes**2 - n_nodes) // 2)
        """
        op_norm = op_weights_norm(activations=self.activations_, n_nodes=self.n_nodes_)
        # smoothness = self._component_pdist_sq(x) / self.n_samples_
        smoothness = (
            self._component_pdist_sq(x)
            / self.n_samples_
            # TODO: this division produces NaN when an atom is never active
            # / self.activations_.mean(1, keepdims=True)  # ** (1 / self.n_components)
        )

        if self.ortho_w > 0:
            # grad_step = (
            #     np.ones((self.n_components, 1)) - np.eye(self.n_components)
            # ) @ self.weights_
            grad_step = weights.sum(0, keepdims=True) - weights
            grad_step *= self.ortho_w
        else:
            grad_step = 0

        # Proximal update
        weights = weights - self.step_w / op_norm * (
            op_adj_weights(self.activations_ @ mc_a.T, dual)  # Dual step
            + grad_step
            + smoothness  # Smoothness step
            + self.l1_w  # L1 step
        )

        # Projection
        weights[weights < 0] = 0
        return weights

    def _fit_step(self, x: NDArray[np.float_]) -> (float, float):
        mc_a = mc_activations(self.activations_, self.mc_samples, self.random_state)

        # primal update
        # x1 = prox_gx(x - step * (op_adjx(x, dualv) + gradf_x(x, y, gz)), step)
        # y1 = prox_gy(y - step * (op_adjy(y, dualv) + gradf_y(x, y, gz)), step)
        activations = self._update_activations(x, self.activations_, mc_a=mc_a, dual=self.dual_)
        weights = self._update_weights(x, self.weights_, mc_a=mc_a, dual=self.dual_)

        # dual update
        # x_overshoot = 2 * activations - self.activations_
        weights_overshoot = 2 * weights - self.weights_

        # z1 = dualv + step * bilinear_op(x_overshoot, y_overshoot)
        # z1 -= step * prox_h(z1 / step, 1 / step)
        # TODO: for fixed activations models I could filter combinations that don't appear
        op_norm = op_weights_norm(
            activations=self.activations_, n_nodes=self.n_nodes_
        )  # * op_activations_norm(lapl=laplacian_squareform_vec(weights_overshoot))
        self.dual_ = self.dual_ + self.step_dual / op_norm * np.einsum(
            "kc,knm->cnm", self._combinations, laplacian_squareform_vec(weights_overshoot)
        )
        self.dual_ = prox_gdet_star(self.dual_, sigma=self.step_dual / op_norm / self.n_samples_)
        # return x1, y1, z1

        a_rel_change = relative_error(self.activations_.ravel(), activations.ravel())
        w_rel_change = relative_error(self.weights_.ravel(), weights.ravel())

        self.activations_ = activations
        self.weights_ = weights

        return a_rel_change, w_rel_change

    def fit(
        self,
        x: NDArray[np.float_],
        _y=None,
        *,
        callback: Callable[[GraphDictionary, int]] = None,
    ) -> GraphDictionary:
        self._initialize(x)
        if callback is not None:
            callback(self, 0)

        start = time()
        for i in range(self.max_iter):
            try:
                self.history_.iloc[i] = a_rel_change, w_rel_change = self._fit_step(x)

                if (a_rel_change < self.tol) and (w_rel_change < self.tol):
                    self.converged_ = i

                if (
                    self.reduce_step_on_plateau
                    and i > 100
                    and (
                        a_rel_change >= self.history_.iloc[i - 10 : i, 0].mean()
                        and w_rel_change >= self.history_.iloc[i - 10 : i, 1].mean()
                    )
                ):
                    warn(f"Divergence detected at step {i}, reducing step size", UserWarning)
                    self.step_a *= 0.7
                    self.step_w *= 0.7
                    self.step_dual *= 0.7

                if callback is not None:
                    callback(self, i + 1)

                if self.converged_ > 0:
                    break

            except KeyboardInterrupt:
                warn("Keyboard interrupt, stopping early")
                break

        self.fit_time_ = time() - start

        return self
