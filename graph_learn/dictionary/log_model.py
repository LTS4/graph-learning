"""Graph components learning original method"""
from __future__ import annotations

from warnings import warn

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from graph_learn.evaluation import relative_error
from graph_learn.operators import squared_pdiffs
from graph_learn.smooth_learning import sum_squareform

from .base_model import GraphDictBase


class GraphDictLog(GraphDictBase):
    """Graph components learning original method"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sum_op: sparse.csr_array  # shape: (n_nodes, n_edges)
        self._sum_op_t: sparse.csr_array  # shape: (n_edges, n_nodes)

    def _init_dual(self, n_samples: int):
        return np.zeros((self.n_nodes_, n_samples))

    def _initialize(self, x: NDArray) -> None:
        super()._initialize(x)

        self._sum_op, self._sum_op_t = sum_squareform(self.n_nodes_)

    def _op_adj_activations(self, weights: NDArray, dualv: NDArray) -> NDArray:
        """Compute the adjoint of the bilinear inst-degree operator wrt activations

        Args:
            weights (NDArray): Array of weights of shape (n_components, n_edges)
            dualv (NDArray): Instantaneous degrees, of shape (n_nodes, n_samples)

        Returns:
            NDArray: Adjoint activations of shape (n_components, n_samples)
        """
        return weights @ (self._sum_op_t @ dualv)

    def _op_adj_weights(self, activations: NDArray, dualv: NDArray) -> NDArray:
        """Compute the adjoint of the bilinear inst-degree operator wrt weights

        Args:
            activations (NDArray): Array of activations of shape (n_components, n_samples)
            dualv (NDArray): Instantaneous degrees, of shape (n_nodes, n_samples)

        Returns:
            NDArray: Dual weights of shape (n_components, n_edges)
        """
        return activations @ (self._sum_op_t @ dualv).T

    def _update_activations(
        self,
        sq_pdiffs: NDArray[np.float_],
        activations: NDArray[np.float_],
        dual: NDArray[np.float_],
    ) -> NDArray[np.float_]:
        """Update activations

        Args:
            sq_pdiffs (NDArray[np.float_]): Squared pairwise differences of shape (n_samples, n_edges)
            activations (NDArray[np.float_]): Current activations of shape (n_atoms, n_samples)
            dual (NDArray[np.float_]): Dual variable (instantaneous Laplacians)
                of shape (n_samples, n_nodes, n_nodes)

        Returns:
            NDArray[np.float_]: Updated activations of shape (n_atoms, n_samples)
        """

        n_samples = sq_pdiffs.shape[0]
        # Smoothness
        step = self.weights_ @ sq_pdiffs.T

        if self.window_size > 1:
            # average non-overlapping windows
            step = step.reshape(self.n_atoms, -1, self.window_size).mean(2)

        # L1 regularization
        step += self.l1_a

        if self.log_a > 0:
            # step -= self.log_a / activations.sum(axis=0, keepdims=True)
            activations[:, activations.sum(0) < 1e-8] = self.log_a

        if self.l1_diff_a > 0:
            step -= self.l1_diff_a * (
                np.diff(np.sign(np.diff(activations, axis=1)), axis=1, prepend=0, append=0)
            )

        dual_step = self._op_adj_activations(self.weights_, dual)

        # Proximal and gradient step
        # NOTE: the step might be divided by the operator norm
        activations = activations - self.step_a / n_samples * (dual_step + step)

        # Projection
        activations[activations < 0] = 0
        activations[activations > 1] = 1

        if np.allclose(activations, 0):
            warn("All activations dropped to 0", UserWarning)
            activations.fill(1)

        return activations

    def _update_weights(
        self,
        sq_pdiffs: NDArray[np.float_],
        weights: NDArray[np.float_],
        dual: NDArray[np.float_],
    ) -> NDArray[np.float_]:
        """Update the weights of the model

        Args:
            x (NDArray[np.float_]): Signal matrix of shape (n_samples, n_nodes)
            dual (NDArray[np.float_]): Dual variable (instantaneous Laplacians)
                of shape (n_samples, n_nodes, n_nodes)

        Returns:
            NDArray[np.float_]: updated weights of shape (n_atoms, (n_nodes**2 - n_nodes) // 2)
        """
        n_samples = sq_pdiffs.shape[0]

        # Smoothness
        step = self.activations_ @ sq_pdiffs
        step += self.l1_w

        if self.ortho_w > 0:
            step += self.ortho_w * (weights.sum(0, keepdims=True) - weights)

        dual_step = self._op_adj_weights(self.activations_, dual)

        # Proximal update
        weights = weights - self.step_w / n_samples * (dual_step + step)

        # Projection
        weights[weights < 0] = 0
        return weights

    def _update_dual(
        self,
        weights: NDArray[np.float_],
        activations: NDArray[np.float_],
        dual: NDArray[np.float_],
    ):
        # z1 = dualv + step * bilinear_op(x_overshoot, y_overshoot)
        # z1 -= step * prox_h(z1 / step, 1 / step)

        sigma = self.step_dual  # / _n_samples / op_norm

        step = self._sum_op @ weights.T @ activations
        dual = dual + sigma * step

        return (dual - np.sqrt(dual**2 + 4 * sigma)) / 2

    def _fit_step(self, sq_pdiffs: NDArray[np.float_]) -> tuple[float, float]:
        # primal update
        # x1 = prox_gx(x - step * (op_adjx(x, dualv) + gradf_x(x, y, gz)), step)
        # y1 = prox_gy(y - step * (op_adjy(y, dualv) + gradf_y(x, y, gz)), step)

        activations = np.repeat(
            self._update_activations(
                sq_pdiffs, self.activations_[:, :: self.window_size], dual=self.dual_
            ),
            repeats=self.window_size,
            axis=1,
        )
        weights = self._update_weights(
            sq_pdiffs, self.weights_, dual=np.repeat(self.dual_, self.window_size, 0)
        )

        # dual update
        # x_overshoot = 2 * activations - self.activations_
        self.dual_ = self._update_dual(
            weights=2 * weights - self.weights_,
            activations=2 * activations[:, :: self.window_size]
            - self.activations_[:, :: self.window_size],
            dual=self.dual_,
        )

        a_rel_change = relative_error(self.activations_.ravel(), activations.ravel())
        w_rel_change = relative_error(self.weights_.ravel(), weights.ravel())

        self.activations_ = activations
        self.weights_ = weights

        return a_rel_change, w_rel_change

    def predict(self, x: NDArray[np.float_]) -> NDArray[np.float_]:
        """Predict activations for a given signal"""
        n_samples, n_nodes = x.shape
        if n_nodes != self.n_nodes_:
            raise ValueError(f"Number of nodes mismatch, got {n_nodes} instead of {self.n_nodes_}")

        activations = self._init_activations(n_samples)
        dual = self._init_dual(n_samples // self.window_size)

        # op_act_norm = op_activations_norm(laplacian_squareform_vec(self.weights_))

        sq_pdiffs = squared_pdiffs(x)

        for _ in range(self.max_iter):
            activations_u = np.repeat(
                self._update_activations(sq_pdiffs, activations[:, :: self.window_size], dual=dual),
                repeats=self.window_size,
                axis=1,
            )

            dual = self._update_dual(
                self.weights_,
                2 * activations_u[:, :: self.window_size] - activations[:, :: self.window_size],
                dual,
            )

            if np.linalg.norm((activations_u - activations).ravel()) < self.tol:
                return activations_u

            activations = activations_u

        return activations
