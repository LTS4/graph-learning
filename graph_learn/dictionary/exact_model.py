"""Graph components learning original method"""
from __future__ import annotations

from warnings import warn

import numpy as np
from numpy.typing import NDArray

from graph_learn.evaluation import relative_error
from graph_learn.operators import (
    laplacian_squareform_vec,
    op_activations_norm,
    op_adj_activations,
    op_adj_weights,
    op_weights_norm,
    prox_gdet_star,
    squared_pdiffs,
)

from .base_model import GraphDictBase


class GraphDictExact(GraphDictBase):
    """Graph components learning original method"""

    def _update_activations(
        self,
        sq_pdiffs: NDArray[np.float_],
        activations: NDArray[np.float_],
        dual: NDArray[np.float_],
        op_norm=1,
    ) -> NDArray[np.float_]:
        """Update activations

        Args:
            x (NDArray[np.float_]): Signal matrix of shape (n_samples, n_nodes)
            activations (NDArray[np.float_]): Current activations of shape (n_atoms, n_samples)
            dual (NDArray[np.float_]): Dual variable (instantaneous Laplacians)
                of shape (n_samples, n_nodes, n_nodes)

        Returns:
            NDArray[np.float_]: Updated activations of shape (n_atoms, n_samples)
        """

        # Smoothness
        step = self.weights_ @ sq_pdiffs.T

        if self.window_size > 1:
            step = np.repeat(
                # average non-overlapping windows
                step.reshape(self.n_atoms, -1, self.window_size).mean(2),
                self.window_size,
                axis=1,
            )

        # L1 regularization
        step += self.l1_a

        if self.log_a > 0:
            step -= self.log_a / activations.sum(axis=0, keepdims=True)

        if self.l1_diff_a > 0:
            step -= self.l1_diff_a * (
                np.diff(np.sign(np.diff(activations, axis=1)), axis=1, prepend=0, append=0)
            )

        dual_step = op_adj_activations(self.weights_, dual)
        # ratio = np.abs(dual_step[np.isfinite(step)].mean() / step[np.isfinite(step)].mean())

        # Proximal and gradient step
        # NOTE: the step might be divided by the operator norm
        activations = activations - self.step_a / self.n_samples_ / op_norm * (dual_step + step)

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
        op_norm=1,
    ) -> NDArray[np.float_]:
        """Update the weights of the model

        Args:
            x (NDArray[np.float_]): Signal matrix of shape (n_samples, n_nodes)
            dual (NDArray[np.float_]): Dual variable (instantaneous Laplacians)
                of shape (n_samples, n_nodes, n_nodes)

        Returns:
            NDArray[np.float_]: updated weights of shape (n_atoms, (n_nodes**2 - n_nodes) // 2)
        """
        # Smoothness
        step = self.activations_ @ sq_pdiffs

        step += self.l1_w

        if self.ortho_w > 0:
            # grad_step = (
            #     np.ones((self.n_atoms, 1)) - np.eye(self.n_atoms)
            # ) @ self.weights_
            step += self.ortho_w(weights.sum(0, keepdims=True) - weights)

        dual_step = op_adj_weights(self.activations_, dual)

        # ratio = np.abs(dual_step[np.isfinite(step)].mean() / step[np.isfinite(step)].mean())
        # Proximal update
        weights = weights - self.step_w / self.n_samples_ / op_norm * (dual_step + step)

        # Projection
        weights[weights < 0] = 0
        return weights

    def _update_dual(
        self,
        weights: NDArray[np.float_],
        activations: NDArray[np.float_],
        dual: NDArray[np.float_],
        op_norm=1,
    ):
        # z1 = dualv + step * bilinear_op(x_overshoot, y_overshoot)
        # z1 -= step * prox_h(z1 / step, 1 / step)

        sigma = self.step_dual / self.n_samples_ / op_norm

        return prox_gdet_star(
            dual + sigma * laplacian_squareform_vec(activations.T @ weights), sigma=sigma
        )

    def _fit_step(self, sq_pdiffs: NDArray[np.float_]) -> (float, float):
        # primal update
        # x1 = prox_gx(x - step * (op_adjx(x, dualv) + gradf_x(x, y, gz)), step)
        # y1 = prox_gy(y - step * (op_adjy(y, dualv) + gradf_y(x, y, gz)), step)
        activations = self._update_activations(sq_pdiffs, self.activations_, dual=self.dual_)
        weights = self._update_weights(sq_pdiffs, self.weights_, dual=self.dual_)

        # dual update
        # x_overshoot = 2 * activations - self.activations_
        # NOTE: I tested overshoot and solutions are either exactly the same, or slightly worse
        op_norm = op_activations_norm(laplacian_squareform_vec(weights)) * op_weights_norm(
            activations, self.n_nodes_
        )
        self.dual_ = self._update_dual(
            weights=weights,
            activations=activations,
            dual=self.dual_,
            op_norm=1 / op_norm,
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
        dual = np.zeros((n_samples, self.n_nodes_, self.n_nodes_))

        op_act_norm = op_activations_norm(laplacian_squareform_vec(self.weights_))

        sq_pdiffs = squared_pdiffs(x)

        for _ in range(self.max_iter):
            activations_u = self._update_activations(sq_pdiffs, activations, dual=dual)

            op_norm = op_act_norm * op_weights_norm(activations_u, self.n_nodes_)
            dual = self._update_dual(self.weights_, activations_u, dual, op_norm=1 / op_norm)

            if np.linalg.norm((activations_u - activations).ravel()) < self.tol:
                return activations_u

            activations = activations_u

        return activations
