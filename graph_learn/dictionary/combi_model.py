"""Graph components learning original method"""
from __future__ import annotations

from warnings import warn

import numpy as np
from numpy.linalg import eigvalsh
from numpy.random import RandomState
from numpy.typing import NDArray

from graph_learn.evaluation import relative_error

# from graph_learn import OptimizationError
from graph_learn.operators import (
    laplacian_squareform_vec,
    op_activations_norm,
    op_adj_weights,
    op_weights_norm,
    prox_gdet_star,
)

from .base_model import GraphDictBase
from .utils import combinations_prob, powerset_matrix


class GraphDictionary(GraphDictBase):
    """Graph components learning original method"""

    def __init__(
        self,
        n_atoms=1,
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
        tol: float = 0.001,
        reduce_step_on_plateau: bool = False,
        random_state: RandomState = None,
        init_strategy: str = "uniform",
        n_init: int = 1,
        weight_prior: float | NDArray[np.float_] = None,
        activation_prior: float | NDArray[np.float_] = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(
            n_atoms,
            window_size=window_size,
            l1_w=l1_w,
            ortho_w=ortho_w,
            smooth_a=smooth_a,
            l1_a=l1_a,
            log_a=log_a,
            l1_diff_a=l1_diff_a,
            max_iter=max_iter,
            step=step,
            step_a=step_a,
            step_w=step_w,
            step_dual=step_dual,
            tol=tol,
            reduce_step_on_plateau=reduce_step_on_plateau,
            random_state=random_state,
            init_strategy=init_strategy,
            n_init=n_init,
            weight_prior=weight_prior,
            activation_prior=activation_prior,
            verbose=verbose,
        )

        # Combinations are binary representation of their column index
        self._combinations = powerset_matrix(n_atoms=self.n_atoms)  # shape (n_atoms, 2**n_atoms)

    def _init_dual(self, n_samples: int):
        return super()._init_dual(2**self.n_atoms)

    def _update_activations(
        self,
        sq_pdiffs: NDArray[np.float_],
        activations: NDArray[np.float_],
        combi_p: NDArray[np.float_],
        op_norm=1,
    ) -> NDArray[np.float_]:
        # pylint: disable=arguments-renamed
        """Update activations

        Args:
            x (NDArray[np.float_]): Signal matrix of shape (n_samples, n_nodes)
            activations (NDArray[np.float_]): Current activations of shape (n_atoms, n_samples)
            combi_p (NDArray[np.float_]): Probabilities of combinations for each sample.
                Array of shape (2**n_atoms, n_samples)
            dual (NDArray[np.float_]): Dual variable (instantaneous Laplacians)
                of shape (2**n_atoms, n_nodes, n_nodes)

        Returns:
            NDArray[np.float_]: Updated activations of shape (n_atoms, n_samples)
        """

        # Smoothness
        step = (self.weights_ / self.weights_.sum(axis=1, keepdims=True)) @ sq_pdiffs.T

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

        inter = np.tile(
            np.where(
                self._combinations[:, :, np.newaxis],
                activations[:, np.newaxis, :],
                1 - activations[:, np.newaxis, :],
            ),
            reps=(self.n_atoms, 1, 1, 1),
        )
        inter[np.arange(self.n_atoms), np.arange(self.n_atoms)] = 1
        inter = inter.prod(1)

        eigvals = eigvalsh(laplacian_squareform_vec(self._combinations.T @ self.weights_))

        dual_step = (inter * (2 * self._combinations[:, :, np.newaxis] - 1)).transpose(
            (0, 2, 1)
        ) @ (-np.sum(np.log(np.where(eigvals > 0, eigvals, 1)), axis=1))

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
        combi_p: NDArray[np.float_],
        dual: NDArray[np.float_],
        op_norm=1,
    ) -> NDArray[np.float_]:
        """Update the weights of the model

        Args:
            x (NDArray[np.float_]): Signal matrix of shape (n_samples, n_nodes)
            combi_p (NDArray[np.float_]): Monte-Carlo probabilities of combinations
                for each sample. Array of shape (2**n_atoms, n_samples)
            dual (NDArray[np.float_]): Dual variable (instantaneous Laplacians)
                of shape (2**n_atoms, n_nodes, n_nodes)

        Returns:
            NDArray[np.float_]: updated weights of shape (n_atoms, (n_nodes**2 - n_nodes) // 2)
        """
        # activations = self.activations_ @ combi_p.T

        # Smoothness
        step = self.activations_ @ sq_pdiffs
        step += self.l1_w

        if self.ortho_w > 0:
            # grad_step = (
            #     np.ones((self.n_atoms, 1)) - np.eye(self.n_atoms)
            # ) @ self.weights_
            step += self.ortho_w(weights.sum(0, keepdims=True) - weights)

        dual_step = op_adj_weights(self._combinations, dual)

        # Proximal update
        weights = weights - self.step_w / self.n_samples_ / op_norm * (dual_step + step)

        # Projection
        weights[weights < 0] = 0
        return weights

    def _update_dual(
        self,
        weights: NDArray[np.float_],
        combi_p: NDArray[np.float_],
        dual: NDArray[np.float_],
        op_norm=1,
    ):
        # z1 = dualv + step * bilinear_op(x_overshoot, y_overshoot)
        # z1 -= step * prox_h(z1 / step, 1 / step)

        sigma = self.step_dual / op_norm  # / self.n_samples_

        # TODONE: I should filter combinations that don't appear, as their combi sum is zero
        combi_e = combi_p.sum(1)
        active = combi_e > 0

        dual = dual.copy()
        dual[active, :, :] = prox_gdet_star(
            dual[active, :, :]
            + sigma * laplacian_squareform_vec(self._combinations[:, combi_e > 0].T @ weights),
            sigma=sigma * combi_e[active, np.newaxis],
        )

        return dual

    def _fit_step(self, sq_pdiffs: NDArray[np.float_]) -> (float, float):
        combi_p = combinations_prob(self.activations_, self._combinations)

        # primal update
        # x1 = prox_gx(x - step * (op_adjx(x, dualv) + gradf_x(x, y, gz)), step)
        # y1 = prox_gy(y - step * (op_adjy(y, dualv) + gradf_y(x, y, gz)), step)
        activations = self._update_activations(sq_pdiffs, self.activations_, combi_p=combi_p)
        weights = self._update_weights(sq_pdiffs, self.weights_, combi_p=combi_p, dual=self.dual_)

        # dual update
        # op_norm = op_activations_norm(laplacian_squareform_vec(weights)) * op_weights_norm(
        #     self._combinations.astype(float), self.n_nodes_
        # )

        combi_p = combinations_prob(activations, self._combinations)

        self.dual_ = self._update_dual(
            weights=weights,
            # weights=weights,
            combi_p=combi_p,
            dual=self.dual_,
            op_norm=1
            / op_weights_norm(
                self._combinations * combi_p.sum(axis=1)[np.newaxis, :], self.n_nodes_
            ),
            # op_norm=1 / op_norm,
        )

        a_rel_change = relative_error(self.activations_.ravel(), activations.ravel())
        w_rel_change = relative_error(self.weights_.ravel(), weights.ravel())

        self.activations_ = activations
        self.weights_ = weights

        return a_rel_change, w_rel_change

    def predict(self, x: NDArray[np.float_]) -> NDArray[np.float_]:
        """Predict activations for a given signal"""
        activations = self._init_activations(x.shape[0])

        for _ in range(self.max_iter):
            combi_p = combinations_prob(activations)
            activations_u = self._update_activations(x, activations, combi_p=combi_p)

            if np.linalg.norm((activations_u - activations).ravel()) < self.tol:
                return activations_u

            activations = activations_u

        return activations
