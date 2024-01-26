"""Graph components learning original method"""
from __future__ import annotations

from warnings import warn

import numpy as np
from numpy.random import RandomState
from numpy.typing import NDArray

from graph_learn.evaluation import relative_error

# from graph_learn import OptimizationError
from graph_learn.operators import (
    laplacian_squareform_vec,
    op_adj_weights,
    prox_gdet_star,
    simplex_projection,
    squared_pdiffs,
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
        random_state: RandomState = None,
        init_strategy: str = "uniform",
        n_init: int = 1,
        weight_prior: float | NDArray[np.float_] = None,
        activation_prior: float | NDArray[np.float_] = None,
        verbose: int = 0,
        combination_update: bool = False,
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
            random_state=random_state,
            init_strategy=init_strategy,
            n_init=n_init,
            weight_prior=weight_prior,
            activation_prior=activation_prior,
            verbose=verbose,
        )

        # Combinations are binary representation of their column index
        self._combinations = powerset_matrix(n_atoms=self.n_atoms)  # shape (n_atoms, 2**n_atoms)
        self.combination_update = combination_update

        self.combi_p_: NDArray[np.float_]
        self.dual_eigvals_: NDArray[np.float_]

        if window_size > 1:
            raise NotImplementedError()

    def _init_dual(self, n_samples: int):
        return super()._init_dual(2**self.n_atoms)

    def _initialize(self, x: NDArray) -> None:
        super()._initialize(x)

        self.combi_p_ = combinations_prob(self.activations_, self._combinations)
        self.dual_eigvals_ = np.zeros(self.dual_.shape[:2])

        # self.combi_p_[0] = 0
        # self.combi_p_[1:] = simplex_projection(self.combi_p_[1:].T).T
        # self.activations_ = self._combinations @ self.combi_p_

    def _update_combi_p(
        self,
        sq_pdiffs: NDArray,
        combi_p: NDArray,
        neg_dual_eigvals: NDArray,
    ) -> NDArray[np.float_]:
        # we do not work with the empty component
        # FIXME: now the steps are on a similar scale, but I don't know why, nor if it is correct

        step_size = self.step_a / self.n_samples_
        inst_weights = self._combinations[:, 1:].T @ self.weights_
        step = np.zeros_like(combi_p)
        combi_sum = combi_p[1:].sum(1, keepdims=True)

        # FIXME: smoothness scale is too big
        # smoothness
        step[1:] = inst_weights @ sq_pdiffs.T  # / inst_weights.sum(1, keepdims=True)

        # dual step
        neg_dual_eigvals = neg_dual_eigvals[1:]
        eigmask = neg_dual_eigvals > 1e-8
        # fmt: off
        dual_step = (
            eigmask.sum(1, keepdims=True) # rank(dual)
            * np.log(np.where(combi_sum > 1e-8, combi_sum, 1))
            - np.sum(np.log(np.where(eigmask, neg_dual_eigvals, 1)), axis=1, keepdims=True)
        )
        # dual_step[np.isnan(dual_step)] = 0
        # fmt: on
        step[1:] -= dual_step

        # FIXME: to be valid combinations prob it is not enough to be in the simplex
        # proximal update (project to simplex)
        combi_p = combi_p - step_size * step

        combi_p = simplex_projection(combi_p.T).T
        # combi_p[1:] = simplex_projection(combi_p[1:].T).T

        return combi_p

    def _update_activations(
        self,
        sq_pdiffs: NDArray[np.float_],
        activations: NDArray[np.float_],
        combi_p: NDArray[np.float_],
        neg_dual_eigvals: NDArray[np.float_],
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
        step_size = self.step_a / self.n_samples_ / op_norm

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

        if self.l1_diff_a > 0:
            step -= self.l1_diff_a * (
                np.diff(np.sign(np.diff(activations, axis=1)), axis=1, prepend=0, append=0)
            )

        # DUAL STEP
        # recompute combi_p
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

        # dual step
        eigmask = neg_dual_eigvals > 1e-8

        # FIXME: I think there is an error in the gradient d/ddelta combi_p
        dual_step = (inter * (2 * self._combinations[:, :, np.newaxis] - 1)).transpose(
            (0, 2, 1)
        ) @ (
            eigmask.sum(1) * np.log(np.where((summed := combi_p.sum(1)) > 1e-8, summed, 1))
            - np.sum(np.log(np.where(eigmask, neg_dual_eigvals, 1)), axis=1)
        )

        # Proximal and gradient step
        # NOTE: the step might be divided by the operator norm
        activations = activations - step_size * (step - dual_step)

        # Projection
        activations[activations < 0] = 0
        activations[activations > 1] = 1

        # Restore dropped activations
        if self.log_a > 0:
            activations[:, activations.sum(0) < 1e-8] = self.log_a

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
        # pylint: disable=arguments-renamed
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
        n_samples = sq_pdiffs.shape[0]

        # Smoothness
        step = self.activations_ @ sq_pdiffs
        step += self.l1_w

        if self.ortho_w > 0:
            step += self.ortho_w * (weights.sum(0, keepdims=True) - weights)

        # FIXME: I think there should be some relation to combi_p.sum(1) here
        dual_step = op_adj_weights(self._combinations, dual)

        # Proximal update
        weights = weights - self.step_w / n_samples / op_norm * (step + dual_step)

        # Projection
        weights[weights < 0] = 0
        return weights

    def _update_dual(
        self,
        weights: NDArray[np.float_],
        combi_p: NDArray[np.float_],
        dual: NDArray[np.float_],
        op_norm=1,
    ) -> tuple[NDArray, NDArray]:
        """This is supposed to be the specific method"""
        # pylint: disable=arguments-renamed
        # z1 = dualv + step * bilinear_op(x_overshoot, y_overshoot)
        # z1 -= step * prox_h(z1 / step, 1 / step)

        n_combi = dual.shape[0]
        # n_atoms = weights.shape[0]
        sigma = self.step_dual / op_norm / n_combi

        step = laplacian_squareform_vec(self._combinations.T @ weights)
        dual = dual.copy()
        eigvals = np.zeros(dual.shape[:2])
        dual, eigvals = prox_gdet_star(dual + sigma * step, sigma=sigma, return_eigvals=True)

        return dual, eigvals

    def _fit_step(self, sq_pdiffs: NDArray[np.float_]) -> (float, float):
        # combi_p = combinations_prob(self.activations_, self._combinations)

        # primal update
        # x1 = prox_gx(x - step * (op_adjx(x, dualv) + gradf_x(x, y, gz)), step)
        # y1 = prox_gy(y - step * (op_adjy(y, dualv) + gradf_y(x, y, gz)), step)

        combi_e = self.combi_p_.sum(1)

        if self.combination_update:
            combi_p = self._update_combi_p(
                sq_pdiffs,
                combi_p=self.combi_p_,
                neg_dual_eigvals=-self.dual_eigvals_ * combi_e[:, np.newaxis],
            )
            activations = self._combinations @ combi_p
        else:
            activations = self._update_activations(
                sq_pdiffs,
                self.activations_,
                combi_p=self.combi_p_,
                neg_dual_eigvals=-self.dual_eigvals_ * combi_e[:, np.newaxis],
            )
            combi_p = combinations_prob(activations, self._combinations)

        weights = self._update_weights(
            sq_pdiffs,
            self.weights_,
            combi_p=self.combi_p_,
            dual=self.dual_ * combi_e[:, np.newaxis, np.newaxis],
        )

        op_norm = 1
        self.dual_, self.dual_eigvals_ = self._update_dual(
            weights=weights,
            combi_p=combi_p,
            dual=self.dual_,
            op_norm=1 / op_norm,
        )

        a_rel_change = relative_error(self.activations_.ravel(), activations.ravel())
        w_rel_change = relative_error(self.weights_.ravel(), weights.ravel())

        self.combi_p_ = combi_p
        self.activations_ = activations
        self.weights_ = weights

        return a_rel_change, w_rel_change

    def predict(self, x: NDArray[np.float_]) -> NDArray[np.float_]:
        """Predict activations for a given signal"""
        _n_samples, n_nodes = x.shape
        if n_nodes != self.n_nodes_:
            raise ValueError(f"Number of nodes mismatch, got {n_nodes} instead of {self.n_nodes_}")

        sq_pdiffs = squared_pdiffs(x)

        activations = self._init_activations(x.shape[0])
        combi_p = combinations_prob(activations)

        for _ in range(self.max_iter):
            combi_e = combi_p.sum(1)

            if self.combination_update:
                combi_p = self._update_combi_p(
                    sq_pdiffs,
                    combi_p=combi_p,
                    neg_dual_eigvals=-self.dual_eigvals_ * combi_e[:, np.newaxis],
                )
                activations_u = self._combinations @ combi_p
            else:
                activations_u = self._update_activations(
                    sq_pdiffs,
                    activations,
                    combi_p=combi_p,
                    neg_dual_eigvals=-self.dual_eigvals_ * combi_e[:, np.newaxis],
                )
                combi_p = combinations_prob(activations)

            if np.linalg.norm((activations_u - activations).ravel()) < self.tol:
                return activations_u

            activations = activations_u

        return activations
