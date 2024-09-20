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


class GraphDictCombi(GraphDictBase):
    """Graph components learning original method"""

    def __init__(
        self,
        n_atoms=1,
        *,
        window_size: int = 1,
        l1_w: float = 0,
        ortho_w: float = 0,
        l1_c: float = 0,
        log_c: float = 0,
        l1_diff_c: float = 0,
        max_iter: int = 1000,
        step: float = 1,
        step_c: float = None,
        step_w: float = None,
        step_dual: float = None,
        tol: float = 0.001,
        random_state: RandomState = None,
        init_strategy: str = "uniform",
        n_init: int = 1,
        weight_prior: float | NDArray[np.float64] = None,
        coefficient_prior: float | NDArray[np.float64] = None,
        combination_update: bool = False,
    ) -> None:
        super().__init__(
            n_atoms,
            window_size=window_size,
            l1_w=l1_w,
            ortho_w=ortho_w,
            l1_c=l1_c,
            log_c=log_c,
            l1_diff_c=l1_diff_c,
            max_iter=max_iter,
            step=step,
            step_c=step_c,
            step_w=step_w,
            step_dual=step_dual,
            tol=tol,
            random_state=random_state,
            init_strategy=init_strategy,
            n_init=n_init,
            weight_prior=weight_prior,
            coefficient_prior=coefficient_prior,
        )

        # Combinations are binary representation of their column index
        self._combinations = powerset_matrix(n_atoms=self.n_atoms)  # shape (n_atoms, 2**n_atoms)
        self.combination_update = combination_update

        self.combi_p_: NDArray[np.float64]
        self.dual_eigvals_: NDArray[np.float64]

        if window_size > 1:
            raise NotImplementedError()

    def _init_dual(self, x: NDArray):
        _n_samples, n_nodes = x.shape
        return np.zeros((2**self.n_atoms // self.window_size, n_nodes, n_nodes))

    def _initialize(self, x: NDArray) -> None:
        super()._initialize(x)

        self.combi_p_ = combinations_prob(self.coefficients_, self._combinations)
        self.dual_eigvals_ = np.zeros(self.dual_.shape[:2])

        # self.combi_p_[0] = 0
        # self.combi_p_[1:] = simplex_projection(self.combi_p_[1:].T).T
        # self.coefficients_ = self._combinations @ self.combi_p_

    def _update_combi_p(
        self,
        sq_pdiffs: NDArray,
        combi_p: NDArray,
        neg_dual_eigvals: NDArray,
    ) -> NDArray[np.float64]:
        # we do not work with the empty component
        # FIXME: now the steps are on a similar scale, but I don't know why, nor if it is correct

        n_samples = sq_pdiffs.shape[0]
        step_size = self.step_c / n_samples

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

    def _update_coefficients(
        self,
        sq_pdiffs: NDArray[np.float64],
        coefficients: NDArray[np.float64],
        combi_p: NDArray[np.float64],
        neg_dual_eigvals: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        # pylint: disable=arguments-renamed
        """Update coefficients

        Args:
            x (NDArray[np.float64]): Signal matrix of shape (n_samples, n_nodes)
            coefficients (NDArray[np.float64]): Current coefficients of shape (n_atoms, n_samples)
            combi_p (NDArray[np.float64]): Probabilities of combinations for each sample.
                Array of shape (2**n_atoms, n_samples)
            dual (NDArray[np.float64]): Dual variable (instantaneous Laplacians)
                of shape (2**n_atoms, n_nodes, n_nodes)

        Returns:
            NDArray[np.float64]: Updated coefficients of shape (n_atoms, n_samples)
        """
        n_samples = sq_pdiffs.shape[0]
        step_size = self.step_c / n_samples

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
        step += self.l1_c

        if self.l1_diff_c > 0:
            step -= self.l1_diff_c * (
                np.diff(np.sign(np.diff(coefficients, axis=1)), axis=1, prepend=0, append=0)
            )

        # DUAL STEP
        # recompute combi_p
        inter = np.tile(
            np.where(
                self._combinations[:, :, np.newaxis],
                coefficients[:, np.newaxis, :],
                1 - coefficients[:, np.newaxis, :],
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
        coefficients = coefficients - step_size * (step - dual_step)

        # Projection
        coefficients[coefficients < 0] = 0
        coefficients[coefficients > 1] = 1

        # Restore dropped coefficients
        if self.log_c > 0:
            coefficients[:, coefficients.sum(0) < 1e-8] = self.log_c

        if np.allclose(coefficients, 0):
            warn("All coefficients dropped to 0", UserWarning)
            coefficients.fill(1)

        return coefficients

    def _update_weights(
        self,
        sq_pdiffs: NDArray[np.float64],
        weights: NDArray[np.float64],
        combi_p: NDArray[np.float64],
        dual: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        # pylint: disable=arguments-renamed
        """Update the weights of the model

        Args:
            x (NDArray[np.float64]): Signal matrix of shape (n_samples, n_nodes)
            combi_p (NDArray[np.float64]): Monte-Carlo probabilities of combinations
                for each sample. Array of shape (2**n_atoms, n_samples)
            dual (NDArray[np.float64]): Dual variable (instantaneous Laplacians)
                of shape (2**n_atoms, n_nodes, n_nodes)

        Returns:
            NDArray[np.float64]: updated weights of shape (n_atoms, (n_nodes**2 - n_nodes) // 2)
        """
        n_samples = sq_pdiffs.shape[0]

        # Smoothness
        step = self.coefficients_ @ sq_pdiffs
        step += self.l1_w

        if self.ortho_w > 0:
            step += self.ortho_w * (weights.sum(0, keepdims=True) - weights)

        # FIXME: I think there should be some relation to combi_p.sum(1) here
        dual_step = op_adj_weights(self._combinations, dual)

        # Proximal update
        weights = weights - self.step_w / n_samples * (step + dual_step)

        # Projection
        weights[weights < 0] = 0
        return weights

    def _update_dual(
        self,
        weights: NDArray[np.float64],
        combi_p: NDArray[np.float64],
        dual: NDArray[np.float64],
    ) -> tuple[NDArray, NDArray]:
        """This is supposed to be the specific method"""
        # pylint: disable=arguments-renamed
        # z1 = dualv + step * bilinear_op(x_overshoot, y_overshoot)
        # z1 -= step * prox_h(z1 / step, 1 / step)

        n_combi = dual.shape[0]
        # n_atoms = weights.shape[0]
        sigma = self.step_dual / n_combi

        step = laplacian_squareform_vec(self._combinations.T @ weights)
        dual = dual.copy()
        eigvals = np.zeros(dual.shape[:2])
        dual, eigvals = prox_gdet_star(dual + sigma * step, sigma=sigma, return_eigvals=True)

        return dual, eigvals

    def _fit_step(self, sq_pdiffs: NDArray[np.float64]) -> tuple[float, float]:
        # combi_p = combinations_prob(self.coefficients_, self._combinations)

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
            coefficients = self._combinations @ combi_p
        else:
            coefficients = self._update_coefficients(
                sq_pdiffs,
                self.coefficients_,
                combi_p=self.combi_p_,
                neg_dual_eigvals=-self.dual_eigvals_ * combi_e[:, np.newaxis],
            )
            combi_p = combinations_prob(coefficients, self._combinations)

        weights = self._update_weights(
            sq_pdiffs,
            self.weights_,
            combi_p=self.combi_p_,
            dual=self.dual_ * combi_e[:, np.newaxis, np.newaxis],
        )

        self.dual_, self.dual_eigvals_ = self._update_dual(
            weights=weights,
            combi_p=combi_p,
            dual=self.dual_,
        )

        a_rel_change = relative_error(self.coefficients_.ravel(), coefficients.ravel())
        w_rel_change = relative_error(self.weights_.ravel(), weights.ravel())

        self.combi_p_ = combi_p
        self.coefficients_ = coefficients
        self.weights_ = weights

        return a_rel_change, w_rel_change

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict coefficients for a given signal"""
        _n_samples, n_nodes = x.shape
        if n_nodes != self.n_nodes_:
            raise ValueError(f"Number of nodes mismatch, got {n_nodes} instead of {self.n_nodes_}")

        sq_pdiffs = squared_pdiffs(x)

        coefficients = self._init_coefficients(x.shape[0])
        combi_p = combinations_prob(coefficients)

        for _ in range(self.max_iter):
            combi_e = combi_p.sum(1)

            if self.combination_update:
                combi_p = self._update_combi_p(
                    sq_pdiffs,
                    combi_p=combi_p,
                    neg_dual_eigvals=-self.dual_eigvals_ * combi_e[:, np.newaxis],
                )
                coefficients_u = self._combinations @ combi_p
            else:
                coefficients_u = self._update_coefficients(
                    sq_pdiffs,
                    coefficients,
                    combi_p=combi_p,
                    neg_dual_eigvals=-self.dual_eigvals_ * combi_e[:, np.newaxis],
                )
                combi_p = combinations_prob(coefficients)

            if np.linalg.norm((coefficients_u - coefficients).ravel()) < self.tol:
                return coefficients_u

            coefficients = coefficients_u

        return coefficients
