"""This module implements the model from yamadaTimeVaryingGraphLearning2020_

_yamadaTimeVaryingGraphLearning2020 : K. Yamada, Y. Tanaka, and A. Ortega,
    “Time-Varying Graph Learning with Constraints on Graph Temporal Variation,”
    ArXiv, 2020.

"""
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from sklearn.base import BaseEstimator

from graph_learn.evaluation import relative_error
from graph_learn.operators import square_to_vec
from graph_learn.smooth_learning import sum_squareform


def relu(x: NDArray) -> NDArray:
    return np.where(x > 0, x, 0)


def prox_l1_pos(x: NDArray, gamma: float) -> NDArray:
    return relu(x - gamma)


def prox_neg_log_sum(x: NDArray, gamma: float) -> NDArray:
    return (x + np.sqrt(x**2 + 4 * gamma)) / 2


def prox_l1(x: NDArray, gamma: float) -> NDArray:
    return np.sign(x) * relu(np.abs(x) - gamma)


def prox_group_l2(X: NDArray, gamma: float) -> NDArray:
    """Proximal operator of what is defined Group lasso in paper"""
    if len(X.shape) < 2:
        raise ValueError("X must be at least 2D")

    norm = np.linalg.norm(X, ord=2, axis=-1)
    norm[norm < gamma] = 1

    return (1 - gamma / norm) * X


class TGFA(BaseEstimator):
    """Time-varying graph factor analysis from yamadaTimeVaryingGraphLearning2020_"""

    def __init__(
        self,
        window_size: int = 1,
        gamma: float = 1,
        degree_reg: float = 0,
        sparse_reg: float = 0,
        l1_time: float = 0,
        l2_time: float = 0,
        max_iter: int = 100,
        tol: float = 1e-3,
        random_state=None,
    ) -> None:
        self.window_size = window_size
        self.gamma = gamma  # step size
        self.degree_reg = degree_reg  # alpha in paper
        self.sparse_reg = sparse_reg  # beta in paper
        self.l1_time = l1_time
        self.l2_time = l2_time
        self.max_iter = max_iter
        self.tol = tol

        self.converged_: int
        self.weights_: NDArray[np.float_]  # shape: (n_windows, n_edges)
        self.dual1_: NDArray[np.float_]  # shape: (n_windows, n_nodes)
        self.dual2_: NDArray[np.float_]  # shape: (n_windows, n_edges)
        self._op_sum: NDArray[np.float_]
        self._op_sum_t: NDArray[np.float_]
        self._op_diff: NDArray[np.float_]
        self._prox_time: Callable[[NDArray[np.float_], float], NDArray[np.float_]]

    def _initialize(self, x: NDArray[np.float_]) -> NDArray:
        self.converged_ = -1
        n_samples, n_nodes = x.shape
        n_windows = int(np.ceil(n_samples / self.window_size))
        n_edges = n_nodes * (n_nodes - 1) // 2

        self._op_sum, self._op_sum_t = sum_squareform(n_nodes)

        self._op_diff = sparse.eye(n_edges * n_windows, dtype=float, format="lil")
        self._op_diff[np.arange(n_edges), np.arange(n_edges)] = 0
        self._op_diff[
            np.arange(n_edges, n_edges * n_windows), np.arange(n_edges * (n_windows - 1))
        ] = -1

        self.weights_ = np.zeros((n_windows, n_edges), dtype=float)
        self.dual1_ = (self._op_sum @ self.weights_.T).T
        self.dual2_ = np.zeros_like(self.weights_)

        if self.l1_time > 0 and self.l2_time > 0:
            raise ValueError("Cannot have both l1_time and l2_time > 0")
        elif self.l1_time > 0:
            self._prox_time = prox_l1
        elif self.l2_time > 0:
            self._prox_time = prox_group_l2
        else:
            raise ValueError("Must have either l1_time or l2_time > 0")

        x_pad = np.zeros((n_windows * self.window_size, n_nodes))
        x_pad[:n_samples] = x
        x_pad = x_pad.reshape((n_windows, n_samples, n_nodes))

        return square_to_vec(
            np.sum(
                (x_pad[:, :, np.newaxis, :] - x_pad[:, :, :, np.newaxis]) ** 2,
                axis=1,
            )
        ).flatten()

    def _primal_step(self, primal, dual1, dual2) -> NDArray:
        return primal - self.gamma * (
            2 * self.sparse_reg * primal + self._op_sum_t @ dual1 + self._op_diff.T @ dual2
        )

    def _dual_step1(self, primal, dual) -> NDArray:
        return dual + self.gamma * self._op_sum @ primal

    def _dual_step2(self, primal, dual) -> NDArray:
        return dual + self.gamma * self._op_diff @ primal

    def _fit_step(
        self, sq_pdiffs: NDArray, weights: NDArray, dual1: NDArray, dual2: NDArray
    ) -> tuple[NDArray, NDArray, NDArray]:
        y = self._primal_step(weights, dual1, dual2)
        yb1 = self._dual_step1(weights, dual1)
        yb2 = self._dual_step2(weights, dual2)

        p = prox_l1_pos(y - 2 * self.gamma * sq_pdiffs, self.gamma)
        # FIXME: here I could use the conjugate of prox_neg_log_sum directly
        pb1 = yb1 - self.gamma * prox_neg_log_sum(yb1 / self.gamma, 1 / self.gamma)
        pb2 = yb2 - self.gamma * self._prox_time(yb2 / self.gamma, 1 / self.gamma)

        q = self._primal_step(p, pb1, pb2)
        qb1 = self._dual_step1(p, pb1)
        qb2 = self._dual_step2(p, pb2)

        return (
            weights - y + q,
            dual1 - yb1 + qb1,
            dual2 - yb2 + qb2,
        )

    def fit(self, x: NDArray[np.float_]):
        sq_pdiffs = self._initialize(x)

        for step in range(self.max_iter):
            weights, self.dual1_, self.dual2_ = self._fit_step(
                sq_pdiffs, self.weights_, self.dual1_, self.dual2_
            )

            if relative_error(self.weights_, weights) < self.tol:
                self.converged_ = step

            self.weights_ = weights

            if self.converged_ > 0:
                break

        return self
