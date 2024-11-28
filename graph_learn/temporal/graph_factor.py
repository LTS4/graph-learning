"""This module implements the model from yamadaTimeVaryingGraphLearning2020_

_yamadaTimeVaryingGraphLearning2020 : K. Yamada, Y. Tanaka, and A. Ortega,
    “Time-Varying Graph Learning with Constraints on Graph Temporal Variation,”
    ArXiv, 2020.

"""

from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator

from graph_learn.evaluation import relative_error
from graph_learn.operators import square_to_vec
from graph_learn.smooth_learning import get_theta, sum_squareform


def relu(x: NDArray) -> NDArray:
    return np.where(x > 0, x, 0)


def prox_l1_pos(x: NDArray, gamma: float) -> NDArray:
    return relu(x - gamma)


def prox_neg_log_sum(x: NDArray, gamma: float) -> NDArray:
    return (x + np.sqrt(x**2 + 4 * gamma)) / 2


def prox_neg_log_sum_conj(x: NDArray, gamma: float) -> NDArray:
    return (x - np.sqrt(x**2 + 4 * gamma)) / 2


def prox_l1(x: NDArray, gamma: float) -> NDArray:
    return np.sign(x) * relu(np.abs(x) - gamma)


def prox_group_l2(X: NDArray, gamma: float) -> NDArray:
    """Proximal operator of what is defined Group lasso in paper"""
    if len(X.shape) < 2:
        raise ValueError("X must be at least 2D")

    norm = np.linalg.norm(X, ord=2, axis=-1)[..., np.newaxis]
    return np.where(norm < gamma, 0, (1 - gamma / norm) * X)


class TGFA(BaseEstimator):
    """Time-varying graph factor analysis from yamadaTimeVaryingGraphLearning2020_"""

    def __init__(
        self,
        window_size: int = 1,
        gamma: float = 1,
        avg_degree: int = None,
        degree_reg: float = 1,
        sparse_reg: float = 1,
        l1_time: float = 0,
        l2_time: float = 0,
        max_iter: int = 1000,
        tol: float = 1e-3,
        random_state=None,
    ) -> None:
        self.window_size = window_size
        self.gamma = gamma  # step size
        self.avg_degree = avg_degree
        self.degree_reg = degree_reg  # alpha in paper
        self.sparse_reg = sparse_reg  # beta in paper
        self.l1_time = l1_time
        self.l2_time = l2_time
        self.max_iter = max_iter
        self.tol = tol

        self.converged_: int
        self.weights_: NDArray[np.float64]  # shape: (n_windows, n_edges)
        self.dual1_: NDArray[np.float64]  # shape: (n_nodes, n_windows)
        self.dual2_: NDArray[np.float64]  # shape: (n_edges, n_windows)
        self._op_sum: NDArray[np.float64]
        self._op_sum_t: NDArray[np.float64]
        self._op_diff: NDArray[np.float64]
        self._op_diff_t: NDArray[np.float64]
        self._prox_time: Callable[[NDArray[np.float64], float], NDArray[np.float64]]
        self._reg_time: float

    def _validate_params(self):
        if self.avg_degree is not None:
            if not np.allclose([self.degree_reg, self.sparse_reg], 1):
                raise ValueError("Cannot have avg_degree and degree_reg or sparse_reg != 1")

    def _initialize(self, x: NDArray[np.float64]) -> NDArray:
        self._validate_params()

        self.converged_ = -1
        n_samples, n_nodes = x.shape
        n_windows = int(np.ceil(n_samples / self.window_size))
        n_edges = n_nodes * (n_nodes - 1) // 2

        self._op_sum, self._op_sum_t = sum_squareform(n_nodes)

        self._op_diff = sparse.eye(n_windows, dtype=float, format="lil") - sparse.eye(
            n_windows, k=-1, dtype=float, format="lil"
        )
        self._op_diff[0, 0] = 0
        self._op_diff_t = self._op_diff.T.tocsr()
        self._op_diff = self._op_diff.tocsr()

        self.weights_ = np.zeros((n_windows, n_edges), dtype=float)
        self.dual1_ = self._op_sum @ self.weights_.T
        self.dual2_ = np.zeros_like(self.weights_.T)

        if self.l1_time > 0 and self.l2_time > 0:
            raise ValueError("Cannot have both l1_time and l2_time > 0")
        elif self.l1_time > 0:
            self._prox_time = prox_l1
            self._reg_time = self.l1_time
        elif self.l2_time > 0:
            self._prox_time = prox_group_l2
            self._reg_time = self.l2_time
        else:
            self._reg_time = 0

        x_pad = np.zeros((n_windows * self.window_size, n_nodes))
        x_pad[:n_samples] = x
        x_pad = x_pad.reshape((n_windows, self.window_size, n_nodes))

        return square_to_vec(
            np.sum(
                (x_pad[:, :, np.newaxis, :] - x_pad[:, :, :, np.newaxis]) ** 2,
                axis=1,
            )
        )

    def _primal_step(self, primal, sq_pdiffs, dual1, dual2) -> NDArray:
        return primal - self.gamma * (
            2 * self.sparse_reg  # * primal
            + 2 * sq_pdiffs
            + self._op_sum_t @ dual1
            + (self._op_diff_t @ dual2.T).T
        )

    def _dual_step1(self, primal, dual) -> NDArray:
        return dual + self.gamma * self._op_sum @ primal

    def _dual_step2(self, primal, dual) -> NDArray:
        return dual + self.gamma * (self._op_diff @ primal.T).T

    def _fit_step(
        self, sq_pdiffs: NDArray, weights: NDArray, dual1: NDArray, dual2: NDArray
    ) -> tuple[NDArray, NDArray, NDArray]:
        y = self._primal_step(weights, sq_pdiffs, dual1, dual2)
        yb1 = self._dual_step1(weights, dual1)
        yb2 = self._dual_step2(weights, dual2)

        # p = prox_l1_pos(y, self.gamma)
        p = relu(y)
        # NOTE: here I use the conjugate of prox_neg_log_sum directly
        # pb1 = yb1 - self.gamma * prox_neg_log_sum(yb1 / self.gamma, 1 / self.gamma)
        pb1 = self.degree_reg * prox_neg_log_sum_conj(
            yb1 / self.degree_reg, self.gamma / self.degree_reg
        )
        if self._reg_time > 0:
            # gamma_time = self.gamma / self._reg_time
            pb2 = (
                yb2
                - self.gamma * self._prox_time(yb2.T / self.gamma, self._reg_time / self.gamma).T
            )
        else:
            pb2 = np.zeros_like(yb2)

        q = self._primal_step(p, sq_pdiffs, pb1, pb2)
        qb1 = self._dual_step1(p, pb1)
        if self._reg_time > 0:
            qb2 = self._dual_step2(p, pb2)
        else:
            qb2 = yb2

        return (
            weights - y + q,
            dual1 - yb1 + qb1,
            dual2 - yb2 + qb2,
        )

    def fit(self, x: NDArray[np.float64]):
        sq_pdiffs = self._initialize(x)

        if self.avg_degree is not None:
            sq_pdiffs *= [
                [get_theta(squareform(sqpd), avg_degree=self.avg_degree)] for sqpd in sq_pdiffs
            ]
        # sq_pdiffs /= self.window_size
        # sq_pdiffs /= sq_pdiffs.mean(axis=1, keepdims=True)

        for step in range(self.max_iter):
            weights, self.dual1_, self.dual2_ = self._fit_step(
                sq_pdiffs.T, self.weights_.T, self.dual1_, self.dual2_
            )

            if relative_error(self.weights_, weights.T) < self.tol:
                self.converged_ = step

            self.weights_ = weights.T

            if self.converged_ > 0:
                break

        return self
