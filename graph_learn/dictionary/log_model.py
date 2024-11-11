"""Graph dictionary learning original method"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from graph_learn.smooth_learning import sum_squareform

from .base_model import GraphDictBase


class GraphDictLog(GraphDictBase):
    """Graph dictionary learning original method"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sum_op: sparse.csr_array  # shape: (n_nodes, n_edges)
        self._sum_op_t: sparse.csr_array  # shape: (n_edges, n_nodes)

    def _init_dual(self, x: NDArray):
        return np.zeros((x.shape[0] // self.window_size, self.n_nodes_))

    def _initialize(self, x: NDArray) -> None:
        super()._initialize(x)

        self._sum_op, self._sum_op_t = sum_squareform(self.n_nodes_)

    def _op_adj_coefficients(self, weights: NDArray, dualv: NDArray) -> NDArray:
        return weights @ (self._sum_op_t @ dualv.T)

    def _op_adj_weights(self, coefficients: NDArray, dualv: NDArray) -> NDArray:
        return coefficients @ (self._sum_op_t @ dualv.T).T

    def _update_dual(
        self,
        weights: NDArray[np.float64],
        coefficients: NDArray[np.float64],
        dual: NDArray[np.float64],
    ):
        # z1 = dualv + step * bilinear_op(x_overshoot, y_overshoot)
        # z1 -= step * prox_h(z1 / step, 1 / step)

        sigma = self.step_dual  # / _n_samples

        step = self._sum_op @ weights.T @ coefficients
        dual = dual + sigma * step.T

        return (dual - np.sqrt(dual**2 + 4 * sigma)) / 2
