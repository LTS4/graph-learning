"""Graph components learning original method"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from graph_learn.smooth_learning import sum_squareform

from .base_model import GraphDictBase


class GraphDictLog(GraphDictBase):
    """Graph components learning original method"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sum_op: sparse.csr_array  # shape: (n_nodes, n_edges)
        self._sum_op_t: sparse.csr_array  # shape: (n_edges, n_nodes)

    def _init_dual(self, n_samples: int):
        return np.zeros((self.n_nodes_, n_samples // self.window_size))

    def _initialize(self, x: NDArray) -> None:
        super()._initialize(x)

        self._sum_op, self._sum_op_t = sum_squareform(self.n_nodes_)

    def _op_adj_activations(self, weights: NDArray, dualv: NDArray) -> NDArray:
        return weights @ (self._sum_op_t @ dualv)

    def _op_adj_weights(self, activations: NDArray, dualv: NDArray) -> NDArray:
        return activations @ (self._sum_op_t @ dualv).T

    def _update_dual(
        self,
        weights: NDArray[np.float_],
        activations: NDArray[np.float_],
        dual: NDArray[np.float_],
    ):
        # z1 = dualv + step * bilinear_op(x_overshoot, y_overshoot)
        # z1 -= step * prox_h(z1 / step, 1 / step)

        sigma = self.step_dual  # / _n_samples

        step = self._sum_op @ weights.T @ activations
        dual = dual + sigma * step

        return (dual - np.sqrt(dual**2 + 4 * sigma)) / 2
