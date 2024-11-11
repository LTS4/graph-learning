"""Graph dictionary learning original method"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from graph_learn.operators import (
    laplacian_squareform_vec,
    op_adj_coefficients,
    op_adj_weights,
    prox_gdet_star,
)

from .base_model import GraphDictBase


class GraphDictExact(GraphDictBase):
    """Graph dictionary learning original method"""

    def _op_adj_coefficients(self, weights: NDArray, dualv: NDArray) -> NDArray:
        return op_adj_coefficients(weights, dualv)

    def _op_adj_weights(self, coefficients: NDArray, dualv: NDArray) -> NDArray:
        return op_adj_weights(coefficients, dualv)

    def _update_dual(
        self,
        weights: NDArray[np.float64],
        coefficients: NDArray[np.float64],
        dual: NDArray[np.float64],
    ) -> NDArray:
        # z1 = dualv + step * bilinear_op(x_overshoot, y_overshoot)
        # z1 -= step * prox_h(z1 / step, 1 / step)

        n_atoms, _n_samples = coefficients.shape
        sigma = self.step_dual / n_atoms

        step = laplacian_squareform_vec(coefficients.T @ weights)
        return prox_gdet_star(dual + sigma * step, sigma=sigma)
