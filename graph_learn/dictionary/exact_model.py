"""Graph components learning original method"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from graph_learn.operators import (
    laplacian_squareform_vec,
    op_adj_activations,
    op_adj_weights,
    prox_gdet_star,
)

from .base_model import GraphDictBase


class GraphDictExact(GraphDictBase):
    """Graph components learning original method"""

    def _op_adj_activations(self, weights: NDArray, dualv: NDArray) -> NDArray:
        return op_adj_activations(weights, dualv)

    def _op_adj_weights(self, activations: NDArray, dualv: NDArray) -> NDArray:
        return op_adj_weights(activations, dualv)

    def _update_dual(
        self,
        weights: NDArray[np.float_],
        activations: NDArray[np.float_],
        dual: NDArray[np.float_],
    ):
        # z1 = dualv + step * bilinear_op(x_overshoot, y_overshoot)
        # z1 -= step * prox_h(z1 / step, 1 / step)

        n_atoms, _n_samples = activations.shape
        sigma = self.step_dual / n_atoms

        step = laplacian_squareform_vec(activations.T @ weights)
        return prox_gdet_star(dual + sigma * step, sigma=sigma)
