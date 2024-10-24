"""Learning time-varying graph from smooth signals"""

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator

from graph_learn.operators import square_to_vec
from graph_learn.smooth_learning import get_theta, gsp_learn_graph_log_degrees


class WindowLogModel(BaseEstimator):
    """Learn a sequence of graphs from signals by applying LogModel on windows"""

    def __init__(
        self,
        window_size: int = 10,
        *,
        avg_degree: int = None,
        edge_init: Optional[NDArray[np.float64]] = None,
        maxit: int = 1000,
        tol: float = 1e-5,
        step_size: float = 0.5,
        edge_tol: float = 1e-3,
        random_state=None,
    ) -> None:
        self.window_size = window_size
        self.avg_degree = avg_degree
        self.edge_init = edge_init
        self.maxit = maxit
        self.tol = tol
        self.step_size = step_size
        self.edge_tol = edge_tol

        self.weights_: NDArray[np.float64]

    def fit(self, x: NDArray[np.float64]):
        n_samples, n_nodes = x.shape

        if (pad_size := n_samples % self.window_size) > 0:
            x = np.vstack((x, np.zeros((self.window_size - pad_size, n_nodes))))

        sq_pdists = np.sum(
            (
                x.reshape(-1, self.window_size, 1, n_nodes)
                - x.reshape(-1, self.window_size, n_nodes, 1)
            )
            ** 2,
            axis=1,
        )

        self.weights_ = gsp_learn_graph_log_degrees(
            square_to_vec(sq_pdists) * [[get_theta(sqpd, self.avg_degree)] for sqpd in sq_pdists],
            1,
            1,
            edge_init=self.edge_init,
            maxit=self.maxit,
            tol=self.tol,
            step_size=self.step_size,
            edge_tol=self.edge_tol,
        )

        return self
