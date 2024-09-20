"""Learning time-varying graph from smooth signals"""

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator

from graph_learn.operators import square_to_vec
from graph_learn.smooth_learning import LogModel, get_theta, gsp_learn_graph_log_degrees


class WindowLogModel(LogModel):
    """Learn a sequence of graphs from signals by applying LogModel on windows"""

    def __init__(
        self,
        window_size: int = 10,
        avg_degree: int = None,
        *,
        edge_init: Optional[NDArray[np.float64]] = None,
        maxit: int = 1000,
        tol: float = 1e-5,
        step_size: float = 0.5,
        edge_tol: float = 1e-3,
        random_state=None,
    ) -> None:
        super().__init__(
            avg_degree=avg_degree,
            edge_init=edge_init,
            maxit=maxit,
            tol=tol,
            step_size=step_size,
            edge_tol=edge_tol,
        )
        self.window_size = window_size

        self.theta_: NDArray[np.float64]
        self.weights_: NDArray[np.float64]

    def _initialize(self, x) -> NDArray[np.float64]:
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

        if self.avg_degree is None:
            self.theta_ = 1
        else:
            self.theta_ = np.array([[get_theta(sqpd, self.avg_degree)] for sqpd in sq_pdists])

        return square_to_vec(sq_pdists)

    def fit(self, x: NDArray[np.float64]):
        sq_pdists = self._initialize(x)

        self.weights_ = gsp_learn_graph_log_degrees(
            sq_pdists * self.theta_,
            1,
            1,
            edge_init=self.edge_init,
            maxit=self.maxit,
            tol=self.tol,
            step_size=self.step_size,
            edge_tol=self.edge_tol,
        )

        return self
