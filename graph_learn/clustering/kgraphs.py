"""Implementation of K-Graphs clustering algorithm"""
# pylint: disable=arguments-renamed
from typing import Optional

import numpy as np
from numpy.random import RandomState
from numpy.typing import NDArray
from scipy.spatial.distance import pdist
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state

from graph_learn.clustering.utils import init_labels
from graph_learn.smooth_learning import gsp_learn_graph_log_degrees


class KGraphs(BaseEstimator, ClusterMixin):
    """K-Graphs clustering algorithm from [araghiGraphsAlgorithmGraph2019]_

    .. [araghiGraphsAlgorithmGraph2019] H. Araghi, M. Sabbaqi, and M.
        Babaie-Zadeh, “K-Graphs: An Algorithm for Graph Signal Clustering and
        Multiple Graph Learning,” IEEE Signal Processing Letters, vol. 26, no.
        10, pp. 1486-1490, Oct. 2019, doi: 10.1109/LSP.2019.2936665."""

    def __init__(
        self,
        n_clusters=1,
        *,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        norm_par: float = 1.5,
        delta: float = 2,
        random_state: Optional[RandomState] = None,
    ) -> None:
        self.n_clusters = n_clusters

        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.norm_par = norm_par
        self.delta = delta
        self.random_state = random_state

        self.labels_: NDArray[np.int_]
        self.laplacians_: NDArray[np.float_]
        self.converged_: bool
        self.score_: float

    def _init_parameters(self, x: NDArray[np.float_]):
        _n_samples, n_nodes = x.shape

        self.random_state = check_random_state(self.random_state)

        self.labels_ = init_labels(
            x,
            self.n_clusters,
            init_params=self.init_params,
            random_state=self.random_state,
        )
        self.laplacians_ = np.empty((self.n_clusters, n_nodes, n_nodes))
        self.converged_ = False

        self.score_ = np.inf

    def _smoothness(self, x: NDArray[np.float_]) -> NDArray[np.float_]:
        return np.einsum("ni,kij,nj->nk", x, self.laplacians_, x)

    def _single_fit(self, x: NDArray[np.float_], _y=None) -> None:
        self._init_parameters(x)

        for _i in range(self.max_iter):
            # Compute Laplacians
            for k in range(self.n_clusters):

                sq_dist = pdist(x[self.labels_ == k].T) ** 2
                theta = np.mean(sq_dist) / self.norm_par
                if np.allclose(theta, 0):
                    theta = 1

                edge_weights = self.delta * gsp_learn_graph_log_degrees(
                    sq_dist / theta, alpha=1, beta=1
                )

                self.laplacians_[k] = np.diag(np.sum(edge_weights, axis=1)) - edge_weights

            # Compute assignments
            # eisum.shape: (n_samples, n_clusters)
            smoothness = self._smoothness(x)
            labels = np.argmin(smoothness, axis=1)

            self.score_ = np.sum(smoothness[np.arange(len(labels)), labels])

            if np.allclose(labels, self.labels_):
                self.converged_ = True
                return

            self.labels_ = labels

    def fit_predict(self, x: NDArray[np.float_], _y=None) -> NDArray[np.int_]:
        n_samples, n_nodes = x.shape

        best_score = np.inf
        best_laplacians = np.empty((self.n_clusters, n_nodes, n_nodes))
        best_labels = np.empty(n_samples, dtype=np.int_)
        best_converged = None

        for _n in range(self.n_init):
            self._single_fit(x)

            if self.score_ < best_score:
                best_score = self.score_
                best_laplacians = self.laplacians_
                best_labels = self.labels_
                best_converged = self.converged_

        self.score_ = best_score
        self.laplacians_ = best_laplacians
        self.labels_ = best_labels
        self.converged_ = best_converged

        return self.labels_

    def fit(self, x: NDArray[np.float_], _y=None):
        self.fit_predict(x)
        return self

    def predict(self, x: NDArray[np.float_]) -> NDArray[np.int_]:
        """Compute labels"""
        return np.argmin(self._smoothness(x), axis=1)

    def predict_proba(self, x: NDArray[np.float_]) -> NDArray[np.float_]:
        "Return softmax of smoothness"
        return softmax(-self._smoothness(x), axis=1)
