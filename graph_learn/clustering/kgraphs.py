"""Implementation of K-Graphs clustering algorithm"""

# pylint: disable=arguments-renamed
from typing import Optional

import numpy as np
from numpy.random import RandomState
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state

from graph_learn.clustering.glmm import _estimate_gauss_laplacian_parameters
from graph_learn.clustering.utils import init_labels, one_hot
from graph_learn.operators import laplacian_squareform_vec
from graph_learn.smooth_learning import get_theta, gsp_learn_graph_log_degrees


class KGraphs(BaseEstimator, ClusterMixin):
    """K-Graphs clustering algorithm from [araghiGraphsAlgorithmGraph2019]_

    Args:
        n_clusters (int, optional): Number of clusters. Defaults to 1.
        avg_degree (float, optional): Average degree of the graph. Defaults to 0.5.
        max_iter (int, optional): Maximum EM steps. Defaults to 100.
        n_init (int, optional): Number of separate initalization. Defaults to 1.
        init_params (str, optional): Label initialization method. Defaults to "kmeans".
        delta (float, optional): _description_. Defaults to 2.
        random_state (Optional[RandomState], optional): _description_. Defaults to None.

    Parameters:
        labels_ (NDArray[np.int64]): Cluster assignments
        laplacians_ (NDArray[np.float64]): Cluster laplacians
        converged_ (bool): Wheter assignment converged
        score_ (float): Total smoothness


    .. [araghiGraphsAlgorithmGraph2019] H. Araghi, M. Sabbaqi, and M.
        Babaie-Zadeh, “K-Graphs: An Algorithm for Graph Signal Clustering and
        Multiple Graph Learning,” IEEE Signal Processing Letters, vol. 26, no.
        10, pp. 1486-1490, Oct. 2019, doi: 10.1109/LSP.2019.2936665."""

    def __init__(
        self,
        n_clusters=1,
        avg_degree: int = 2,
        *,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        # norm_par: float = 1.5,
        delta: float = 1,
        random_state: Optional[RandomState] = None,
    ) -> None:
        self.n_clusters = n_clusters

        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        # self.norm_par = norm_par
        self.delta = delta
        self.avg_degree = avg_degree
        self.random_state = random_state

        self.labels_: NDArray[np.int64]
        self.laplacians_: NDArray[np.float64]
        self.converged_: bool
        self.score_: float

    def _init_parameters(self, x: NDArray[np.float64]):
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

    def _smoothness(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.einsum("ni,kij,nj->nk", x, self.laplacians_, x)

    def _single_fit(self, x: NDArray[np.float64], _y=None) -> None:
        self._init_parameters(x)

        for _i in range(self.max_iter):
            # Compute Laplacians

            sq_dist = np.stack([pdist(x[self.labels_ == k].T) ** 2 for k in range(self.n_clusters)])
            # theta = np.mean(sq_dist) / self.norm_par
            # if np.allclose(theta, 0):
            #     theta = 1

            # FIXME: this parameterization ends with collapsing most labels to a single graph
            edge_weights = self.delta * gsp_learn_graph_log_degrees(
                sq_dist * [[get_theta(squareform(sqd), self.avg_degree)] for sqd in sq_dist],
                alpha=1,
                beta=1,
            )

            self.laplacians_ = laplacian_squareform_vec(edge_weights)

            # Compute assignments
            # eisum.shape: (n_samples, n_clusters)
            smoothness = self._smoothness(x)
            labels = np.argmin(smoothness, axis=1)

            self.score_ = np.sum(smoothness[np.arange(len(labels)), labels])

            if np.allclose(labels, self.labels_):
                self.converged_ = True
                return

            self.labels_ = labels

    def fit_predict(self, x: NDArray[np.float64], _y=None) -> NDArray[np.int64]:
        best_score = np.inf
        best_params = {}

        for _n in range(self.n_init):
            self._single_fit(x)

            if self.score_ < best_score:
                best_score = self.score_
                best_params = self.get_params()

        self.score_ = best_score
        self.set_params(**best_params)

        return self.labels_

    def fit(self, x: NDArray[np.float64], _y=None):
        self.fit_predict(x)
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.int64]:
        """Compute labels"""
        return np.argmin(self._smoothness(x), axis=1)

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        "Return softmax of smoothness"
        return softmax(-self._smoothness(x), axis=1)


class KGraphsV2(KGraphs):
    """Extension of KGraphs to allow for centers estimation"""

    def __init__(
        self,
        n_clusters=1,
        avg_degree: int = 2,
        *,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        delta: float = 1,
        theta: Optional[float] = None,
        random_state: RandomState | None = None,
    ) -> None:
        super().__init__(
            n_clusters,
            avg_degree,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            delta=delta,
            random_state=random_state,
        )

        self.theta = theta

        self.means_: NDArray[np.float64]  # shape: (n_clusters, n_nodes)

    def _init_parameters(self, x: NDArray[np.float64]):
        self.random_state = check_random_state(self.random_state)

        self.labels_ = init_labels(
            x,
            self.n_clusters,
            init_params=self.init_params,
            random_state=self.random_state,
        )

        _, self.means_, self.laplacians_ = _estimate_gauss_laplacian_parameters(
            x,
            one_hot(self.labels_, self.n_clusters),
            self.delta,
            theta=self.theta,
            avg_degree=self.avg_degree,
        )

        self.converged_ = False

        self.score_ = np.inf

    def _centered_smoothness(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x = x[np.newaxis, ...] - self.means_[:, np.newaxis, :]
        return np.einsum("kni,kij,knj->nk", x, self.laplacians_, x)

    def _single_fit(self, x: NDArray[np.float64], _y=None) -> None:
        self._init_parameters(x)

        for _i in range(self.max_iter):
            # Compute assignments
            # eisum.shape: (n_samples, n_clusters)

            # FIXME: Why should I use smoothness instead of centerd one?
            smoothness = self._smoothness(x)
            labels = np.argmin(smoothness, axis=1)

            self.score_ = np.sum(smoothness[np.arange(len(labels)), labels])

            # Compute means and Laplacians
            _, self.means_, self.laplacians_ = _estimate_gauss_laplacian_parameters(
                x,
                one_hot(labels, self.n_clusters),
                self.delta,
                theta=self.theta,
                avg_degree=self.avg_degree,
            )

            if np.allclose(labels, self.labels_):
                self.converged_ = True
                return

            self.labels_ = labels
