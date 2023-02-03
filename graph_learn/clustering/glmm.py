"""Implementation of Graph Laplacian Mixture Model"""
# pylint: disable=arguments-renamed

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import pdist
from sklearn.mixture._base import BaseMixture

from graph_learn.smooth_learning import gsp_learn_graph_log_degrees


def _sample_laplacian(n_nodes: int, random_state: np.random.Generator):
    out = random_state.uniform(size=(n_nodes, n_nodes))
    np.fill_diagonal(out, 0)
    out += out.T
    np.fill_diagonal(out, -out.sum(axis=-1))
    return -out


def _estimate_gauss_laplacian_parameters(
    x: NDArray[np.float64], resp: NDArray[np.float64], norm_par: float, delta: float
):
    _n_samples, n_nodes = x.shape
    _n_samples, n_components = resp.shape
    laplacians = np.empty((n_components, n_nodes, n_nodes))

    weights: NDArray[np.float64] = np.sum(resp, axis=0)  # shape: n_components

    means = resp.T @ x / weights[:, np.newaxis]  # shape: n_components, n_nodes
    weights = weights / n_nodes

    # Estimate Laplacians
    for k in range(n_components):
        y = resp[:, k, np.newaxis] * (x - means[np.newaxis, k])
        sq_dist = pdist(y.T) ** 2

        theta = np.mean(sq_dist) / norm_par
        edge_weights = delta * gsp_learn_graph_log_degrees(
            sq_dist / theta,
            alpha=1,
            beta=1,
        )

        laplacians[k] = np.diag(np.sum(edge_weights, axis=1)) - edge_weights

    return weights, means, laplacians


class GLMM(BaseMixture):
    """Graph Laplacian Mixture model from [mareticGraphLaplacianMixture2020]_

    .. [mareticGraphLaplacianMixture2020] H. P. Maretic and P. Frossard, “Graph
        Laplacian mixture model,” arXiv:1810.10053 [cs, stat], Mar. 2020,
        Available: http://arxiv.org/abs/1810.10053
    """

    def __init__(
        self,
        n_components=1,
        *,
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        regul: float = 0.15,
        norm_par: float = 1.5,
        delta: float = 2,
        laplacian_init: Optional[float | str] = None,
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10,
    ):
        super().__init__(
            n_components,
            tol,
            reg_covar,
            max_iter,
            n_init,
            init_params,
            random_state,
            warm_start,
            verbose,
            verbose_interval,
        )

        self.regul = regul
        self.norm_par = norm_par
        self.delta = delta

        self.laplacian_init = laplacian_init

        # Init empty variables for typechecking
        self.n_nodes_: int
        self.weights_: NDArray[np.float64]
        self.means_: NDArray[np.float64]
        self.laplacians_: NDArray[np.float64]
        self.labels_: NDArray[np.int64]

    def _check_parameters(self, X):
        pass

    def _initialize(self, x, resp):
        _n_samples, self.n_nodes_ = x.shape

        self.weights_, self.means_, laplacians = _estimate_gauss_laplacian_parameters(
            x, resp, self.norm_par, self.delta
        )

        if self.laplacian_init is None:
            self.laplacians_ = laplacians
        elif isinstance(spread := self.laplacian_init, float):
            self.laplacians_ = np.tile(
                spread * np.eye(self.n_nodes_)
                - spread / self.n_nodes_ * np.ones((self.n_nodes_, self.n_nodes_)),
                (self.n_components, 1, 1),
            )
        elif self.laplacian_init == "random":
            self.laplacians_ = np.stack(
                [
                    _sample_laplacian(self.n_nodes_, self.random_state)
                    for _ in range(self.n_components)
                ]
            )
        else:
            raise ValueError("Invalid Laplacian init")

    def _m_step(self, x: NDArray[np.float64], log_resp: NDArray[np.float64]) -> None:
        (
            self.weights_,
            self.means_,
            self.laplacians_,
        ) = _estimate_gauss_laplacian_parameters(x, np.exp(log_resp), self.norm_par, self.delta)
        self.weights_ /= self.weights_.sum()

    def _estimate_log_prob(self, x: ArrayLike) -> NDArray[np.float64]:
        """Compute assignment log-probabilities (Expectation step).

        Args:
            x (ArrayLike): Design matrix, shape (n_samples, n_nodes)

        Returns:
            NDArray[np.float64]: array, shape (n_samples, n_components)
                Log-probabilities of the point of each sample in x
        """
        n_samples, _n_nodes = x.shape
        log_prob = np.zeros((n_samples, self.n_components))

        # NOTE: This is the smooth Laplacian setting
        # shapes: (n_components, n_nodes), (n_components, n_nodes, n_nodes)
        eigval, eigvec = np.linalg.eigh(self.laplacians_)
        eigval += self.regul

        # shape: (n_components, n_nodes-1, n_nodes-1)
        prec = np.stack([np.diag(ev[1:]) for ev in eigval])

        # shape: n_components
        # log_weights = np.log(self.weights_)

        for k in range(self.n_components):
            # Compute pdf

            # shape: (n_samples, n_nodes) @ (n_nodes, n_nodes-1) = n_samples, n_nodes-1
            y = (x - self.means_[k]) @ eigvec[k, :, 1:]

            log_prob[:, k] = (
                -(self.n_nodes_ - 1) * np.log(2 * np.pi)  # Dimensionality term
                + np.log(np.prod(eigval[k, 1:]))  # Det of inv(precision)
                - np.sum(y @ prec[k] * y, axis=-1)  # Data part, shape: nb_samples
                # - np.diag(y @ prec[k] @ y.T)
            ) / 2

        # Normalization should be handled by BaseMixture
        # log_prob += log_weights[np.newaxis, :]
        # log_prob -= np.log(np.sum(np.exp(log_prob), axis=1))[:, np.newaxis]
        return log_prob

    def _estimate_log_weights(self):
        return np.log(self.weights_)

    def _get_parameters(self):
        return (
            self.weights_,
            self.means_,
            self.laplacians_,
        )

    def _set_parameters(self, params):
        (
            self.weights_,
            self.means_,
            self.laplacians_,
        ) = params

        self.n_nodes_ = self.means_.shape[-1]

    def _compute_lower_bound(self, _, log_prob_norm):
        return log_prob_norm
