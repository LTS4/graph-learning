"""Implementation of Graph Laplacian Mixture Model"""

# pylint: disable=arguments-renamed

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

# from scipy.spatial.distance import pdist, squareform
from sklearn.mixture._base import BaseMixture

from graph_learn.operators import laplacian_squareform_vec, square_to_vec
from graph_learn.sampling.graphs import sample_uniform_laplacian
from graph_learn.smooth_learning import get_theta, gsp_learn_graph_log_degrees


def _estimate_gauss_laplacian_parameters(
    x: NDArray[np.float64],
    resp: NDArray[np.float64],
    delta: float,
    *,
    theta: float | NDArray[np.float64] | None = None,
    avg_degree: Optional[int | dict[tuple[int, int], int]] = None,
    blocks: NDArray[np.int64] = None,
    laplacians: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Estimate the parameters of Gaussian-Laplacian Mixture model, given the associations.

    Args:
        x (NDArray[np.float64]): Design matrix, shape (n_samples, n_nodes)
        resp (NDArray[np.float64]): Association probabilities, shape (n_samples, n_components)
        delta (float): Scale parameters of learned graphs
        theta (float | NDArray[np.float64] | None, optional): Scale parameter of signals, or array
            of them. Defaults to None. Incompatible with avg_degree.
        avg_degree (int | dict, optional): Expected average degree of the graphs. Defaults to
            None. Incompatible with theta, as the latter is estimated with the :method:`get_theta`.
        blocks (NDArray[np.int64], optional): Node assignments to blocks. Defaults to None.
        laplacians (NDArray[np.float64] | None, optional): Priors on the Laplacians, or previous
            estimates. Defaults to None.

    Raises:
        ValueError: In case both theta and avg_degree are provided.
        NotImplementedError: If laplacians are provided.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: Weights, means and
            Laplacians of each component.
    """
    if (theta is None) == (avg_degree is None):
        raise ValueError("Exactly one of theta and avg_degree should be provided")

    _n_samples, n_nodes = x.shape

    if laplacians is None:
        edge_init = None
    else:
        # Must transpose as gsp_learn_graph_log_degrees expects edge weights on first axis
        edge_init = -square_to_vec(laplacians).T

    weights: NDArray[np.float64] = np.sum(resp, axis=0)  # shape: n_components

    means = resp.T @ x / weights[:, np.newaxis]  # shape: n_components, n_nodes
    weights = weights / n_nodes

    # Estimate Laplacians
    y = resp[:, :, np.newaxis] * (x[:, np.newaxis, :] - means[np.newaxis, ...])
    sq_dist = np.sum((y[..., np.newaxis] - y[..., np.newaxis, :]) ** 2, axis=0)

    # Theta should be in the order of np.mean(sq_dist)
    if avg_degree is not None:
        # Get Theta returns inversed thetas
        theta_inv = np.array([get_theta(sqd, avg_degree, blocks=blocks) for sqd in sq_dist])
        if len(theta_inv.shape) < 2:
            theta_inv = theta_inv[:, np.newaxis]
    else:
        theta_inv = 1 / theta

    edge_weights = delta * gsp_learn_graph_log_degrees(
        square_to_vec(sq_dist) * theta_inv, alpha=1, beta=1, edge_init=edge_init
    )

    laplacians = laplacian_squareform_vec(edge_weights)

    return weights, means, laplacians


class GLMM(BaseMixture):
    """Graph Laplacian Mixture model from [mareticGraphLaplacianMixture2020]_

    Args:
        n_components (int, optional): Number of components. Defaults to 1.
        avg_degree (float, optional): Average degree of the graph. Defaults to 0.5.
        tol (float, optional): EM convergence tolerance. Defaults to 1e-3.
        reg_covar (float, optional): Covariance regularization. Defaults to 1e-6.
        max_iter (int, optional): Max EM iterations. Defaults to 100.
        n_init (int, optional): Number of random initializations. Defaults to 1.
        init_params (str, optional): Label initialization method. Defaults to "kmeans".
            Accepts same values as :class:`GaussianMixture`.
        regul (float, optional): GLMM regularization. Defaults to 0.15.
        theta (float | NDArray[np.float64], optional): Alternative parameterization to
            :arg:`avg_degree`, is thr scale parameter of signals. Defaults to None.
        delta (float, optional): Scale parameter of learned graphs. Defaults to 2.
        laplacian_init (Optional[float  |  str], optional): Method for laplacian initialization.
            Defaults to None. Options are:
            - None: Estimate Laplacians from first assignments estimate, given by :arg:`init_params`
            - float: Initialize as fully connected with weights eaual to arg value
            - 'random': Edge weights are sampled as uniform random variables.
        random_state (_type_, optional): Random state. Defaults to None.
        warm_start (bool, optional): Wheter to use warm start in EM. Defaults to False.
        verbose (int, optional): Verobsity level. Defaults to 0.
        verbose_interval (int, optional): Verbosity interval. Defaults to 10.

    Parameters:
        n_nodes_ (int): Number of nodes in the graph
        weights_ (NDArray[np.float64]): GLMM class weights
        means_ (NDArray[np.float64]): Components mean
        laplacians_ (NDArray[np.float64]): COmponents Laplacians
        labels_ (NDArray[np.int64]): Input assignments based on MAP


    .. [mareticGraphLaplacianMixture2020] H. P. Maretic and P. Frossard, “Graph
        Laplacian mixture model,” arXiv:1810.10053 [cs, stat], Mar. 2020,
        Available: http://arxiv.org/abs/1810.10053
    """

    def __init__(
        self,
        n_components=1,
        avg_degree: int | dict[tuple[int, int], int] = 2,
        *,
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        regul: float = 0.15,
        theta: NDArray[np.float64] = None,
        delta: float = 2,
        laplacian_init: Optional[float | str] = None,
        blocks: NDArray[np.int64] = None,
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
        self.theta = theta
        self.avg_degree = avg_degree
        self.delta = delta
        self.blocks = blocks

        self.laplacian_init = laplacian_init

        # Init empty variables for typechecking
        self.n_nodes_: int
        self.weights_: NDArray[np.float64]
        self.means_: NDArray[np.float64]
        self.laplacians_: NDArray[np.float64]

        self._propagate_laplacians = False

    def _check_parameters(self, X):
        pass

    def _initialize(self, x, resp):
        _n_samples, self.n_nodes_ = x.shape

        if self.laplacian_init is None:
            self.laplacians_ = None

        elif isinstance(self.laplacian_init, np.ndarray):
            if self.laplacian_init.shape != (self.n_components, self.n_nodes_, self.n_nodes_):
                raise ValueError("Laplacians must have shape (n_components, n_nodes, n_nodes)")

            self._propagate_laplacians = True
            self.laplacians_ = self.laplacian_init

            raise NotImplementedError("Laplacian order should be related to resp to make sense")

        elif isinstance(spread := self.laplacian_init, float):
            self.laplacians_ = np.tile(
                spread * np.eye(self.n_nodes_)
                - spread / self.n_nodes_ * np.ones((self.n_nodes_, self.n_nodes_)),
                (self.n_components, 1, 1),
            )

        elif self.laplacian_init == "random":
            self.laplacians_ = np.stack(
                [
                    sample_uniform_laplacian(self.n_nodes_, self.random_state)
                    for _ in range(self.n_components)
                ]
            )

        else:
            raise ValueError("Invalid Laplacian init")

        self.weights_, self.means_, self.laplacians_ = _estimate_gauss_laplacian_parameters(
            x,
            resp,
            self.delta,
            theta=self.theta,
            avg_degree=self.avg_degree,
            blocks=self.blocks,
            laplacians=self.laplacians_ if self._propagate_laplacians else None,
        )

    def _m_step(self, x: NDArray[np.float64], log_resp: NDArray[np.float64]) -> None:
        (
            self.weights_,
            self.means_,
            self.laplacians_,
        ) = _estimate_gauss_laplacian_parameters(
            x,
            np.exp(log_resp),
            self.delta,
            theta=self.theta,
            avg_degree=self.avg_degree,
            blocks=self.blocks,
            laplacians=self.laplacians_ if self._propagate_laplacians else None,
        )
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

        # FIXME: This could be vectorized
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
