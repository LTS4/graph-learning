"""Module for learning graphs from signals"""
from typing import Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.spatial.distance import squareform


def sum_squareform(n: int) -> Tuple[sparse.csr_array, sparse.csr_array]:
    """For *z* vectorform of *Z*, return the operator S such that S @ z =
    np.sum(Z, axis=-1) and its transpose"""
    # pylint: disable=invalid-name
    nnz = n * (n - 1)

    J = np.concatenate([i * np.ones(n - 1) for i in range(n)])

    slices = []
    offsets = []
    start = 0
    stop = 0
    for i in range(n):
        offsets.append([sl[i - j - 1] for j, sl in enumerate(slices)])
        stop = start + n - i - 1
        slices.append(list(range(start, stop)))
        start = stop

    I = np.concatenate([off + sl for sl, off in zip(slices, offsets)])

    # # offset
    # k = 0
    # for i in range(2, n):
    #     I[k : k + (n - i)] = np.arange(i, n)
    #     k = k + (n - i + 1)

    # St = sparse([1:ncols, 1:ncols], [I, J], 1, ncols, n)
    St = sparse.coo_array(
        (np.ones(nnz), (I, J)),
        shape=(nnz // 2, n),
    )
    return St.T.tocsr(), St.tocsr()


def gsp_learn_graph_log_degrees(
    distances: np.ndarray,
    alpha: float,
    beta: float,
    edge_init: Optional[np.ndarray] = None,
    maxit: int = 1000,
    tol: float = 1e-5,
    step_size: float = 0.5,
    edge_tol: float = 1e-3,
) -> np.ndarray:
    r"""Computes a weighted adjacency matrix $W$ from squared pairwise distances
    in $Z$, using the smoothness assumption that $\text{trace}(X^TLX)$ is small,
    where $X$ is the data (columns) changing smoothly from node to node on the
    graph and $L = D-W$ is the combinatorial graph Laplacian. See the paper of
    the references for the theory behind the algorithm.


    Args:
        distances (np.ndarray): Matrix with (squared) pairwise distances of
            nodes, or its vectorization.
        alpha (float): Log prior constant  (bigger a -> bigger weights in W)
        beta (float): :math:`\|W\|_F^2` prior constant  (bigger b -> more dense W)
        edge_init (Optional[np.ndarray]): Initialization point. default:
            `zeros_like(distances)`
        maxit (int): Maximum number of iterations. Default: 1000
        tol (float): Tolerance for stopping criterion. Defaul: 1e-5
        step_size (float): Step size from the interval (0,1). Default: 0.5

    Returns:
        np.ndarray: Weighted adjacency matrix


    Alternatively, Z can contain other types of distances and use the
    smoothness assumption that

    .. sum(sum(W .* Z))

    .. math:: \sum_i\sum_j W_{ij}Z_{ij}

    is small.

    The minimization problem solved is

    .. minimize_W sum(sum(W .* Z)) - a * sum(log(sum(W))) + b * ||W||_F^2/2 + c
    * ||W-W_0||_F^2/2

    .. math:: \min_W \sum_i\sum_j W_{ij}Z_{ij} -\alpha\sum_i\log(\sum_jW_{ij}) +
        \frac{\beta}{2} \|W\|_F^2 + c/2 ||W - W0||_F^2

    subject to $W$ being a valid weighted adjacency matrix (non-negative,
    symmetric, with zero diagonal).

    The algorithm used is forward-backward-forward (FBF) based primal dual
    optimization (see references).

    Example:::

          G = gsp_sensor(256);
          f1 = @(x,y) sin((2-x-y).^2);
          f2 = @(x,y) cos((x+y).^2);
          f3 = @(x,y) (x-.5).^2 + (y-.5).^3 + x - y;
          f4 = @(x,y) sin(3*((x-.5).^2+(y-.5).^2));
          X = [f1(G.coords(:,1), G.coords(:,2)), f2(G.coords(:,1), G.coords(:,2)), f3(G.coords(:,1), G.coords(:,2)), f4(G.coords(:,1), G.coords(:,2))];
          figure; subplot(2,2,1); gsp_plot_signal(G, X(:,1)); title('1st smooth signal');
          subplot(2,2,2); gsp_plot_signal(G, X(:,2)); title('2nd smooth signal');
          subplot(2,2,3); gsp_plot_signal(G, X(:,3)); title('3rd smooth signal');
          subplot(2,2,4); gsp_plot_signal(G, X(:,4)); title('4th smooth signal');
          Z = gsp_distanz(X').^2;
          % we can multiply the pairwise distances with a number to control sparsity
          [W] = gsp_learn_graph_log_degrees(Z*25, 1, 1);
          % clean up zeros
          W(W<1e-5) = 0;
          G2 = gsp_update_weights(G, W);
          figure; gsp_plot_graph(G2); title('Graph with edges learned from above 4 signals');


    The stopping criterion is whether both relative primal and dual
    distance between two iterations are below a given tolerance.

    To set the step size use the following rule of thumb: Set it so that
    relative change of primal and dual converge with similar rates.

    Adapted from matlab implementation in `GSP toolbox`_ by William Cappelletti

    References:
    - V. Kalofolias and N. Perraudin, “Large Scale Graph Learning from Smooth
        Signals.” arXiv, May 01, 2019. doi: 10.48550/arXiv.1710.05654.
    - V. Kalofolias, “How to Learn a Graph from Smooth Signals,” in Proceedings
        of the 19th International Conference on Artificial Intelligence and
        Statistics, May 2016, pp. 920–929. doi: 10.48550/arXiv.1601.02513.
    - N. Komodakis and J.-C. Pesquet, “Playing with Duality: An overview of
        recent primal?dual approaches for solving large-scale optimization
        problems,” IEEE Signal Processing Magazine, vol. 32, no. 6, pp. 31–54, Nov.
        2015, doi: 10.1109/MSP.2014.2377273.

    .. _`GSP toolbox`: https://epfl-lts2.github.io/gspbox-html/
    """
    # pylint: disable=too-many-locals

    # distances shall be vectorform of pairwise distances
    if (n_dim := len(distances.shape)) == 2:
        sum_op, sum_op_t = sum_squareform(distances.shape[0])
        distances = squareform(distances)
    elif n_dim == 1:
        sum_op, sum_op_t = sum_squareform(
            int(np.round((1 + np.sqrt(1 + 8 * len(distances))) / 2))
        )  # TODO: check
    else:
        raise ValueError(f"Distances must be square matrix or vector, got {n_dim} dimensions")

    step_size /= 2 * beta + np.sqrt(
        2 * (sum_op.shape[0] - 1)  # This approximate sparse.linalg.norm(sum_op, ord=2)
    )
    # epsilon = 0

    # Variable and dual
    if edge_init is not None:
        edge_w = edge_init
    else:
        edge_w = np.zeros_like(distances)
    d_n = sum_op @ edge_w

    for i in range(maxit):
        y_n = (1 - step_size * 2 * beta) * edge_w - step_size * (sum_op_t @ d_n)
        yb_n = d_n + step_size * (sum_op @ edge_w)

        p_n = np.where((prox := y_n - 2 * step_size * distances) > 0, prox, 0)
        pb_n = (yb_n - np.sqrt(yb_n**2 + 4 * alpha * step_size)) / 2

        q_n = (1 - step_size * 2 * beta) * p_n - step_size * (sum_op_t @ pb_n)
        qb_n = pb_n + step_size * (sum_op @ p_n)

        if i > 0:
            # Denominators are previous iteration ones
            # TODO: Verify why in original they use ord="fro" for vectorized primal vars
            rel_norm_primal = np.linalg.norm(-y_n + q_n, ord=2) / np.linalg.norm(edge_w, ord=2)
            rel_norm_dual = np.linalg.norm(-yb_n + qb_n, ord=2) / np.linalg.norm(d_n, ord=2)
        else:
            rel_norm_primal = rel_norm_dual = np.inf

        edge_w = edge_w - y_n + q_n  # TODO: check since q_n is p_n in paper
        d_n = d_n - yb_n + qb_n

        if rel_norm_primal < tol and rel_norm_dual < tol:
            break

    edge_w[edge_w < edge_tol] = 0

    return squareform(edge_w)
