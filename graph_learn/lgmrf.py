"""Model for Laplacian-constrained Gaussian Markov Random Field"""

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator

from graph_learn.evaluation import relative_error

_LAPLACIAN_SET = [
    "g",
    "generalized",
    "dd",
    "diagonally-dominant",
    "c",
    "combinatorial",
]


def nonnegative_qp_solver(
    mat: NDArray[np.float64],
    vec: NDArray[np.float64],
    step: float = 0.1,
    tol: float = 1e-6,
    max_iter=100,
) -> NDArray[np.float64]:
    r"""Find non-negative solution of the following quadratic form
    .. :math:
        \frac{1}{2}\beta^\top Q \beta - \beta^\top p, \quad \beta \geq 0

    If conditioning number is low we solve with inv(mat) @ vec, otherwise we
    solve with ADMM.

    Args:
        mat (NDArray[np.float64]): Symmetric matrix
        vec (NDArray[np.float64]): Offset
        tol (float): Convergence tolerance

    Returns:
        NDArray[np.float64]: Solution of quadratic problem
    """
    n = mat.shape[0]

    # # We take the cond number as regularization
    # evaln, *_, eval1 = sorted(np.abs(np.linalg.eigvalsh(mat)))
    # if eval1 / evaln < 100:  # arbitrary value
    #     z_k = np.linalg.solve(mat, vec)
    #     return np.where(z_k > 0, z_k, 0)
    # else:
    #     lambda_ = 1 / eval1
    # # except np.linalg.LinAlgError:
    # #     # lambda_ = 1 / mat.sum()

    inv = np.linalg.inv(np.eye(n) + step * mat)

    x_k = z_k = u_k = np.zeros(n)
    for it in range(max_iter):
        x_kp = inv @ ((z_k - u_k) + step * vec)
        z_kp = x_kp + u_k
        z_kp[z_kp < 0] = 0
        u_k += x_kp - z_kp

        rel_change_x = relative_error(x_k, x_kp)
        rel_change_z = relative_error(z_k, z_kp)

        x_k = x_kp
        z_k = z_kp

        if it > 0 and rel_change_x < tol and rel_change_z < tol:
            return z_k

    return z_k


class LGMRF(BaseEstimator):
    """Graph learning from Laplacian-constrained Gaussian Markov Random Field

    Args:
        norm_par (float, optional): Normalization parmater that control
            sparsity of the network, the lower the sparser. Defaults to 1.
        prob_tol (float, optional): Laplacian learning tolerance, stop
            optimization if change in Fro norm is lower. Defaults to 1e-3.
        max_cycle (int, optional): Maximun number of Laplacian optimization
            cycles. Defaults to 50.
        qp_step (float, optional): Inner quadratic program optimizaiton step. Defaults to 0.1.
            Note: does not seem to affect learning at all.
        qp_tol (float, optional): Inner quadratic program tolerance. Defaults to 1e-6.
        regularization_type (int, optional): Type of regularization. Defaults to 1,
            Set the regularization matrix to one of the following
                - 1: Regularize with $\\mathbf{H} = 2 \\mathrm{Id} - \\mathbf{11}^\\top$;
                - 2: Regularize with $\\mathbf{H} = \\mathrm{Id} - \\mathbf{11}^\\top$;
        laplacian_set (str, optional): Which kind of Laplacian is to be
            learned. Defaults to 'generalized'. Options are:
                - 'generalized', or 'g': Generalized Laplacian,
        adj_mask (NDArray[np.int64], optional): Prior on graph connectivity.
            Defaults to a fully connected network.

    Parameters:
        ...

    Raises:
        NotImplementedError: _description_

    .. [egilmezGraphLearningData2017] H. E. Egilmez, E. Pavez, and A.
        Ortega, “Graph Learning From Data Under Laplacian and Structural
        Constraints,” IEEE Journal of Selected Topics in Signal Processing,
        vol.  11, no. 6, pp. 825-841, Sep. 2017, doi: 10.1109/JSTSP.2017.2726975.
    """

    def __init__(
        self,
        norm_par: float = 1,
        prob_tol=1e-3,
        max_cycle=50,
        qp_step=0.1,
        qp_tol=1e-6,
        regularization_type=1,
        laplacian_set: str = "generalized",
        adj_mask: NDArray[np.int64] = None,
    ) -> None:
        self.norm_par = norm_par
        self.prob_tol = prob_tol
        self.qp_step = qp_step
        self.qp_tol = qp_tol
        self.max_cycle = max_cycle
        self.regularization_type = regularization_type
        self.laplacian_set = laplacian_set
        self.adj_mask = adj_mask

        self.n_nodes_: int
        self.laplacian_: NDArray[np.float64]
        self.inv_laplacian_: NDArray[np.float64]
        self.converged_: int

        self._h_alpha: NDArray[np.float64]

    def _check_parameters(self, x):
        if self.laplacian_set not in _LAPLACIAN_SET:
            raise ValueError(
                f"invalid Laplacian set, got {self.laplacian_set}, must be one of [{_LAPLACIAN_SET}]"
            )

        if self.adj_mask is None:
            self.adj_mask = np.ones_like(x)
        if self.adj_mask.shape != x.shape:
            raise ValueError("Prior shape does not match data")

        if self.regularization_type not in (1, 2):
            raise ValueError(
                f"regularization type can be either 1 or 2, got {self.regularization_type}"
            )

    def _initialize(self, x):
        self._check_parameters(x)
        self.n_nodes_ = x.shape[0]

        if self.regularization_type == 1:
            self._h_alpha = 2 * np.eye(self.n_nodes_) - np.ones(self.n_nodes_)
        elif self.regularization_type == 2:
            self._h_alpha = np.eye(self.n_nodes_) - np.ones(self.n_nodes_)

        x += self._h_alpha

        self.inv_laplacian_ = np.diag(np.diag(x))
        self.laplacian_ = np.diag(1 / np.diag(x))
        self.converged_ = -1

        return x

    def _ggl_fit(self, x: NDArray[np.float64]):
        indices = np.arange(self.n_nodes_)

        for cycle in range(self.max_cycle):
            l_pre = self.laplacian_.copy()

            for u in indices:
                # minus_u = np.concatenate([indices[:u], indices[u + 1 :]])
                minus_u = indices[indices != u]

                # input matrix variables
                x_u = x[minus_u, u]
                x_uu = x[u, u]

                # update lapl_u_inv
                c_u = self.inv_laplacian_[minus_u, u]
                c_uu = self.inv_laplacian_[u, u]
                lapl_u_inv = self.inv_laplacian_[minus_u[:, np.newaxis], minus_u[np.newaxis, :]] - (
                    np.outer(c_u, c_u) / c_uu
                )

                # block-descent variables
                ind_nz = indices[:-1][
                    self.adj_mask[minus_u, u] == 1
                ]  # non-zero indices in [0, n_nodes-1]

                # block-descent step, note the sign flip
                lapl_u = np.zeros_like(minus_u, dtype=float)
                b_opt = -nonnegative_qp_solver(
                    mat=lapl_u_inv[ind_nz[:, np.newaxis], ind_nz[np.newaxis, :]],
                    vec=x_u[ind_nz] / x_uu,
                    step=self.qp_step,
                    tol=self.qp_tol,
                )
                lapl_u[ind_nz] = b_opt

                # Update the current Theta
                # lapl_u has a minus sign, disappears as quadratic form
                self.laplacian_[u, u] = (1.0 / x_uu) + lapl_u.T @ lapl_u_inv @ lapl_u
                self.laplacian_[minus_u, u] = lapl_u
                self.laplacian_[u, minus_u] = lapl_u

                # Update the current Theta inverse
                inv_lapl_u = (lapl_u_inv @ lapl_u) * x_uu
                self.inv_laplacian_[u, u] = x_uu  # 1 / (lapl_uu - lapl_u_quadratic)
                self.inv_laplacian_[u, minus_u] = -inv_lapl_u
                self.inv_laplacian_[minus_u, u] = -inv_lapl_u
                # use Sherman-Woodbury
                self.inv_laplacian_[minus_u[:, np.newaxis], minus_u[np.newaxis, :]] = lapl_u_inv + (
                    np.outer(inv_lapl_u, inv_lapl_u) / x_uu
                )

            if cycle > 4 and relative_error(l_pre, self.laplacian_) < self.prob_tol:

                self.converged_ = cycle
                break

        return self

    def fit(self, x: NDArray[np.float64], _y=None):
        x = x / np.mean(x) * self.norm_par
        x = self._initialize(x)

        if self.laplacian_set in ("g", "generalized"):
            return self._ggl_fit(x)
        else:
            raise NotImplementedError(f"{self.laplacian_set} is not available yet")
