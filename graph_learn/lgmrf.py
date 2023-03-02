"""Model for Laplacian-constrained Gaussian Markov Random Field"""
import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator

_LAPLACIAN_SET = [
    "g",
    "generalized",
    "dd",
    "diagonally-dominant",
    "c",
    "combinatorial",
]


def nonnegative_qp_solver(
    mat: NDArray[np.float_],
    vec: NDArray[np.float_],
    tol: float = 1e-6,
    max_iter=100,
) -> NDArray[np.float_]:
    r"""Find non-negative solution of the following quadratic form
    .. :math:
        \frac{1}{2}\beta^\top Q \beta - \beta^\top p, \quad \beta \geq 0

    We solve with ADMM

    Args:
        mat (NDArray[np.float_]): Symmetric matrix
        vec (NDArray[np.float_]): Offset
        tol (float): Convergence tolerance

    Returns:
        NDArray[np.float_]: Solution of quadratic problem
    """
    lambda_ = 0.1
    n = mat.shape[0]
    inv = np.linalg.inv(np.eye(n) + lambda_ * mat)

    x_k = z_k = u_k = np.zeros(n)
    for it in range(max_iter):
        x_kp = inv @ ((z_k - u_k) + lambda_ * vec)
        z_kp = x_kp + u_k
        z_kp[z_kp < 0] = 0
        u_k += x_kp - z_kp

        rel_change_x = (
            np.linalg.norm(x_kp - x_k) / np.linalg.norm(x_k) if not np.allclose(x_k, 0) else np.inf
        )
        rel_change_z = (
            np.linalg.norm(z_kp - z_k) / np.linalg.norm(z_k) if not np.allclose(z_k, 0) else np.inf
        )

        x_k = x_kp
        z_k = z_kp

        if it > 0 and rel_change_x < tol and rel_change_z < tol:
            return z_k

    return z_k


class LGMRF(BaseEstimator):
    def __init__(
        self,
        alpha: float = 0,
        prob_tol=1e-4,
        inner_tol=1e-6,
        max_cycle=20,
        regularization_type=1,
        laplacian_set: str = "ggl",
        adj_mask: NDArray[np.int_] = None,
    ) -> None:
        """Graph learning from Laplacian-constrained Gaussian Markov Random Field

        Args:
            alpha (float, optional): _description_. Defaults to None.
            tol (float, optional): _description_. Defaults to 1e-4.
            laplacian_set (str, optional): _description_. Defaults to 'ggl'.
            connectivity_prior (NDArray[np.int_], optional): _description_. Defaults to None.

        Parameters:
            ...

        Raises:
            NotImplementedError: _description_

        .. [egilmezGraphLearningData2017] H. E. Egilmez, E. Pavez, and A.
            Ortega, “Graph Learning From Data Under Laplacian and Structural
            Constraints,” IEEE Journal of Selected Topics in Signal Processing,
            vol.  11, no. 6, pp. 825-841, Sep. 2017, doi: 10.1109/JSTSP.2017.2726975.

        """
        self.alpha = alpha
        self.prob_tol = prob_tol
        self.inner_tol = inner_tol
        self.max_cycle = max_cycle
        self.regularization_type = regularization_type
        self.laplacian_set = laplacian_set
        self.adj_mask = adj_mask

        self.n_nodes_: int
        self.laplacian_: NDArray[np.float_]
        self.inv_laplacian_: NDArray[np.float_]

        self._h_alpha: NDArray[np.float_]

    def _check_parameters(self, x):
        if self.alpha < 0:
            raise ValueError(f"Alpha must be greater than 0, got {self.alpha}")

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
            self._h_alpha = (self.alpha) * (2 * np.eye(self.n_nodes_) - np.ones(self.n_nodes_))
        elif self.regularization_type == 2:
            self._h_alpha = (self.alpha) * (np.eye(self.n_nodes_) - np.ones(self.n_nodes_))

        x += self._h_alpha

        self.inv_laplacian_ = np.diag(np.diag(x))
        self.laplacian_ = np.diag(1 / np.diag(x))

        return x

    def fit(self, x: NDArray[np.float_], _y):
        x = self._initialize(x)

        indices = np.arange(self.n_nodes_)
        for _cycle in range(self.max_cycle):
            l_pre = self.laplacian_

            for u in indices:
                minus_u = np.concatenate([indices[:u], indices[u + 1 :]])

                # input matrix variables
                x_u = x[minus_u, u]
                x_uu = x[u, u]

                # update Ou_inv
                c_u = self.inv_laplacian_[minus_u, u]
                c_uu = self.inv_laplacian_[u, u]
                lapl_u_inv = self.inv_laplacian_[minus_u, minus_u] - (c_u @ c_u.T / c_uu)

                # block-descent variables
                beta = np.zeros((n - 1, 1))
                ind_nz = self.adj_mask[minus_u, u] == 1  # non-zero indices
                a_nnls = lapl_u_inv[ind_nz, ind_nz]
                b_nnls = x_u[ind_nz] / x_uu

                # block-descent step
                b_opt = nonnegative_qp_solver(a_nnls, b_nnls, self.inner_tol)

                beta_nnls = -b_opt  # sign flip
                beta[ind_nz] = beta_nnls
                lapl_u = beta
                lapl_uu = (1 / x_uu) + lapl_u.T @ lapl_u_inv * lapl_u

                # Update the current Theta
                self.laplacian_ = 1
                self.laplacian_[u, u] = lapl_uu
                self.laplacian_[minus_u, u] = lapl_u
                self.laplacian_[u, minus_u] = lapl_u

                # Update the current Theta inverse
                inv_lapl_u = (lapl_u_inv * lapl_u) * x_uu
                inv_lapl_uu = 1 / (lapl_uu - (lapl_u.T @ lapl_u_inv @ lapl_u))
                self.inv_laplacian_[u, u] = inv_lapl_uu  # C(u,u) = k_uu
                self.inv_laplacian_[u, minus_u] = -inv_lapl_u
                self.inv_laplacian_[minus_u, u] = -inv_lapl_u
                # use Sherman-Woodbury
                self.inv_laplacian_[minus_u, minus_u] = lapl_u_inv + (
                    (inv_lapl_u * inv_lapl_u.T) / (x_uu)
                )

            if (
                _cycle > 4
                and np.linalg.norm(l_pre - self.laplacian_, ord="fro")
                / np.linalg.norm(l_pre, ord="fro")
                < self.prob_tol
            ):
                break
