"""Graph components learning original method"""
from __future__ import annotations

from time import time
from typing import Callable
from warnings import warn

import numpy as np
import pandas as pd
from numpy.random import RandomState
from numpy.typing import NDArray
from sklearn.base import BaseEstimator

from graph_learn.evaluation import relative_error

# from graph_learn import OptimizationError
from graph_learn.operators import (
    laplacian_squareform_vec,
    op_adj_activations,
    op_adj_weights,
    op_weights_norm,
    prox_gdet_star,
    square_to_vec,
)

from .utils import combinations_prob, powerset_matrix


class GraphDictionary(BaseEstimator):
    """Graph components learning original method"""

    def __init__(
        self,
        n_atoms=1,
        *,
        window_size: int = 1,
        l1_w: float = 0,
        ortho_w: float = 0,
        smooth_a: float = 1,
        l1_a: float = 0,
        log_a: float = 0,
        l1_diff_a: float = 0,
        max_iter: int = 1000,
        step: float = 1,
        step_a: float = None,
        step_w: float = None,
        step_dual: float = None,
        tol: float = 1e-3,
        reduce_step_on_plateau: bool = False,
        random_state: RandomState = None,
        init_strategy: str = "uniform",
        n_init: int = 1,
        weight_prior: float | NDArray[np.float_] = None,
        activation_prior: float | NDArray[np.float_] = None,
        verbose: int = 0,
    ) -> None:
        super().__init__()

        self.n_atoms = n_atoms
        self.window_size = window_size
        self.ortho_w = ortho_w
        self.l1_w = l1_w
        self.smooth_a = smooth_a
        self.l1_a = l1_a
        self.log_a = log_a
        self.l1_diff_a = l1_diff_a

        self.max_iter = max_iter

        self.step_a = step_a or step
        self.step_w = step_w or step
        self.step_dual = step_dual or step

        self.tol = tol
        self.reduce_step_on_plateau = reduce_step_on_plateau

        if isinstance(random_state, RandomState):
            self.random_state = random_state
        else:
            self.random_state = RandomState(random_state)

        self.init_strategy = init_strategy
        self.n_init = n_init
        self.weight_prior = weight_prior
        self.activation_prior = activation_prior

        self.verbose = verbose

        # Combinations are binary representation of their column index
        self._combinations = powerset_matrix(n_atoms=self.n_atoms)  # shape (n_atoms, 2**n_atoms)
        self._sq_pdiffs: NDArray[np.float_]  # shape (n_atoms, n_edges )

        self.activations_: NDArray[np.float_]  # shape (n_atoms, n_samples)
        self.weights_: NDArray[np.float_]  # shape (n_atoms, n_edges )
        self.dual_: NDArray[np.float_]  # shape (2**n_atoms, n_nodes, n_nodes)
        self.n_nodes_: int
        self.n_samples_: int
        self.scale_: float
        self.converged_: int
        self.history_: pd.DataFrame
        self.fit_time_: float

    def _init_activations(self, n_samples) -> NDArray[np.float_]:
        activations = np.ones((self.n_atoms, n_samples))

        match self.init_strategy:
            case "uniform":
                if self.activation_prior is None:
                    activations = self.random_state.uniform(size=(self.n_atoms, n_samples))
                else:
                    self.activations_ *= self.activation_prior

            case "discrete":
                activations = self.random_state.uniform(size=(self.n_atoms, n_samples)) < (
                    self.activation_prior or 0.5
                )
                activations = activations.astype(float)

            case "exp":
                activations = self.random_state.exponential(
                    scale=self.activation_prior or 1, size=activations.shape
                )

            case "exact":
                if self.activation_prior is not None:
                    activations *= self.activation_prior
            case _:
                raise ValueError(f"Invalid init strategy {self.init_strategy}")

        if self.window_size > 1:
            activations = np.repeat(activations[:, :: self.window_size], self.window_size, axis=1)[
                :, :n_samples
            ]
        return activations

    def _init_weigths(self) -> NDArray[np.float_]:
        weights = np.ones((self.n_atoms, (self.n_nodes_**2 - self.n_nodes_) // 2))

        match self.init_strategy:
            case "uniform":
                if self.weight_prior is not None and self.activation_prior is not None:
                    raise ValueError("Need one free parameter for uniform initialization")

                if self.weight_prior is None:
                    weights = self.random_state.uniform(
                        size=(self.n_atoms, (self.n_nodes_**2 - self.n_nodes_) // 2)
                    )
                else:
                    weights *= self.weight_prior

            case "discrete":
                weights = self.random_state.uniform(
                    size=(self.n_atoms, (self.n_nodes_**2 - self.n_nodes_) // 2)
                ) < (self.weight_prior or 0.5)
                weights = weights.astype(float)

            case "exp":
                weights = self.random_state.exponential(
                    scale=self.weight_prior or 1, size=weights.shape
                )

            case "exact":
                if self.weight_prior is not None:
                    weights *= self.weight_prior

            case _:
                raise ValueError(f"Invalid init strategy {self.init_strategy}")

        return weights

    def _initialize(self, x: NDArray) -> None:
        self.n_samples_, self.n_nodes_ = x.shape

        self.weights_ = self._init_weigths()
        self.activations_ = self._init_activations(self.n_samples_)
        self.dual_ = np.zeros((2**self.n_atoms, self.n_nodes_, self.n_nodes_))

        self._sq_pdiffs = self._pairwise_sq_diff(x)

        self.converged_ = -1
        self.fit_time_ = -1
        self.history_ = pd.DataFrame(
            data=-np.ones((self.max_iter, 2), dtype=float),
            columns=["activ_change", "weight_change"],
            index=np.arange(self.max_iter),
        )

    def _pairwise_sq_diff(self, x: NDArray[np.float_]) -> NDArray[np.float_]:
        """Compute pairwise square differences between all signals, on all nodes

        Args:
            x (NDArray[np.float_]): Signals matrix of shape (n_samples, n_nodes)

        Returns:
            NDArray[np.float_]: Pairwise squared diferences on all edges of shape
                (n_samples, n_edges), w/ ``n_edges = n_nodes * (n_nodes - 1) / 2``
        """
        # This works with continuous activations
        pdiffs = x[:, np.newaxis, :] - x[:, :, np.newaxis]
        return square_to_vec(pdiffs) ** 2

    def _component_pdist_sq(
        self, activations: NDArray[np.float_], x: NDArray[np.float_] = None
    ) -> NDArray[np.float_]:
        """Compute pairwise square distances on each componend, based on activations

        Args:
            x (NDArray[np.float_]): Design matrix of shape (n_samples, n_nodes)

        Returns:
            NDArray[np.float_]: Pairwise node squared distances of shape
                (n_atoms, n_edges), w/ ``n_edges = n_nodes * (n_nodes - 1) / 2``
        """
        if x is None:
            sq_pdiffs = self._sq_pdiffs
        else:
            sq_pdiffs = self._pairwise_sq_diff(x)
        return activations @ sq_pdiffs

    def _update_activations(
        self,
        x: NDArray[np.float_],
        activations: NDArray[np.float_],
        combi_p: NDArray[np.float_],
        dual: NDArray[np.float_],
    ) -> NDArray[np.float_]:
        """Update activations

        Args:
            x (NDArray[np.float_]): Signal matrix of shape (n_samples, n_nodes)
            activations (NDArray[np.float_]): Current activations of shape (n_atoms, n_samples)
            combi_p (NDArray[np.float_]): Probabilities of combinations for each sample.
                Array of shape (2**n_atoms, n_samples)
            dual (NDArray[np.float_]): Dual variable (instantaneous Laplacians)
                of shape (2**n_atoms, n_nodes, n_nodes)

        Returns:
            NDArray[np.float_]: Updated activations of shape (n_atoms, n_samples)
        """
        laplacians = laplacian_squareform_vec(self.weights_)

        # Smoothness
        step = np.einsum("ktn,tn->kt", x @ laplacians, x) * self.smooth_a
        if self.window_size > 1:
            step = np.repeat(
                # average non-overlapping windows
                step.reshape(self.n_atoms, -1, self.window_size).mean(2),
                self.window_size,
                axis=1,
            )

        # L1 regularization
        step += self.l1_a

        if self.log_a > 0:
            step -= self.log_a / activations.sum(axis=0, keepdims=True)

        if self.l1_diff_a > 0:
            step -= self.l1_diff_a * (
                np.diff(np.sign(np.diff(activations, axis=1)), axis=1, prepend=0, append=0)
            )

        # Proximal and gradient step
        # NOTE: the step might be divided by the operator norm
        activations = activations - self.step_a * (
            op_adj_activations(self.weights_, dual) @ combi_p + step / self.n_samples_
        )

        # Projection
        activations[activations < 0] = 0
        activations[activations > 1] = 1

        if np.allclose(activations, 0):
            warn("All activations dropped to 0", UserWarning)
            activations.fill(1)

        return activations

    def _update_weights(
        self,
        x: NDArray[np.float_],
        weights: NDArray[np.float_],
        combi_p: NDArray[np.float_],
        dual: NDArray[np.float_],
    ) -> NDArray[np.float_]:
        """Update the weights of the model

        Args:
            x (NDArray[np.float_]): Signal matrix of shape (n_samples, n_nodes)
            combi_p (NDArray[np.float_]): Monte-Carlo probabilities of combinations
                for each sample. Array of shape (2**n_atoms, n_samples)
            dual (NDArray[np.float_]): Dual variable (instantaneous Laplacians)
                of shape (2**n_atoms, n_nodes, n_nodes)

        Returns:
            NDArray[np.float_]: updated weights of shape (n_atoms, (n_nodes**2 - n_nodes) // 2)
        """
        # FIXEDME: here I use activations (K x T), while for prox update I use activations_ @ combi_p.T (K x 2**k)
        activations = self.activations_ @ combi_p.T
        op_norm = op_weights_norm(activations=activations, n_nodes=self.n_nodes_)

        step = self._component_pdist_sq(self.activations_)
        step += self.l1_w

        if self.ortho_w > 0:
            # grad_step = (
            #     np.ones((self.n_atoms, 1)) - np.eye(self.n_atoms)
            # ) @ self.weights_
            step += self.ortho_w(weights.sum(0, keepdims=True) - weights)

        # Proximal update
        weights = weights - self.step_w / op_norm * (
            op_adj_weights(activations, dual) + step / self.n_samples_
        )

        # Projection
        weights[weights < 0] = 0
        return weights

    def _update_dual(
        self, weights: NDArray[np.float_], activations: NDArray[np.float_], dual: NDArray[np.float_]
    ):
        # z1 = dualv + step * bilinear_op(x_overshoot, y_overshoot)
        # z1 -= step * prox_h(z1 / step, 1 / step)

        # NOTE: Isn't it weird that activations have so little influence on the dual update?

        # TODO: I should filter combinations that don't appear, but will they never appear?
        # Filtering would work for fixed activations

        op_norm = op_weights_norm(
            activations=activations, n_nodes=self.n_nodes_
        )  # * op_activations_norm(lapl=laplacian_squareform_vec(weights))
        dual = dual + self.step_dual / op_norm * np.einsum(
            "kc,knm->cnm", self._combinations, laplacian_squareform_vec(weights)
        )
        return prox_gdet_star(dual, sigma=self.step_dual / op_norm / self.n_samples_)

    def _fit_step(self, x: NDArray[np.float_]) -> (float, float):
        combi_p = combinations_prob(self.activations_)

        # primal update
        # x1 = prox_gx(x - step * (op_adjx(x, dualv) + gradf_x(x, y, gz)), step)
        # y1 = prox_gy(y - step * (op_adjy(y, dualv) + gradf_y(x, y, gz)), step)
        activations = self._update_activations(
            x, self.activations_, combi_p=combi_p, dual=self.dual_
        )
        weights = self._update_weights(x, self.weights_, combi_p=combi_p, dual=self.dual_)

        # dual update
        # x_overshoot = 2 * activations - self.activations_
        self.dual_ = self._update_dual(
            weights=2 * weights - self.weights_,
            # weights=weights,
            # TODO: understand what is going on here
            # activations=2 * activations - self.activations_,
            activations=self.activations_,
            dual=self.dual_,
        )

        a_rel_change = relative_error(self.activations_.ravel(), activations.ravel())
        w_rel_change = relative_error(self.weights_.ravel(), weights.ravel())

        self.activations_ = activations
        self.weights_ = weights

        return a_rel_change, w_rel_change

    def _single_fit(
        self,
        x: NDArray[np.float_],
        _y=None,
        *,
        callback: Callable[[GraphDictionary, int]] = None,
    ) -> GraphDictionary:
        self._initialize(x)
        if callback is not None:
            callback(self, 0)

        start = time()
        for i in range(self.max_iter):
            try:
                self.history_.iloc[i] = a_rel_change, w_rel_change = self._fit_step(x)

                if (a_rel_change < self.tol) and (w_rel_change < self.tol):
                    self.converged_ = i

                if (
                    self.reduce_step_on_plateau
                    and i > 100
                    and (
                        a_rel_change >= self.history_.iloc[i - 10 : i, 0].mean()
                        and w_rel_change >= self.history_.iloc[i - 10 : i, 1].mean()
                    )
                ):
                    warn(f"Divergence detected at step {i}, reducing step size", UserWarning)
                    self.step_a *= 0.7
                    self.step_w *= 0.7
                    self.step_dual *= 0.7

                if callback is not None:
                    callback(self, i + 1)

                if self.converged_ > 0:
                    break

            except KeyboardInterrupt:
                warn("Keyboard interrupt, stopping early")
                break

        self.fit_time_ = time() - start

        return self

    def score(self, x: NDArray[np.float_], _y=None) -> float:
        """Compute the negative log-likelihood of the model

        Args:
            x (NDArray[np.float_]): Design matrix of shape (n_samples, n_nodes)
            y (None, optional): Ignored. Defaults to None.

        Returns:
            float: Score of the model
        """
        # lw(self.weights_) + 1/self.n_samples * (
        #     la(self.activations_) + sum_t(
        #        - np.log(gdet(L_inst)) + x.T @ L_inst @ x
        #    )
        # )
        if np.allclose(self.weights_, 0):
            warn("All weights are 0, log determinant is undefined", UserWarning)
            return np.inf

        # sum of L1 norm, orthogonality and smoothness
        weight_loss = (
            self.l1_w * np.abs(self.weights_).sum()  # L1
            + np.sum(self.weights_ * self._component_pdist_sq(self.activations_, x))
            / self.n_samples_  # smoothness
        )
        if self.ortho_w > 0:
            # gradient
            weight_loss += self.ortho_w * (
                np.sum(self.weights_.T @ self.weights_) - np.linalg.norm(self.weights_) ** 2
            )

        activation_loss = self.l1_a * np.abs(self.activations_).sum()
        if self.log_a > 0:
            activation_loss -= self.log_a * np.sum(np.log(self.activations_.sum(0)))
        if self.l1_diff_a > 0:
            warn("Check diff loss", UserWarning)
            activation_loss -= self.l1_diff_a * np.linalg.norm(
                np.diff(np.sign(np.diff(self.activations_, axis=1)), axis=1, prepend=0, append=0)
            )

        # Sum of log determinants
        combi_p = combinations_prob(self.activations_)

        eigvals = np.linalg.eigvalsh(
            np.einsum("kc,knm->cnm", self._combinations, laplacian_squareform_vec(self.weights_))
        )
        # shape: (n_samples, n_combi) x (n_combi,)
        loggdet = np.sum(combi_p.T @ np.log(np.where(eigvals > 0, eigvals, 1)).sum(-1))

        return weight_loss + (activation_loss - loggdet) / self.n_samples_

    def fit(
        self, x: NDArray[np.float_], _y=None, *, callback: Callable[[GraphDictionary, int]] = None
    ):
        """Fit the model to the data

        Args:
            x (NDArray[np.float_]): Design matrix of shape (n_samples, n_nodes)
            y (None, optional): Ignored. Defaults to None.
            callback (Callable[[GraphDictionary, int]], optional): Callback function
                called at each iteration. Defaults to None.

        Returns:
            GraphDictionary: self
        """
        if self.n_init == 1:
            self._single_fit(x, callback=callback)

        else:
            best = {
                "score": None,
                "weights": None,
                "activations": None,
                "dual": None,
                "converged": None,
                "fit_time": None,
                "history": None,
            }

            for _ in range(self.n_init):
                self._single_fit(x, callback=callback)
                score = self.score(x)

                if best["score"] is None or score < best["score"]:
                    best["score"] = score
                    best["weights"] = self.weights_
                    best["activations"] = self.activations_
                    best["dual"] = self.dual_
                    best["converged"] = self.converged_
                    best["fit_time"] = self.fit_time_
                    best["history"] = self.history_

            self.weights_ = best["weights"]
            self.activations_ = best["activations"]
            self.dual_ = best["dual"]
            self.converged_ = best["converged"]
            self.fit_time_ = best["fit_time"]
            self.history_ = best["history"]

        return self

    def predict(self, x: NDArray[np.float_]) -> NDArray[np.float_]:
        """Predict activations for a given signal"""
        activations = self._init_activations(x.shape[0])

        for _ in range(self.max_iter):
            combi_p = combinations_prob(activations)
            activations_u = self._update_activations(
                x, activations, combi_p=combi_p, dual=self.dual_
            )

            if np.linalg.norm((activations_u - activations).ravel()) < self.tol:
                return activations_u

            activations = activations_u

        return activations
