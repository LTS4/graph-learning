"""Graph components learning original method"""
from __future__ import annotations

from abc import ABC, abstractmethod
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
    dictionary_smoothness,
    laplacian_squareform_vec,
    op_activations_norm,
    op_weights_norm,
    squared_pdiffs,
)


class GraphDictBase(ABC, BaseEstimator):
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
        early_stop_keyboard: bool = False,
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

        self.step = step
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
        self.early_stop_keyboard = early_stop_keyboard

        self._sq_pdiffs: NDArray[np.float_]  # shape (n_atoms, n_edges )

        self.activations_: NDArray[np.float_]  # shape (n_atoms, n_samples)
        self.weights_: NDArray[np.float_]  # shape (n_atoms, n_edges )
        self.dual_: NDArray[np.float_]  # shape (n_samples, n_nodes, n_nodes)
        self.n_nodes_: int
        self.n_samples_: int
        self.scale_: float
        self.converged_: int
        self.history_: pd.DataFrame
        self.fit_time_: float

    def _init_activations(self, n_samples) -> NDArray[np.float_]:
        activations = np.ones((self.n_atoms, n_samples // self.window_size))

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

        activations[activations < 0] = 0
        activations[activations > 1] = 1

        if self.window_size > 1:
            activations = np.repeat(activations, self.window_size, axis=1)[:, :n_samples]
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

        weights[weights < 0] = 0

        return weights

    def _init_dual(self, n_samples: int):
        return np.zeros((n_samples // self.window_size, self.n_nodes_, self.n_nodes_))

    def _initialize(self, x: NDArray) -> None:
        self.n_samples_, self.n_nodes_ = x.shape

        self.weights_ = self._init_weigths()
        self.activations_ = self._init_activations(self.n_samples_)
        self.dual_ = self._init_dual(self.n_samples_)

        self.converged_ = -1
        self.fit_time_ = -1
        self.history_ = pd.DataFrame(
            data=-np.ones((self.max_iter, 2), dtype=float),
            columns=["activ_change", "weight_change"],
            index=np.arange(self.max_iter),
        )

    @abstractmethod
    def _update_activations(
        self,
        sq_pdiffs: NDArray[np.float_],
        activations: NDArray[np.float_],
        dual: NDArray[np.float_],
        op_norm=1,
    ) -> NDArray[np.float_]:
        """Update activations

        Args:
            sq_pdiffs (NDArray[np.float_]): Edgewise squared dstances betweens
                signal matrix of shape (n_edges, n_nodes)
            activations (NDArray[np.float_]): Current activations of shape (n_atoms, n_samples)
            dual (NDArray[np.float_]): Dual variable (instantaneous Laplacians)
                of shape (n_samples, n_nodes, n_nodes)

        Returns:
            NDArray[np.float_]: Updated activations of shape (n_atoms, n_samples)
        """
        raise NotImplementedError

    @abstractmethod
    def _update_weights(
        self,
        sq_pdiffs: NDArray[np.float_],
        weights: NDArray[np.float_],
        dual: NDArray[np.float_],
        op_norm=1,
    ) -> NDArray[np.float_]:
        """Update the weights of the model

        Args:
            x (NDArray[np.float_]): Signal matrix of shape (n_samples, n_nodes)
            dual (NDArray[np.float_]): Dual variable (instantaneous Laplacians)
                of shape (n_samples, n_nodes, n_nodes)

        Returns:
            NDArray[np.float_]: updated weights of shape (n_atoms, (n_nodes**2 - n_nodes) // 2)
        """
        raise NotImplementedError

    @abstractmethod
    def _update_dual(
        self,
        weights: NDArray[np.float_],
        activations: NDArray[np.float_],
        dual: NDArray[np.float_],
        op_norm=1,
    ):
        raise NotImplementedError

    def _fit_step(self, sq_pdiffs: NDArray[np.float_]) -> (float, float):
        # primal update
        # x1 = prox_gx(x - step * (op_adjx(x, dualv) + gradf_x(x, y, gz)), step)
        # y1 = prox_gy(y - step * (op_adjy(y, dualv) + gradf_y(x, y, gz)), step)
        activations = self._update_activations(sq_pdiffs, self.activations_, dual=self.dual_)
        weights = self._update_weights(sq_pdiffs, self.weights_, dual=self.dual_)

        # dual update
        # x_overshoot = 2 * activations - self.activations_
        # NOTE: I tested overshoot and solutions are either exactly the same, or slightly worse
        op_norm = op_activations_norm(laplacian_squareform_vec(weights)) * op_weights_norm(
            activations, self.n_nodes_
        )
        self.dual_ = self._update_dual(
            weights=weights,
            activations=activations,
            dual=self.dual_,
            op_norm=1 / op_norm,
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
        callback: Callable[[GraphDictBase, int]] = None,
    ) -> GraphDictBase:
        self._initialize(x)

        sq_pdiffs = squared_pdiffs(x)

        if callback is not None:
            callback(self, 0)

        start = time()
        for i in range(self.max_iter):
            try:
                self.history_.iloc[i] = a_rel_change, w_rel_change = self._fit_step(sq_pdiffs)

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
                if self.early_stop_keyboard:
                    warn("Keyboard interrupt, stopping early")
                    break
                else:
                    raise

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
        weight_loss = self.l1_w * np.abs(self.weights_).sum() + dictionary_smoothness(  # L1
            self.activations_, self.weights_, x
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
        eigvals = np.linalg.eigvalsh(
            np.einsum("kc,knm->cnm", self.activations_, laplacian_squareform_vec(self.weights_))
        )
        # shape: (n_samples, n_combi) x (n_combi,)
        loggdet = np.sum(np.log(np.where(eigvals > 0, eigvals, 1)).sum(-1))

        return weight_loss + (activation_loss - loggdet)

    def fit(
        self, x: NDArray[np.float_], _y=None, *, callback: Callable[[GraphDictBase, int]] = None
    ):
        """Fit the model to the data

        Args:
            x (NDArray[np.float_]): Design matrix of shape (n_samples, n_nodes)
            y (None, optional): Ignored. Defaults to None.
            callback (Callable[[GraphDictBase, int]], optional): Callback function
                called at each iteration. Defaults to None.

        Returns:
            GraphDictBase: self
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

    @abstractmethod
    def predict(self, x: NDArray[np.float_]) -> NDArray[np.float_]:
        """Predict activations for a given signal"""
        raise NotImplementedError
