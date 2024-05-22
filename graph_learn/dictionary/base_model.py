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
from sklearn.cluster import kmeans_plusplus

from graph_learn.evaluation import relative_error

# from graph_learn import OptimizationError
from graph_learn.operators import (
    autocorr,
    dictionary_smoothness,
    laplacian_squareform_vec,
    square_to_vec,
    squared_pdiffs,
)


class GraphDictBase(ABC, BaseEstimator):
    """Graph components learning original method"""

    # INITIALIZATION ###############################################################################

    def __init__(
        self,
        n_atoms=1,
        *,
        window_size: int = 1,
        l1_w: float = 0,
        ortho_w: float = 0,
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
        init_strategy_a: str = None,
        init_strategy_w: str = None,
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

        self.init_strategy_a = init_strategy_a or init_strategy
        self.init_strategy_w = init_strategy_w or init_strategy
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

    def _init_activations(self, x: NDArray) -> NDArray[np.float_]:
        n_samples = x.shape[0]
        activations = np.ones((self.n_atoms, n_samples // self.window_size))

        match self.init_strategy_a:
            case "uniform":
                if not isinstance(self.activation_prior, (list, np.ndarray)):
                    activations = self.random_state.uniform(size=(self.n_atoms, n_samples))

                if self.activation_prior is not None:
                    activations *= self.activation_prior

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

            case "k-means++":
                centers, _ = kmeans_plusplus(x.T, self.n_atoms, random_state=self.random_state)
                # shape: (n_atoms, n_samples)
                # TODO: correlations would make more sense with weight definition
                dists = np.sum((x[np.newaxis, :, :] - centers[:, :, np.newaxis]) ** 2, axis=2)

                activations.fill(0)
                activations[np.argmax(dists, axis=0), np.arange(n_samples)] = 1

            case _:
                raise ValueError(f"Invalid init_strategy for activations: {self.init_strategy_a}")

        activations[activations < 0] = 0
        activations[activations > 1] = 1

        if self.window_size > 1:
            activations = np.repeat(activations, self.window_size, axis=1)[:, :n_samples]
        return activations

    def _init_weigths(self, x: NDArray = None) -> NDArray[np.float_]:
        weights = np.ones((self.n_atoms, (self.n_nodes_**2 - self.n_nodes_) // 2))

        match self.init_strategy_w.split("_"):
            case ["uniform"]:
                if not isinstance(self.weight_prior, (list, np.ndarray)):
                    weights = self.random_state.uniform(
                        size=(self.n_atoms, (self.n_nodes_**2 - self.n_nodes_) // 2)
                    )

                if self.weight_prior is not None:
                    weights *= self.weight_prior

            case ["discrete"]:
                weights = self.random_state.uniform(
                    size=(self.n_atoms, (self.n_nodes_**2 - self.n_nodes_) // 2)
                ) < (self.weight_prior or 0.5)
                weights = weights.astype(float)

            case ["exp"]:
                weights = self.random_state.exponential(
                    scale=self.weight_prior or 1, size=weights.shape
                )

            case ["exact"]:
                if self.weight_prior is not None:
                    weights *= self.weight_prior

            case ["correlation", choice, *other]:
                match choice:
                    case "a":
                        # Stands for Activations
                        choices = np.argmax(self.activations_, axis=0)
                    case "d":
                        # Stands for Discrete
                        choices = self.random_state.choice(
                            self.n_atoms, size=self.n_samples_, replace=False
                        )
                    case _:
                        raise ValueError("Invalid choice of correlation for init_strategy")

                weights = np.stack([autocorr(x[choices == i]) for i in range(self.n_atoms)])

                if other == ["pinv"]:
                    weights = np.linalg.pinv(weights)
                elif other:
                    raise ValueError(f"Additional option not recognized: {other}")

                weights = -square_to_vec(weights)

            case _:
                raise ValueError(f"Invalid init_strategy for weights: {self.init_strategy_w}")

        weights[weights < 0] = 0

        return weights

    def _init_dual(self, n_samples: int):
        return np.zeros((n_samples // self.window_size, self.n_nodes_, self.n_nodes_))

    def _initialize(self, x: NDArray) -> None:
        self.n_samples_, self.n_nodes_ = x.shape

        self.activations_ = self._init_activations(x)
        self.weights_ = self._init_weigths(x)
        self.dual_ = self._init_dual(self.n_samples_)

        self.converged_ = -1
        self.fit_time_ = -1
        self.history_ = pd.DataFrame(
            data=-np.ones((self.max_iter, 2), dtype=float),
            columns=["activ_change", "weight_change"],
            index=np.arange(self.max_iter),
        )

    # UPDATE ACTIVATIONS ###########################################################################

    @abstractmethod
    def _op_adj_activations(self, weights: NDArray, dualv: NDArray) -> NDArray:
        """Compute the adjoint of the bilinear inst-degree operator wrt activations

        Args:
            weights (NDArray): Array of weights of shape (n_components, n_edges)
            dualv (NDArray): Instantaneous degrees, of shape (n_nodes, n_samples)

        Returns:
            NDArray: Adjoint activations of shape (n_components, n_samples)
        """
        raise NotImplementedError

    def _update_activations(
        self,
        sq_pdiffs: NDArray[np.float_],
        activations: NDArray[np.float_],
        dual: NDArray[np.float_],
    ) -> NDArray[np.float_]:
        """Update activations

        Args:
            sq_pdiffs (NDArray[np.float_]): Squared pairwise differences of
                shape (n_samples, n_edges)
            activations (NDArray[np.float_]): Current activations of shape (n_atoms, n_samples)
            dual (NDArray[np.float_]): Dual variable (instantaneous Laplacians)
                of shape (n_samples, n_nodes, n_nodes)

        Returns:
            NDArray[np.float_]: Updated activations of shape (n_atoms, n_samples)
        """

        n_samples = sq_pdiffs.shape[0]
        # Smoothness
        step = self.weights_ @ sq_pdiffs.T

        if self.window_size > 1:
            # average non-overlapping windows
            step = step.reshape(self.n_atoms, -1, self.window_size).mean(2)

        # L1 regularization
        step += self.l1_a * self.n_samples_

        if self.log_a > 0:
            # step -= self.log_a / activations.sum(axis=0, keepdims=True)
            activations[:, activations.sum(0) < 1e-8] = self.log_a

        if self.l1_diff_a > 0:
            step -= (
                self.l1_diff_a
                * self.n_samples_
                * (np.diff(np.sign(np.diff(activations, axis=1)), axis=1, prepend=0, append=0))
            )

        dual_step = self._op_adj_activations(self.weights_, dual)

        # Proximal and gradient step
        # NOTE: the step might be divided by the operator norm
        activations = activations - self.step_a / n_samples * (dual_step + step)

        # Projection
        activations[activations < 0] = 0
        activations[activations > 1] = 1

        if np.allclose(activations, 0):
            warn("All activations dropped to 0", UserWarning)
            activations.fill(self.log_a)

        return activations

    # UPDATE WEIGHTS ###############################################################################

    @abstractmethod
    def _op_adj_weights(self, activations: NDArray, dualv: NDArray) -> NDArray:
        """Compute the adjoint of the bilinear inst-degree operator wrt weights

        Args:
            activations (NDArray): Array of activations of shape (n_components, n_samples)
            dualv (NDArray): Instantaneous degrees, of shape (n_nodes, n_samples)

        Returns:
            NDArray: Dual weights of shape (n_components, n_edges)
        """
        raise NotImplementedError()

    def _update_weights(
        self,
        sq_pdiffs: NDArray[np.float_],
        weights: NDArray[np.float_],
        dual: NDArray[np.float_],
    ) -> NDArray[np.float_]:
        """Update the weights of the model

        Args:
            x (NDArray[np.float_]): Signal matrix of shape (n_samples, n_nodes)
            dual (NDArray[np.float_]): Dual variable (instantaneous Laplacians)
                of shape (n_samples, n_nodes, n_nodes)

        Returns:
            NDArray[np.float_]: updated weights of shape (n_atoms, (n_nodes**2 - n_nodes) // 2)
        """
        n_samples = sq_pdiffs.shape[0]

        # Smoothness
        step = self.activations_ @ sq_pdiffs
        step += self.l1_w * self.n_samples_

        if self.ortho_w > 0:
            step += self.ortho_w * self.n_samples_ * (weights.sum(0, keepdims=True) - weights)

        dual_step = self._op_adj_weights(self.activations_, dual)

        # Proximal update
        weights = weights - self.step_w / n_samples * (dual_step + step)

        # Projection
        weights[weights < 0] = 0
        return weights

    # UPDATE DUAL ##################################################################################

    @abstractmethod
    def _update_dual(
        self,
        weights: NDArray[np.float_],
        activations: NDArray[np.float_],
        dual: NDArray[np.float_],
    ):
        raise NotImplementedError

    def _fit_step(self, sq_pdiffs: NDArray[np.float_]) -> tuple[float, float]:
        # primal update
        # x1 = prox_gx(x - step * (op_adjx(x, dualv) + gradf_x(x, y, gz)), step)
        # y1 = prox_gy(y - step * (op_adjy(y, dualv) + gradf_y(x, y, gz)), step)

        activations = np.repeat(
            self._update_activations(
                sq_pdiffs, self.activations_[:, :: self.window_size], dual=self.dual_
            ),
            repeats=self.window_size,
            axis=1,
        )
        weights = self._update_weights(
            sq_pdiffs, self.weights_, dual=np.repeat(self.dual_, self.window_size, 1)
        )

        # dual update
        # x_overshoot = 2 * activations - self.activations_
        self.dual_ = self._update_dual(
            weights=2 * weights - self.weights_,
            activations=2 * activations[:, :: self.window_size]
            - self.activations_[:, :: self.window_size],
            dual=self.dual_,
        )

        a_rel_change = relative_error(self.activations_.ravel(), activations.ravel())
        w_rel_change = relative_error(self.weights_.ravel(), weights.ravel())

        self.activations_ = activations
        self.weights_ = weights

        return a_rel_change, w_rel_change

    # FITTING FUNCTIONS ############################################################################

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
        n_samples = x.shape[0]
        pad_size = n_samples % self.window_size
        if pad_size > 0:
            x = np.concatenate(
                (x, np.zeros((self.window_size - pad_size, x.shape[1]), dtype=float))
            )
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

        self.activations_ = self.activations_[:, :n_samples]
        return self

    # PREDICT ######################################################################################

    def predict(self, x: NDArray[np.float_]) -> NDArray[np.float_]:
        """Predict activations for a given signal"""
        n_samples, n_nodes = x.shape
        if n_nodes != self.n_nodes_:
            raise ValueError(f"Number of nodes mismatch, got {n_nodes} instead of {self.n_nodes_}")

        activations = self._init_activations(x)
        dual = self._init_dual(n_samples)

        # op_act_norm = op_activations_norm(laplacian_squareform_vec(self.weights_))

        sq_pdiffs = squared_pdiffs(x)

        for _ in range(self.max_iter):
            activations_u = np.repeat(
                self._update_activations(sq_pdiffs, activations[:, :: self.window_size], dual=dual),
                repeats=self.window_size,
                axis=1,
            )

            dual = self._update_dual(
                self.weights_,
                2 * activations_u[:, :: self.window_size] - activations[:, :: self.window_size],
                dual,
            )

            if np.linalg.norm((activations_u - activations).ravel()) < self.tol:
                return activations_u

            activations = activations_u

        return activations[:, :n_samples]

    # SCORING ######################################################################################

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
