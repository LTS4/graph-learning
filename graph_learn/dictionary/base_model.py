"""Graph dictionary learning original method"""

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
    """Graph dictionary learning original method

    Args:
        n_atoms (int, optional): Number of atoms to learn. Defaults to 1.
        window_size (int, optional): Number of consecutive samples sharing coefficients.
            Defaults to 1.
        l1_w (float, optional): L1 regularization on the weights. Defaults to 0.
        ortho_w (float, optional): Orthogonality constraint on the weights. Defaults to 0.
        l1_c (float, optional): L1 regularization on the coefficients. Defaults to 0.
        log_c (float, optional): Logbarrier regularization on the coefficients. Defaults to 0.
        l1_diff_c (float, optional): L1 regularization on the difference of consecutive
            coefficients. Defaults to 0.
        max_iter (int, optional): Maximum number of pds iterations. Defaults to 1000.
        step (float, optional): Step size for PDS updates. Defaults to 1, ignored if step_c, step_w
            or step_dual are set.
        step_c (float, optional): Step size for coefficients updates.
            Defaults to :attr:`step` if None.
        step_w (float, optional): Step size for weights updates. Defaults to :attr:`step` if None.
        step_dual (float, optional): Step size for dual updates. Defaults to :attr:`step` if None.
        tol (float, optional): Tolerance for convergence. Defaults to 1e-3.
        reduce_step_on_plateau (bool, optional): Reduce step size if no improvement is seen.
            Defaults to False.
        random_state (RandomState, optional): Random state for reproducibility. Defaults to None.
        init_strategy (str, optional): Initialization strategy for both weights and coefficients.
            Defaults to "uniform". Available stategies are explaned for :attr:`init_strategy_c` and
            :attr:`init_strategy_w`.
        init_strategy_c (str, optional): Initialization strategy for coefficients. Defaults to None.
            Possible values are:
            - "uniform": Uniform random values between 0 and 1.
            - "discrete": Random binary values.
            - "exp": Exponential random values.
            - "exact": Exact values from :attr:`coefficient_prior`.
            - "k-means++d": K-means++ based on euclidean distance between signals.
            - "k-means++c": K-means++ based on correlation between signals.
        init_strategy_w (str, optional): Initialization strategy for weights. Defaults to None.
            Possible values are:
            - "uniform": Uniform random values between 0 and 1.
            - "discrete": Random binary values.
            - "exp": Exponential random values.
            - "exact": Exact values from :attr:`weight_prior`.
            - "correlation_a": Correlation based on coefficients.
            - "correlation_d": Correlation based on random choice of coefficients.
            - "correlation_a_pinv": Correlation based on coefficients with pseudo-inverse.
            - "correlation_d_pinv": Correlation based on random choice of coefficients with
                pseudo-inverse.
        n_init (int, optional): Number of initializations. Defaults to 1.
        weight_prior (float | NDArray, optional): Weight prior for initialization,
        coefficient_prior (float | NDArray, optional): Coefficient prior for initialization.
        early_stop_keyboard (bool, optional): Wether to stop training on keyboard interrupt.
            Defaults to False, in which case KeiboardInterrupt is raised.

    Attributes:
        coefficients_ (NDArray[np.float64]): Coefficients of the model of shape
            (n_atoms, n_samples).
        weights_ (NDArray[np.float64]): Weights of the model of shape (n_atoms, n_edges).
        dual_ (NDArray[np.float64]): Dual variable of the model of shape
            (n_samples, n_nodes, n_nodes).
        n_nodes_ (int): Number of nodes in the graph.
        converged_ (int): Number of iterations before convergence.
        history_ (pd.DataFrame): History of the model during training.
        fit_time_ (float): Time taken to fit the model.

    """

    # INITIALIZATION ###############################################################################

    def __init__(
        self,
        n_atoms=1,
        *,
        window_size: int = 1,
        l1_w: float = 0,
        ortho_w: float = 0,
        l1_c: float = 0,
        log_c: float = 0,
        l1_diff_c: float = 0,
        max_iter: int = 1000,
        step: float = 1,
        step_c: float = None,
        step_w: float = None,
        step_dual: float = None,
        tol: float = 1e-3,
        reduce_step_on_plateau: bool = False,
        random_state: RandomState = None,
        init_strategy: str = "uniform",
        init_strategy_c: str = None,
        init_strategy_w: str = None,
        n_init: int = 1,
        weight_prior: float | NDArray[np.float64] = None,
        coefficient_prior: float | NDArray[np.float64] = None,
        early_stop_keyboard: bool = False,
    ) -> None:
        super().__init__()

        self.n_atoms = n_atoms
        self.window_size = window_size
        self.ortho_w = ortho_w
        self.l1_w = l1_w
        self.l1_c = l1_c
        self.log_c = log_c
        self.l1_diff_c = l1_diff_c

        self.max_iter = max_iter

        self.step = step
        self.step_c = step_c or step
        self.step_w = step_w or step
        self.step_dual = step_dual or step

        self.tol = tol
        self.reduce_step_on_plateau = reduce_step_on_plateau

        if isinstance(random_state, RandomState):
            self.random_state = random_state
        else:
            self.random_state = RandomState(random_state)

        self.init_strategy_c = init_strategy_c or init_strategy
        self.init_strategy_w = init_strategy_w or init_strategy
        self.n_init = n_init
        self.weight_prior = weight_prior
        self.coefficient_prior = coefficient_prior

        self.early_stop_keyboard = early_stop_keyboard

        self._sq_pdiffs: NDArray[np.float64]  # shape (n_atoms, n_edges )

        self.coefficients_: NDArray[np.float64]  # shape (n_atoms, n_samples)
        self.weights_: NDArray[np.float64]  # shape (n_atoms, n_edges )
        self.dual_: NDArray[np.float64]  # shape (n_samples, n_nodes, n_nodes)
        self.n_nodes_: int
        self.converged_: int
        self.history_: pd.DataFrame
        self.fit_time_: float

    def _init_coefficients(self, x: NDArray) -> NDArray[np.float64]:
        """Initialize coefficients based on :attr:`init_strategy_c`

        Args:
            x (NDArray): Input signal of shape (n_samples, n_nodes)

        Raises:
            ValueError: Invalid init_strategy for coefficients

        Returns:
            NDArray[np.float64]: Initialized coefficients of shape (n_atoms, n_samples)
        """
        n_samples = x.shape[0]
        coefficients = np.ones((self.n_atoms, n_samples // self.window_size))

        match self.init_strategy_c:
            case "uniform":
                if not isinstance(self.coefficient_prior, (list, np.ndarray)):
                    coefficients = self.random_state.uniform(size=(self.n_atoms, n_samples))

                if self.coefficient_prior is not None:
                    coefficients *= self.coefficient_prior

            case "discrete":
                coefficients = self.random_state.uniform(size=(self.n_atoms, n_samples)) < (
                    self.coefficient_prior or 0.5
                )
                coefficients = coefficients.astype(float)

            case "exp":
                coefficients = self.random_state.exponential(
                    scale=self.coefficient_prior or 1, size=coefficients.shape
                )

            case "exact":
                if self.coefficient_prior is not None:
                    coefficients *= self.coefficient_prior

            case "k-means++d":
                # Distance based
                centers, _ = kmeans_plusplus(x, self.n_atoms, random_state=self.random_state)
                # shape: (n_atoms, n_samples)
                dists = np.sum((x[np.newaxis, :, :] - centers[:, np.newaxis, :]) ** 2, axis=2)

                coefficients.fill(0)
                coefficients[np.argmin(dists, axis=0), np.arange(n_samples)] = 1

            case "k-means++c":
                # Alignement/correlation based
                centers, _ = kmeans_plusplus(x, self.n_atoms, random_state=self.random_state)
                # shape: (n_atoms, n_samples)
                # NOTE: correlations should make more sense with weight definition
                corr_abs = np.abs(
                    (x / np.linalg.norm(x, axis=1, keepdims=True))
                    @ (centers / np.linalg.norm(centers, axis=1, keepdims=True)).T
                )

                coefficients.fill(0)
                coefficients[np.argmax(corr_abs, axis=1), np.arange(n_samples)] = 1

            case _:
                raise ValueError(f"Invalid init_strategy for coefficients: {self.init_strategy_c}")

        coefficients[coefficients < 0] = 0
        coefficients[coefficients > 1] = 1

        if self.window_size > 1:
            coefficients = np.repeat(coefficients, self.window_size, axis=1)[:, :n_samples]
        return coefficients

    def _init_weigths(self, x: NDArray = None) -> NDArray[np.float64]:
        """Initialize weights based on :attr:`init_strategy_w`

        Args:
            x (NDArray, optional): Input signal of shape (n_samples, n_nodes). Defaults to None.

        Raises:
            ValueError: Invalid init_strategy for weights

        Returns:
            NDArray[np.float64]: Initialized weights of shape (n_atoms, n_edges)
        """
        n_samples, n_nodes = x.shape
        weights = np.ones((self.n_atoms, (n_nodes**2 - n_nodes) // 2))

        match self.init_strategy_w.split("_"):
            case ["uniform"]:
                if not isinstance(self.weight_prior, (list, np.ndarray)):
                    weights = self.random_state.uniform(
                        size=(self.n_atoms, (n_nodes**2 - n_nodes) // 2)
                    )

                if self.weight_prior is not None:
                    weights *= self.weight_prior

            case ["discrete"]:
                weights = self.random_state.uniform(
                    size=(self.n_atoms, (n_nodes**2 - n_nodes) // 2)
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
                        # Stands for Coefficients
                        choices = np.argmax(self.coefficients_, axis=0)
                    case "d":
                        # Stands for Discrete
                        choices = self.random_state.choice(
                            self.n_atoms, size=n_samples, replace=False
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

    def _init_dual(self, x: NDArray) -> NDArray:
        """Initialize dual variable to zeros.

        Args:
            x (NDArray): Input signal of shape (n_samples, n_nodes)

        Returns:
            NDArray: Initialized dual variable of shape (n_samples, n_nodes, n_nodes)
        """
        n_samples, n_nodes = x.shape
        return np.zeros((n_samples // self.window_size, n_nodes, n_nodes))

    def _squared_pdiffs(self, x: NDArray) -> NDArray:
        """This function is provided to allow submodels to use other parameterization"""
        return squared_pdiffs(x)

    def _initialize(self, x: NDArray) -> None:
        """Initialize the model

        Args:
            x (NDArray): Input signal of shape (n_samples, n_nodes)
        """
        self.n_nodes_ = x.shape[1]

        self.coefficients_ = self._init_coefficients(x)
        self.weights_ = self._init_weigths(x)
        self.dual_ = self._init_dual(x)

        self.converged_ = -1
        self.fit_time_ = -1
        self.history_ = pd.DataFrame(
            data=-np.ones((self.max_iter, 2), dtype=float),
            columns=["coefficient_change", "weight_change"],
            index=np.arange(self.max_iter),
        )

    # UPDATE ACTIVATIONS ###########################################################################

    @abstractmethod
    def _op_adj_coefficients(self, weights: NDArray, dualv: NDArray) -> NDArray:
        """Compute the adjoint of the bilinear inst-degree operator wrt coefficients

        Args:
            weights (NDArray): Array of weights of shape (n_atoms, n_edges)
            dualv (NDArray): Dual variable of shape (n_samples, n_nodes, n_nodes)

        Returns:
            NDArray: Adjoint coefficients of shape (n_atoms, n_samples)
        """
        raise NotImplementedError

    def _update_coefficients(
        self,
        sq_pdiffs: NDArray[np.float64],
        coefficients: NDArray[np.float64],
        dual: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Update coefficients

        Args:
            sq_pdiffs (NDArray[np.float64]): Squared pairwise differences of
                shape (n_samples, n_edges)
            coefficients (NDArray[np.float64]): Current coefficients of shape (n_atoms, n_samples)
            dual (NDArray[np.float64]): Dual variable (instantaneous Laplacians)
                of shape (n_samples, n_nodes, n_nodes)

        Returns:
            NDArray[np.float64]: Updated coefficients of shape (n_atoms, n_samples)
        """

        n_samples = sq_pdiffs.shape[0]
        # Smoothness
        step = self.weights_ @ sq_pdiffs.T

        if self.window_size > 1:
            # average non-overlapping windows
            step = step.reshape(self.n_atoms, -1, self.window_size).mean(2)

        # L1 regularization
        step += self.l1_c * n_samples

        if self.log_c > 0:
            # step -= self.log_c / coefficients.sum(axis=0, keepdims=True)
            coefficients[:, coefficients.sum(0) < 1e-8] = self.log_c

        if self.l1_diff_c > 0:
            step -= (
                self.l1_diff_c
                * n_samples
                * (np.diff(np.sign(np.diff(coefficients, axis=1)), axis=1, prepend=0, append=0))
            )

        dual_step = self._op_adj_coefficients(self.weights_, dual)

        # Proximal and gradient step
        # NOTE: the step might be divided by the operator norm
        coefficients = coefficients - self.step_c / n_samples * (dual_step + step)

        # Projection
        coefficients[coefficients < 0] = 0
        coefficients[coefficients > 1] = 1

        if np.allclose(coefficients, 0):
            warn("All coefficients dropped to 0", UserWarning)
            coefficients.fill(self.log_c)

        return coefficients

    # UPDATE WEIGHTS ###############################################################################

    @abstractmethod
    def _op_adj_weights(self, coefficients: NDArray, dualv: NDArray) -> NDArray:
        """Compute the adjoint of the bilinear inst-degree operator wrt weights

        Args:
            coefficients (NDArray): Array of coefficients of shape (n_atoms, n_samples)
            dualv (NDArray): Dual variable of shape (n_samples, n_nodes, n_nodes)

        Returns:
            NDArray: Dual weights of shape (n_atoms, n_edges)
        """
        raise NotImplementedError()

    def _update_weights(
        self,
        sq_pdiffs: NDArray[np.float64],
        weights: NDArray[np.float64],
        dual: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Update the weights of the model

        Args:
            sq_pdiffs (NDArray[np.float64]): Squared pairwise differences of
                shape (n_samples, n_edges)
            weights (NDArray[np.float64]): Current weights of shape (n_atoms, n_edges)
            dual (NDArray[np.float64]): Dual variable (instantaneous Laplacians)
                of shape (n_samples, n_nodes, n_nodes)

        Returns:
            NDArray[np.float64]: updated weights of shape (n_atoms, (n_nodes**2 - n_nodes) // 2)
        """
        n_samples = sq_pdiffs.shape[0]

        # Smoothness
        step = self.coefficients_ @ sq_pdiffs
        step += self.l1_w * n_samples

        if self.ortho_w > 0:
            step += self.ortho_w * n_samples * (weights.sum(0, keepdims=True) - weights)

        dual_step = self._op_adj_weights(self.coefficients_, dual)

        # Proximal update
        weights = weights - self.step_w / n_samples * (dual_step + step)

        # Projection
        weights[weights < 0] = 0
        return weights

    # UPDATE DUAL ##################################################################################

    @abstractmethod
    def _update_dual(
        self,
        weights: NDArray[np.float64],
        coefficients: NDArray[np.float64],
        dual: NDArray[np.float64],
    ):
        """Update the dual variable

        Args:
            weights (NDArray[np.float64]): Weights of shape (n_atoms, n_edges)
            coefficients (NDArray[np.float64]): Coefficients of shape (n_atoms, n_samples)
            dual (NDArray[np.float64]): Dual variable of shape (n_samples, n_nodes, n_nodes)

        Returns:
            NDArray[np.float64]: Updated dual variable of shape (n_samples, n_nodes, n_nodes)
        """

        raise NotImplementedError

    def _fit_step(self, sq_pdiffs: NDArray[np.float64]) -> tuple[float, float]:
        """Single step of PDS optimization

        Args:
            sq_pdiffs (NDArray[np.float64]): Squared pairwise differences of
                shape (n_samples, n_edges)

        Returns:
            tuple[float, float]: Relative change in coefficients and weights
        """
        # primal update
        # x1 = prox_gx(x - step * (op_adjx(x, dualv) + gradf_x(x, y, gz)), step)
        # y1 = prox_gy(y - step * (op_adjy(y, dualv) + gradf_y(x, y, gz)), step)

        coefficients = np.repeat(
            self._update_coefficients(
                sq_pdiffs, self.coefficients_[:, :: self.window_size], dual=self.dual_
            ),
            repeats=self.window_size,
            axis=1,
        )
        weights = self._update_weights(
            sq_pdiffs, self.weights_, dual=np.repeat(self.dual_, self.window_size, 0)
        )

        # dual update
        # x_overshoot = 2 * coefficients - self.coefficients_
        self.dual_ = self._update_dual(
            weights=2 * weights - self.weights_,
            coefficients=2 * coefficients[:, :: self.window_size]
            - self.coefficients_[:, :: self.window_size],
            dual=self.dual_,
        )

        a_rel_change = relative_error(self.coefficients_.ravel(), coefficients.ravel())
        w_rel_change = relative_error(self.weights_.ravel(), weights.ravel())

        self.coefficients_ = coefficients
        self.weights_ = weights

        return a_rel_change, w_rel_change

    # FITTING FUNCTIONS ############################################################################

    def _single_fit(
        self,
        x: NDArray[np.float64],
        _y=None,
        *,
        callback: Callable[[GraphDictBase, int]] = None,
    ) -> GraphDictBase:
        """Fit the model to the data with a single initialization

        Args:
            x (NDArray[np.float64]): Design matrix of shape (n_samples, n_nodes)
            y (None, optional): Ignored. Defaults to None.
            callback (Callable[[GraphDictBase, int]], optional): Callback function
                called at each iteration. Defaults to None.
        """
        self._initialize(x)

        sq_pdiffs = self._squared_pdiffs(x)  # shape: (n_samples, n_edges)

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
                    self.step_c *= 0.7
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
        self, x: NDArray[np.float64], _y=None, *, callback: Callable[[GraphDictBase, int]] = None
    ):
        """Fit the model to the data, possibly on multiple initializations

        Args:
            x (NDArray[np.float64]): Design matrix of shape (n_samples, n_nodes)
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
                "coefficients": None,
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
                    best["coefficients"] = self.coefficients_
                    best["dual"] = self.dual_
                    best["converged"] = self.converged_
                    best["fit_time"] = self.fit_time_
                    best["history"] = self.history_

            self.weights_ = best["weights"]
            self.coefficients_ = best["coefficients"]
            self.dual_ = best["dual"]
            self.converged_ = best["converged"]
            self.fit_time_ = best["fit_time"]
            self.history_ = best["history"]

        self.coefficients_ = self.coefficients_[:, :n_samples]
        return self

    # PREDICT ######################################################################################

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict coefficients for a given signal"""
        n_samples, n_nodes = x.shape
        if n_nodes != self.n_nodes_:
            raise ValueError(f"Number of nodes mismatch, got {n_nodes} instead of {self.n_nodes_}")

        coefficients = self._init_coefficients(x)
        dual = self._init_dual(x)

        # op_act_norm = op_coefficients_norm(laplacian_squareform_vec(self.weights_))

        sq_pdiffs = self._squared_pdiffs(x)

        for _ in range(self.max_iter):
            coefficients_u = np.repeat(
                self._update_coefficients(
                    sq_pdiffs, coefficients[:, :: self.window_size], dual=dual
                ),
                repeats=self.window_size,
                axis=1,
            )

            dual = self._update_dual(
                self.weights_,
                2 * coefficients_u[:, :: self.window_size] - coefficients[:, :: self.window_size],
                dual,
            )

            if np.linalg.norm((coefficients_u - coefficients).ravel()) < self.tol:
                return coefficients_u

            coefficients = coefficients_u

        return coefficients[:, :n_samples]

    # SCORING ######################################################################################

    def score(self, x: NDArray[np.float64], _y=None) -> float:
        """Compute the negative log-likelihood of the model

        Args:
            x (NDArray[np.float64]): Design matrix of shape (n_samples, n_nodes)
            y (None, optional): Ignored. Defaults to None.

        Returns:
            float: Score of the model
        """
        # lw(self.weights_) + 1/self.n_samples * (
        #     la(self.coefficients_) + sum_t(
        #        - np.log(gdet(L_inst)) + x.T @ L_inst @ x
        #    )
        # )
        if np.allclose(self.weights_, 0):
            warn("All weights are 0, log determinant is undefined", UserWarning)
            return np.inf

        # sum of L1 norm, orthogonality and smoothness
        weight_loss = self.l1_w * np.abs(self.weights_).sum() + dictionary_smoothness(  # L1
            self.coefficients_, self.weights_, x
        )

        if self.ortho_w > 0:
            # gradient
            weight_loss += self.ortho_w * (
                np.sum(self.weights_.T @ self.weights_) - np.linalg.norm(self.weights_) ** 2
            )

        coefficient_loss = self.l1_c * np.abs(self.coefficients_).sum()
        if self.log_c > 0:
            coefficient_loss -= self.log_c * np.sum(np.log(self.coefficients_.sum(0)))
        if self.l1_diff_c > 0:
            warn("Check diff loss", UserWarning)
            coefficient_loss -= self.l1_diff_c * np.linalg.norm(
                np.diff(np.sign(np.diff(self.coefficients_, axis=1)), axis=1, prepend=0, append=0)
            )

        # Sum of log determinants
        eigvals = np.linalg.eigvalsh(
            np.einsum("kc,knm->cnm", self.coefficients_, laplacian_squareform_vec(self.weights_))
        )
        # shape: (n_samples, n_combi) x (n_combi,)
        loggdet = np.sum(np.log(np.where(eigvals > 0, eigvals, 1)).sum(-1))

        return weight_loss + (coefficient_loss - loggdet)
