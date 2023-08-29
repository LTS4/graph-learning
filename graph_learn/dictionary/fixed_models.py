"""Partially fixed component models"""
import numpy as np
from numpy.linalg import eigh
from numpy.typing import NDArray

from graph_learn.operators import laplacian_squareform_vec, op_weights_norm

from .base_model import GraphDictionary


class FixedWeights(GraphDictionary):
    """Subclass of :class:`GraphDictionary` with fixed weights.

    Only Activations are optimized.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._eigvecs: NDArray
        self._eigvals: NDArray
        self._combi_lapl: NDArray

    def _initialize(self, x: NDArray):
        super()._initialize(x)

        if isinstance(self.weight_prior, np.ndarray):
            if self.weights_.shape != self.weight_prior.shape:
                raise ValueError(
                    f"Invalid weight prior shape, expected {self.activations_.shape},"
                    f" got {self.weight_prior.shape}"
                )

            self.weights_ = self.weight_prior

        elif isinstance(self.activation_prior, (float, int)):
            self.weights_ = self.weight_prior * np.ones_like(self.weights_)

        else:
            raise TypeError(
                "Weight prior must be a real number or a numpy array,"
                f"got {type(self.activation_prior)}"
            )

        self._combi_lapl = np.einsum(
            "kc,knm->cnm", self._combinations, laplacian_squareform_vec(self.weights_)
        )
        self._eigvals, self._eigvecs = eigh(self._combi_lapl)

    def _update_weights(self, *_args, **_kwargs) -> NDArray:
        return self.weights_

    def _update_dual(self, weights: NDArray, activations: NDArray, dual: NDArray):
        op_norm = op_weights_norm(
            activations=activations, n_nodes=self.n_nodes_
        )  # * op_activations_norm(lapl=laplacian_squareform_vec(weights))

        # return prox_gdet_star(dual, sigma=self.step_dual / op_norm / self.n_samples_)
        sigma = self.step_dual / op_norm / self.n_samples_

        _shape = dual.shape
        eigvals = np.nanmedian(dual @ self._eigvecs / self._eigvecs)
        eigvals += self.step_dual / op_norm * self._eigvals

        # Input shall be SPD, so negative values come from numerical erros
        eigvals[eigvals < 0] = 0

        # Proximal step
        # eigvals = (eigvals - np.sqrt(eigvals**2 + 4 * sigma)) / 2
        eigvals -= np.sqrt(eigvals**2 + 4 * sigma)
        eigvals /= 2
        dual = np.stack(
            [eigvec @ np.diag(eigval) @ eigvec.T for eigval, eigvec in zip(eigvals, self._eigvecs)]
        )
        assert dual.shape == _shape

        # Remove constant eignevector. Initial eigval was 0, with prox step is  -np.sqrt(sigma)
        # Note that the norm of the eigenvector is sqrt(n_nodes)
        return dual + np.sqrt(sigma) / _shape[1]

    def fit(self, *_args, **_kwargs) -> "FixedWeights":
        return super().fit(*_args, **_kwargs)


class FixedActivations(GraphDictionary):
    """Subclass of :class:`GraphDictionary` with fixed activations.

    Only weights are optimized.
    """

    def _initialize(self, x: NDArray):
        super()._initialize(x)

        if isinstance(self.activation_prior, np.ndarray):
            if self.activations_.shape != self.activation_prior.shape:
                raise ValueError(
                    f"Invalid activation prior shape, expected {self.activations_.shape},"
                    f" got {self.activation_prior.shape}"
                )
            self.activations_ = self.activation_prior

        elif isinstance(self.activation_prior, (float, int)):
            self.activations_ = self.activation_prior * np.ones_like(self.activations_)

        else:
            raise TypeError(
                f"Activation prior must be real number or a numpy array, got {type(self.activation_prior)}"
            )

    def _update_activations(self, *_args, **_kwargs) -> NDArray:
        return self.activations_
