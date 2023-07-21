"""Partially fixed component models"""
import numpy as np
from numpy.typing import NDArray

from graph_learn.dictionary.base_model import GraphDictionary


class FixedWeights(GraphDictionary):
    """Subclass of :class:`GraphDictionary` with fixed weights.

    Only Activations are optimized.
    """

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
                f"Weight prior must be a real number or a numpy array, got {type(self.activation_prior)}"
            )

    def _update_weights(self, x: NDArray, mc_activations: NDArray, dual: NDArray) -> NDArray:
        return self.weights_


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

    def _update_activations(self, x: NDArray, mc_activations: NDArray, dual: NDArray) -> NDArray:
        return self.activations_
