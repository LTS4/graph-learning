"""Partially fixed component models"""
import numpy as np
from numpy.typing import NDArray

from . import GraphDictExact
from .utils import powerset_matrix


class FixedWeights(GraphDictExact):
    """Subclass of :class:`GraphDictionary` with fixed weights.

    Only Activations are optimized.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._eigvecs: NDArray
        self._eigvals: NDArray
        self._combi_lapl: NDArray

    def _init_weigths(self) -> NDArray[np.float_]:
        expected_shape = (self.n_atoms, self.n_nodes_ * (self.n_nodes_ - 1) // 2)

        if isinstance(self.weight_prior, np.ndarray):
            if self.weight_prior.shape != expected_shape:
                raise ValueError(
                    f"Invalid weight prior shape, expected {expected_shape},"
                    f" got {self.weight_prior.shape}"
                )

            weights = self.weight_prior

        else:
            raise TypeError(
                "Weight prior must be a a numpy array," f"got {type(self.weight_prior)}"
            )

        return weights

    # def _initialize(self, x: NDArray):
    #     super()._initialize(x)

    #     self._combi_lapl = np.einsum(
    #         "kc,knm->cnm", self._combinations, laplacian_squareform_vec(self.weights_)
    #     )
    #     self._eigvals, self._eigvecs = eigh(self._combi_lapl)

    def _update_weights(self, *_args, **_kwargs) -> NDArray:
        return self.weights_

    # def _update_dual(
    #     self,
    #     weights: NDArray[np.float_],
    #     combi_p: NDArray[np.float_],
    #     dual: NDArray[np.float_],
    #     op_norm=1,
    # ):
    #     # return prox_gdet_star(dual, sigma=self.step_dual / op_norm / self.n_samples_)
    #     sigma = self.step_dual / op_norm  # / self.n_samples_

    #     combi_e = combi_p.sum(1)
    #     active = combi_e > 0
    #     active[0] = 0  # This ignores the empty component

    #     _shape = dual.shape
    #     eigvals = np.nanmean(
    #         np.where(self._eigvecs, dual @ self._eigvecs / self._eigvecs, np.nan), axis=1
    #     )
    #     eigvals += self.step_dual / op_norm * self._eigvals

    #     # Input shall be SPD, so negative values come from numerical erros
    #     eigvals[eigvals < 0] = 0

    #     # Proximal step
    #     # eigvals = (eigvals - np.sqrt(eigvals**2 + 4 * sigma)) / 2
    #     eigvals -= np.sqrt(eigvals**2 + 4 * sigma)
    #     eigvals /= 2
    #     dual = np.stack(
    #         [eigvec @ np.diag(eigval) @ eigvec.T for eigval, eigvec in zip(eigvals, self._eigvecs)]
    #     )
    #     assert dual.shape == _shape

    #     # Remove constant eignevector. Initial eigval was 0, with prox step is  -np.sqrt(sigma)
    #     # Note that the norm of the eigenvector is sqrt(n_nodes)
    #     return dual + np.sqrt(sigma) / _shape[1]

    def fit(self, *_args, **_kwargs) -> "FixedWeights":
        return super().fit(*_args, **_kwargs)


class FixedActivations(GraphDictExact):
    """Subclass of :class:`GraphDictionary` with fixed activations.

    Only weights are optimized.
    """

    def _init_activations(self, n_samples) -> NDArray[np.float_]:
        expected_shape = (self.n_atoms, n_samples)

        if isinstance(self.activation_prior, (np.ndarray, list)):
            self.activation_prior = np.array(self.activation_prior, dtype=float)
            if self.activation_prior.shape != expected_shape:
                raise ValueError(
                    f"Invalid activation prior shape, expected {expected_shape},"
                    f" got {self.activation_prior.shape}"
                )
            activations = self.activation_prior

        elif isinstance(self.activation_prior, (float, int)):
            activations = self.activation_prior * np.ones_like(activations)

        else:
            raise TypeError(
                "Activation prior must be real number or a numpy array"
                f", got {type(self.activation_prior)}"
            )

        return activations

    # def _update_combi_p(self, *_args, **_kwargs) -> NDArray[np.float_]:
    #     return self.combi_p_

    def _update_activations(
        self, sq_pdiffs: NDArray, activations: NDArray, dual: NDArray, op_norm=1
    ) -> NDArray:
        return activations

    def predict(self, x) -> NDArray[np.float_]:
        raise NotImplementedError


def fixw_from_full(model: GraphDictExact) -> FixedWeights:
    """Create a :class:`FixedWeights` model from a :class:`GraphDictionary` model.

    Parameters
    ----------
    model : GraphDictionary
        Model to copy.

    Returns
    -------
    FixedWeights
        Model with fixed weights.
    """
    if not isinstance(model, GraphDictExact):
        raise TypeError(f"Expected GraphDictionary, got {type(model)}")

    return FixedWeights(
        n_atoms=model.n_atoms,
        weight_prior=model.weights_,
        window_size=model.window_size,
        l1_w=model.l1_w,
        ortho_w=model.ortho_w,
        smooth_a=model.smooth_a,
        l1_a=model.l1_a,
        log_a=model.log_a,
        l1_diff_a=model.l1_diff_a,
        max_iter=model.max_iter,
        step_a=model.step_a,
        step_w=model.step_w,
        step_dual=model.step_dual,
        tol=model.tol,
        reduce_step_on_plateau=model.reduce_step_on_plateau,
        random_state=model.random_state,
        init_strategy=model.init_strategy,
        activation_prior=model.activation_prior,
        verbose=model.verbose,
    )


class GraphDictHier(FixedActivations):
    def __init__(self, depth: int, **kwargs) -> None:
        super().__init__(
            n_atoms=2 ** (depth) - 1,
            **kwargs,
        )
        self.depth = depth

    def _initialize(self, x: NDArray) -> None:
        n_samples, _n_nodes = x.shape
        n_windows = 2 ** (self.depth - 1)
        self.window_size = int(np.ceil(n_samples / n_windows))

        # shape (n_atoms, n_samples)
        self.activation_prior = np.zeros((self.n_atoms, n_windows))
        atom = 0
        for d in range(self.depth):
            n_rep = 2**d
            win_width = n_windows // n_rep
            for rep in range(n_rep):
                start = rep * win_width
                self.activation_prior[atom, start : start + win_width] = 1
                atom += 1

        self.activation_prior = np.repeat(self.activation_prior, repeats=self.window_size, axis=1)[
            :, :n_samples
        ]

        super()._initialize(x)
