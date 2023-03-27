"""Implementation of Temporal Multiresolution Graph Learning"""
import numpy as np
from numpy.random import RandomState
from numpy.typing import NDArray

from graph_learn.components_newer import GraphComponents


class TMGL(GraphComponents):
    def __init__(
        self,
        n_levels=1,
        alpha: float = 0.5,
        *,
        max_iter: int = 50,
        tol: float = 0.01,
        max_iter_pds: int = 100,
        tol_pds: float = 0.001,
        pds_relaxation: float = None,
        random_state: RandomState = None,
        init_strategy: str = "uniform",
        weight_prior=None,
        discretize: bool = False,
        verbose: int = 0,
    ) -> None:
        super().__init__(
            n_components=2**n_levels - 1,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            max_iter_pds=max_iter_pds,
            tol_pds=tol_pds,
            pds_relaxation=pds_relaxation,
            random_state=random_state,
            init_strategy=init_strategy,
            weight_prior=weight_prior,
            discretize=discretize,
            verbose=verbose,
        )

        self.n_levels = n_levels

    def _e_step(self, x: NDArray[np.float_], laplacians: NDArray[np.float_]):
        return

    def _initialize(self, x: NDArray):
        super()._initialize(x)

        self.activations_.fill(0)
        n_samples = x.shape[0]
        n_windows = 2 ** (self.n_levels - 1)
        win_size = n_samples // n_windows
        self.activations_[0, :] = 1
        for k in range(n_windows):
            index = n_windows - 1 + k
            while index > 0:
                self.activations_[index, k * win_size : (k + 1) * win_size] = 1
                index = (index - 1) // 2
