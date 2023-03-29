"""Implementation of Temporal Multiresolution Graph Learning"""
import numpy as np
from numpy.typing import NDArray

from graph_learn.components_newer import GraphComponents


class TMGL(GraphComponents):
    def __init__(self, n_levels=1, *args, **kwargs) -> None:
        super().__init__(n_components=2**n_levels - 1, *args, **kwargs)

        self.n_levels = n_levels

    def _e_step(self, x: NDArray[np.float_]) -> int:
        return 0

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
