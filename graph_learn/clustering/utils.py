"""Utility functions for clustering learning"""

import numpy as np
from numpy.random import RandomState
from numpy.typing import NDArray
from sklearn.cluster import k_means, kmeans_plusplus


def init_centers(
    x: NDArray[np.float_],
    n_clusters: int,
    init_params: str,
    random_state: RandomState,
) -> NDArray[np.int_]:
    n_samples, _n_nodes = x.shape

    match init_params:
        case "random":
            labels = np.zeros(n_samples, dtype=np.int_)
            n_per_class = n_samples // n_clusters
            for i in range(n_clusters):
                labels[i * n_per_class : (i + 1) * n_per_class] = i
            random_state.shuffle(labels)
            return labels
        case "kmeans":
            return k_means(x, n_clusters, init="random", n_init=1, random_state=random_state)[1]
        case "k-means++":
            return kmeans_plusplus(x, n_clusters, random_state=random_state)
        case _:
            raise ValueError(f"Invalid init_params: {init_params}")
