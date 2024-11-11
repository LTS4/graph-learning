"""Utility functions for clustering learning"""

import numpy as np
from numpy.random import RandomState
from numpy.typing import NDArray
from sklearn.cluster import k_means, kmeans_plusplus


def init_labels(
    x: NDArray[np.float64],
    n_clusters: int,
    init_params: str,
    random_state: RandomState,
) -> NDArray[np.int64]:
    """Get initial estimate for cluster assignments

    Args:
        x (NDArray[np.float64]): Input data
        n_clusters (int): Number of clusters
        init_params (str): Keyword for cluster initialization. Options are
            - 'random': returns random stratified assignments
            - 'kmeans': Use k-means to gest first estimate
            - 'k-means++': Use k-means++ to gest first estimate
        random_state (RandomState): Random state

    Raises:
        ValueError: If 'init_params' is not supported

    Returns:
        NDArray[np.int64]: Vector of cluster assignments
    """
    n_samples, _n_nodes = x.shape

    match init_params:
        case "random":
            labels = np.zeros(n_samples, dtype=np.int64)
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


def one_hot(labels: NDArray[np.int_], n_labels: int = None) -> NDArray:
    """One hot encode labels"""
    n_samples = labels.shape[0]
    n_labels = n_labels or labels.max() + 1

    y_one_hot = np.zeros((n_samples, n_labels))
    y_one_hot[np.arange(n_samples), labels] = 1

    return y_one_hot
