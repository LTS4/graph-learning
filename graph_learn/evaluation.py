"""Functions for evaluation and scoring of graph learning methods"""
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import squareform
from sklearn.metrics import f1_score, precision_score, recall_score


def relative_error(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    r"""Relative error between matrices or vectors
    .. :math:
        \operatorname{RE}(\hat{\mathbf y}, \bm y^*)
            = {\norm{\hat{\mathbf y} - \bm y^*}_F} / \norm{\bm y^*}_F

    Args:
        y_true (NDArray[np.float64]): Target matrix/vector
        y_pred (NDArray[np.float64]): Predicted matrix/vector

    Returns:
        float: Error
    """
    err_norm = np.linalg.norm(y_pred - y_true)
    if np.allclose(y_true, 0):
        return err_norm

    return err_norm / np.linalg.norm(y_true)


def edge_f1_score(y_true, y_pred) -> float:
    """F1 score on edge discovery. (Does not consider edge weights)"""
    return f1_score(
        squareform(y_true, checks=False) > 0, squareform(y_pred, checks=False) > 0
    ).item()


def edge_precision(y_true, y_pred) -> float:
    """Precision score on edge discovery. (Does not consider edge weights)"""
    return precision_score(
        squareform(y_true, checks=False) > 0, squareform(y_pred, checks=False) > 0
    ).item()


def edge_recall(y_true, y_pred) -> float:
    """Precision score on edge discovery. (Does not consider edge weights)"""
    return recall_score(
        squareform(y_true, checks=False) > 0, squareform(y_pred, checks=False) > 0
    ).item()
