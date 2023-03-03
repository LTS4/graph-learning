"""Functions for evaluation and scoring of graph learning methods"""
import numpy as np
from numpy.typing import NDArray


def relative_error(y_true: NDArray[np.float_], y_pred: NDArray[np.float_]) -> float:
    r"""Relative error between matrices or vectors
    .. :math:
        \operatorname{RE}(\hat{\mathbf y}, \bm y^*)
            = {\norm{\hat{\mathbf y} - \bm y^*}_F} / \norm{\bm y^*}_F

    Args:
        y_true (NDArray[np.float_]): Target matrix/vector
        y_pred (NDArray[np.float_]): Predicted matrix/vector

    Returns:
        float: Error
    """
    err_norm = np.linalg.norm(y_pred - y_true)
    if np.allclose(y_true, 0):
        return np.inf

    return err_norm / np.linalg.norm(y_true)
