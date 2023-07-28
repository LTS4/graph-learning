"""Tests for bilinear operators and adjoints"""
import numpy as np
from numpy.random import default_rng

from graph_learn.utils import (
    laplacian_squareform,
    laplacian_squareform_adj,
    laplacian_squareform_adj_vec,
    laplacian_squareform_vec,
    op_adj_activations,
    op_adj_weights,
)

################################################################################
# Operator adjoint tests


def test_laplacian_squareform_adj():
    for seed in range(1000):
        rng = default_rng(seed)
        n_nodes = rng.integers(20, 100)

        weights = rng.standard_normal((n_nodes * (n_nodes - 1)) // 2)
        lapl = rng.standard_normal((n_nodes, n_nodes))

        assert np.allclose(
            np.sum(lapl * laplacian_squareform(weights)),
            np.sum(weights * laplacian_squareform_adj(lapl)),
        )


def test_laplacian_squareform_adj_vec():
    for seed in range(1000):
        rng = default_rng(seed)
        n_nodes = rng.integers(20, 100)
        n_components = rng.integers(1, 10)

        weights = rng.standard_normal((n_components, (n_nodes * (n_nodes - 1)) // 2))
        dual = rng.standard_normal((n_components, n_nodes, n_nodes))

        assert np.allclose(
            np.sum(weights * laplacian_squareform_adj_vec(dual)),
            np.sum(dual * laplacian_squareform_vec(weights)),
        )


def test_op_adj_weights():
    """Verify that the operator adjoint of the weights is correct"""
    for seed in range(1000):
        rng = default_rng(seed)
        n_components = rng.integers(1, 10)
        n_nodes = rng.integers(20, 100)
        n_samples = rng.integers(100, 1000)

        activations = rng.standard_normal((n_components, n_samples))
        weights = rng.standard_normal((n_components, (n_nodes * (n_nodes - 1)) // 2))
        dual = rng.standard_normal((n_samples, n_nodes, n_nodes))

        assert np.allclose(
            np.sum(weights * op_adj_weights(activations, dual)),
            np.sum(dual * np.einsum("kt,knm->tnm", activations, laplacian_squareform_vec(weights))),
        )


def test_op_adj_activations():
    """Verify that the operator adjoint of the activations is correct"""
    for seed in range(1000):
        rng = default_rng(seed)
        n_components = rng.integers(1, 10)
        n_nodes = rng.integers(20, 100)
        n_samples = rng.integers(100, 1000)

        activations = rng.standard_normal((n_components, n_samples))
        weights = rng.standard_normal((n_components, (n_nodes * (n_nodes - 1)) // 2))
        dual = rng.standard_normal((n_samples, n_nodes, n_nodes))

        assert np.allclose(
            np.sum(activations * op_adj_activations(weights, dual)),
            np.sum(dual * np.einsum("kt,knm->tnm", activations, laplacian_squareform_vec(weights))),
        )


################################################################################
# Operator norm tests


def test_operator_norm_weights():
    """Verify that the operator norm of the weights is correct"""
    n_samples = 200
    n_nodes = 33
    n_components = 13

    for seed in range(1000):
        rng = default_rng(seed)
        activations = rng.integers(n_components, n_samples)
        weights = rng.integers(n_components, (n_nodes * (n_nodes - 1)) // 2)
        dual = np.einsum("kt,knm->tnm", activations, laplacian_squareform_vec(weights))

        assert np.allclose(
            2
            * (np.linalg.norm(weights.ravel(), ord=1) * np.linalg.norm(activations.ravel(), ord=2))
            ** 2,
            np.linalg.norm(dual.ravel(), ord=2) ** 2,
        )
