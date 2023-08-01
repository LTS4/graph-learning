"""Tests for bilinear operators and adjoints"""
import numpy as np
from numpy.random import default_rng
from scipy.sparse.linalg import svds

from graph_learn.operators import (
    laplacian_squareform,
    laplacian_squareform_adj,
    laplacian_squareform_adj_vec,
    laplacian_squareform_vec,
    op_activations_norm,
    op_adj_activations,
    op_adj_weights,
    op_weights_norm,
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


def test_full_adjoint():
    for seed in range(1000):
        rng = default_rng(seed)
        n_components = rng.integers(1, 10)
        n_nodes = rng.integers(20, 100)
        n_samples = rng.integers(100, 1000)

        activations = rng.standard_normal((n_components, n_samples))
        weights = rng.standard_normal((n_components, (n_nodes * (n_nodes - 1)) // 2))
        dual = rng.standard_normal((n_samples, n_nodes, n_nodes))

        prod1 = np.sum(
            dual * np.einsum("kt,knm->tnm", activations, laplacian_squareform_vec(weights))
        )
        prod2 = 0.5 * (
            np.sum(weights * op_adj_weights(activations, dual))
            + np.sum(activations * op_adj_activations(weights, dual))
        )

        assert np.allclose(prod1, prod2)


################################################################################
# Operator norm tests


def test_operator_norm_weights():
    """Verify that the operator norm, wrt weights is correct"""

    for seed in range(10):
        rng = default_rng(seed)
        n_components = rng.integers(1, 10)
        n_nodes = rng.integers(20, 100)
        n_samples = rng.integers(100, 1000)

        activations = rng.standard_normal((n_components, n_samples))
        weights = rng.standard_normal((n_components, (n_nodes * (n_nodes - 1)) // 2))

        estimate = op_weights_norm(activations=activations, n_nodes=n_nodes) ** 2

        for _ in range(100):
            weights = weights / np.linalg.norm(weights)
            _weights = op_adj_weights(
                activations,
                np.einsum("kt,knm->tnm", activations, laplacian_squareform_vec(weights)),
            )

            alpha = np.mean(_weights / weights)
            weights = _weights

        assert np.isclose(estimate, alpha, rtol=1e-2)


def test_operator_norm_activations():
    """Verify that the operator norm, wrt activations is correct"""

    for seed in range(10):
        rng = default_rng(seed)
        n_components = rng.integers(1, 10)
        n_nodes = rng.integers(20, 100)
        n_samples = rng.integers(100, 1000)

        activations = rng.standard_normal((n_components, n_samples))
        weights = rng.standard_normal((n_components, (n_nodes * (n_nodes - 1)) // 2))

        lapl = laplacian_squareform_vec(weights)
        estimate = op_activations_norm(lapl) ** 2

        for _ in range(200):
            activations = activations / np.linalg.norm(activations)
            _activations = op_adj_activations(
                weights,
                np.einsum("kt,knm->tnm", activations, laplacian_squareform_vec(weights)),
            )

            alpha = np.mean(_activations / activations)
            activations = _activations

        assert np.isclose(estimate, alpha, rtol=1e-2)
