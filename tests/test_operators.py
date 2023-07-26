"""Tests for bilinear operators and adjoints"""
import numpy as np

from graph_learn.components.utils import laplacian_squareform_vec
from graph_learn.dictionary.base_model import GraphDictionary


def test_graph_dictionary_op_adj_weights():
    """Verify that the operator adjoint of the weights is correct"""
    n_samples = 200
    n_nodes = 33
    n_components = 13

    model = GraphDictionary(n_components=n_components, alpha_a=1)

    for _ in range(1000):
        activations = np.random.rand(n_components, n_samples)
        weights = np.random.rand(n_components, (n_nodes * (n_nodes - 1)) // 2)
        dual = np.random.randn(n_samples, n_nodes, n_nodes)

        assert np.allclose(
            np.sum(weights * model.op_adj_weights(activations, dual)),
            np.sum(dual * np.einsum("kt,knm->tnm", activations, laplacian_squareform_vec(weights))),
        )
