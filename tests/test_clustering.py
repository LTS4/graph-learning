"""Tests for clustering methods"""

# pylint: disable=redefined-outer-name

import numpy as np
import pytest
from numpy.random import default_rng
from sklearn.metrics import adjusted_mutual_info_score

from graph_learn.clustering import GLMM, KGraphs, KGraphsV2
from graph_learn.sampling.graphs import sample_er_laplacian
from graph_learn.sampling.signals import sample_lgmrf

N_NODES = 13
N_SAMPLES = 100
N_CLUSTERS = 2
EDGE_P = 0.2

MIN_SCORE = 0


@pytest.fixture
def rng():
    return default_rng(2410241319)


@pytest.fixture
def y_true(rng):
    return rng.choice(N_CLUSTERS, size=N_SAMPLES, replace=True)


@pytest.fixture
def data(y_true, rng):
    "Generate data"
    laplacians = sample_er_laplacian(
        N_NODES, edge_p=EDGE_P, edge_w_min=0.3, edge_w_max=2, n_graphs=y_true.max() + 1, seed=rng
    )

    x = np.zeros((N_SAMPLES, N_NODES), dtype=float)

    for i, lapl in enumerate(laplacians):
        mask = y_true == i
        x[mask] = sample_lgmrf(lapl, n_samples=mask.sum(), seed=rng)

    return x


def test_glmm(data, y_true, rng):
    """Test GLMM clustering"""
    model = GLMM(
        n_components=N_CLUSTERS,
        avg_degree=int(EDGE_P * N_NODES) + 1,
        n_init=5,
        random_state=rng.integers(10**6),
    )

    y_pred = model.fit_predict(data)

    # assert np.mean(y_true == y_pred) > MIN_SCORE
    assert adjusted_mutual_info_score(y_true, y_pred) > MIN_SCORE


def test_kgraphs(data, y_true, rng):
    """Test KGraphs clustering"""
    model = KGraphs(
        n_clusters=N_CLUSTERS,
        avg_degree=int(EDGE_P * N_NODES) + 1,
        n_init=5,
        random_state=rng.integers(10**6),
    )

    y_pred = model.fit_predict(data)

    # assert np.mean(y_true == y_pred) > MIN_SCORE
    assert adjusted_mutual_info_score(y_true, y_pred) > MIN_SCORE


def test_kgraphsv2(data, y_true, rng):
    """Test KGraphsV2 clustering"""
    model = KGraphsV2(
        n_clusters=N_CLUSTERS,
        avg_degree=int(EDGE_P * N_NODES) + 1,
        random_state=rng.integers(10**6),
    )

    y_pred = model.fit_predict(data)

    assert adjusted_mutual_info_score(y_true, y_pred) > MIN_SCORE
