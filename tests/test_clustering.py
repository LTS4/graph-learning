"""Tests for clustering methods"""

# pylint: disable=redefined-outer-name

import numpy as np
import pytest
from numpy.random import default_rng
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.mixture import GaussianMixture

from graph_learn.clustering import GLMM, KGraphs, KGraphsV2
from graph_learn.sampling.graphs import sample_er_laplacian
from graph_learn.sampling.signals import sample_lgmrf

N_NODES = 25
N_SAMPLES = 100
N_CLUSTERS = 3
EDGE_P = 0.2
MEANS_VAR = 0

MAX_ITER = 1000


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

    means = MEANS_VAR * rng.standard_normal(size=(y_true.max() + 1, N_NODES))

    for i, lapl in enumerate(laplacians):
        mask = y_true == i
        x[mask] = means[i, np.newaxis, :] + sample_lgmrf(lapl, n_samples=mask.sum(), seed=rng)

    return x


@pytest.fixture
def min_score(data, y_true, rng):
    """Compute baseline (Gaussian Mixture) score."""
    model = GaussianMixture(
        n_components=N_CLUSTERS,
        n_init=5,
        max_iter=MAX_ITER,
        random_state=rng.integers(10**6),
    )

    y_pred = model.fit_predict(data)

    # This is 0.07 if MEANS_VAR == 0
    return adjusted_mutual_info_score(y_true, y_pred)


def test_glmm(data, y_true, min_score, rng):
    """Test GLMM clustering"""
    model = GLMM(
        n_components=N_CLUSTERS,
        avg_degree=int(EDGE_P * N_NODES) + 1,
        n_init=5,
        max_iter=MAX_ITER,
        random_state=rng.integers(10**6),
    )

    y_pred = model.fit_predict(data)

    # This is 0.61 if MEANS_VAR == 0
    assert adjusted_mutual_info_score(y_true, y_pred) >= min_score


def test_glmm_blocks(data, y_true, min_score, rng):
    """Test GLMM clustering"""
    blocks = np.array(10 * [0] + 8 * [1] + 7 * [2])
    avg_degrees = {(0, 0): 6, (0, 1): 3, (0, 2): 3, (1, 1): 4, (1, 2): 2, (2, 2): 3}

    model = GLMM(
        n_components=N_CLUSTERS,
        avg_degree=avg_degrees,
        n_init=5,
        max_iter=MAX_ITER,
        random_state=rng.integers(10**6),
        blocks=blocks,
    )

    y_pred = model.fit_predict(data)

    # This is 0.61 if MEANS_VAR == 0
    assert adjusted_mutual_info_score(y_true, y_pred) >= min_score


def test_kgraphs(data, y_true, min_score, rng):
    """Test KGraphs clustering"""
    model = KGraphs(
        n_clusters=N_CLUSTERS,
        avg_degree=int(EDGE_P * N_NODES) + 1,
        n_init=5,
        max_iter=MAX_ITER,
        random_state=rng.integers(10**6),
    )

    y_pred = model.fit_predict(data)

    # This is 0.11 if MEANS_VAR == 0
    assert adjusted_mutual_info_score(y_true, y_pred) >= min_score


def test_kgraphsv2(data, y_true, min_score, rng):
    """Test KGraphsV2 clustering"""
    model = KGraphsV2(
        n_clusters=N_CLUSTERS,
        avg_degree=int(EDGE_P * N_NODES) + 1,
        n_init=5,
        max_iter=MAX_ITER,
        random_state=rng.integers(10**6),
    )

    y_pred = model.fit_predict(data)

    # This is 0.38 if MEANS_VAR == 0
    assert adjusted_mutual_info_score(y_true, y_pred) >= min_score


# array([[-0.14153019, -0.22397099,  0.22772982,  0.06249811, -0.06384377,
#          0.05714588,  0.51780199,  0.39499243,  0.02803168, -0.18984876,
#         -0.48554256, -0.10164939, -0.08181426],
#        [ 0.17566147,  0.05249482, -0.28870163,  0.14559845, -0.01851911,
#          0.15573786, -0.87927826, -0.23088974,  0.09902922,  0.10506065,
#          0.26892766,  0.10224784,  0.31263077]])
