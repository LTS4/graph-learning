# pylint: disable=redefined-outer-name
from itertools import product

import pytest
from numpy.random import default_rng

from graph_learn.dictionary import GraphDictLog


@pytest.fixture
def seed():
    return 240920


@pytest.fixture
def data(seed):
    return default_rng(seed).standard_normal((23, 7))


@pytest.mark.parametrize("n_atoms, l1_w", list(product([1, 3], [0.0, 0.01])))
def test_graphdictlog_fit_l1_w(n_atoms, l1_w, data, seed):
    GraphDictLog(n_atoms=n_atoms, l1_w=l1_w, random_state=seed, max_iter=100).fit(data)
