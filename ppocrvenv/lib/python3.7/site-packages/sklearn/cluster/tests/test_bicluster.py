"""Testing for Spectral Biclustering methods"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix, issparse

from sklearn.model_selection import ParameterGrid

from sklearn.utils._testing import assert_almost_equal
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_array_almost_equal

from sklearn.base import BaseEstimator, BiclusterMixin

from sklearn.cluster import SpectralCoclustering
from sklearn.cluster import SpectralBiclustering
from sklearn.cluster._bicluster import _scale_normalize
from sklearn.cluster._bicluster import _bistochastic_normalize
from sklearn.cluster._bicluster import _log_normalize

from sklearn.metrics import consensus_score, v_measure_score

from sklearn.datasets import make_biclusters, make_checkerboard


class MockBiclustering(BiclusterMixin, BaseEstimator):
    # Mock object for testing get_submatrix.
    def __init__(self):
        pass

    def get_indices(self, i):
        # Overridden to reproduce old get_submatrix test.
        return (
            np.where([True, True, False, False, True])[0],
            np.where([False, False, True, True])[0],
        )


def test_get_submatrix():
    data = np.arange(20).reshape(5, 4)
    model = MockBiclustering()

    for X in (data, csr_matrix(data), data.tolist()):
        submatrix = model.get_submatrix(0, X)
        if issparse(submatrix):
            submatrix = submatrix.toarray()
        assert_array_equal(submatrix, [[2, 3], [6, 7], [18, 19]])
        submatrix[:] = -1
        if issparse(X):
            X = X.toarray()
        assert np.all(X != -1)


def _test_shape_indices(model):
    # Test get_shape and get_indices on fitted model.
    for i in range(model.n_clusters):
        m, n = model.get_shape(i)
        i_ind, j_ind = model.get_indices(i)
        assert len(i_ind) == m
        assert len(j_ind) == n


def test_spectral_coclustering():
    # Test Dhillon's Spectral CoClustering on a simple problem.
    param_grid = {
        "svd_method": ["randomized", "arpack"],
        "n_svd_vecs": [None, 20],
        "mini_batch": [False, True],
        "init": ["k-means++"],
        "n_init": [10],
    }
    random_state = 0
    S, rows, cols = make_biclusters((30, 30), 3, noise=0.5, random_state=random_state)
    S -= S.min()  # needs to be nonnegative before making it sparse
    S = np.where(S < 1, 0, S)  # threshold some values
    for mat in (S, csr_matrix(S)):
        for kwargs in ParameterGrid(param_grid):
            model = SpectralCoclustering(
                n_clusters=3, random_state=random_state, **kwargs
            )
            model.fit(mat)

            assert model.rows_.shape == (3, 30)
            assert_array_equal(model.rows_.sum(axis=0), np.ones(30))
            assert_array_equal(model.columns_.sum(axis=0), np.ones(30))
            assert consensus_score(model.biclusters_, (rows, cols)) == 1

            _test_shape_indices(model)


def test_spectral_biclustering():
    # Test Kluger methods on a checkerboard dataset.
    S, rows, cols = make_checkerboard((30, 30), 3, noise=0.5, random_state=0)

    non_default_params = {
        "method": ["scale", "log"],
        "svd_method": ["arpack"],
        "n_svd_vecs": [20],
        "mini_batch": [True],
    }

    for mat in (S, csr_matrix(S)):
        for param_name, param_values in non_default_params.items():
            for param_value in param_values:

                model = SpectralBiclustering(
                    n_clusters=3,
                    n_init=3,
                    init="k-means++",
                    random_state=0,
                )
                model.set_params(**dict([(param_name, param_value)]))

                if issparse(mat) and model.get_params().get("method") == "log":
                    # cannot take log of sparse matrix
                    with pytest.raises(ValueError):
                        model.fit(mat)
                    continue
                else:
                    model.fit(mat)

                assert model.rows_.shape == (9, 30)
                assert model.columns_.shape == (9, 30)
                assert_array_equal(model.rows_.sum(axis=0), np.repeat(3, 30))
                assert_array_equal(model.columns_.sum(axis=0), np.repeat(3, 30))
                assert consensus_score(model.biclusters_, (rows, cols)) == 1

                _test_shape_indices(model)


def _do_scale_test(scaled):
    """Check that rows sum to one constant, and columns to another."""
    row_sum = scaled.sum(axis=1)
    col_sum = scaled.sum(axis=0)
    if issparse(scaled):
        row_sum = np.asarray(row_sum).squeeze()
        col_sum = np.asarray(col_sum).squeeze()
    assert_array_almost_equal(row_sum, np.tile(row_sum.mean(), 100), decimal=1)
    assert_array_almost_equal(col_sum, np.tile(col_sum.mean(), 100), decimal=1)


def _do_bistochastic_test(scaled):
    """Check that rows and columns sum to the same constant."""
    _do_scale_test(scaled)
    assert_almost_equal(scaled.sum(axis=0).mean(), scaled.sum(axis=1).mean(), decimal=1)


def test_scale_normalize():
    generator = np.random.RandomState(0)
    X = generator.rand(100, 100)
    for mat in (X, csr_matrix(X)):
        scaled, _, _ = _scale_normalize(mat)
        _do_scale_test(scaled)
        if issparse(mat):
            assert issparse(scaled)


def test_bistochastic_normalize():
    generator = np.random.RandomState(0)
    X = generator.rand(100, 100)
    for mat in (X, csr_matrix(X)):
        scaled = _bistochastic_normalize(mat)
        _do_bistochastic_test(scaled)
        if issparse(mat):
            assert issparse(scaled)


def test_log_normalize():
    # adding any constant to a log-scaled matrix should make it
    # bistochastic
    generator = np.random.RandomState(0)
    mat = generator.rand(100, 100)
    scaled = _log_normalize(mat) + 1
    _do_bistochastic_test(scaled)


def test_fit_best_piecewise():
    model = SpectralBiclustering(random_state=0)
    vectors = np.array([[0, 0, 0, 1, 1, 1], [2, 2, 2, 3, 3, 3], [0, 1, 2, 3, 4, 5]])
    best = model._fit_best_piecewise(vectors, n_best=2, n_clusters=2)
    assert_array_equal(best, vectors[:2])


def test_project_and_cluster():
    model = SpectralBiclustering(random_state=0)
    data = np.array([[1, 1, 1], [1, 1, 1], [3, 6, 3], [3, 6, 3]])
    vectors = np.array([[1, 0], [0, 1], [0, 0]])
    for mat in (data, csr_matrix(data)):
        labels = model._project_and_cluster(mat, vectors, n_clusters=2)
        assert_almost_equal(v_measure_score(labels, [0, 0, 1, 1]), 1.0)


def test_perfect_checkerboard():
    # XXX Previously failed on build bot (not reproducible)
    model = SpectralBiclustering(3, svd_method="arpack", random_state=0)

    S, rows, cols = make_checkerboard((30, 30), 3, noise=0, random_state=0)
    model.fit(S)
    assert consensus_score(model.biclusters_, (rows, cols)) == 1

    S, rows, cols = make_checkerboard((40, 30), 3, noise=0, random_state=0)
    model.fit(S)
    assert consensus_score(model.biclusters_, (rows, cols)) == 1

    S, rows, cols = make_checkerboard((30, 40), 3, noise=0, random_state=0)
    model.fit(S)
    assert consensus_score(model.biclusters_, (rows, cols)) == 1


@pytest.mark.parametrize(
    "args",
    [
        {"n_clusters": (3, 3, 3)},
        {"n_clusters": "abc"},
        {"n_clusters": (3, "abc")},
        {"method": "unknown"},
        {"n_components": 0},
        {"n_best": 0},
        {"svd_method": "unknown"},
        {"n_components": 3, "n_best": 4},
    ],
)
def test_errors(args):
    data = np.arange(25).reshape((5, 5))

    model = SpectralBiclustering(**args)
    with pytest.raises(ValueError):
        model.fit(data)


def test_wrong_shape():
    model = SpectralBiclustering()
    data = np.arange(27).reshape((3, 3, 3))
    with pytest.raises(ValueError):
        model.fit(data)


@pytest.mark.parametrize("est", (SpectralBiclustering(), SpectralCoclustering()))
def test_n_features_in_(est):

    X, _, _ = make_biclusters((3, 3), 3, random_state=0)

    assert not hasattr(est, "n_features_in_")
    est.fit(X)
    assert est.n_features_in_ == 3
