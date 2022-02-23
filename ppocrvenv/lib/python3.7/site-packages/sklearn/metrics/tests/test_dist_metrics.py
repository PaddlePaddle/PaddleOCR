import itertools
import pickle
import copy

import numpy as np
from numpy.testing import assert_array_almost_equal

import pytest

from scipy.spatial.distance import cdist
from sklearn.metrics import DistanceMetric
from sklearn.utils import check_random_state
from sklearn.utils._testing import create_memmap_backed_data
from sklearn.utils.fixes import sp_version, parse_version


def dist_func(x1, x2, p):
    return np.sum((x1 - x2) ** p) ** (1.0 / p)


rng = check_random_state(0)
d = 4
n1 = 20
n2 = 25
X1 = rng.random_sample((n1, d)).astype("float64", copy=False)
X2 = rng.random_sample((n2, d)).astype("float64", copy=False)

[X1_mmap, X2_mmap] = create_memmap_backed_data([X1, X2])

# make boolean arrays: ones and zeros
X1_bool = X1.round(0)
X2_bool = X2.round(0)

[X1_bool_mmap, X2_bool_mmap] = create_memmap_backed_data([X1_bool, X2_bool])


V = rng.random_sample((d, d))
VI = np.dot(V, V.T)

BOOL_METRICS = [
    "matching",
    "jaccard",
    "dice",
    "kulsinski",
    "rogerstanimoto",
    "russellrao",
    "sokalmichener",
    "sokalsneath",
]

METRICS_DEFAULT_PARAMS = [
    ("euclidean", {}),
    ("cityblock", {}),
    ("minkowski", dict(p=(1, 1.5, 2, 3))),
    ("chebyshev", {}),
    ("seuclidean", dict(V=(rng.random_sample(d),))),
    ("mahalanobis", dict(VI=(VI,))),
    ("hamming", {}),
    ("canberra", {}),
    ("braycurtis", {}),
]
if sp_version >= parse_version("1.8.0.dev0"):
    # Starting from scipy 1.8.0.dev0, minkowski now accepts w, the weighting
    # parameter directly and using it is preferred over using wminkowski.
    METRICS_DEFAULT_PARAMS.append(
        ("minkowski", dict(p=(1, 1.5, 3), w=(rng.random_sample(d),))),
    )
else:
    # For previous versions of scipy, this was possible through a dedicated
    # metric (deprecated in 1.6 and removed in 1.8).
    METRICS_DEFAULT_PARAMS.append(
        ("wminkowski", dict(p=(1, 1.5, 3), w=(rng.random_sample(d),))),
    )


def check_cdist(metric, kwargs, X1, X2):
    if metric == "wminkowski":
        # wminkoski is deprecated in SciPy 1.6.0 and removed in 1.8.0
        WarningToExpect = None
        if sp_version >= parse_version("1.6.0"):
            WarningToExpect = DeprecationWarning
        with pytest.warns(WarningToExpect):
            D_scipy_cdist = cdist(X1, X2, metric, **kwargs)
    else:
        D_scipy_cdist = cdist(X1, X2, metric, **kwargs)

    dm = DistanceMetric.get_metric(metric, **kwargs)
    D_sklearn = dm.pairwise(X1, X2)
    assert_array_almost_equal(D_sklearn, D_scipy_cdist)


@pytest.mark.parametrize("metric_param_grid", METRICS_DEFAULT_PARAMS)
@pytest.mark.parametrize("X1, X2", [(X1, X2), (X1_mmap, X2_mmap)])
def test_cdist(metric_param_grid, X1, X2):
    metric, param_grid = metric_param_grid
    keys = param_grid.keys()
    for vals in itertools.product(*param_grid.values()):
        kwargs = dict(zip(keys, vals))
        if metric == "mahalanobis":
            # See: https://github.com/scipy/scipy/issues/13861
            # Possibly caused by: https://github.com/joblib/joblib/issues/563
            pytest.xfail(
                "scipy#13861: cdist with 'mahalanobis' fails on joblib memmap data"
            )
        check_cdist(metric, kwargs, X1, X2)


@pytest.mark.parametrize("metric", BOOL_METRICS)
@pytest.mark.parametrize(
    "X1_bool, X2_bool", [(X1_bool, X2_bool), (X1_bool_mmap, X2_bool_mmap)]
)
def test_cdist_bool_metric(metric, X1_bool, X2_bool):
    D_true = cdist(X1_bool, X2_bool, metric)
    check_cdist_bool(metric, D_true)


def check_cdist_bool(metric, D_true):
    dm = DistanceMetric.get_metric(metric)
    D12 = dm.pairwise(X1_bool, X2_bool)
    assert_array_almost_equal(D12, D_true)


@pytest.mark.parametrize("metric_param_grid", METRICS_DEFAULT_PARAMS)
@pytest.mark.parametrize("X1, X2", [(X1, X2), (X1_mmap, X2_mmap)])
def test_pdist(metric_param_grid, X1, X2):
    metric, param_grid = metric_param_grid
    keys = param_grid.keys()
    for vals in itertools.product(*param_grid.values()):
        kwargs = dict(zip(keys, vals))
        if metric == "mahalanobis":
            # See: https://github.com/scipy/scipy/issues/13861
            pytest.xfail("scipy#13861: pdist with 'mahalanobis' fails onmemmap data")
        elif metric == "wminkowski":
            if sp_version >= parse_version("1.8.0"):
                pytest.skip("wminkowski will be removed in SciPy 1.8.0")

            # wminkoski is deprecated in SciPy 1.6.0 and removed in 1.8.0
            ExceptionToAssert = None
            if sp_version >= parse_version("1.6.0"):
                ExceptionToAssert = DeprecationWarning
            with pytest.warns(ExceptionToAssert):
                D_true = cdist(X1, X1, metric, **kwargs)
        else:
            D_true = cdist(X1, X1, metric, **kwargs)

        check_pdist(metric, kwargs, D_true)


@pytest.mark.parametrize("metric", BOOL_METRICS)
@pytest.mark.parametrize("X1_bool", [X1_bool, X1_bool_mmap])
def test_pdist_bool_metrics(metric, X1_bool):
    D_true = cdist(X1_bool, X1_bool, metric)
    check_pdist_bool(metric, D_true)


def check_pdist(metric, kwargs, D_true):
    dm = DistanceMetric.get_metric(metric, **kwargs)
    D12 = dm.pairwise(X1)
    assert_array_almost_equal(D12, D_true)


def check_pdist_bool(metric, D_true):
    dm = DistanceMetric.get_metric(metric)
    D12 = dm.pairwise(X1_bool)
    # Based on https://github.com/scipy/scipy/pull/7373
    # When comparing two all-zero vectors, scipy>=1.2.0 jaccard metric
    # was changed to return 0, instead of nan.
    if metric == "jaccard" and sp_version < parse_version("1.2.0"):
        D_true[np.isnan(D_true)] = 0
    assert_array_almost_equal(D12, D_true)


@pytest.mark.parametrize("writable_kwargs", [True, False])
@pytest.mark.parametrize("metric_param_grid", METRICS_DEFAULT_PARAMS)
def test_pickle(writable_kwargs, metric_param_grid):
    metric, param_grid = metric_param_grid
    keys = param_grid.keys()
    for vals in itertools.product(*param_grid.values()):
        if any(isinstance(val, np.ndarray) for val in vals):
            vals = copy.deepcopy(vals)
            for val in vals:
                if isinstance(val, np.ndarray):
                    val.setflags(write=writable_kwargs)
        kwargs = dict(zip(keys, vals))
        check_pickle(metric, kwargs)


@pytest.mark.parametrize("metric", BOOL_METRICS)
@pytest.mark.parametrize("X1_bool", [X1_bool, X1_bool_mmap])
def test_pickle_bool_metrics(metric, X1_bool):
    dm = DistanceMetric.get_metric(metric)
    D1 = dm.pairwise(X1_bool)
    dm2 = pickle.loads(pickle.dumps(dm))
    D2 = dm2.pairwise(X1_bool)
    assert_array_almost_equal(D1, D2)


def check_pickle(metric, kwargs):
    dm = DistanceMetric.get_metric(metric, **kwargs)
    D1 = dm.pairwise(X1)
    dm2 = pickle.loads(pickle.dumps(dm))
    D2 = dm2.pairwise(X1)
    assert_array_almost_equal(D1, D2)


def test_haversine_metric():
    def haversine_slow(x1, x2):
        return 2 * np.arcsin(
            np.sqrt(
                np.sin(0.5 * (x1[0] - x2[0])) ** 2
                + np.cos(x1[0]) * np.cos(x2[0]) * np.sin(0.5 * (x1[1] - x2[1])) ** 2
            )
        )

    X = np.random.random((10, 2))

    haversine = DistanceMetric.get_metric("haversine")

    D1 = haversine.pairwise(X)
    D2 = np.zeros_like(D1)
    for i, x1 in enumerate(X):
        for j, x2 in enumerate(X):
            D2[i, j] = haversine_slow(x1, x2)

    assert_array_almost_equal(D1, D2)
    assert_array_almost_equal(haversine.dist_to_rdist(D1), np.sin(0.5 * D2) ** 2)


def test_pyfunc_metric():
    X = np.random.random((10, 3))

    euclidean = DistanceMetric.get_metric("euclidean")
    pyfunc = DistanceMetric.get_metric("pyfunc", func=dist_func, p=2)

    # Check if both callable metric and predefined metric initialized
    # DistanceMetric object is picklable
    euclidean_pkl = pickle.loads(pickle.dumps(euclidean))
    pyfunc_pkl = pickle.loads(pickle.dumps(pyfunc))

    D1 = euclidean.pairwise(X)
    D2 = pyfunc.pairwise(X)

    D1_pkl = euclidean_pkl.pairwise(X)
    D2_pkl = pyfunc_pkl.pairwise(X)

    assert_array_almost_equal(D1, D2)
    assert_array_almost_equal(D1_pkl, D2_pkl)


def test_input_data_size():
    # Regression test for #6288
    # Previously, a metric requiring a particular input dimension would fail
    def custom_metric(x, y):
        assert x.shape[0] == 3
        return np.sum((x - y) ** 2)

    rng = check_random_state(0)
    X = rng.rand(10, 3)

    pyfunc = DistanceMetric.get_metric("pyfunc", func=custom_metric)
    eucl = DistanceMetric.get_metric("euclidean")
    assert_array_almost_equal(pyfunc.pairwise(X), eucl.pairwise(X) ** 2)


def test_readonly_kwargs():
    # Non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/21685

    rng = check_random_state(0)

    weights = rng.rand(100)
    VI = rng.rand(10, 10)
    weights.setflags(write=False)
    VI.setflags(write=False)

    # Those distances metrics have to support readonly buffers.
    DistanceMetric.get_metric("seuclidean", V=weights)
    DistanceMetric.get_metric("wminkowski", p=1, w=weights)
    DistanceMetric.get_metric("mahalanobis", VI=VI)
