import numpy as np

import pytest

from sklearn.utils._testing import assert_allclose
from sklearn.neighbors import KernelDensity, KDTree, NearestNeighbors
from sklearn.neighbors._ball_tree import kernel_norm
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_blobs
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
import joblib


# XXX Duplicated in test_neighbors_tree, test_kde
def compute_kernel_slow(Y, X, kernel, h):
    d = np.sqrt(((Y[:, None, :] - X) ** 2).sum(-1))
    norm = kernel_norm(h, X.shape[1], kernel) / X.shape[0]

    if kernel == "gaussian":
        return norm * np.exp(-0.5 * (d * d) / (h * h)).sum(-1)
    elif kernel == "tophat":
        return norm * (d < h).sum(-1)
    elif kernel == "epanechnikov":
        return norm * ((1.0 - (d * d) / (h * h)) * (d < h)).sum(-1)
    elif kernel == "exponential":
        return norm * (np.exp(-d / h)).sum(-1)
    elif kernel == "linear":
        return norm * ((1 - d / h) * (d < h)).sum(-1)
    elif kernel == "cosine":
        return norm * (np.cos(0.5 * np.pi * d / h) * (d < h)).sum(-1)
    else:
        raise ValueError("kernel not recognized")


def check_results(kernel, bandwidth, atol, rtol, X, Y, dens_true):
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth, atol=atol, rtol=rtol)
    log_dens = kde.fit(X).score_samples(Y)
    assert_allclose(np.exp(log_dens), dens_true, atol=atol, rtol=max(1e-7, rtol))
    assert_allclose(
        np.exp(kde.score(Y)), np.prod(dens_true), atol=atol, rtol=max(1e-7, rtol)
    )


@pytest.mark.parametrize(
    "kernel", ["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"]
)
@pytest.mark.parametrize("bandwidth", [0.01, 0.1, 1])
def test_kernel_density(kernel, bandwidth):
    n_samples, n_features = (100, 3)

    rng = np.random.RandomState(0)
    X = rng.randn(n_samples, n_features)
    Y = rng.randn(n_samples, n_features)

    dens_true = compute_kernel_slow(Y, X, kernel, bandwidth)

    for rtol in [0, 1e-5]:
        for atol in [1e-6, 1e-2]:
            for breadth_first in (True, False):
                check_results(kernel, bandwidth, atol, rtol, X, Y, dens_true)


def test_kernel_density_sampling(n_samples=100, n_features=3):
    rng = np.random.RandomState(0)
    X = rng.randn(n_samples, n_features)

    bandwidth = 0.2

    for kernel in ["gaussian", "tophat"]:
        # draw a tophat sample
        kde = KernelDensity(bandwidth=bandwidth, kernel=kernel).fit(X)
        samp = kde.sample(100)
        assert X.shape == samp.shape

        # check that samples are in the right range
        nbrs = NearestNeighbors(n_neighbors=1).fit(X)
        dist, ind = nbrs.kneighbors(X, return_distance=True)

        if kernel == "tophat":
            assert np.all(dist < bandwidth)
        elif kernel == "gaussian":
            # 5 standard deviations is safe for 100 samples, but there's a
            # very small chance this test could fail.
            assert np.all(dist < 5 * bandwidth)

    # check unsupported kernels
    for kernel in ["epanechnikov", "exponential", "linear", "cosine"]:
        kde = KernelDensity(bandwidth=bandwidth, kernel=kernel).fit(X)
        with pytest.raises(NotImplementedError):
            kde.sample(100)

    # non-regression test: used to return a scalar
    X = rng.randn(4, 1)
    kde = KernelDensity(kernel="gaussian").fit(X)
    assert kde.sample().shape == (1, 1)


@pytest.mark.parametrize("algorithm", ["auto", "ball_tree", "kd_tree"])
@pytest.mark.parametrize(
    "metric", ["euclidean", "minkowski", "manhattan", "chebyshev", "haversine"]
)
def test_kde_algorithm_metric_choice(algorithm, metric):
    # Smoke test for various metrics and algorithms
    rng = np.random.RandomState(0)
    X = rng.randn(10, 2)  # 2 features required for haversine dist.
    Y = rng.randn(10, 2)

    if algorithm == "kd_tree" and metric not in KDTree.valid_metrics:
        with pytest.raises(ValueError):
            KernelDensity(algorithm=algorithm, metric=metric)
    else:
        kde = KernelDensity(algorithm=algorithm, metric=metric)
        kde.fit(X)
        y_dens = kde.score_samples(Y)
        assert y_dens.shape == Y.shape[:1]


def test_kde_score(n_samples=100, n_features=3):
    pass
    # FIXME
    # rng = np.random.RandomState(0)
    # X = rng.random_sample((n_samples, n_features))
    # Y = rng.random_sample((n_samples, n_features))


def test_kde_badargs():
    with pytest.raises(ValueError):
        KernelDensity(algorithm="blah")
    with pytest.raises(ValueError):
        KernelDensity(bandwidth=0)
    with pytest.raises(ValueError):
        KernelDensity(kernel="blah")
    with pytest.raises(ValueError):
        KernelDensity(metric="blah")
    with pytest.raises(ValueError):
        KernelDensity(algorithm="kd_tree", metric="blah")
    kde = KernelDensity()
    with pytest.raises(ValueError):
        kde.fit(np.random.random((200, 10)), sample_weight=np.random.random((200, 10)))
    with pytest.raises(ValueError):
        kde.fit(np.random.random((200, 10)), sample_weight=-np.random.random(200))


def test_kde_pipeline_gridsearch():
    # test that kde plays nice in pipelines and grid-searches
    X, _ = make_blobs(cluster_std=0.1, random_state=1, centers=[[0, 1], [1, 0], [0, 0]])
    pipe1 = make_pipeline(
        StandardScaler(with_mean=False, with_std=False),
        KernelDensity(kernel="gaussian"),
    )
    params = dict(kerneldensity__bandwidth=[0.001, 0.01, 0.1, 1, 10])
    search = GridSearchCV(pipe1, param_grid=params)
    search.fit(X)
    assert search.best_params_["kerneldensity__bandwidth"] == 0.1


def test_kde_sample_weights():
    n_samples = 400
    size_test = 20
    weights_neutral = np.full(n_samples, 3.0)
    for d in [1, 2, 10]:
        rng = np.random.RandomState(0)
        X = rng.rand(n_samples, d)
        weights = 1 + (10 * X.sum(axis=1)).astype(np.int8)
        X_repetitions = np.repeat(X, weights, axis=0)
        n_samples_test = size_test // d
        test_points = rng.rand(n_samples_test, d)
        for algorithm in ["auto", "ball_tree", "kd_tree"]:
            for metric in ["euclidean", "minkowski", "manhattan", "chebyshev"]:
                if algorithm != "kd_tree" or metric in KDTree.valid_metrics:
                    kde = KernelDensity(algorithm=algorithm, metric=metric)

                    # Test that adding a constant sample weight has no effect
                    kde.fit(X, sample_weight=weights_neutral)
                    scores_const_weight = kde.score_samples(test_points)
                    sample_const_weight = kde.sample(random_state=1234)
                    kde.fit(X)
                    scores_no_weight = kde.score_samples(test_points)
                    sample_no_weight = kde.sample(random_state=1234)
                    assert_allclose(scores_const_weight, scores_no_weight)
                    assert_allclose(sample_const_weight, sample_no_weight)

                    # Test equivalence between sampling and (integer) weights
                    kde.fit(X, sample_weight=weights)
                    scores_weight = kde.score_samples(test_points)
                    sample_weight = kde.sample(random_state=1234)
                    kde.fit(X_repetitions)
                    scores_ref_sampling = kde.score_samples(test_points)
                    sample_ref_sampling = kde.sample(random_state=1234)
                    assert_allclose(scores_weight, scores_ref_sampling)
                    assert_allclose(sample_weight, sample_ref_sampling)

                    # Test that sample weights has a non-trivial effect
                    diff = np.max(np.abs(scores_no_weight - scores_weight))
                    assert diff > 0.001

                    # Test invariance with respect to arbitrary scaling
                    scale_factor = rng.rand()
                    kde.fit(X, sample_weight=(scale_factor * weights))
                    scores_scaled_weight = kde.score_samples(test_points)
                    assert_allclose(scores_scaled_weight, scores_weight)


def test_sample_weight_invalid():
    # Check sample weighting raises errors.
    kde = KernelDensity()
    data = np.reshape([1.0, 2.0, 3.0], (-1, 1))

    sample_weight = [0.1, -0.2, 0.3]
    expected_err = "sample_weight must have positive values"
    with pytest.raises(ValueError, match=expected_err):
        kde.fit(data, sample_weight=sample_weight)


@pytest.mark.parametrize("sample_weight", [None, [0.1, 0.2, 0.3]])
def test_pickling(tmpdir, sample_weight):
    # Make sure that predictions are the same before and after pickling. Used
    # to be a bug because sample_weights wasn't pickled and the resulting tree
    # would miss some info.

    kde = KernelDensity()
    data = np.reshape([1.0, 2.0, 3.0], (-1, 1))
    kde.fit(data, sample_weight=sample_weight)

    X = np.reshape([1.1, 2.1], (-1, 1))
    scores = kde.score_samples(X)

    file_path = str(tmpdir.join("dump.pkl"))
    joblib.dump(kde, file_path)
    kde = joblib.load(file_path)
    scores_pickled = kde.score_samples(X)

    assert_allclose(scores, scores_pickled)


@pytest.mark.parametrize("method", ["score_samples", "sample"])
def test_check_is_fitted(method):
    # Check that predict raises an exception in an unfitted estimator.
    # Unfitted estimators should raise a NotFittedError.
    rng = np.random.RandomState(0)
    X = rng.randn(10, 2)
    kde = KernelDensity()

    with pytest.raises(NotFittedError):
        getattr(kde, method)(X)
