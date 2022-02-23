# Author: Vlad Niculae
# License: BSD 3 clause

import sys
import pytest

import numpy as np

from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_allclose
from sklearn.utils._testing import if_safe_multiprocessing_with_blas

from sklearn.decomposition import SparsePCA, MiniBatchSparsePCA, PCA
from sklearn.utils import check_random_state


def generate_toy_data(n_components, n_samples, image_size, random_state=None):
    n_features = image_size[0] * image_size[1]

    rng = check_random_state(random_state)
    U = rng.randn(n_samples, n_components)
    V = rng.randn(n_components, n_features)

    centers = [(3, 3), (6, 7), (8, 1)]
    sz = [1, 2, 1]
    for k in range(n_components):
        img = np.zeros(image_size)
        xmin, xmax = centers[k][0] - sz[k], centers[k][0] + sz[k]
        ymin, ymax = centers[k][1] - sz[k], centers[k][1] + sz[k]
        img[xmin:xmax][:, ymin:ymax] = 1.0
        V[k, :] = img.ravel()

    # Y is defined by : Y = UV + noise
    Y = np.dot(U, V)
    Y += 0.1 * rng.randn(Y.shape[0], Y.shape[1])  # Add noise
    return Y, U, V


# SparsePCA can be a bit slow. To avoid having test times go up, we
# test different aspects of the code in the same test


def test_correct_shapes():
    rng = np.random.RandomState(0)
    X = rng.randn(12, 10)
    spca = SparsePCA(n_components=8, random_state=rng)
    U = spca.fit_transform(X)
    assert spca.components_.shape == (8, 10)
    assert U.shape == (12, 8)
    # test overcomplete decomposition
    spca = SparsePCA(n_components=13, random_state=rng)
    U = spca.fit_transform(X)
    assert spca.components_.shape == (13, 10)
    assert U.shape == (12, 13)


def test_fit_transform():
    alpha = 1
    rng = np.random.RandomState(0)
    Y, _, _ = generate_toy_data(3, 10, (8, 8), random_state=rng)  # wide array
    spca_lars = SparsePCA(n_components=3, method="lars", alpha=alpha, random_state=0)
    spca_lars.fit(Y)

    # Test that CD gives similar results
    spca_lasso = SparsePCA(n_components=3, method="cd", random_state=0, alpha=alpha)
    spca_lasso.fit(Y)
    assert_array_almost_equal(spca_lasso.components_, spca_lars.components_)


@if_safe_multiprocessing_with_blas
def test_fit_transform_parallel():
    alpha = 1
    rng = np.random.RandomState(0)
    Y, _, _ = generate_toy_data(3, 10, (8, 8), random_state=rng)  # wide array
    spca_lars = SparsePCA(n_components=3, method="lars", alpha=alpha, random_state=0)
    spca_lars.fit(Y)
    U1 = spca_lars.transform(Y)
    # Test multiple CPUs
    spca = SparsePCA(
        n_components=3, n_jobs=2, method="lars", alpha=alpha, random_state=0
    ).fit(Y)
    U2 = spca.transform(Y)
    assert not np.all(spca_lars.components_ == 0)
    assert_array_almost_equal(U1, U2)


def test_transform_nan():
    # Test that SparsePCA won't return NaN when there is 0 feature in all
    # samples.
    rng = np.random.RandomState(0)
    Y, _, _ = generate_toy_data(3, 10, (8, 8), random_state=rng)  # wide array
    Y[:, 0] = 0
    estimator = SparsePCA(n_components=8)
    assert not np.any(np.isnan(estimator.fit_transform(Y)))


def test_fit_transform_tall():
    rng = np.random.RandomState(0)
    Y, _, _ = generate_toy_data(3, 65, (8, 8), random_state=rng)  # tall array
    spca_lars = SparsePCA(n_components=3, method="lars", random_state=rng)
    U1 = spca_lars.fit_transform(Y)
    spca_lasso = SparsePCA(n_components=3, method="cd", random_state=rng)
    U2 = spca_lasso.fit(Y).transform(Y)
    assert_array_almost_equal(U1, U2)


def test_initialization():
    rng = np.random.RandomState(0)
    U_init = rng.randn(5, 3)
    V_init = rng.randn(3, 4)
    model = SparsePCA(
        n_components=3, U_init=U_init, V_init=V_init, max_iter=0, random_state=rng
    )
    model.fit(rng.randn(5, 4))
    assert_allclose(model.components_, V_init / np.linalg.norm(V_init, axis=1)[:, None])


def test_mini_batch_correct_shapes():
    rng = np.random.RandomState(0)
    X = rng.randn(12, 10)
    pca = MiniBatchSparsePCA(n_components=8, random_state=rng)
    U = pca.fit_transform(X)
    assert pca.components_.shape == (8, 10)
    assert U.shape == (12, 8)
    # test overcomplete decomposition
    pca = MiniBatchSparsePCA(n_components=13, random_state=rng)
    U = pca.fit_transform(X)
    assert pca.components_.shape == (13, 10)
    assert U.shape == (12, 13)


# XXX: test always skipped
@pytest.mark.skipif(True, reason="skipping mini_batch_fit_transform.")
def test_mini_batch_fit_transform():
    alpha = 1
    rng = np.random.RandomState(0)
    Y, _, _ = generate_toy_data(3, 10, (8, 8), random_state=rng)  # wide array
    spca_lars = MiniBatchSparsePCA(n_components=3, random_state=0, alpha=alpha).fit(Y)
    U1 = spca_lars.transform(Y)
    # Test multiple CPUs
    if sys.platform == "win32":  # fake parallelism for win32
        import joblib

        _mp = joblib.parallel.multiprocessing
        joblib.parallel.multiprocessing = None
        try:
            spca = MiniBatchSparsePCA(
                n_components=3, n_jobs=2, alpha=alpha, random_state=0
            )
            U2 = spca.fit(Y).transform(Y)
        finally:
            joblib.parallel.multiprocessing = _mp
    else:  # we can efficiently use parallelism
        spca = MiniBatchSparsePCA(n_components=3, n_jobs=2, alpha=alpha, random_state=0)
        U2 = spca.fit(Y).transform(Y)
    assert not np.all(spca_lars.components_ == 0)
    assert_array_almost_equal(U1, U2)
    # Test that CD gives similar results
    spca_lasso = MiniBatchSparsePCA(
        n_components=3, method="cd", alpha=alpha, random_state=0
    ).fit(Y)
    assert_array_almost_equal(spca_lasso.components_, spca_lars.components_)


def test_scaling_fit_transform():
    alpha = 1
    rng = np.random.RandomState(0)
    Y, _, _ = generate_toy_data(3, 1000, (8, 8), random_state=rng)
    spca_lars = SparsePCA(n_components=3, method="lars", alpha=alpha, random_state=rng)
    results_train = spca_lars.fit_transform(Y)
    results_test = spca_lars.transform(Y[:10])
    assert_allclose(results_train[0], results_test[0])


def test_pca_vs_spca():
    rng = np.random.RandomState(0)
    Y, _, _ = generate_toy_data(3, 1000, (8, 8), random_state=rng)
    Z, _, _ = generate_toy_data(3, 10, (8, 8), random_state=rng)
    spca = SparsePCA(alpha=0, ridge_alpha=0, n_components=2)
    pca = PCA(n_components=2)
    pca.fit(Y)
    spca.fit(Y)
    results_test_pca = pca.transform(Z)
    results_test_spca = spca.transform(Z)
    assert_allclose(
        np.abs(spca.components_.dot(pca.components_.T)), np.eye(2), atol=1e-5
    )
    results_test_pca *= np.sign(results_test_pca[0, :])
    results_test_spca *= np.sign(results_test_spca[0, :])
    assert_allclose(results_test_pca, results_test_spca)


@pytest.mark.parametrize("SPCA", [SparsePCA, MiniBatchSparsePCA])
@pytest.mark.parametrize("n_components", [None, 3])
def test_spca_n_components_(SPCA, n_components):
    rng = np.random.RandomState(0)
    n_samples, n_features = 12, 10
    X = rng.randn(n_samples, n_features)

    model = SPCA(n_components=n_components).fit(X)

    if n_components is not None:
        assert model.n_components_ == n_components
    else:
        assert model.n_components_ == n_features
