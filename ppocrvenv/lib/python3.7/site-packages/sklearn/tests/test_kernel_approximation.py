import re

import numpy as np
from scipy.sparse import csr_matrix
import pytest

from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_array_almost_equal

from sklearn.metrics.pairwise import kernel_metrics
from sklearn.kernel_approximation import RBFSampler
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.kernel_approximation import SkewedChi2Sampler
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import PolynomialCountSketch
from sklearn.datasets import make_classification
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel, chi2_kernel

# generate data
rng = np.random.RandomState(0)
X = rng.random_sample(size=(300, 50))
Y = rng.random_sample(size=(300, 50))
X /= X.sum(axis=1)[:, np.newaxis]
Y /= Y.sum(axis=1)[:, np.newaxis]


@pytest.mark.parametrize("degree", [-1, 0])
def test_polynomial_count_sketch_raises_if_degree_lower_than_one(degree):
    with pytest.raises(ValueError, match=f"degree={degree} should be >=1."):
        ps_transform = PolynomialCountSketch(degree=degree)
        ps_transform.fit(X, Y)


@pytest.mark.parametrize("X", [X, csr_matrix(X)])
@pytest.mark.parametrize("Y", [Y, csr_matrix(Y)])
@pytest.mark.parametrize("gamma", [0.1, 1, 2.5])
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("coef0", [0, 1, 2.5])
def test_polynomial_count_sketch(X, Y, gamma, degree, coef0):
    # test that PolynomialCountSketch approximates polynomial
    # kernel on random data

    # compute exact kernel
    kernel = polynomial_kernel(X, Y, gamma=gamma, degree=degree, coef0=coef0)

    # approximate kernel mapping
    ps_transform = PolynomialCountSketch(
        n_components=5000, gamma=gamma, coef0=coef0, degree=degree, random_state=42
    )
    X_trans = ps_transform.fit_transform(X)
    Y_trans = ps_transform.transform(Y)
    kernel_approx = np.dot(X_trans, Y_trans.T)

    error = kernel - kernel_approx
    assert np.abs(np.mean(error)) <= 0.05  # close to unbiased
    np.abs(error, out=error)
    assert np.max(error) <= 0.1  # nothing too far off
    assert np.mean(error) <= 0.05  # mean is fairly close


def _linear_kernel(X, Y):
    return np.dot(X, Y.T)


def test_additive_chi2_sampler():
    # test that AdditiveChi2Sampler approximates kernel on random data

    # compute exact kernel
    # abbreviations for easier formula
    X_ = X[:, np.newaxis, :]
    Y_ = Y[np.newaxis, :, :]

    large_kernel = 2 * X_ * Y_ / (X_ + Y_)

    # reduce to n_samples_x x n_samples_y by summing over features
    kernel = large_kernel.sum(axis=2)

    # approximate kernel mapping
    transform = AdditiveChi2Sampler(sample_steps=3)
    X_trans = transform.fit_transform(X)
    Y_trans = transform.transform(Y)

    kernel_approx = np.dot(X_trans, Y_trans.T)

    assert_array_almost_equal(kernel, kernel_approx, 1)

    X_sp_trans = transform.fit_transform(csr_matrix(X))
    Y_sp_trans = transform.transform(csr_matrix(Y))

    assert_array_equal(X_trans, X_sp_trans.A)
    assert_array_equal(Y_trans, Y_sp_trans.A)

    # test error is raised on negative input
    Y_neg = Y.copy()
    Y_neg[0, 0] = -1
    msg = "Negative values in data passed to"
    with pytest.raises(ValueError, match=msg):
        transform.transform(Y_neg)

    # test error on invalid sample_steps
    transform = AdditiveChi2Sampler(sample_steps=4)
    msg = re.escape(
        "If sample_steps is not in [1, 2, 3], you need to provide sample_interval"
    )
    with pytest.raises(ValueError, match=msg):
        transform.fit(X)

    # test that the sample interval is set correctly
    sample_steps_available = [1, 2, 3]
    for sample_steps in sample_steps_available:

        # test that the sample_interval is initialized correctly
        transform = AdditiveChi2Sampler(sample_steps=sample_steps)
        assert transform.sample_interval is None

        # test that the sample_interval is changed in the fit method
        transform.fit(X)
        assert transform.sample_interval_ is not None

    # test that the sample_interval is set correctly
    sample_interval = 0.3
    transform = AdditiveChi2Sampler(sample_steps=4, sample_interval=sample_interval)
    assert transform.sample_interval == sample_interval
    transform.fit(X)
    assert transform.sample_interval_ == sample_interval


def test_skewed_chi2_sampler():
    # test that RBFSampler approximates kernel on random data

    # compute exact kernel
    c = 0.03
    # set on negative component but greater than c to ensure that the kernel
    # approximation is valid on the group (-c; +\infty) endowed with the skewed
    # multiplication.
    Y[0, 0] = -c / 2.0

    # abbreviations for easier formula
    X_c = (X + c)[:, np.newaxis, :]
    Y_c = (Y + c)[np.newaxis, :, :]

    # we do it in log-space in the hope that it's more stable
    # this array is n_samples_x x n_samples_y big x n_features
    log_kernel = (
        (np.log(X_c) / 2.0) + (np.log(Y_c) / 2.0) + np.log(2.0) - np.log(X_c + Y_c)
    )
    # reduce to n_samples_x x n_samples_y by summing over features in log-space
    kernel = np.exp(log_kernel.sum(axis=2))

    # approximate kernel mapping
    transform = SkewedChi2Sampler(skewedness=c, n_components=1000, random_state=42)
    X_trans = transform.fit_transform(X)
    Y_trans = transform.transform(Y)

    kernel_approx = np.dot(X_trans, Y_trans.T)
    assert_array_almost_equal(kernel, kernel_approx, 1)
    assert np.isfinite(kernel).all(), "NaNs found in the Gram matrix"
    assert np.isfinite(kernel_approx).all(), "NaNs found in the approximate Gram matrix"

    # test error is raised on when inputs contains values smaller than -c
    Y_neg = Y.copy()
    Y_neg[0, 0] = -c * 2.0
    msg = "X may not contain entries smaller than -skewedness"
    with pytest.raises(ValueError, match=msg):
        transform.transform(Y_neg)


def test_additive_chi2_sampler_exceptions():
    """Ensures correct error message"""
    transformer = AdditiveChi2Sampler()
    X_neg = X.copy()
    X_neg[0, 0] = -1
    with pytest.raises(ValueError, match="X in AdditiveChi2Sampler.fit"):
        transformer.fit(X_neg)
    with pytest.raises(ValueError, match="X in AdditiveChi2Sampler.transform"):
        transformer.fit(X)
        transformer.transform(X_neg)


def test_rbf_sampler():
    # test that RBFSampler approximates kernel on random data
    # compute exact kernel
    gamma = 10.0
    kernel = rbf_kernel(X, Y, gamma=gamma)

    # approximate kernel mapping
    rbf_transform = RBFSampler(gamma=gamma, n_components=1000, random_state=42)
    X_trans = rbf_transform.fit_transform(X)
    Y_trans = rbf_transform.transform(Y)
    kernel_approx = np.dot(X_trans, Y_trans.T)

    error = kernel - kernel_approx
    assert np.abs(np.mean(error)) <= 0.01  # close to unbiased
    np.abs(error, out=error)
    assert np.max(error) <= 0.1  # nothing too far off
    assert np.mean(error) <= 0.05  # mean is fairly close


def test_input_validation():
    # Regression test: kernel approx. transformers should work on lists
    # No assertions; the old versions would simply crash
    X = [[1, 2], [3, 4], [5, 6]]
    AdditiveChi2Sampler().fit(X).transform(X)
    SkewedChi2Sampler().fit(X).transform(X)
    RBFSampler().fit(X).transform(X)

    X = csr_matrix(X)
    RBFSampler().fit(X).transform(X)


def test_nystroem_approximation():
    # some basic tests
    rnd = np.random.RandomState(0)
    X = rnd.uniform(size=(10, 4))

    # With n_components = n_samples this is exact
    X_transformed = Nystroem(n_components=X.shape[0]).fit_transform(X)
    K = rbf_kernel(X)
    assert_array_almost_equal(np.dot(X_transformed, X_transformed.T), K)

    trans = Nystroem(n_components=2, random_state=rnd)
    X_transformed = trans.fit(X).transform(X)
    assert X_transformed.shape == (X.shape[0], 2)

    # test callable kernel
    trans = Nystroem(n_components=2, kernel=_linear_kernel, random_state=rnd)
    X_transformed = trans.fit(X).transform(X)
    assert X_transformed.shape == (X.shape[0], 2)

    # test that available kernels fit and transform
    kernels_available = kernel_metrics()
    for kern in kernels_available:
        trans = Nystroem(n_components=2, kernel=kern, random_state=rnd)
        X_transformed = trans.fit(X).transform(X)
        assert X_transformed.shape == (X.shape[0], 2)


def test_nystroem_default_parameters():
    rnd = np.random.RandomState(42)
    X = rnd.uniform(size=(10, 4))

    # rbf kernel should behave as gamma=None by default
    # aka gamma = 1 / n_features
    nystroem = Nystroem(n_components=10)
    X_transformed = nystroem.fit_transform(X)
    K = rbf_kernel(X, gamma=None)
    K2 = np.dot(X_transformed, X_transformed.T)
    assert_array_almost_equal(K, K2)

    # chi2 kernel should behave as gamma=1 by default
    nystroem = Nystroem(kernel="chi2", n_components=10)
    X_transformed = nystroem.fit_transform(X)
    K = chi2_kernel(X, gamma=1)
    K2 = np.dot(X_transformed, X_transformed.T)
    assert_array_almost_equal(K, K2)


def test_nystroem_singular_kernel():
    # test that nystroem works with singular kernel matrix
    rng = np.random.RandomState(0)
    X = rng.rand(10, 20)
    X = np.vstack([X] * 2)  # duplicate samples

    gamma = 100
    N = Nystroem(gamma=gamma, n_components=X.shape[0]).fit(X)
    X_transformed = N.transform(X)

    K = rbf_kernel(X, gamma=gamma)

    assert_array_almost_equal(K, np.dot(X_transformed, X_transformed.T))
    assert np.all(np.isfinite(Y))


def test_nystroem_poly_kernel_params():
    # Non-regression: Nystroem should pass other parameters beside gamma.
    rnd = np.random.RandomState(37)
    X = rnd.uniform(size=(10, 4))

    K = polynomial_kernel(X, degree=3.1, coef0=0.1)
    nystroem = Nystroem(
        kernel="polynomial", n_components=X.shape[0], degree=3.1, coef0=0.1
    )
    X_transformed = nystroem.fit_transform(X)
    assert_array_almost_equal(np.dot(X_transformed, X_transformed.T), K)


def test_nystroem_callable():
    # Test Nystroem on a callable.
    rnd = np.random.RandomState(42)
    n_samples = 10
    X = rnd.uniform(size=(n_samples, 4))

    def logging_histogram_kernel(x, y, log):
        """Histogram kernel that writes to a log."""
        log.append(1)
        return np.minimum(x, y).sum()

    kernel_log = []
    X = list(X)  # test input validation
    Nystroem(
        kernel=logging_histogram_kernel,
        n_components=(n_samples - 1),
        kernel_params={"log": kernel_log},
    ).fit(X)
    assert len(kernel_log) == n_samples * (n_samples - 1) / 2

    # if degree, gamma or coef0 is passed, we raise a ValueError
    msg = "Don't pass gamma, coef0 or degree to Nystroem"
    params = ({"gamma": 1}, {"coef0": 1}, {"degree": 2})
    for param in params:
        ny = Nystroem(kernel=_linear_kernel, n_components=(n_samples - 1), **param)
        with pytest.raises(ValueError, match=msg):
            ny.fit(X)


def test_nystroem_precomputed_kernel():
    # Non-regression: test Nystroem on precomputed kernel.
    # PR - 14706
    rnd = np.random.RandomState(12)
    X = rnd.uniform(size=(10, 4))

    K = polynomial_kernel(X, degree=2, coef0=0.1)
    nystroem = Nystroem(kernel="precomputed", n_components=X.shape[0])
    X_transformed = nystroem.fit_transform(K)
    assert_array_almost_equal(np.dot(X_transformed, X_transformed.T), K)

    # if degree, gamma or coef0 is passed, we raise a ValueError
    msg = "Don't pass gamma, coef0 or degree to Nystroem"
    params = ({"gamma": 1}, {"coef0": 1}, {"degree": 2})
    for param in params:
        ny = Nystroem(kernel="precomputed", n_components=X.shape[0], **param)
        with pytest.raises(ValueError, match=msg):
            ny.fit(K)


def test_nystroem_component_indices():
    """Check that `component_indices_` corresponds to the subset of
    training points used to construct the feature map.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/20474
    """
    X, _ = make_classification(n_samples=100, n_features=20)
    feature_map_nystroem = Nystroem(
        n_components=10,
        random_state=0,
    )
    feature_map_nystroem.fit(X)
    assert feature_map_nystroem.component_indices_.shape == (10,)
