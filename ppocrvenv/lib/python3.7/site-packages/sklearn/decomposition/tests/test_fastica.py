"""
Test the fastica algorithm.
"""
import itertools
import warnings
import pytest

import numpy as np
from scipy import stats

from sklearn.utils._testing import assert_almost_equal
from sklearn.utils._testing import assert_array_almost_equal

from sklearn.decomposition import FastICA, fastica, PCA
from sklearn.decomposition._fastica import _gs_decorrelation
from sklearn.exceptions import ConvergenceWarning


def center_and_norm(x, axis=-1):
    """Centers and norms x **in place**

    Parameters
    -----------
    x: ndarray
        Array with an axis of observations (statistical units) measured on
        random variables.
    axis: int, optional
        Axis along which the mean and variance are calculated.
    """
    x = np.rollaxis(x, axis)
    x -= x.mean(axis=0)
    x /= x.std(axis=0)


def test_gs():
    # Test gram schmidt orthonormalization
    # generate a random orthogonal  matrix
    rng = np.random.RandomState(0)
    W, _, _ = np.linalg.svd(rng.randn(10, 10))
    w = rng.randn(10)
    _gs_decorrelation(w, W, 10)
    assert (w ** 2).sum() < 1.0e-10
    w = rng.randn(10)
    u = _gs_decorrelation(w, W, 5)
    tmp = np.dot(u, W.T)
    assert (tmp[:5] ** 2).sum() < 1.0e-10


@pytest.mark.parametrize("add_noise", [True, False])
@pytest.mark.parametrize("seed", range(1))
def test_fastica_simple(add_noise, seed):
    # Test the FastICA algorithm on very simple data.
    rng = np.random.RandomState(seed)
    # scipy.stats uses the global RNG:
    n_samples = 1000
    # Generate two sources:
    s1 = (2 * np.sin(np.linspace(0, 100, n_samples)) > 0) - 1
    s2 = stats.t.rvs(1, size=n_samples)
    s = np.c_[s1, s2].T
    center_and_norm(s)
    s1, s2 = s

    # Mixing angle
    phi = 0.6
    mixing = np.array([[np.cos(phi), np.sin(phi)], [np.sin(phi), -np.cos(phi)]])
    m = np.dot(mixing, s)

    if add_noise:
        m += 0.1 * rng.randn(2, 1000)

    center_and_norm(m)

    # function as fun arg
    def g_test(x):
        return x ** 3, (3 * x ** 2).mean(axis=-1)

    algos = ["parallel", "deflation"]
    nls = ["logcosh", "exp", "cube", g_test]
    whitening = [True, False]
    for algo, nl, whiten in itertools.product(algos, nls, whitening):
        if whiten:
            k_, mixing_, s_ = fastica(m.T, fun=nl, algorithm=algo, random_state=rng)
            with pytest.raises(ValueError):
                fastica(m.T, fun=np.tanh, algorithm=algo)
        else:
            pca = PCA(n_components=2, whiten=True, random_state=rng)
            X = pca.fit_transform(m.T)
            k_, mixing_, s_ = fastica(
                X, fun=nl, algorithm=algo, whiten=False, random_state=rng
            )
            with pytest.raises(ValueError):
                fastica(X, fun=np.tanh, algorithm=algo)
        s_ = s_.T
        # Check that the mixing model described in the docstring holds:
        if whiten:
            assert_almost_equal(s_, np.dot(np.dot(mixing_, k_), m))

        center_and_norm(s_)
        s1_, s2_ = s_
        # Check to see if the sources have been estimated
        # in the wrong order
        if abs(np.dot(s1_, s2)) > abs(np.dot(s1_, s1)):
            s2_, s1_ = s_
        s1_ *= np.sign(np.dot(s1_, s1))
        s2_ *= np.sign(np.dot(s2_, s2))

        # Check that we have estimated the original sources
        if not add_noise:
            assert_almost_equal(np.dot(s1_, s1) / n_samples, 1, decimal=2)
            assert_almost_equal(np.dot(s2_, s2) / n_samples, 1, decimal=2)
        else:
            assert_almost_equal(np.dot(s1_, s1) / n_samples, 1, decimal=1)
            assert_almost_equal(np.dot(s2_, s2) / n_samples, 1, decimal=1)

    # Test FastICA class
    _, _, sources_fun = fastica(m.T, fun=nl, algorithm=algo, random_state=seed)
    ica = FastICA(fun=nl, algorithm=algo, random_state=seed)
    sources = ica.fit_transform(m.T)
    assert ica.components_.shape == (2, 2)
    assert sources.shape == (1000, 2)

    assert_array_almost_equal(sources_fun, sources)
    assert_array_almost_equal(sources, ica.transform(m.T))

    assert ica.mixing_.shape == (2, 2)

    for fn in [np.tanh, "exp(-.5(x^2))"]:
        ica = FastICA(fun=fn, algorithm=algo)
        with pytest.raises(ValueError):
            ica.fit(m.T)

    with pytest.raises(TypeError):
        FastICA(fun=range(10)).fit(m.T)


def test_fastica_nowhiten():
    m = [[0, 1], [1, 0]]

    # test for issue #697
    ica = FastICA(n_components=1, whiten=False, random_state=0)
    warn_msg = "Ignoring n_components with whiten=False."
    with pytest.warns(UserWarning, match=warn_msg):
        ica.fit(m)
    assert hasattr(ica, "mixing_")


def test_fastica_convergence_fail():
    # Test the FastICA algorithm on very simple data
    # (see test_non_square_fastica).
    # Ensure a ConvergenceWarning raised if the tolerance is sufficiently low.
    rng = np.random.RandomState(0)

    n_samples = 1000
    # Generate two sources:
    t = np.linspace(0, 100, n_samples)
    s1 = np.sin(t)
    s2 = np.ceil(np.sin(np.pi * t))
    s = np.c_[s1, s2].T
    center_and_norm(s)

    # Mixing matrix
    mixing = rng.randn(6, 2)
    m = np.dot(mixing, s)

    # Do fastICA with tolerance 0. to ensure failing convergence
    warn_msg = (
        "FastICA did not converge. Consider increasing tolerance "
        "or the maximum number of iterations."
    )
    with pytest.warns(ConvergenceWarning, match=warn_msg):
        ica = FastICA(
            algorithm="parallel", n_components=2, random_state=rng, max_iter=2, tol=0.0
        )
        ica.fit(m.T)


@pytest.mark.parametrize("add_noise", [True, False])
def test_non_square_fastica(add_noise):
    # Test the FastICA algorithm on very simple data.
    rng = np.random.RandomState(0)

    n_samples = 1000
    # Generate two sources:
    t = np.linspace(0, 100, n_samples)
    s1 = np.sin(t)
    s2 = np.ceil(np.sin(np.pi * t))
    s = np.c_[s1, s2].T
    center_and_norm(s)
    s1, s2 = s

    # Mixing matrix
    mixing = rng.randn(6, 2)
    m = np.dot(mixing, s)

    if add_noise:
        m += 0.1 * rng.randn(6, n_samples)

    center_and_norm(m)

    k_, mixing_, s_ = fastica(m.T, n_components=2, random_state=rng)
    s_ = s_.T

    # Check that the mixing model described in the docstring holds:
    assert_almost_equal(s_, np.dot(np.dot(mixing_, k_), m))

    center_and_norm(s_)
    s1_, s2_ = s_
    # Check to see if the sources have been estimated
    # in the wrong order
    if abs(np.dot(s1_, s2)) > abs(np.dot(s1_, s1)):
        s2_, s1_ = s_
    s1_ *= np.sign(np.dot(s1_, s1))
    s2_ *= np.sign(np.dot(s2_, s2))

    # Check that we have estimated the original sources
    if not add_noise:
        assert_almost_equal(np.dot(s1_, s1) / n_samples, 1, decimal=3)
        assert_almost_equal(np.dot(s2_, s2) / n_samples, 1, decimal=3)


def test_fit_transform():
    # Test FastICA.fit_transform
    rng = np.random.RandomState(0)
    X = rng.random_sample((100, 10))
    for whiten, n_components in [[True, 5], [False, None]]:
        n_components_ = n_components if n_components is not None else X.shape[1]

        ica = FastICA(n_components=n_components, whiten=whiten, random_state=0)
        Xt = ica.fit_transform(X)
        assert ica.components_.shape == (n_components_, 10)
        assert Xt.shape == (100, n_components_)

        ica = FastICA(n_components=n_components, whiten=whiten, random_state=0)
        ica.fit(X)
        assert ica.components_.shape == (n_components_, 10)
        Xt2 = ica.transform(X)

        assert_array_almost_equal(Xt, Xt2)


def test_inverse_transform():
    # Test FastICA.inverse_transform
    n_features = 10
    n_samples = 100
    n1, n2 = 5, 10
    rng = np.random.RandomState(0)
    X = rng.random_sample((n_samples, n_features))
    expected = {
        (True, n1): (n_features, n1),
        (True, n2): (n_features, n2),
        (False, n1): (n_features, n2),
        (False, n2): (n_features, n2),
    }
    for whiten in [True, False]:
        for n_components in [n1, n2]:
            n_components_ = n_components if n_components is not None else X.shape[1]
            ica = FastICA(n_components=n_components, random_state=rng, whiten=whiten)
            with warnings.catch_warnings(record=True):
                # catch "n_components ignored" warning
                Xt = ica.fit_transform(X)
            expected_shape = expected[(whiten, n_components_)]
            assert ica.mixing_.shape == expected_shape
            X2 = ica.inverse_transform(Xt)
            assert X.shape == X2.shape

            # reversibility test in non-reduction case
            if n_components == X.shape[1]:
                assert_array_almost_equal(X, X2)


def test_fastica_errors():
    n_features = 3
    n_samples = 10
    rng = np.random.RandomState(0)
    X = rng.random_sample((n_samples, n_features))
    w_init = rng.randn(n_features + 1, n_features + 1)
    with pytest.raises(ValueError, match="max_iter should be greater than 1"):
        FastICA(max_iter=0)
    with pytest.raises(ValueError, match=r"alpha must be in \[1,2\]"):
        fastica(X, fun_args={"alpha": 0})
    with pytest.raises(
        ValueError, match="w_init has invalid shape.+" r"should be \(3L?, 3L?\)"
    ):
        fastica(X, w_init=w_init)
    with pytest.raises(
        ValueError, match="Invalid algorithm.+must be.+parallel.+or.+deflation"
    ):
        fastica(X, algorithm="pizza")


@pytest.mark.parametrize("whiten", [True, False])
@pytest.mark.parametrize("return_X_mean", [True, False])
@pytest.mark.parametrize("return_n_iter", [True, False])
def test_fastica_output_shape(whiten, return_X_mean, return_n_iter):
    n_features = 3
    n_samples = 10
    rng = np.random.RandomState(0)
    X = rng.random_sample((n_samples, n_features))

    expected_len = 3 + return_X_mean + return_n_iter

    out = fastica(
        X, whiten=whiten, return_n_iter=return_n_iter, return_X_mean=return_X_mean
    )

    assert len(out) == expected_len
    if not whiten:
        assert out[0] is None
