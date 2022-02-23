# Author: Vlad Niculae
# License: BSD 3 clause

import numpy as np
import pytest

from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import ignore_warnings


from sklearn.linear_model import (
    orthogonal_mp,
    orthogonal_mp_gram,
    OrthogonalMatchingPursuit,
    OrthogonalMatchingPursuitCV,
    LinearRegression,
)
from sklearn.utils import check_random_state
from sklearn.datasets import make_sparse_coded_signal

n_samples, n_features, n_nonzero_coefs, n_targets = 25, 35, 5, 3
y, X, gamma = make_sparse_coded_signal(
    n_samples=n_targets,
    n_components=n_features,
    n_features=n_samples,
    n_nonzero_coefs=n_nonzero_coefs,
    random_state=0,
)
# Make X not of norm 1 for testing
X *= 10
y *= 10
G, Xy = np.dot(X.T, X), np.dot(X.T, y)
# this makes X (n_samples, n_features)
# and y (n_samples, 3)


# FIXME: 'normalize' to set to False in 1.2 and removed in 1.4
@pytest.mark.parametrize(
    "OmpModel", [OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV]
)
@pytest.mark.parametrize(
    "normalize, n_warnings", [(True, 0), (False, 0), ("deprecated", 1)]
)
def test_assure_warning_when_normalize(OmpModel, normalize, n_warnings):
    # check that we issue a FutureWarning when normalize was set
    rng = check_random_state(0)
    n_samples = 200
    n_features = 2
    X = rng.randn(n_samples, n_features)
    X[X < 0.1] = 0.0
    y = rng.rand(n_samples)

    model = OmpModel(normalize=normalize)
    with pytest.warns(None) as record:
        model.fit(X, y)

    record = [r for r in record if r.category == FutureWarning]
    assert len(record) == n_warnings


def test_correct_shapes():
    assert orthogonal_mp(X, y[:, 0], n_nonzero_coefs=5).shape == (n_features,)
    assert orthogonal_mp(X, y, n_nonzero_coefs=5).shape == (n_features, 3)


def test_correct_shapes_gram():
    assert orthogonal_mp_gram(G, Xy[:, 0], n_nonzero_coefs=5).shape == (n_features,)
    assert orthogonal_mp_gram(G, Xy, n_nonzero_coefs=5).shape == (n_features, 3)


def test_n_nonzero_coefs():
    assert np.count_nonzero(orthogonal_mp(X, y[:, 0], n_nonzero_coefs=5)) <= 5
    assert (
        np.count_nonzero(orthogonal_mp(X, y[:, 0], n_nonzero_coefs=5, precompute=True))
        <= 5
    )


def test_tol():
    tol = 0.5
    gamma = orthogonal_mp(X, y[:, 0], tol=tol)
    gamma_gram = orthogonal_mp(X, y[:, 0], tol=tol, precompute=True)
    assert np.sum((y[:, 0] - np.dot(X, gamma)) ** 2) <= tol
    assert np.sum((y[:, 0] - np.dot(X, gamma_gram)) ** 2) <= tol


def test_with_without_gram():
    assert_array_almost_equal(
        orthogonal_mp(X, y, n_nonzero_coefs=5),
        orthogonal_mp(X, y, n_nonzero_coefs=5, precompute=True),
    )


def test_with_without_gram_tol():
    assert_array_almost_equal(
        orthogonal_mp(X, y, tol=1.0), orthogonal_mp(X, y, tol=1.0, precompute=True)
    )


def test_unreachable_accuracy():
    assert_array_almost_equal(
        orthogonal_mp(X, y, tol=0), orthogonal_mp(X, y, n_nonzero_coefs=n_features)
    )
    warning_message = (
        "Orthogonal matching pursuit ended prematurely "
        "due to linear dependence in the dictionary. "
        "The requested precision might not have been met."
    )
    with pytest.warns(RuntimeWarning, match=warning_message):
        assert_array_almost_equal(
            orthogonal_mp(X, y, tol=0, precompute=True),
            orthogonal_mp(X, y, precompute=True, n_nonzero_coefs=n_features),
        )


@pytest.mark.parametrize("positional_params", [(X, y), (G, Xy)])
@pytest.mark.parametrize(
    "keyword_params",
    [{"tol": -1}, {"n_nonzero_coefs": -1}, {"n_nonzero_coefs": n_features + 1}],
)
def test_bad_input(positional_params, keyword_params):
    with pytest.raises(ValueError):
        orthogonal_mp(*positional_params, **keyword_params)


def test_perfect_signal_recovery():
    (idx,) = gamma[:, 0].nonzero()
    gamma_rec = orthogonal_mp(X, y[:, 0], n_nonzero_coefs=5)
    gamma_gram = orthogonal_mp_gram(G, Xy[:, 0], n_nonzero_coefs=5)
    assert_array_equal(idx, np.flatnonzero(gamma_rec))
    assert_array_equal(idx, np.flatnonzero(gamma_gram))
    assert_array_almost_equal(gamma[:, 0], gamma_rec, decimal=2)
    assert_array_almost_equal(gamma[:, 0], gamma_gram, decimal=2)


def test_orthogonal_mp_gram_readonly():
    # Non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/5956
    (idx,) = gamma[:, 0].nonzero()
    G_readonly = G.copy()
    G_readonly.setflags(write=False)
    Xy_readonly = Xy.copy()
    Xy_readonly.setflags(write=False)
    gamma_gram = orthogonal_mp_gram(
        G_readonly, Xy_readonly[:, 0], n_nonzero_coefs=5, copy_Gram=False, copy_Xy=False
    )
    assert_array_equal(idx, np.flatnonzero(gamma_gram))
    assert_array_almost_equal(gamma[:, 0], gamma_gram, decimal=2)


# FIXME: 'normalize' to be removed in 1.4
@pytest.mark.filterwarnings("ignore:The default of 'normalize'")
def test_estimator():
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
    omp.fit(X, y[:, 0])
    assert omp.coef_.shape == (n_features,)
    assert omp.intercept_.shape == ()
    assert np.count_nonzero(omp.coef_) <= n_nonzero_coefs

    omp.fit(X, y)
    assert omp.coef_.shape == (n_targets, n_features)
    assert omp.intercept_.shape == (n_targets,)
    assert np.count_nonzero(omp.coef_) <= n_targets * n_nonzero_coefs

    coef_normalized = omp.coef_[0].copy()
    omp.set_params(fit_intercept=True, normalize=False)
    omp.fit(X, y[:, 0])
    assert_array_almost_equal(coef_normalized, omp.coef_)

    omp.set_params(fit_intercept=False, normalize=False)
    omp.fit(X, y[:, 0])
    assert np.count_nonzero(omp.coef_) <= n_nonzero_coefs
    assert omp.coef_.shape == (n_features,)
    assert omp.intercept_ == 0

    omp.fit(X, y)
    assert omp.coef_.shape == (n_targets, n_features)
    assert omp.intercept_ == 0
    assert np.count_nonzero(omp.coef_) <= n_targets * n_nonzero_coefs


def test_identical_regressors():
    newX = X.copy()
    newX[:, 1] = newX[:, 0]
    gamma = np.zeros(n_features)
    gamma[0] = gamma[1] = 1.0
    newy = np.dot(newX, gamma)
    warning_message = (
        "Orthogonal matching pursuit ended prematurely "
        "due to linear dependence in the dictionary. "
        "The requested precision might not have been met."
    )
    with pytest.warns(RuntimeWarning, match=warning_message):
        orthogonal_mp(newX, newy, n_nonzero_coefs=2)


def test_swapped_regressors():
    gamma = np.zeros(n_features)
    # X[:, 21] should be selected first, then X[:, 0] selected second,
    # which will take X[:, 21]'s place in case the algorithm does
    # column swapping for optimization (which is the case at the moment)
    gamma[21] = 1.0
    gamma[0] = 0.5
    new_y = np.dot(X, gamma)
    new_Xy = np.dot(X.T, new_y)
    gamma_hat = orthogonal_mp(X, new_y, n_nonzero_coefs=2)
    gamma_hat_gram = orthogonal_mp_gram(G, new_Xy, n_nonzero_coefs=2)
    assert_array_equal(np.flatnonzero(gamma_hat), [0, 21])
    assert_array_equal(np.flatnonzero(gamma_hat_gram), [0, 21])


def test_no_atoms():
    y_empty = np.zeros_like(y)
    Xy_empty = np.dot(X.T, y_empty)
    gamma_empty = ignore_warnings(orthogonal_mp)(X, y_empty, n_nonzero_coefs=1)
    gamma_empty_gram = ignore_warnings(orthogonal_mp)(G, Xy_empty, n_nonzero_coefs=1)
    assert np.all(gamma_empty == 0)
    assert np.all(gamma_empty_gram == 0)


def test_omp_path():
    path = orthogonal_mp(X, y, n_nonzero_coefs=5, return_path=True)
    last = orthogonal_mp(X, y, n_nonzero_coefs=5, return_path=False)
    assert path.shape == (n_features, n_targets, 5)
    assert_array_almost_equal(path[:, :, -1], last)
    path = orthogonal_mp_gram(G, Xy, n_nonzero_coefs=5, return_path=True)
    last = orthogonal_mp_gram(G, Xy, n_nonzero_coefs=5, return_path=False)
    assert path.shape == (n_features, n_targets, 5)
    assert_array_almost_equal(path[:, :, -1], last)


def test_omp_return_path_prop_with_gram():
    path = orthogonal_mp(X, y, n_nonzero_coefs=5, return_path=True, precompute=True)
    last = orthogonal_mp(X, y, n_nonzero_coefs=5, return_path=False, precompute=True)
    assert path.shape == (n_features, n_targets, 5)
    assert_array_almost_equal(path[:, :, -1], last)


# FIXME: 'normalize' to be removed in 1.4
@pytest.mark.filterwarnings("ignore:The default of 'normalize'")
def test_omp_cv():
    y_ = y[:, 0]
    gamma_ = gamma[:, 0]
    ompcv = OrthogonalMatchingPursuitCV(
        normalize=True, fit_intercept=False, max_iter=10
    )
    ompcv.fit(X, y_)
    assert ompcv.n_nonzero_coefs_ == n_nonzero_coefs
    assert_array_almost_equal(ompcv.coef_, gamma_)
    omp = OrthogonalMatchingPursuit(
        normalize=True, fit_intercept=False, n_nonzero_coefs=ompcv.n_nonzero_coefs_
    )
    omp.fit(X, y_)
    assert_array_almost_equal(ompcv.coef_, omp.coef_)


# FIXME: 'normalize' to be removed in 1.4
@pytest.mark.filterwarnings("ignore:The default of 'normalize'")
def test_omp_reaches_least_squares():
    # Use small simple data; it's a sanity check but OMP can stop early
    rng = check_random_state(0)
    n_samples, n_features = (10, 8)
    n_targets = 3
    X = rng.randn(n_samples, n_features)
    Y = rng.randn(n_samples, n_targets)
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_features)
    lstsq = LinearRegression()
    omp.fit(X, Y)
    lstsq.fit(X, Y)
    assert_array_almost_equal(omp.coef_, lstsq.coef_)
