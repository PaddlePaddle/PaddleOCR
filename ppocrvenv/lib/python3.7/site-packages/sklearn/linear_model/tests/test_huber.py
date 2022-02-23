# Authors: Manoj Kumar mks542@nyu.edu
# License: BSD 3 clause

import numpy as np
from scipy import optimize, sparse

from sklearn.utils._testing import assert_almost_equal
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_array_almost_equal

from sklearn.datasets import make_regression
from sklearn.linear_model import HuberRegressor, LinearRegression, SGDRegressor, Ridge
from sklearn.linear_model._huber import _huber_loss_and_gradient


def make_regression_with_outliers(n_samples=50, n_features=20):
    rng = np.random.RandomState(0)
    # Generate data with outliers by replacing 10% of the samples with noise.
    X, y = make_regression(
        n_samples=n_samples, n_features=n_features, random_state=0, noise=0.05
    )

    # Replace 10% of the sample with noise.
    num_noise = int(0.1 * n_samples)
    random_samples = rng.randint(0, n_samples, num_noise)
    X[random_samples, :] = 2.0 * rng.normal(0, 1, (num_noise, X.shape[1]))
    return X, y


def test_huber_equals_lr_for_high_epsilon():
    # Test that Ridge matches LinearRegression for large epsilon
    X, y = make_regression_with_outliers()
    lr = LinearRegression()
    lr.fit(X, y)
    huber = HuberRegressor(epsilon=1e3, alpha=0.0)
    huber.fit(X, y)
    assert_almost_equal(huber.coef_, lr.coef_, 3)
    assert_almost_equal(huber.intercept_, lr.intercept_, 2)


def test_huber_max_iter():
    X, y = make_regression_with_outliers()
    huber = HuberRegressor(max_iter=1)
    huber.fit(X, y)
    assert huber.n_iter_ == huber.max_iter


def test_huber_gradient():
    # Test that the gradient calculated by _huber_loss_and_gradient is correct
    rng = np.random.RandomState(1)
    X, y = make_regression_with_outliers()
    sample_weight = rng.randint(1, 3, (y.shape[0]))

    def loss_func(x, *args):
        return _huber_loss_and_gradient(x, *args)[0]

    def grad_func(x, *args):
        return _huber_loss_and_gradient(x, *args)[1]

    # Check using optimize.check_grad that the gradients are equal.
    for _ in range(5):
        # Check for both fit_intercept and otherwise.
        for n_features in [X.shape[1] + 1, X.shape[1] + 2]:
            w = rng.randn(n_features)
            w[-1] = np.abs(w[-1])
            grad_same = optimize.check_grad(
                loss_func, grad_func, w, X, y, 0.01, 0.1, sample_weight
            )
            assert_almost_equal(grad_same, 1e-6, 4)


def test_huber_sample_weights():
    # Test sample_weights implementation in HuberRegressor"""

    X, y = make_regression_with_outliers()
    huber = HuberRegressor()
    huber.fit(X, y)
    huber_coef = huber.coef_
    huber_intercept = huber.intercept_

    # Rescale coefs before comparing with assert_array_almost_equal to make
    # sure that the number of decimal places used is somewhat insensitive to
    # the amplitude of the coefficients and therefore to the scale of the
    # data and the regularization parameter
    scale = max(np.mean(np.abs(huber.coef_)), np.mean(np.abs(huber.intercept_)))

    huber.fit(X, y, sample_weight=np.ones(y.shape[0]))
    assert_array_almost_equal(huber.coef_ / scale, huber_coef / scale)
    assert_array_almost_equal(huber.intercept_ / scale, huber_intercept / scale)

    X, y = make_regression_with_outliers(n_samples=5, n_features=20)
    X_new = np.vstack((X, np.vstack((X[1], X[1], X[3]))))
    y_new = np.concatenate((y, [y[1]], [y[1]], [y[3]]))
    huber.fit(X_new, y_new)
    huber_coef = huber.coef_
    huber_intercept = huber.intercept_
    sample_weight = np.ones(X.shape[0])
    sample_weight[1] = 3
    sample_weight[3] = 2
    huber.fit(X, y, sample_weight=sample_weight)

    assert_array_almost_equal(huber.coef_ / scale, huber_coef / scale)
    assert_array_almost_equal(huber.intercept_ / scale, huber_intercept / scale)

    # Test sparse implementation with sample weights.
    X_csr = sparse.csr_matrix(X)
    huber_sparse = HuberRegressor()
    huber_sparse.fit(X_csr, y, sample_weight=sample_weight)
    assert_array_almost_equal(huber_sparse.coef_ / scale, huber_coef / scale)


def test_huber_sparse():
    X, y = make_regression_with_outliers()
    huber = HuberRegressor(alpha=0.1)
    huber.fit(X, y)

    X_csr = sparse.csr_matrix(X)
    huber_sparse = HuberRegressor(alpha=0.1)
    huber_sparse.fit(X_csr, y)
    assert_array_almost_equal(huber_sparse.coef_, huber.coef_)
    assert_array_equal(huber.outliers_, huber_sparse.outliers_)


def test_huber_scaling_invariant():
    # Test that outliers filtering is scaling independent.
    X, y = make_regression_with_outliers()
    huber = HuberRegressor(fit_intercept=False, alpha=0.0)
    huber.fit(X, y)
    n_outliers_mask_1 = huber.outliers_
    assert not np.all(n_outliers_mask_1)

    huber.fit(X, 2.0 * y)
    n_outliers_mask_2 = huber.outliers_
    assert_array_equal(n_outliers_mask_2, n_outliers_mask_1)

    huber.fit(2.0 * X, 2.0 * y)
    n_outliers_mask_3 = huber.outliers_
    assert_array_equal(n_outliers_mask_3, n_outliers_mask_1)


def test_huber_and_sgd_same_results():
    # Test they should converge to same coefficients for same parameters

    X, y = make_regression_with_outliers(n_samples=10, n_features=2)

    # Fit once to find out the scale parameter. Scale down X and y by scale
    # so that the scale parameter is optimized to 1.0
    huber = HuberRegressor(fit_intercept=False, alpha=0.0, epsilon=1.35)
    huber.fit(X, y)
    X_scale = X / huber.scale_
    y_scale = y / huber.scale_
    huber.fit(X_scale, y_scale)
    assert_almost_equal(huber.scale_, 1.0, 3)

    sgdreg = SGDRegressor(
        alpha=0.0,
        loss="huber",
        shuffle=True,
        random_state=0,
        max_iter=10000,
        fit_intercept=False,
        epsilon=1.35,
        tol=None,
    )
    sgdreg.fit(X_scale, y_scale)
    assert_array_almost_equal(huber.coef_, sgdreg.coef_, 1)


def test_huber_warm_start():
    X, y = make_regression_with_outliers()
    huber_warm = HuberRegressor(alpha=1.0, max_iter=10000, warm_start=True, tol=1e-1)

    huber_warm.fit(X, y)
    huber_warm_coef = huber_warm.coef_.copy()
    huber_warm.fit(X, y)

    # SciPy performs the tol check after doing the coef updates, so
    # these would be almost same but not equal.
    assert_array_almost_equal(huber_warm.coef_, huber_warm_coef, 1)

    assert huber_warm.n_iter_ == 0


def test_huber_better_r2_score():
    # Test that huber returns a better r2 score than non-outliers"""
    X, y = make_regression_with_outliers()
    huber = HuberRegressor(alpha=0.01)
    huber.fit(X, y)
    linear_loss = np.dot(X, huber.coef_) + huber.intercept_ - y
    mask = np.abs(linear_loss) < huber.epsilon * huber.scale_
    huber_score = huber.score(X[mask], y[mask])
    huber_outlier_score = huber.score(X[~mask], y[~mask])

    # The Ridge regressor should be influenced by the outliers and hence
    # give a worse score on the non-outliers as compared to the huber
    # regressor.
    ridge = Ridge(alpha=0.01)
    ridge.fit(X, y)
    ridge_score = ridge.score(X[mask], y[mask])
    ridge_outlier_score = ridge.score(X[~mask], y[~mask])
    assert huber_score > ridge_score

    # The huber model should also fit poorly on the outliers.
    assert ridge_outlier_score > huber_outlier_score


def test_huber_bool():
    # Test that it does not crash with bool data
    X, y = make_regression(n_samples=200, n_features=2, noise=4.0, random_state=0)
    X_bool = X > 0
    HuberRegressor().fit(X_bool, y)
