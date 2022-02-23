# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Fabian Pedregosa <fabian.pedregosa@inria.fr>
#
# License: BSD 3 clause

from math import log

import numpy as np
from scipy.linalg import pinvh
import pytest


from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_almost_equal
from sklearn.utils._testing import assert_array_less
from sklearn.utils import check_random_state
from sklearn.linear_model import BayesianRidge, ARDRegression
from sklearn.linear_model import Ridge
from sklearn import datasets
from sklearn.utils.extmath import fast_logdet

diabetes = datasets.load_diabetes()


def test_n_iter():
    """Check value of n_iter."""
    X = np.array([[1], [2], [6], [8], [10]])
    y = np.array([1, 2, 6, 8, 10])
    clf = BayesianRidge(n_iter=0)
    msg = "n_iter should be greater than or equal to 1."
    with pytest.raises(ValueError, match=msg):
        clf.fit(X, y)


def test_bayesian_ridge_scores():
    """Check scores attribute shape"""
    X, y = diabetes.data, diabetes.target

    clf = BayesianRidge(compute_score=True)
    clf.fit(X, y)

    assert clf.scores_.shape == (clf.n_iter_ + 1,)


def test_bayesian_ridge_score_values():
    """Check value of score on toy example.

    Compute log marginal likelihood with equation (36) in Sparse Bayesian
    Learning and the Relevance Vector Machine (Tipping, 2001):

    - 0.5 * (log |Id/alpha + X.X^T/lambda| +
             y^T.(Id/alpha + X.X^T/lambda).y + n * log(2 * pi))
    + lambda_1 * log(lambda) - lambda_2 * lambda
    + alpha_1 * log(alpha) - alpha_2 * alpha

    and check equality with the score computed during training.
    """

    X, y = diabetes.data, diabetes.target
    n_samples = X.shape[0]
    # check with initial values of alpha and lambda (see code for the values)
    eps = np.finfo(np.float64).eps
    alpha_ = 1.0 / (np.var(y) + eps)
    lambda_ = 1.0

    # value of the parameters of the Gamma hyperpriors
    alpha_1 = 0.1
    alpha_2 = 0.1
    lambda_1 = 0.1
    lambda_2 = 0.1

    # compute score using formula of docstring
    score = lambda_1 * log(lambda_) - lambda_2 * lambda_
    score += alpha_1 * log(alpha_) - alpha_2 * alpha_
    M = 1.0 / alpha_ * np.eye(n_samples) + 1.0 / lambda_ * np.dot(X, X.T)
    M_inv = pinvh(M)
    score += -0.5 * (
        fast_logdet(M) + np.dot(y.T, np.dot(M_inv, y)) + n_samples * log(2 * np.pi)
    )

    # compute score with BayesianRidge
    clf = BayesianRidge(
        alpha_1=alpha_1,
        alpha_2=alpha_2,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        n_iter=1,
        fit_intercept=False,
        compute_score=True,
    )
    clf.fit(X, y)

    assert_almost_equal(clf.scores_[0], score, decimal=9)


def test_bayesian_ridge_parameter():
    # Test correctness of lambda_ and alpha_ parameters (GitHub issue #8224)
    X = np.array([[1, 1], [3, 4], [5, 7], [4, 1], [2, 6], [3, 10], [3, 2]])
    y = np.array([1, 2, 3, 2, 0, 4, 5]).T

    # A Ridge regression model using an alpha value equal to the ratio of
    # lambda_ and alpha_ from the Bayesian Ridge model must be identical
    br_model = BayesianRidge(compute_score=True).fit(X, y)
    rr_model = Ridge(alpha=br_model.lambda_ / br_model.alpha_).fit(X, y)
    assert_array_almost_equal(rr_model.coef_, br_model.coef_)
    assert_almost_equal(rr_model.intercept_, br_model.intercept_)


def test_bayesian_sample_weights():
    # Test correctness of the sample_weights method
    X = np.array([[1, 1], [3, 4], [5, 7], [4, 1], [2, 6], [3, 10], [3, 2]])
    y = np.array([1, 2, 3, 2, 0, 4, 5]).T
    w = np.array([4, 3, 3, 1, 1, 2, 3]).T

    # A Ridge regression model using an alpha value equal to the ratio of
    # lambda_ and alpha_ from the Bayesian Ridge model must be identical
    br_model = BayesianRidge(compute_score=True).fit(X, y, sample_weight=w)
    rr_model = Ridge(alpha=br_model.lambda_ / br_model.alpha_).fit(
        X, y, sample_weight=w
    )
    assert_array_almost_equal(rr_model.coef_, br_model.coef_)
    assert_almost_equal(rr_model.intercept_, br_model.intercept_)


def test_toy_bayesian_ridge_object():
    # Test BayesianRidge on toy
    X = np.array([[1], [2], [6], [8], [10]])
    Y = np.array([1, 2, 6, 8, 10])
    clf = BayesianRidge(compute_score=True)
    clf.fit(X, Y)

    # Check that the model could approximately learn the identity function
    test = [[1], [3], [4]]
    assert_array_almost_equal(clf.predict(test), [1, 3, 4], 2)


def test_bayesian_initial_params():
    # Test BayesianRidge with initial values (alpha_init, lambda_init)
    X = np.vander(np.linspace(0, 4, 5), 4)
    y = np.array([0.0, 1.0, 0.0, -1.0, 0.0])  # y = (x^3 - 6x^2 + 8x) / 3

    # In this case, starting from the default initial values will increase
    # the bias of the fitted curve. So, lambda_init should be small.
    reg = BayesianRidge(alpha_init=1.0, lambda_init=1e-3)
    # Check the R2 score nearly equals to one.
    r2 = reg.fit(X, y).score(X, y)
    assert_almost_equal(r2, 1.0)


def test_prediction_bayesian_ridge_ard_with_constant_input():
    # Test BayesianRidge and ARDRegression predictions for edge case of
    # constant target vectors
    n_samples = 4
    n_features = 5
    random_state = check_random_state(42)
    constant_value = random_state.rand()
    X = random_state.random_sample((n_samples, n_features))
    y = np.full(n_samples, constant_value, dtype=np.array(constant_value).dtype)
    expected = np.full(n_samples, constant_value, dtype=np.array(constant_value).dtype)

    for clf in [BayesianRidge(), ARDRegression()]:
        y_pred = clf.fit(X, y).predict(X)
        assert_array_almost_equal(y_pred, expected)


def test_std_bayesian_ridge_ard_with_constant_input():
    # Test BayesianRidge and ARDRegression standard dev. for edge case of
    # constant target vector
    # The standard dev. should be relatively small (< 0.01 is tested here)
    n_samples = 10
    n_features = 5
    random_state = check_random_state(42)
    constant_value = random_state.rand()
    X = random_state.random_sample((n_samples, n_features))
    y = np.full(n_samples, constant_value, dtype=np.array(constant_value).dtype)
    expected_upper_boundary = 0.01

    for clf in [BayesianRidge(), ARDRegression()]:
        _, y_std = clf.fit(X, y).predict(X, return_std=True)
        assert_array_less(y_std, expected_upper_boundary)


def test_update_of_sigma_in_ard():
    # Checks that `sigma_` is updated correctly after the last iteration
    # of the ARDRegression algorithm. See issue #10128.
    X = np.array([[1, 0], [0, 0]])
    y = np.array([0, 0])
    clf = ARDRegression(n_iter=1)
    clf.fit(X, y)
    # With the inputs above, ARDRegression prunes both of the two coefficients
    # in the first iteration. Hence, the expected shape of `sigma_` is (0, 0).
    assert clf.sigma_.shape == (0, 0)
    # Ensure that no error is thrown at prediction stage
    clf.predict(X, return_std=True)


def test_toy_ard_object():
    # Test BayesianRegression ARD classifier
    X = np.array([[1], [2], [3]])
    Y = np.array([1, 2, 3])
    clf = ARDRegression(compute_score=True)
    clf.fit(X, Y)

    # Check that the model could approximately learn the identity function
    test = [[1], [3], [4]]
    assert_array_almost_equal(clf.predict(test), [1, 3, 4], 2)


@pytest.mark.parametrize("seed", range(100))
@pytest.mark.parametrize("n_samples, n_features", ((10, 100), (100, 10)))
def test_ard_accuracy_on_easy_problem(seed, n_samples, n_features):
    # Check that ARD converges with reasonable accuracy on an easy problem
    # (Github issue #14055)
    X = np.random.RandomState(seed=seed).normal(size=(250, 3))
    y = X[:, 1]

    regressor = ARDRegression()
    regressor.fit(X, y)

    abs_coef_error = np.abs(1 - regressor.coef_[1])
    assert abs_coef_error < 1e-10


def test_return_std():
    # Test return_std option for both Bayesian regressors
    def f(X):
        return np.dot(X, w) + b

    def f_noise(X, noise_mult):
        return f(X) + np.random.randn(X.shape[0]) * noise_mult

    d = 5
    n_train = 50
    n_test = 10

    w = np.array([1.0, 0.0, 1.0, -1.0, 0.0])
    b = 1.0

    X = np.random.random((n_train, d))
    X_test = np.random.random((n_test, d))

    for decimal, noise_mult in enumerate([1, 0.1, 0.01]):
        y = f_noise(X, noise_mult)

        m1 = BayesianRidge()
        m1.fit(X, y)
        y_mean1, y_std1 = m1.predict(X_test, return_std=True)
        assert_array_almost_equal(y_std1, noise_mult, decimal=decimal)

        m2 = ARDRegression()
        m2.fit(X, y)
        y_mean2, y_std2 = m2.predict(X_test, return_std=True)
        assert_array_almost_equal(y_std2, noise_mult, decimal=decimal)


@pytest.mark.parametrize("seed", range(10))
def test_update_sigma(seed):
    # make sure the two update_sigma() helpers are equivalent. The woodbury
    # formula is used when n_samples < n_features, and the other one is used
    # otherwise.

    rng = np.random.RandomState(seed)

    # set n_samples == n_features to avoid instability issues when inverting
    # the matrices. Using the woodbury formula would be unstable when
    # n_samples > n_features
    n_samples = n_features = 10
    X = rng.randn(n_samples, n_features)
    alpha = 1
    lmbda = np.arange(1, n_features + 1)
    keep_lambda = np.array([True] * n_features)

    reg = ARDRegression()

    sigma = reg._update_sigma(X, alpha, lmbda, keep_lambda)
    sigma_woodbury = reg._update_sigma_woodbury(X, alpha, lmbda, keep_lambda)

    np.testing.assert_allclose(sigma, sigma_woodbury)


# FIXME: 'normalize' to be removed in 1.2 in LinearRegression
@pytest.mark.filterwarnings("ignore:'normalize' was deprecated")
def test_ard_regression_predict_normalize_true():
    """Check that we can predict with `normalize=True` and `return_std=True`.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/18605
    """
    clf = ARDRegression(normalize=True)
    clf.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
    clf.predict([[1, 1]], return_std=True)
