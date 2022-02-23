# Authors: David Dale <dale.david@mail.ru>
#          Christian Lorentzen <lorentzen.ch@gmail.com>
# License: BSD 3 clause

import numpy as np
import pytest
from pytest import approx
from scipy.optimize import minimize

from sklearn.datasets import make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor, QuantileRegressor
from sklearn.metrics import mean_pinball_loss
from sklearn.utils._testing import assert_allclose
from sklearn.utils.fixes import parse_version, sp_version


@pytest.fixture
def X_y_data():
    X, y = make_regression(n_samples=10, n_features=1, random_state=0, noise=1)
    return X, y


@pytest.mark.parametrize(
    "params, err_msg",
    [
        ({"quantile": 2}, "Quantile should be strictly between 0.0 and 1.0"),
        ({"quantile": 1}, "Quantile should be strictly between 0.0 and 1.0"),
        ({"quantile": 0}, "Quantile should be strictly between 0.0 and 1.0"),
        ({"quantile": -1}, "Quantile should be strictly between 0.0 and 1.0"),
        ({"alpha": -1.5}, "Penalty alpha must be a non-negative number"),
        ({"fit_intercept": "blah"}, "The argument fit_intercept must be bool"),
        ({"fit_intercept": 0}, "The argument fit_intercept must be bool"),
        ({"solver": "blah"}, "Invalid value for argument solver"),
        (
            {"solver_options": "blah"},
            "Invalid value for argument solver_options",
        ),
    ],
)
def test_init_parameters_validation(X_y_data, params, err_msg):
    """Test that invalid init parameters raise errors."""
    X, y = X_y_data
    with pytest.raises(ValueError, match=err_msg):
        QuantileRegressor(**params).fit(X, y)


@pytest.mark.parametrize("solver", ("highs-ds", "highs-ipm", "highs"))
@pytest.mark.skipif(
    sp_version >= parse_version("1.6.0"),
    reason="Solvers are available as of scipy 1.6.0",
)
def test_too_new_solver_methods_raise_error(X_y_data, solver):
    """Test that highs solver raises for scipy<1.6.0."""
    X, y = X_y_data
    with pytest.raises(ValueError, match="scipy>=1.6.0"):
        QuantileRegressor(solver=solver).fit(X, y)


@pytest.mark.parametrize(
    "quantile, alpha, intercept, coef",
    [
        # for 50% quantile w/o regularization, any slope in [1, 10] is okay
        [0.5, 0, 1, None],
        # if positive error costs more, the slope is maximal
        [0.51, 0, 1, 10],
        # if negative error costs more, the slope is minimal
        [0.49, 0, 1, 1],
        # for a small lasso penalty, the slope is also minimal
        [0.5, 0.01, 1, 1],
        # for a large lasso penalty, the model predicts the constant median
        [0.5, 100, 2, 0],
    ],
)
def test_quantile_toy_example(quantile, alpha, intercept, coef):
    # test how different parameters affect a small intuitive example
    X = [[0], [1], [1]]
    y = [1, 2, 11]
    model = QuantileRegressor(quantile=quantile, alpha=alpha).fit(X, y)
    assert_allclose(model.intercept_, intercept, atol=1e-2)
    if coef is not None:
        assert_allclose(model.coef_[0], coef, atol=1e-2)
    if alpha < 100:
        assert model.coef_[0] >= 1
    assert model.coef_[0] <= 10


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_quantile_equals_huber_for_low_epsilon(fit_intercept):
    X, y = make_regression(n_samples=100, n_features=20, random_state=0, noise=1.0)
    alpha = 1e-4
    huber = HuberRegressor(
        epsilon=1 + 1e-4, alpha=alpha, fit_intercept=fit_intercept
    ).fit(X, y)
    quant = QuantileRegressor(alpha=alpha, fit_intercept=fit_intercept).fit(X, y)
    assert_allclose(huber.coef_, quant.coef_, atol=1e-1)
    if fit_intercept:
        assert huber.intercept_ == approx(quant.intercept_, abs=1e-1)
        # check that we still predict fraction
        assert np.mean(y < quant.predict(X)) == approx(0.5, abs=1e-1)


@pytest.mark.parametrize("q", [0.5, 0.9, 0.05])
def test_quantile_estimates_calibration(q):
    # Test that model estimates percentage of points below the prediction
    X, y = make_regression(n_samples=1000, n_features=20, random_state=0, noise=1.0)
    quant = QuantileRegressor(
        quantile=q,
        alpha=0,
        solver_options={"lstsq": False},
    ).fit(X, y)
    assert np.mean(y < quant.predict(X)) == approx(q, abs=1e-2)


def test_quantile_sample_weight():
    # test that with unequal sample weights we still estimate weighted fraction
    n = 1000
    X, y = make_regression(n_samples=n, n_features=5, random_state=0, noise=10.0)
    weight = np.ones(n)
    # when we increase weight of upper observations,
    # estimate of quantile should go up
    weight[y > y.mean()] = 100
    quant = QuantileRegressor(quantile=0.5, alpha=1e-8, solver_options={"lstsq": False})
    quant.fit(X, y, sample_weight=weight)
    fraction_below = np.mean(y < quant.predict(X))
    assert fraction_below > 0.5
    weighted_fraction_below = np.average(y < quant.predict(X), weights=weight)
    assert weighted_fraction_below == approx(0.5, abs=3e-2)


@pytest.mark.skipif(
    sp_version < parse_version("1.6.0"),
    reason="The `highs` solver is available from the 1.6.0 scipy version",
)
@pytest.mark.parametrize("quantile", [0.2, 0.5, 0.8])
def test_asymmetric_error(quantile):
    """Test quantile regression for asymmetric distributed targets."""
    n_samples = 1000
    rng = np.random.RandomState(42)
    X = np.concatenate(
        (
            np.abs(rng.randn(n_samples)[:, None]),
            -rng.randint(2, size=(n_samples, 1)),
        ),
        axis=1,
    )
    intercept = 1.23
    coef = np.array([0.5, -2])
    #  Take care that X @ coef + intercept > 0
    assert np.min(X @ coef + intercept) > 0
    # For an exponential distribution with rate lambda, e.g. exp(-lambda * x),
    # the quantile at level q is:
    #   quantile(q) = - log(1 - q) / lambda
    #   scale = 1/lambda = -quantile(q) / log(1 - q)
    y = rng.exponential(
        scale=-(X @ coef + intercept) / np.log(1 - quantile), size=n_samples
    )
    model = QuantileRegressor(
        quantile=quantile,
        alpha=0,
        solver="highs",
    ).fit(X, y)
    # This test can be made to pass with any solver but in the interest
    # of sparing continuous integration resources, the test is performed
    # with the fastest solver only.

    assert model.intercept_ == approx(intercept, rel=0.2)
    assert_allclose(model.coef_, coef, rtol=0.6)
    assert_allclose(np.mean(model.predict(X) > y), quantile, atol=1e-2)

    # Now compare to Nelder-Mead optimization with L1 penalty
    alpha = 0.01
    model.set_params(alpha=alpha).fit(X, y)
    model_coef = np.r_[model.intercept_, model.coef_]

    def func(coef):
        loss = mean_pinball_loss(y, X @ coef[1:] + coef[0], alpha=quantile)
        L1 = np.sum(np.abs(coef[1:]))
        return loss + alpha * L1

    res = minimize(
        fun=func,
        x0=[1, 0, -1],
        method="Nelder-Mead",
        tol=1e-12,
        options={"maxiter": 2000},
    )

    assert func(model_coef) == approx(func(res.x))
    assert_allclose(model.intercept_, res.x[0])
    assert_allclose(model.coef_, res.x[1:])
    assert_allclose(np.mean(model.predict(X) > y), quantile, atol=1e-2)


@pytest.mark.parametrize("quantile", [0.2, 0.5, 0.8])
def test_equivariance(quantile):
    """Test equivariace of quantile regression.

    See Koenker (2005) Quantile Regression, Chapter 2.2.3.
    """
    rng = np.random.RandomState(42)
    n_samples, n_features = 100, 5
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        noise=0,
        random_state=rng,
        shuffle=False,
    )
    # make y asymmetric
    y += rng.exponential(scale=100, size=y.shape)
    params = dict(alpha=0, solver_options={"lstsq": True, "tol": 1e-10})
    model1 = QuantileRegressor(quantile=quantile, **params).fit(X, y)

    # coef(q; a*y, X) = a * coef(q; y, X)
    a = 2.5
    model2 = QuantileRegressor(quantile=quantile, **params).fit(X, a * y)
    assert model2.intercept_ == approx(a * model1.intercept_, rel=1e-5)
    assert_allclose(model2.coef_, a * model1.coef_, rtol=1e-5)

    # coef(1-q; -a*y, X) = -a * coef(q; y, X)
    model2 = QuantileRegressor(quantile=1 - quantile, **params).fit(X, -a * y)
    assert model2.intercept_ == approx(-a * model1.intercept_, rel=1e-5)
    assert_allclose(model2.coef_, -a * model1.coef_, rtol=1e-5)

    # coef(q; y + X @ g, X) = coef(q; y, X) + g
    g_intercept, g_coef = rng.randn(), rng.randn(n_features)
    model2 = QuantileRegressor(quantile=quantile, **params)
    model2.fit(X, y + X @ g_coef + g_intercept)
    assert model2.intercept_ == approx(model1.intercept_ + g_intercept)
    assert_allclose(model2.coef_, model1.coef_ + g_coef, rtol=1e-6)

    # coef(q; y, X @ A) = A^-1 @ coef(q; y, X)
    A = rng.randn(n_features, n_features)
    model2 = QuantileRegressor(quantile=quantile, **params)
    model2.fit(X @ A, y)
    assert model2.intercept_ == approx(model1.intercept_, rel=1e-5)
    assert_allclose(model2.coef_, np.linalg.solve(A, model1.coef_), rtol=1e-5)


def test_linprog_failure():
    """Test that linprog fails."""
    X = np.linspace(0, 10, num=10).reshape(-1, 1)
    y = np.linspace(0, 10, num=10)
    reg = QuantileRegressor(
        alpha=0, solver="interior-point", solver_options={"maxiter": 1}
    )

    msg = "Linear programming for QuantileRegressor did not succeed."
    with pytest.warns(ConvergenceWarning, match=msg):
        reg.fit(X, y)
