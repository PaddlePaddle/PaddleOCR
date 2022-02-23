# Authors: Christian Lorentzen <lorentzen.ch@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from numpy.testing import assert_allclose
import pytest
import warnings

from sklearn.datasets import make_regression
from sklearn.linear_model._glm import GeneralizedLinearRegressor
from sklearn.linear_model import TweedieRegressor, PoissonRegressor, GammaRegressor
from sklearn.linear_model._glm.link import (
    IdentityLink,
    LogLink,
)
from sklearn._loss.glm_distribution import (
    TweedieDistribution,
    NormalDistribution,
    PoissonDistribution,
    GammaDistribution,
    InverseGaussianDistribution,
)
from sklearn.linear_model import Ridge
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split


@pytest.fixture(scope="module")
def regression_data():
    X, y = make_regression(
        n_samples=107, n_features=10, n_informative=80, noise=0.5, random_state=2
    )
    return X, y


def test_sample_weights_validation():
    """Test the raised errors in the validation of sample_weight."""
    # scalar value but not positive
    X = [[1]]
    y = [1]
    weights = 0
    glm = GeneralizedLinearRegressor()

    # Positive weights are accepted
    glm.fit(X, y, sample_weight=1)

    # 2d array
    weights = [[0]]
    with pytest.raises(ValueError, match="must be 1D array or scalar"):
        glm.fit(X, y, weights)

    # 1d but wrong length
    weights = [1, 0]
    msg = r"sample_weight.shape == \(2,\), expected \(1,\)!"
    with pytest.raises(ValueError, match=msg):
        glm.fit(X, y, weights)


@pytest.mark.parametrize(
    "name, instance",
    [
        ("normal", NormalDistribution()),
        ("poisson", PoissonDistribution()),
        ("gamma", GammaDistribution()),
        ("inverse-gaussian", InverseGaussianDistribution()),
    ],
)
def test_glm_family_argument(name, instance):
    """Test GLM family argument set as string."""
    y = np.array([0.1, 0.5])  # in range of all distributions
    X = np.array([[1], [2]])
    glm = GeneralizedLinearRegressor(family=name, alpha=0).fit(X, y)
    assert isinstance(glm._family_instance, instance.__class__)

    glm = GeneralizedLinearRegressor(family="not a family")
    with pytest.raises(ValueError, match="family must be"):
        glm.fit(X, y)


@pytest.mark.parametrize(
    "name, instance", [("identity", IdentityLink()), ("log", LogLink())]
)
def test_glm_link_argument(name, instance):
    """Test GLM link argument set as string."""
    y = np.array([0.1, 0.5])  # in range of all distributions
    X = np.array([[1], [2]])
    glm = GeneralizedLinearRegressor(family="normal", link=name).fit(X, y)
    assert isinstance(glm._link_instance, instance.__class__)

    glm = GeneralizedLinearRegressor(family="normal", link="not a link")
    with pytest.raises(ValueError, match="link must be"):
        glm.fit(X, y)


@pytest.mark.parametrize(
    "family, expected_link_class",
    [
        ("normal", IdentityLink),
        ("poisson", LogLink),
        ("gamma", LogLink),
        ("inverse-gaussian", LogLink),
    ],
)
def test_glm_link_auto(family, expected_link_class):
    # Make sure link='auto' delivers the expected link function
    y = np.array([0.1, 0.5])  # in range of all distributions
    X = np.array([[1], [2]])
    glm = GeneralizedLinearRegressor(family=family, link="auto").fit(X, y)
    assert isinstance(glm._link_instance, expected_link_class)


@pytest.mark.parametrize("alpha", ["not a number", -4.2])
def test_glm_alpha_argument(alpha):
    """Test GLM for invalid alpha argument."""
    y = np.array([1, 2])
    X = np.array([[1], [2]])
    glm = GeneralizedLinearRegressor(family="normal", alpha=alpha)
    with pytest.raises(ValueError, match="Penalty term must be a non-negative"):
        glm.fit(X, y)


@pytest.mark.parametrize("fit_intercept", ["not bool", 1, 0, [True]])
def test_glm_fit_intercept_argument(fit_intercept):
    """Test GLM for invalid fit_intercept argument."""
    y = np.array([1, 2])
    X = np.array([[1], [1]])
    glm = GeneralizedLinearRegressor(fit_intercept=fit_intercept)
    with pytest.raises(ValueError, match="fit_intercept must be bool"):
        glm.fit(X, y)


@pytest.mark.parametrize("solver", ["not a solver", 1, [1]])
def test_glm_solver_argument(solver):
    """Test GLM for invalid solver argument."""
    y = np.array([1, 2])
    X = np.array([[1], [2]])
    glm = GeneralizedLinearRegressor(solver=solver)
    with pytest.raises(ValueError):
        glm.fit(X, y)


@pytest.mark.parametrize("max_iter", ["not a number", 0, -1, 5.5, [1]])
def test_glm_max_iter_argument(max_iter):
    """Test GLM for invalid max_iter argument."""
    y = np.array([1, 2])
    X = np.array([[1], [2]])
    glm = GeneralizedLinearRegressor(max_iter=max_iter)
    with pytest.raises(ValueError, match="must be a positive integer"):
        glm.fit(X, y)


@pytest.mark.parametrize("tol", ["not a number", 0, -1.0, [1e-3]])
def test_glm_tol_argument(tol):
    """Test GLM for invalid tol argument."""
    y = np.array([1, 2])
    X = np.array([[1], [2]])
    glm = GeneralizedLinearRegressor(tol=tol)
    with pytest.raises(ValueError, match="stopping criteria must be positive"):
        glm.fit(X, y)


@pytest.mark.parametrize("warm_start", ["not bool", 1, 0, [True]])
def test_glm_warm_start_argument(warm_start):
    """Test GLM for invalid warm_start argument."""
    y = np.array([1, 2])
    X = np.array([[1], [1]])
    glm = GeneralizedLinearRegressor(warm_start=warm_start)
    with pytest.raises(ValueError, match="warm_start must be bool"):
        glm.fit(X, y)


@pytest.mark.parametrize("fit_intercept", [False, True])
def test_glm_identity_regression(fit_intercept):
    """Test GLM regression with identity link on a simple dataset."""
    coef = [1.0, 2.0]
    X = np.array([[1, 1, 1, 1, 1], [0, 1, 2, 3, 4]]).T
    y = np.dot(X, coef)
    glm = GeneralizedLinearRegressor(
        alpha=0,
        family="normal",
        link="identity",
        fit_intercept=fit_intercept,
        tol=1e-12,
    )
    if fit_intercept:
        glm.fit(X[:, 1:], y)
        assert_allclose(glm.coef_, coef[1:], rtol=1e-10)
        assert_allclose(glm.intercept_, coef[0], rtol=1e-10)
    else:
        glm.fit(X, y)
        assert_allclose(glm.coef_, coef, rtol=1e-12)


@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("alpha", [0.0, 1.0])
@pytest.mark.parametrize("family", ["normal", "poisson", "gamma"])
def test_glm_sample_weight_consistentcy(fit_intercept, alpha, family):
    """Test that the impact of sample_weight is consistent"""
    rng = np.random.RandomState(0)
    n_samples, n_features = 10, 5

    X = rng.rand(n_samples, n_features)
    y = rng.rand(n_samples)
    glm_params = dict(
        alpha=alpha, family=family, link="auto", fit_intercept=fit_intercept
    )

    glm = GeneralizedLinearRegressor(**glm_params).fit(X, y)
    coef = glm.coef_.copy()

    # sample_weight=np.ones(..) should be equivalent to sample_weight=None
    sample_weight = np.ones(y.shape)
    glm.fit(X, y, sample_weight=sample_weight)
    assert_allclose(glm.coef_, coef, rtol=1e-12)

    # sample_weight are normalized to 1 so, scaling them has no effect
    sample_weight = 2 * np.ones(y.shape)
    glm.fit(X, y, sample_weight=sample_weight)
    assert_allclose(glm.coef_, coef, rtol=1e-12)

    # setting one element of sample_weight to 0 is equivalent to removing
    # the corresponding sample
    sample_weight = np.ones(y.shape)
    sample_weight[-1] = 0
    glm.fit(X, y, sample_weight=sample_weight)
    coef1 = glm.coef_.copy()
    glm.fit(X[:-1], y[:-1])
    assert_allclose(glm.coef_, coef1, rtol=1e-12)

    # check that multiplying sample_weight by 2 is equivalent
    # to repeating corresponding samples twice
    X2 = np.concatenate([X, X[: n_samples // 2]], axis=0)
    y2 = np.concatenate([y, y[: n_samples // 2]])
    sample_weight_1 = np.ones(len(y))
    sample_weight_1[: n_samples // 2] = 2

    glm1 = GeneralizedLinearRegressor(**glm_params).fit(
        X, y, sample_weight=sample_weight_1
    )

    glm2 = GeneralizedLinearRegressor(**glm_params).fit(X2, y2, sample_weight=None)
    assert_allclose(glm1.coef_, glm2.coef_)


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize(
    "family",
    [
        NormalDistribution(),
        PoissonDistribution(),
        GammaDistribution(),
        InverseGaussianDistribution(),
        TweedieDistribution(power=1.5),
        TweedieDistribution(power=4.5),
    ],
)
def test_glm_log_regression(fit_intercept, family):
    """Test GLM regression with log link on a simple dataset."""
    coef = [0.2, -0.1]
    X = np.array([[1, 1, 1, 1, 1], [0, 1, 2, 3, 4]]).T
    y = np.exp(np.dot(X, coef))
    glm = GeneralizedLinearRegressor(
        alpha=0, family=family, link="log", fit_intercept=fit_intercept, tol=1e-7
    )
    if fit_intercept:
        res = glm.fit(X[:, 1:], y)
        assert_allclose(res.coef_, coef[1:], rtol=1e-6)
        assert_allclose(res.intercept_, coef[0], rtol=1e-6)
    else:
        res = glm.fit(X, y)
        assert_allclose(res.coef_, coef, rtol=2e-6)


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_warm_start(fit_intercept):
    n_samples, n_features = 110, 10
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features - 2,
        noise=0.5,
        random_state=42,
    )

    glm1 = GeneralizedLinearRegressor(
        warm_start=False, fit_intercept=fit_intercept, max_iter=1000
    )
    glm1.fit(X, y)

    glm2 = GeneralizedLinearRegressor(
        warm_start=True, fit_intercept=fit_intercept, max_iter=1
    )
    # As we intentionally set max_iter=1, L-BFGS-B will issue a
    # ConvergenceWarning which we here simply ignore.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        glm2.fit(X, y)
    assert glm1.score(X, y) > glm2.score(X, y)
    glm2.set_params(max_iter=1000)
    glm2.fit(X, y)
    # The two model are not exactly identical since the lbfgs solver
    # computes the approximate hessian from previous iterations, which
    # will not be strictly identical in the case of a warm start.
    assert_allclose(glm1.coef_, glm2.coef_, rtol=1e-5)
    assert_allclose(glm1.score(X, y), glm2.score(X, y), rtol=1e-4)


# FIXME: 'normalize' to be removed in 1.2 in LinearRegression
@pytest.mark.filterwarnings("ignore:'normalize' was deprecated")
@pytest.mark.parametrize("n_samples, n_features", [(100, 10), (10, 100)])
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("sample_weight", [None, True])
def test_normal_ridge_comparison(
    n_samples, n_features, fit_intercept, sample_weight, request
):
    """Compare with Ridge regression for Normal distributions."""
    test_size = 10
    X, y = make_regression(
        n_samples=n_samples + test_size,
        n_features=n_features,
        n_informative=n_features - 2,
        noise=0.5,
        random_state=42,
    )

    if n_samples > n_features:
        ridge_params = {"solver": "svd"}
    else:
        ridge_params = {"solver": "saga", "max_iter": 1000000, "tol": 1e-7}

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(X, y, test_size=test_size, random_state=0)

    alpha = 1.0
    if sample_weight is None:
        sw_train = None
        alpha_ridge = alpha * n_samples
    else:
        sw_train = np.random.RandomState(0).rand(len(y_train))
        alpha_ridge = alpha * sw_train.sum()

    # GLM has 1/(2*n) * Loss + 1/2*L2, Ridge has Loss + L2
    ridge = Ridge(
        alpha=alpha_ridge,
        normalize=False,
        random_state=42,
        fit_intercept=fit_intercept,
        **ridge_params,
    )
    ridge.fit(X_train, y_train, sample_weight=sw_train)

    glm = GeneralizedLinearRegressor(
        alpha=alpha,
        family="normal",
        link="identity",
        fit_intercept=fit_intercept,
        max_iter=300,
        tol=1e-5,
    )
    glm.fit(X_train, y_train, sample_weight=sw_train)
    assert glm.coef_.shape == (X.shape[1],)
    assert_allclose(glm.coef_, ridge.coef_, atol=5e-5)
    assert_allclose(glm.intercept_, ridge.intercept_, rtol=1e-5)
    assert_allclose(glm.predict(X_train), ridge.predict(X_train), rtol=2e-4)
    assert_allclose(glm.predict(X_test), ridge.predict(X_test), rtol=2e-4)


def test_poisson_glmnet():
    """Compare Poisson regression with L2 regularization and LogLink to glmnet"""
    # library("glmnet")
    # options(digits=10)
    # df <- data.frame(a=c(-2,-1,1,2), b=c(0,0,1,1), y=c(0,1,1,2))
    # x <- data.matrix(df[,c("a", "b")])
    # y <- df$y
    # fit <- glmnet(x=x, y=y, alpha=0, intercept=T, family="poisson",
    #               standardize=F, thresh=1e-10, nlambda=10000)
    # coef(fit, s=1)
    # (Intercept) -0.12889386979
    # a            0.29019207995
    # b            0.03741173122
    X = np.array([[-2, -1, 1, 2], [0, 0, 1, 1]]).T
    y = np.array([0, 1, 1, 2])
    glm = GeneralizedLinearRegressor(
        alpha=1,
        fit_intercept=True,
        family="poisson",
        link="log",
        tol=1e-7,
        max_iter=300,
    )
    glm.fit(X, y)
    assert_allclose(glm.intercept_, -0.12889386979, rtol=1e-5)
    assert_allclose(glm.coef_, [0.29019207995, 0.03741173122], rtol=1e-5)


def test_convergence_warning(regression_data):
    X, y = regression_data

    est = GeneralizedLinearRegressor(max_iter=1, tol=1e-20)
    with pytest.warns(ConvergenceWarning):
        est.fit(X, y)


def test_poisson_regression_family(regression_data):
    # Make sure the family attribute is read-only to prevent searching over it
    # e.g. in a grid search
    est = PoissonRegressor()
    est.family == "poisson"

    msg = "PoissonRegressor.family must be 'poisson'!"
    with pytest.raises(ValueError, match=msg):
        est.family = 0


def test_gamma_regression_family(regression_data):
    # Make sure the family attribute is read-only to prevent searching over it
    # e.g. in a grid search
    est = GammaRegressor()
    est.family == "gamma"

    msg = "GammaRegressor.family must be 'gamma'!"
    with pytest.raises(ValueError, match=msg):
        est.family = 0


def test_tweedie_regression_family(regression_data):
    # Make sure the family attribute is always a TweedieDistribution and that
    # the power attribute is properly updated
    power = 2.0
    est = TweedieRegressor(power=power)
    assert isinstance(est.family, TweedieDistribution)
    assert est.family.power == power
    assert est.power == power

    new_power = 0
    new_family = TweedieDistribution(power=new_power)
    est.family = new_family
    assert isinstance(est.family, TweedieDistribution)
    assert est.family.power == new_power
    assert est.power == new_power

    msg = "TweedieRegressor.family must be of type TweedieDistribution!"
    with pytest.raises(TypeError, match=msg):
        est.family = None


@pytest.mark.parametrize(
    "estimator, value",
    [
        (PoissonRegressor(), True),
        (GammaRegressor(), True),
        (TweedieRegressor(power=1.5), True),
        (TweedieRegressor(power=0), False),
    ],
)
def test_tags(estimator, value):
    assert estimator._get_tags()["requires_positive_y"] is value
