import numpy as np
import scipy.sparse as sp
from scipy import linalg
from itertools import product

import pytest

from sklearn.utils import _IS_32BIT
from sklearn.utils._testing import assert_almost_equal
from sklearn.utils._testing import assert_allclose
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.estimator_checks import check_sample_weights_invariance

from sklearn.exceptions import ConvergenceWarning

from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.metrics import get_scorer

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ridge_regression
from sklearn.linear_model import Ridge
from sklearn.linear_model._ridge import _RidgeGCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model._ridge import _solve_cholesky
from sklearn.linear_model._ridge import _solve_cholesky_kernel
from sklearn.linear_model._ridge import _solve_svd
from sklearn.linear_model._ridge import _solve_lbfgs
from sklearn.linear_model._ridge import _check_gcv_mode
from sklearn.linear_model._ridge import _X_CenterStackOp
from sklearn.datasets import make_regression
from sklearn.datasets import make_classification

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import LeaveOneOut

from sklearn.preprocessing import minmax_scale
from sklearn.utils import check_random_state
from sklearn.datasets import make_multilabel_classification

diabetes = datasets.load_diabetes()
X_diabetes, y_diabetes = diabetes.data, diabetes.target
ind = np.arange(X_diabetes.shape[0])
rng = np.random.RandomState(0)
rng.shuffle(ind)
ind = ind[:200]
X_diabetes, y_diabetes = X_diabetes[ind], y_diabetes[ind]

iris = datasets.load_iris()

X_iris = sp.csr_matrix(iris.data)
y_iris = iris.target


def DENSE_FILTER(X):
    return X


def SPARSE_FILTER(X):
    return sp.csr_matrix(X)


def _accuracy_callable(y_test, y_pred):
    return np.mean(y_test == y_pred)


def _mean_squared_error_callable(y_test, y_pred):
    return ((y_test - y_pred) ** 2).mean()


@pytest.mark.parametrize("solver", ("svd", "sparse_cg", "cholesky", "lsqr", "sag"))
def test_ridge(solver):
    # Ridge regression convergence test using score
    # TODO: for this test to be robust, we should use a dataset instead
    # of np.random.
    rng = np.random.RandomState(0)
    alpha = 1.0

    # With more samples than features
    n_samples, n_features = 6, 5
    y = rng.randn(n_samples)
    X = rng.randn(n_samples, n_features)

    ridge = Ridge(alpha=alpha, solver=solver)
    ridge.fit(X, y)
    assert ridge.coef_.shape == (X.shape[1],)
    assert ridge.score(X, y) > 0.47

    if solver in ("cholesky", "sag"):
        # Currently the only solvers to support sample_weight.
        ridge.fit(X, y, sample_weight=np.ones(n_samples))
        assert ridge.score(X, y) > 0.47

    # With more features than samples
    n_samples, n_features = 5, 10
    y = rng.randn(n_samples)
    X = rng.randn(n_samples, n_features)
    ridge = Ridge(alpha=alpha, solver=solver)
    ridge.fit(X, y)
    assert ridge.score(X, y) > 0.9

    if solver in ("cholesky", "sag"):
        # Currently the only solvers to support sample_weight.
        ridge.fit(X, y, sample_weight=np.ones(n_samples))
        assert ridge.score(X, y) > 0.9


def test_primal_dual_relationship():
    y = y_diabetes.reshape(-1, 1)
    coef = _solve_cholesky(X_diabetes, y, alpha=[1e-2])
    K = np.dot(X_diabetes, X_diabetes.T)
    dual_coef = _solve_cholesky_kernel(K, y, alpha=[1e-2])
    coef2 = np.dot(X_diabetes.T, dual_coef).T
    assert_array_almost_equal(coef, coef2)


def test_ridge_singular():
    # test on a singular matrix
    rng = np.random.RandomState(0)
    n_samples, n_features = 6, 6
    y = rng.randn(n_samples // 2)
    y = np.concatenate((y, y))
    X = rng.randn(n_samples // 2, n_features)
    X = np.concatenate((X, X), axis=0)

    ridge = Ridge(alpha=0)
    ridge.fit(X, y)
    assert ridge.score(X, y) > 0.9


def test_ridge_regression_sample_weights():
    rng = np.random.RandomState(0)

    for solver in ("cholesky",):
        for n_samples, n_features in ((6, 5), (5, 10)):
            for alpha in (1.0, 1e-2):
                y = rng.randn(n_samples)
                X = rng.randn(n_samples, n_features)
                sample_weight = 1.0 + rng.rand(n_samples)

                coefs = ridge_regression(
                    X, y, alpha=alpha, sample_weight=sample_weight, solver=solver
                )

                # Sample weight can be implemented via a simple rescaling
                # for the square loss.
                coefs2 = ridge_regression(
                    X * np.sqrt(sample_weight)[:, np.newaxis],
                    y * np.sqrt(sample_weight),
                    alpha=alpha,
                    solver=solver,
                )
                assert_array_almost_equal(coefs, coefs2)


def test_ridge_regression_convergence_fail():
    rng = np.random.RandomState(0)
    y = rng.randn(5)
    X = rng.randn(5, 10)
    warning_message = r"sparse_cg did not converge after" r" [0-9]+ iterations."
    with pytest.warns(ConvergenceWarning, match=warning_message):
        ridge_regression(
            X, y, alpha=1.0, solver="sparse_cg", tol=0.0, max_iter=None, verbose=1
        )


def test_ridge_sample_weights():
    # TODO: loop over sparse data as well
    # Note: parametrizing this test with pytest results in failed
    #       assertions, meaning that is is not extremely robust

    rng = np.random.RandomState(0)
    param_grid = product(
        (1.0, 1e-2), (True, False), ("svd", "cholesky", "lsqr", "sparse_cg")
    )

    for n_samples, n_features in ((6, 5), (5, 10)):

        y = rng.randn(n_samples)
        X = rng.randn(n_samples, n_features)
        sample_weight = 1.0 + rng.rand(n_samples)

        for (alpha, intercept, solver) in param_grid:

            # Ridge with explicit sample_weight
            est = Ridge(alpha=alpha, fit_intercept=intercept, solver=solver, tol=1e-12)
            est.fit(X, y, sample_weight=sample_weight)
            coefs = est.coef_
            inter = est.intercept_

            # Closed form of the weighted regularized least square
            # theta = (X^T W X + alpha I)^(-1) * X^T W y
            W = np.diag(sample_weight)
            if intercept is False:
                X_aug = X
                D = np.eye(n_features)
            else:
                dummy_column = np.ones(shape=(n_samples, 1))
                X_aug = np.concatenate((dummy_column, X), axis=1)
                D = np.eye(n_features + 1)
                D[0, 0] = 0

            cf_coefs = linalg.solve(
                X_aug.T.dot(W).dot(X_aug) + alpha * D, X_aug.T.dot(W).dot(y)
            )

            if intercept is False:
                assert_array_almost_equal(coefs, cf_coefs)
            else:
                assert_array_almost_equal(coefs, cf_coefs[1:])
                assert_almost_equal(inter, cf_coefs[0])


def test_ridge_shapes():
    # Test shape of coef_ and intercept_
    rng = np.random.RandomState(0)
    n_samples, n_features = 5, 10
    X = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples)
    Y1 = y[:, np.newaxis]
    Y = np.c_[y, 1 + y]

    ridge = Ridge()

    ridge.fit(X, y)
    assert ridge.coef_.shape == (n_features,)
    assert ridge.intercept_.shape == ()

    ridge.fit(X, Y1)
    assert ridge.coef_.shape == (1, n_features)
    assert ridge.intercept_.shape == (1,)

    ridge.fit(X, Y)
    assert ridge.coef_.shape == (2, n_features)
    assert ridge.intercept_.shape == (2,)


def test_ridge_intercept():
    # Test intercept with multiple targets GH issue #708
    rng = np.random.RandomState(0)
    n_samples, n_features = 5, 10
    X = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples)
    Y = np.c_[y, 1.0 + y]

    ridge = Ridge()

    ridge.fit(X, y)
    intercept = ridge.intercept_

    ridge.fit(X, Y)
    assert_almost_equal(ridge.intercept_[0], intercept)
    assert_almost_equal(ridge.intercept_[1], intercept + 1.0)


def test_toy_ridge_object():
    # Test BayesianRegression ridge classifier
    # TODO: test also n_samples > n_features
    X = np.array([[1], [2]])
    Y = np.array([1, 2])
    reg = Ridge(alpha=0.0)
    reg.fit(X, Y)
    X_test = [[1], [2], [3], [4]]
    assert_almost_equal(reg.predict(X_test), [1.0, 2, 3, 4])

    assert len(reg.coef_.shape) == 1
    assert type(reg.intercept_) == np.float64

    Y = np.vstack((Y, Y)).T

    reg.fit(X, Y)
    X_test = [[1], [2], [3], [4]]

    assert len(reg.coef_.shape) == 2
    assert type(reg.intercept_) == np.ndarray


def test_ridge_vs_lstsq():
    # On alpha=0., Ridge and OLS yield the same solution.

    rng = np.random.RandomState(0)
    # we need more samples than features
    n_samples, n_features = 5, 4
    y = rng.randn(n_samples)
    X = rng.randn(n_samples, n_features)

    ridge = Ridge(alpha=0.0, fit_intercept=False)
    ols = LinearRegression(fit_intercept=False)

    ridge.fit(X, y)
    ols.fit(X, y)
    assert_almost_equal(ridge.coef_, ols.coef_)

    ridge.fit(X, y)
    ols.fit(X, y)
    assert_almost_equal(ridge.coef_, ols.coef_)


def test_ridge_individual_penalties():
    # Tests the ridge object using individual penalties

    rng = np.random.RandomState(42)

    n_samples, n_features, n_targets = 20, 10, 5
    X = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples, n_targets)

    penalties = np.arange(n_targets)

    coef_cholesky = np.array(
        [
            Ridge(alpha=alpha, solver="cholesky").fit(X, target).coef_
            for alpha, target in zip(penalties, y.T)
        ]
    )

    coefs_indiv_pen = [
        Ridge(alpha=penalties, solver=solver, tol=1e-12).fit(X, y).coef_
        for solver in ["svd", "sparse_cg", "lsqr", "cholesky", "sag", "saga"]
    ]
    for coef_indiv_pen in coefs_indiv_pen:
        assert_array_almost_equal(coef_cholesky, coef_indiv_pen)

    # Test error is raised when number of targets and penalties do not match.
    ridge = Ridge(alpha=penalties[:-1])
    with pytest.raises(ValueError):
        ridge.fit(X, y)


@pytest.mark.parametrize("n_col", [(), (1,), (3,)])
def test_X_CenterStackOp(n_col):
    rng = np.random.RandomState(0)
    X = rng.randn(11, 8)
    X_m = rng.randn(8)
    sqrt_sw = rng.randn(len(X))
    Y = rng.randn(11, *n_col)
    A = rng.randn(9, *n_col)
    operator = _X_CenterStackOp(sp.csr_matrix(X), X_m, sqrt_sw)
    reference_operator = np.hstack([X - sqrt_sw[:, None] * X_m, sqrt_sw[:, None]])
    assert_allclose(reference_operator.dot(A), operator.dot(A))
    assert_allclose(reference_operator.T.dot(Y), operator.T.dot(Y))


@pytest.mark.parametrize("shape", [(10, 1), (13, 9), (3, 7), (2, 2), (20, 20)])
@pytest.mark.parametrize("uniform_weights", [True, False])
def test_compute_gram(shape, uniform_weights):
    rng = np.random.RandomState(0)
    X = rng.randn(*shape)
    if uniform_weights:
        sw = np.ones(X.shape[0])
    else:
        sw = rng.chisquare(1, shape[0])
    sqrt_sw = np.sqrt(sw)
    X_mean = np.average(X, axis=0, weights=sw)
    X_centered = (X - X_mean) * sqrt_sw[:, None]
    true_gram = X_centered.dot(X_centered.T)
    X_sparse = sp.csr_matrix(X * sqrt_sw[:, None])
    gcv = _RidgeGCV(fit_intercept=True)
    computed_gram, computed_mean = gcv._compute_gram(X_sparse, sqrt_sw)
    assert_allclose(X_mean, computed_mean)
    assert_allclose(true_gram, computed_gram)


@pytest.mark.parametrize("shape", [(10, 1), (13, 9), (3, 7), (2, 2), (20, 20)])
@pytest.mark.parametrize("uniform_weights", [True, False])
def test_compute_covariance(shape, uniform_weights):
    rng = np.random.RandomState(0)
    X = rng.randn(*shape)
    if uniform_weights:
        sw = np.ones(X.shape[0])
    else:
        sw = rng.chisquare(1, shape[0])
    sqrt_sw = np.sqrt(sw)
    X_mean = np.average(X, axis=0, weights=sw)
    X_centered = (X - X_mean) * sqrt_sw[:, None]
    true_covariance = X_centered.T.dot(X_centered)
    X_sparse = sp.csr_matrix(X * sqrt_sw[:, None])
    gcv = _RidgeGCV(fit_intercept=True)
    computed_cov, computed_mean = gcv._compute_covariance(X_sparse, sqrt_sw)
    assert_allclose(X_mean, computed_mean)
    assert_allclose(true_covariance, computed_cov)


def _make_sparse_offset_regression(
    n_samples=100,
    n_features=100,
    proportion_nonzero=0.5,
    n_informative=10,
    n_targets=1,
    bias=13.0,
    X_offset=30.0,
    noise=30.0,
    shuffle=True,
    coef=False,
    positive=False,
    random_state=None,
):
    X, y, c = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_targets=n_targets,
        bias=bias,
        noise=noise,
        shuffle=shuffle,
        coef=True,
        random_state=random_state,
    )
    if n_features == 1:
        c = np.asarray([c])
    X += X_offset
    mask = (
        np.random.RandomState(random_state).binomial(1, proportion_nonzero, X.shape) > 0
    )
    removed_X = X.copy()
    X[~mask] = 0.0
    removed_X[mask] = 0.0
    y -= removed_X.dot(c)
    if positive:
        y += X.dot(np.abs(c) + 1 - c)
        c = np.abs(c) + 1
    if n_features == 1:
        c = c[0]
    if coef:
        return X, y, c
    return X, y


# FIXME: 'normalize' to be removed in 1.2
@pytest.mark.filterwarnings("ignore:'normalize' was deprecated")
@pytest.mark.parametrize(
    "solver, sparse_X",
    (
        (solver, sparse_X)
        for (solver, sparse_X) in product(
            ["cholesky", "sag", "sparse_cg", "lsqr", "saga", "ridgecv"],
            [False, True],
        )
        if not (sparse_X and solver not in ["sparse_cg", "ridgecv"])
    ),
)
@pytest.mark.parametrize(
    "n_samples,dtype,proportion_nonzero",
    [(20, "float32", 0.1), (40, "float32", 1.0), (20, "float64", 0.2)],
)
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("seed", np.arange(3))
def test_solver_consistency(
    solver, proportion_nonzero, n_samples, dtype, sparse_X, seed, normalize
):
    alpha = 1.0
    noise = 50.0 if proportion_nonzero > 0.9 else 500.0
    X, y = _make_sparse_offset_regression(
        bias=10,
        n_features=30,
        proportion_nonzero=proportion_nonzero,
        noise=noise,
        random_state=seed,
        n_samples=n_samples,
    )
    if not normalize:
        # Manually scale the data to avoid pathological cases. We use
        # minmax_scale to deal with the sparse case without breaking
        # the sparsity pattern.
        X = minmax_scale(X)
    svd_ridge = Ridge(solver="svd", normalize=normalize, alpha=alpha).fit(X, y)
    X = X.astype(dtype, copy=False)
    y = y.astype(dtype, copy=False)
    if sparse_X:
        X = sp.csr_matrix(X)
    if solver == "ridgecv":
        ridge = RidgeCV(alphas=[alpha], normalize=normalize)
    else:
        ridge = Ridge(solver=solver, tol=1e-10, normalize=normalize, alpha=alpha)
    ridge.fit(X, y)
    assert_allclose(ridge.coef_, svd_ridge.coef_, atol=1e-3, rtol=1e-3)
    assert_allclose(ridge.intercept_, svd_ridge.intercept_, atol=1e-3, rtol=1e-3)


# FIXME: 'normalize' to be removed in 1.2
@pytest.mark.filterwarnings("ignore:'normalize' was deprecated")
@pytest.mark.parametrize("gcv_mode", ["svd", "eigen"])
@pytest.mark.parametrize("X_constructor", [np.asarray, sp.csr_matrix])
@pytest.mark.parametrize("X_shape", [(11, 8), (11, 20)])
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize(
    "y_shape, normalize, noise",
    [
        ((11,), True, 1.0),
        ((11, 1), False, 30.0),
        ((11, 3), False, 150.0),
    ],
)
def test_ridge_gcv_vs_ridge_loo_cv(
    gcv_mode, X_constructor, X_shape, y_shape, fit_intercept, normalize, noise
):
    n_samples, n_features = X_shape
    n_targets = y_shape[-1] if len(y_shape) == 2 else 1
    X, y = _make_sparse_offset_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_targets=n_targets,
        random_state=0,
        shuffle=False,
        noise=noise,
        n_informative=5,
    )
    y = y.reshape(y_shape)

    alphas = [1e-3, 0.1, 1.0, 10.0, 1e3]
    loo_ridge = RidgeCV(
        cv=n_samples,
        fit_intercept=fit_intercept,
        alphas=alphas,
        scoring="neg_mean_squared_error",
        normalize=normalize,
    )
    gcv_ridge = RidgeCV(
        gcv_mode=gcv_mode,
        fit_intercept=fit_intercept,
        alphas=alphas,
        normalize=normalize,
    )

    loo_ridge.fit(X, y)

    X_gcv = X_constructor(X)
    gcv_ridge.fit(X_gcv, y)

    assert gcv_ridge.alpha_ == pytest.approx(loo_ridge.alpha_)
    assert_allclose(gcv_ridge.coef_, loo_ridge.coef_, rtol=1e-3)
    assert_allclose(gcv_ridge.intercept_, loo_ridge.intercept_, rtol=1e-3)


def test_ridge_loo_cv_asym_scoring():
    # checking on asymmetric scoring
    scoring = "explained_variance"
    n_samples, n_features = 10, 5
    n_targets = 1
    X, y = _make_sparse_offset_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_targets=n_targets,
        random_state=0,
        shuffle=False,
        noise=1,
        n_informative=5,
    )

    alphas = [1e-3, 0.1, 1.0, 10.0, 1e3]
    loo_ridge = RidgeCV(
        cv=n_samples, fit_intercept=True, alphas=alphas, scoring=scoring
    )

    gcv_ridge = RidgeCV(fit_intercept=True, alphas=alphas, scoring=scoring)

    loo_ridge.fit(X, y)
    gcv_ridge.fit(X, y)

    assert gcv_ridge.alpha_ == pytest.approx(loo_ridge.alpha_)
    assert_allclose(gcv_ridge.coef_, loo_ridge.coef_, rtol=1e-3)
    assert_allclose(gcv_ridge.intercept_, loo_ridge.intercept_, rtol=1e-3)


@pytest.mark.parametrize("gcv_mode", ["svd", "eigen"])
@pytest.mark.parametrize("X_constructor", [np.asarray, sp.csr_matrix])
@pytest.mark.parametrize("n_features", [8, 20])
@pytest.mark.parametrize(
    "y_shape, fit_intercept, noise",
    [
        ((11,), True, 1.0),
        ((11, 1), True, 20.0),
        ((11, 3), True, 150.0),
        ((11, 3), False, 30.0),
    ],
)
def test_ridge_gcv_sample_weights(
    gcv_mode, X_constructor, fit_intercept, n_features, y_shape, noise
):
    alphas = [1e-3, 0.1, 1.0, 10.0, 1e3]
    rng = np.random.RandomState(0)
    n_targets = y_shape[-1] if len(y_shape) == 2 else 1
    X, y = _make_sparse_offset_regression(
        n_samples=11,
        n_features=n_features,
        n_targets=n_targets,
        random_state=0,
        shuffle=False,
        noise=noise,
    )
    y = y.reshape(y_shape)

    sample_weight = 3 * rng.randn(len(X))
    sample_weight = (sample_weight - sample_weight.min() + 1).astype(int)
    indices = np.repeat(np.arange(X.shape[0]), sample_weight)
    sample_weight = sample_weight.astype(float)
    X_tiled, y_tiled = X[indices], y[indices]

    cv = GroupKFold(n_splits=X.shape[0])
    splits = cv.split(X_tiled, y_tiled, groups=indices)
    kfold = RidgeCV(
        alphas=alphas,
        cv=splits,
        scoring="neg_mean_squared_error",
        fit_intercept=fit_intercept,
    )
    kfold.fit(X_tiled, y_tiled)

    ridge_reg = Ridge(alpha=kfold.alpha_, fit_intercept=fit_intercept)
    splits = cv.split(X_tiled, y_tiled, groups=indices)
    predictions = cross_val_predict(ridge_reg, X_tiled, y_tiled, cv=splits)
    kfold_errors = (y_tiled - predictions) ** 2
    kfold_errors = [
        np.sum(kfold_errors[indices == i], axis=0) for i in np.arange(X.shape[0])
    ]
    kfold_errors = np.asarray(kfold_errors)

    X_gcv = X_constructor(X)
    gcv_ridge = RidgeCV(
        alphas=alphas,
        store_cv_values=True,
        gcv_mode=gcv_mode,
        fit_intercept=fit_intercept,
    )
    gcv_ridge.fit(X_gcv, y, sample_weight=sample_weight)
    if len(y_shape) == 2:
        gcv_errors = gcv_ridge.cv_values_[:, :, alphas.index(kfold.alpha_)]
    else:
        gcv_errors = gcv_ridge.cv_values_[:, alphas.index(kfold.alpha_)]

    assert kfold.alpha_ == pytest.approx(gcv_ridge.alpha_)
    assert_allclose(gcv_errors, kfold_errors, rtol=1e-3)
    assert_allclose(gcv_ridge.coef_, kfold.coef_, rtol=1e-3)
    assert_allclose(gcv_ridge.intercept_, kfold.intercept_, rtol=1e-3)


@pytest.mark.parametrize("mode", [True, 1, 5, "bad", "gcv"])
def test_check_gcv_mode_error(mode):
    X, y = make_regression(n_samples=5, n_features=2)
    gcv = RidgeCV(gcv_mode=mode)
    with pytest.raises(ValueError, match="Unknown value for 'gcv_mode'"):
        gcv.fit(X, y)
    with pytest.raises(ValueError, match="Unknown value for 'gcv_mode'"):
        _check_gcv_mode(X, mode)


@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize(
    "mode, mode_n_greater_than_p, mode_p_greater_than_n",
    [
        (None, "svd", "eigen"),
        ("auto", "svd", "eigen"),
        ("eigen", "eigen", "eigen"),
        ("svd", "svd", "svd"),
    ],
)
def test_check_gcv_mode_choice(
    sparse, mode, mode_n_greater_than_p, mode_p_greater_than_n
):
    X, _ = make_regression(n_samples=5, n_features=2)
    if sparse:
        X = sp.csr_matrix(X)
    assert _check_gcv_mode(X, mode) == mode_n_greater_than_p
    assert _check_gcv_mode(X.T, mode) == mode_p_greater_than_n


def _test_ridge_loo(filter_):
    # test that can work with both dense or sparse matrices
    n_samples = X_diabetes.shape[0]

    ret = []

    fit_intercept = filter_ == DENSE_FILTER
    ridge_gcv = _RidgeGCV(fit_intercept=fit_intercept)

    # check best alpha
    ridge_gcv.fit(filter_(X_diabetes), y_diabetes)
    alpha_ = ridge_gcv.alpha_
    ret.append(alpha_)

    # check that we get same best alpha with custom loss_func
    f = ignore_warnings
    scoring = make_scorer(mean_squared_error, greater_is_better=False)
    ridge_gcv2 = RidgeCV(fit_intercept=False, scoring=scoring)
    f(ridge_gcv2.fit)(filter_(X_diabetes), y_diabetes)
    assert ridge_gcv2.alpha_ == pytest.approx(alpha_)

    # check that we get same best alpha with custom score_func
    def func(x, y):
        return -mean_squared_error(x, y)

    scoring = make_scorer(func)
    ridge_gcv3 = RidgeCV(fit_intercept=False, scoring=scoring)
    f(ridge_gcv3.fit)(filter_(X_diabetes), y_diabetes)
    assert ridge_gcv3.alpha_ == pytest.approx(alpha_)

    # check that we get same best alpha with a scorer
    scorer = get_scorer("neg_mean_squared_error")
    ridge_gcv4 = RidgeCV(fit_intercept=False, scoring=scorer)
    ridge_gcv4.fit(filter_(X_diabetes), y_diabetes)
    assert ridge_gcv4.alpha_ == pytest.approx(alpha_)

    # check that we get same best alpha with sample weights
    if filter_ == DENSE_FILTER:
        ridge_gcv.fit(filter_(X_diabetes), y_diabetes, sample_weight=np.ones(n_samples))
        assert ridge_gcv.alpha_ == pytest.approx(alpha_)

    # simulate several responses
    Y = np.vstack((y_diabetes, y_diabetes)).T

    ridge_gcv.fit(filter_(X_diabetes), Y)
    Y_pred = ridge_gcv.predict(filter_(X_diabetes))
    ridge_gcv.fit(filter_(X_diabetes), y_diabetes)
    y_pred = ridge_gcv.predict(filter_(X_diabetes))

    assert_allclose(np.vstack((y_pred, y_pred)).T, Y_pred, rtol=1e-5)

    return ret


# FIXME: 'normalize' to be removed in 1.2
def _test_ridge_cv_normalize(filter_):
    ridge_cv = RidgeCV(normalize=True, cv=3)
    ridge_cv.fit(filter_(10.0 * X_diabetes), y_diabetes)

    gs = GridSearchCV(
        Ridge(normalize=True, solver="sparse_cg"),
        cv=3,
        param_grid={"alpha": ridge_cv.alphas},
    )
    gs.fit(filter_(10.0 * X_diabetes), y_diabetes)
    assert gs.best_estimator_.alpha == ridge_cv.alpha_


def _test_ridge_cv(filter_):
    ridge_cv = RidgeCV()
    ridge_cv.fit(filter_(X_diabetes), y_diabetes)
    ridge_cv.predict(filter_(X_diabetes))

    assert len(ridge_cv.coef_.shape) == 1
    assert type(ridge_cv.intercept_) == np.float64

    cv = KFold(5)
    ridge_cv.set_params(cv=cv)
    ridge_cv.fit(filter_(X_diabetes), y_diabetes)
    ridge_cv.predict(filter_(X_diabetes))

    assert len(ridge_cv.coef_.shape) == 1
    assert type(ridge_cv.intercept_) == np.float64


@pytest.mark.parametrize(
    "ridge, make_dataset",
    [
        (RidgeCV(store_cv_values=False), make_regression),
        (RidgeClassifierCV(store_cv_values=False), make_classification),
    ],
)
def test_ridge_gcv_cv_values_not_stored(ridge, make_dataset):
    # Check that `cv_values_` is not stored when store_cv_values is False
    X, y = make_dataset(n_samples=6, random_state=42)
    ridge.fit(X, y)
    assert not hasattr(ridge, "cv_values_")


@pytest.mark.parametrize(
    "ridge, make_dataset",
    [(RidgeCV(), make_regression), (RidgeClassifierCV(), make_classification)],
)
@pytest.mark.parametrize("cv", [None, 3])
def test_ridge_best_score(ridge, make_dataset, cv):
    # check that the best_score_ is store
    X, y = make_dataset(n_samples=6, random_state=42)
    ridge.set_params(store_cv_values=False, cv=cv)
    ridge.fit(X, y)
    assert hasattr(ridge, "best_score_")
    assert isinstance(ridge.best_score_, float)


def test_ridge_cv_individual_penalties():
    # Tests the ridge_cv object optimizing individual penalties for each target

    rng = np.random.RandomState(42)

    # Create random dataset with multiple targets. Each target should have
    # a different optimal alpha.
    n_samples, n_features, n_targets = 20, 5, 3
    y = rng.randn(n_samples, n_targets)
    X = (
        np.dot(y[:, [0]], np.ones((1, n_features)))
        + np.dot(y[:, [1]], 0.05 * np.ones((1, n_features)))
        + np.dot(y[:, [2]], 0.001 * np.ones((1, n_features)))
        + rng.randn(n_samples, n_features)
    )

    alphas = (1, 100, 1000)

    # Find optimal alpha for each target
    optimal_alphas = [RidgeCV(alphas=alphas).fit(X, target).alpha_ for target in y.T]

    # Find optimal alphas for all targets simultaneously
    ridge_cv = RidgeCV(alphas=alphas, alpha_per_target=True).fit(X, y)
    assert_array_equal(optimal_alphas, ridge_cv.alpha_)

    # The resulting regression weights should incorporate the different
    # alpha values.
    assert_array_almost_equal(
        Ridge(alpha=ridge_cv.alpha_).fit(X, y).coef_, ridge_cv.coef_
    )

    # Test shape of alpha_ and cv_values_
    ridge_cv = RidgeCV(alphas=alphas, alpha_per_target=True, store_cv_values=True).fit(
        X, y
    )
    assert ridge_cv.alpha_.shape == (n_targets,)
    assert ridge_cv.best_score_.shape == (n_targets,)
    assert ridge_cv.cv_values_.shape == (n_samples, len(alphas), n_targets)

    # Test edge case of there being only one alpha value
    ridge_cv = RidgeCV(alphas=1, alpha_per_target=True, store_cv_values=True).fit(X, y)
    assert ridge_cv.alpha_.shape == (n_targets,)
    assert ridge_cv.best_score_.shape == (n_targets,)
    assert ridge_cv.cv_values_.shape == (n_samples, n_targets, 1)

    # Test edge case of there being only one target
    ridge_cv = RidgeCV(alphas=alphas, alpha_per_target=True, store_cv_values=True).fit(
        X, y[:, 0]
    )
    assert np.isscalar(ridge_cv.alpha_)
    assert np.isscalar(ridge_cv.best_score_)
    assert ridge_cv.cv_values_.shape == (n_samples, len(alphas))

    # Try with a custom scoring function
    ridge_cv = RidgeCV(alphas=alphas, alpha_per_target=True, scoring="r2").fit(X, y)
    assert_array_equal(optimal_alphas, ridge_cv.alpha_)
    assert_array_almost_equal(
        Ridge(alpha=ridge_cv.alpha_).fit(X, y).coef_, ridge_cv.coef_
    )

    # Using a custom CV object should throw an error in combination with
    # alpha_per_target=True
    ridge_cv = RidgeCV(alphas=alphas, cv=LeaveOneOut(), alpha_per_target=True)
    msg = "cv!=None and alpha_per_target=True are incompatible"
    with pytest.raises(ValueError, match=msg):
        ridge_cv.fit(X, y)
    ridge_cv = RidgeCV(alphas=alphas, cv=6, alpha_per_target=True)
    with pytest.raises(ValueError, match=msg):
        ridge_cv.fit(X, y)


def _test_ridge_diabetes(filter_):
    ridge = Ridge(fit_intercept=False)
    ridge.fit(filter_(X_diabetes), y_diabetes)
    return np.round(ridge.score(filter_(X_diabetes), y_diabetes), 5)


def _test_multi_ridge_diabetes(filter_):
    # simulate several responses
    Y = np.vstack((y_diabetes, y_diabetes)).T
    n_features = X_diabetes.shape[1]

    ridge = Ridge(fit_intercept=False)
    ridge.fit(filter_(X_diabetes), Y)
    assert ridge.coef_.shape == (2, n_features)
    Y_pred = ridge.predict(filter_(X_diabetes))
    ridge.fit(filter_(X_diabetes), y_diabetes)
    y_pred = ridge.predict(filter_(X_diabetes))
    assert_array_almost_equal(np.vstack((y_pred, y_pred)).T, Y_pred, decimal=3)


def _test_ridge_classifiers(filter_):
    n_classes = np.unique(y_iris).shape[0]
    n_features = X_iris.shape[1]
    for reg in (RidgeClassifier(), RidgeClassifierCV()):
        reg.fit(filter_(X_iris), y_iris)
        assert reg.coef_.shape == (n_classes, n_features)
        y_pred = reg.predict(filter_(X_iris))
        assert np.mean(y_iris == y_pred) > 0.79

    cv = KFold(5)
    reg = RidgeClassifierCV(cv=cv)
    reg.fit(filter_(X_iris), y_iris)
    y_pred = reg.predict(filter_(X_iris))
    assert np.mean(y_iris == y_pred) >= 0.8


@pytest.mark.parametrize("scoring", [None, "accuracy", _accuracy_callable])
@pytest.mark.parametrize("cv", [None, KFold(5)])
@pytest.mark.parametrize("filter_", [DENSE_FILTER, SPARSE_FILTER])
def test_ridge_classifier_with_scoring(filter_, scoring, cv):
    # non-regression test for #14672
    # check that RidgeClassifierCV works with all sort of scoring and
    # cross-validation
    scoring_ = make_scorer(scoring) if callable(scoring) else scoring
    clf = RidgeClassifierCV(scoring=scoring_, cv=cv)
    # Smoke test to check that fit/predict does not raise error
    clf.fit(filter_(X_iris), y_iris).predict(filter_(X_iris))


@pytest.mark.parametrize("cv", [None, KFold(5)])
@pytest.mark.parametrize("filter_", [DENSE_FILTER, SPARSE_FILTER])
def test_ridge_regression_custom_scoring(filter_, cv):
    # check that custom scoring is working as expected
    # check the tie breaking strategy (keep the first alpha tried)

    def _dummy_score(y_test, y_pred):
        return 0.42

    alphas = np.logspace(-2, 2, num=5)
    clf = RidgeClassifierCV(alphas=alphas, scoring=make_scorer(_dummy_score), cv=cv)
    clf.fit(filter_(X_iris), y_iris)
    assert clf.best_score_ == pytest.approx(0.42)
    # In case of tie score, the first alphas will be kept
    assert clf.alpha_ == pytest.approx(alphas[0])


def _test_tolerance(filter_):
    ridge = Ridge(tol=1e-5, fit_intercept=False)
    ridge.fit(filter_(X_diabetes), y_diabetes)
    score = ridge.score(filter_(X_diabetes), y_diabetes)

    ridge2 = Ridge(tol=1e-3, fit_intercept=False)
    ridge2.fit(filter_(X_diabetes), y_diabetes)
    score2 = ridge2.score(filter_(X_diabetes), y_diabetes)

    assert score >= score2


def check_dense_sparse(test_func):
    # test dense matrix
    ret_dense = test_func(DENSE_FILTER)
    # test sparse matrix
    ret_sparse = test_func(SPARSE_FILTER)
    # test that the outputs are the same
    if ret_dense is not None and ret_sparse is not None:
        assert_array_almost_equal(ret_dense, ret_sparse, decimal=3)


# FIXME: 'normalize' to be removed in 1.2
@pytest.mark.filterwarnings("ignore:'normalize' was deprecated")
@pytest.mark.parametrize(
    "test_func",
    (
        _test_ridge_loo,
        _test_ridge_cv,
        _test_ridge_cv_normalize,
        _test_ridge_diabetes,
        _test_multi_ridge_diabetes,
        _test_ridge_classifiers,
        _test_tolerance,
    ),
)
def test_dense_sparse(test_func):
    check_dense_sparse(test_func)


def test_ridge_sparse_svd():
    X = sp.csc_matrix(rng.rand(100, 10))
    y = rng.rand(100)
    ridge = Ridge(solver="svd", fit_intercept=False)
    with pytest.raises(TypeError):
        ridge.fit(X, y)


def test_class_weights():
    # Test class weights.
    X = np.array([[-1.0, -1.0], [-1.0, 0], [-0.8, -1.0], [1.0, 1.0], [1.0, 0.0]])
    y = [1, 1, 1, -1, -1]

    reg = RidgeClassifier(class_weight=None)
    reg.fit(X, y)
    assert_array_equal(reg.predict([[0.2, -1.0]]), np.array([1]))

    # we give a small weights to class 1
    reg = RidgeClassifier(class_weight={1: 0.001})
    reg.fit(X, y)

    # now the hyperplane should rotate clock-wise and
    # the prediction on this point should shift
    assert_array_equal(reg.predict([[0.2, -1.0]]), np.array([-1]))

    # check if class_weight = 'balanced' can handle negative labels.
    reg = RidgeClassifier(class_weight="balanced")
    reg.fit(X, y)
    assert_array_equal(reg.predict([[0.2, -1.0]]), np.array([1]))

    # class_weight = 'balanced', and class_weight = None should return
    # same values when y has equal number of all labels
    X = np.array([[-1.0, -1.0], [-1.0, 0], [-0.8, -1.0], [1.0, 1.0]])
    y = [1, 1, -1, -1]
    reg = RidgeClassifier(class_weight=None)
    reg.fit(X, y)
    rega = RidgeClassifier(class_weight="balanced")
    rega.fit(X, y)
    assert len(rega.classes_) == 2
    assert_array_almost_equal(reg.coef_, rega.coef_)
    assert_array_almost_equal(reg.intercept_, rega.intercept_)


@pytest.mark.parametrize("reg", (RidgeClassifier, RidgeClassifierCV))
def test_class_weight_vs_sample_weight(reg):
    """Check class_weights resemble sample_weights behavior."""

    # Iris is balanced, so no effect expected for using 'balanced' weights
    reg1 = reg()
    reg1.fit(iris.data, iris.target)
    reg2 = reg(class_weight="balanced")
    reg2.fit(iris.data, iris.target)
    assert_almost_equal(reg1.coef_, reg2.coef_)

    # Inflate importance of class 1, check against user-defined weights
    sample_weight = np.ones(iris.target.shape)
    sample_weight[iris.target == 1] *= 100
    class_weight = {0: 1.0, 1: 100.0, 2: 1.0}
    reg1 = reg()
    reg1.fit(iris.data, iris.target, sample_weight)
    reg2 = reg(class_weight=class_weight)
    reg2.fit(iris.data, iris.target)
    assert_almost_equal(reg1.coef_, reg2.coef_)

    # Check that sample_weight and class_weight are multiplicative
    reg1 = reg()
    reg1.fit(iris.data, iris.target, sample_weight ** 2)
    reg2 = reg(class_weight=class_weight)
    reg2.fit(iris.data, iris.target, sample_weight)
    assert_almost_equal(reg1.coef_, reg2.coef_)


def test_class_weights_cv():
    # Test class weights for cross validated ridge classifier.
    X = np.array([[-1.0, -1.0], [-1.0, 0], [-0.8, -1.0], [1.0, 1.0], [1.0, 0.0]])
    y = [1, 1, 1, -1, -1]

    reg = RidgeClassifierCV(class_weight=None, alphas=[0.01, 0.1, 1])
    reg.fit(X, y)

    # we give a small weights to class 1
    reg = RidgeClassifierCV(class_weight={1: 0.001}, alphas=[0.01, 0.1, 1, 10])
    reg.fit(X, y)

    assert_array_equal(reg.predict([[-0.2, 2]]), np.array([-1]))


@pytest.mark.parametrize(
    "scoring", [None, "neg_mean_squared_error", _mean_squared_error_callable]
)
def test_ridgecv_store_cv_values(scoring):
    rng = np.random.RandomState(42)

    n_samples = 8
    n_features = 5
    x = rng.randn(n_samples, n_features)
    alphas = [1e-1, 1e0, 1e1]
    n_alphas = len(alphas)

    scoring_ = make_scorer(scoring) if callable(scoring) else scoring

    r = RidgeCV(alphas=alphas, cv=None, store_cv_values=True, scoring=scoring_)

    # with len(y.shape) == 1
    y = rng.randn(n_samples)
    r.fit(x, y)
    assert r.cv_values_.shape == (n_samples, n_alphas)

    # with len(y.shape) == 2
    n_targets = 3
    y = rng.randn(n_samples, n_targets)
    r.fit(x, y)
    assert r.cv_values_.shape == (n_samples, n_targets, n_alphas)

    r = RidgeCV(cv=3, store_cv_values=True, scoring=scoring)
    with pytest.raises(ValueError, match="cv!=None and store_cv_values"):
        r.fit(x, y)


@pytest.mark.parametrize("scoring", [None, "accuracy", _accuracy_callable])
def test_ridge_classifier_cv_store_cv_values(scoring):
    x = np.array([[-1.0, -1.0], [-1.0, 0], [-0.8, -1.0], [1.0, 1.0], [1.0, 0.0]])
    y = np.array([1, 1, 1, -1, -1])

    n_samples = x.shape[0]
    alphas = [1e-1, 1e0, 1e1]
    n_alphas = len(alphas)

    scoring_ = make_scorer(scoring) if callable(scoring) else scoring

    r = RidgeClassifierCV(
        alphas=alphas, cv=None, store_cv_values=True, scoring=scoring_
    )

    # with len(y.shape) == 1
    n_targets = 1
    r.fit(x, y)
    assert r.cv_values_.shape == (n_samples, n_targets, n_alphas)

    # with len(y.shape) == 2
    y = np.array(
        [[1, 1, 1, -1, -1], [1, -1, 1, -1, 1], [-1, -1, 1, -1, -1]]
    ).transpose()
    n_targets = y.shape[1]
    r.fit(x, y)
    assert r.cv_values_.shape == (n_samples, n_targets, n_alphas)


def test_ridgecv_sample_weight():
    rng = np.random.RandomState(0)
    alphas = (0.1, 1.0, 10.0)

    # There are different algorithms for n_samples > n_features
    # and the opposite, so test them both.
    for n_samples, n_features in ((6, 5), (5, 10)):
        y = rng.randn(n_samples)
        X = rng.randn(n_samples, n_features)
        sample_weight = 1.0 + rng.rand(n_samples)

        cv = KFold(5)
        ridgecv = RidgeCV(alphas=alphas, cv=cv)
        ridgecv.fit(X, y, sample_weight=sample_weight)

        # Check using GridSearchCV directly
        parameters = {"alpha": alphas}
        gs = GridSearchCV(Ridge(), parameters, cv=cv)
        gs.fit(X, y, sample_weight=sample_weight)

        assert ridgecv.alpha_ == gs.best_estimator_.alpha
        assert_array_almost_equal(ridgecv.coef_, gs.best_estimator_.coef_)


def test_raises_value_error_if_sample_weights_greater_than_1d():
    # Sample weights must be either scalar or 1D

    n_sampless = [2, 3]
    n_featuress = [3, 2]

    rng = np.random.RandomState(42)

    for n_samples, n_features in zip(n_sampless, n_featuress):
        X = rng.randn(n_samples, n_features)
        y = rng.randn(n_samples)
        sample_weights_OK = rng.randn(n_samples) ** 2 + 1
        sample_weights_OK_1 = 1.0
        sample_weights_OK_2 = 2.0
        sample_weights_not_OK = sample_weights_OK[:, np.newaxis]
        sample_weights_not_OK_2 = sample_weights_OK[np.newaxis, :]

        ridge = Ridge(alpha=1)

        # make sure the "OK" sample weights actually work
        ridge.fit(X, y, sample_weights_OK)
        ridge.fit(X, y, sample_weights_OK_1)
        ridge.fit(X, y, sample_weights_OK_2)

        def fit_ridge_not_ok():
            ridge.fit(X, y, sample_weights_not_OK)

        def fit_ridge_not_ok_2():
            ridge.fit(X, y, sample_weights_not_OK_2)

        err_msg = "Sample weights must be 1D array or scalar"
        with pytest.raises(ValueError, match=err_msg):
            fit_ridge_not_ok()

        err_msg = "Sample weights must be 1D array or scalar"
        with pytest.raises(ValueError, match=err_msg):
            fit_ridge_not_ok_2()


def test_sparse_design_with_sample_weights():
    # Sample weights must work with sparse matrices

    n_sampless = [2, 3]
    n_featuress = [3, 2]

    rng = np.random.RandomState(42)

    sparse_matrix_converters = [
        sp.coo_matrix,
        sp.csr_matrix,
        sp.csc_matrix,
        sp.lil_matrix,
        sp.dok_matrix,
    ]

    sparse_ridge = Ridge(alpha=1.0, fit_intercept=False)
    dense_ridge = Ridge(alpha=1.0, fit_intercept=False)

    for n_samples, n_features in zip(n_sampless, n_featuress):
        X = rng.randn(n_samples, n_features)
        y = rng.randn(n_samples)
        sample_weights = rng.randn(n_samples) ** 2 + 1
        for sparse_converter in sparse_matrix_converters:
            X_sparse = sparse_converter(X)
            sparse_ridge.fit(X_sparse, y, sample_weight=sample_weights)
            dense_ridge.fit(X, y, sample_weight=sample_weights)

            assert_array_almost_equal(sparse_ridge.coef_, dense_ridge.coef_, decimal=6)


def test_ridgecv_int_alphas():
    X = np.array([[-1.0, -1.0], [-1.0, 0], [-0.8, -1.0], [1.0, 1.0], [1.0, 0.0]])
    y = [1, 1, 1, -1, -1]

    # Integers
    ridge = RidgeCV(alphas=(1, 10, 100))
    ridge.fit(X, y)


def test_ridgecv_negative_alphas():
    X = np.array([[-1.0, -1.0], [-1.0, 0], [-0.8, -1.0], [1.0, 1.0], [1.0, 0.0]])
    y = [1, 1, 1, -1, -1]

    # Negative integers
    ridge = RidgeCV(alphas=(-1, -10, -100))
    with pytest.raises(ValueError, match="alphas must be strictly positive"):
        ridge.fit(X, y)

    # Negative floats
    ridge = RidgeCV(alphas=(-0.1, -1.0, -10.0))
    with pytest.raises(ValueError, match="alphas must be strictly positive"):
        ridge.fit(X, y)


def test_raises_value_error_if_solver_not_supported():
    # Tests whether a ValueError is raised if a non-identified solver
    # is passed to ridge_regression

    wrong_solver = "This is not a solver (MagritteSolveCV QuantumBitcoin)"

    exception = ValueError
    message = (
        "Known solvers are 'sparse_cg', 'cholesky', 'svd'"
        " 'lsqr', 'sag' or 'saga'. Got %s." % wrong_solver
    )

    def func():
        X = np.eye(3)
        y = np.ones(3)
        ridge_regression(X, y, alpha=1.0, solver=wrong_solver)

        with pytest.raises(exception, match=message):
            func()


def test_sparse_cg_max_iter():
    reg = Ridge(solver="sparse_cg", max_iter=1)
    reg.fit(X_diabetes, y_diabetes)
    assert reg.coef_.shape[0] == X_diabetes.shape[1]


@ignore_warnings
def test_n_iter():
    # Test that self.n_iter_ is correct.
    n_targets = 2
    X, y = X_diabetes, y_diabetes
    y_n = np.tile(y, (n_targets, 1)).T

    for max_iter in range(1, 4):
        for solver in ("sag", "saga", "lsqr"):
            reg = Ridge(solver=solver, max_iter=max_iter, tol=1e-12)
            reg.fit(X, y_n)
            assert_array_equal(reg.n_iter_, np.tile(max_iter, n_targets))

    for solver in ("sparse_cg", "svd", "cholesky"):
        reg = Ridge(solver=solver, max_iter=1, tol=1e-1)
        reg.fit(X, y_n)
        assert reg.n_iter_ is None


@pytest.mark.parametrize("solver", ["sparse_cg", "lbfgs", "auto"])
def test_ridge_fit_intercept_sparse(solver):
    positive = solver == "lbfgs"
    X, y = _make_sparse_offset_regression(
        n_features=20, random_state=0, positive=positive
    )
    X_csr = sp.csr_matrix(X)

    # for now only sparse_cg and lbfgs can correctly fit an intercept
    # with sparse X with default tol and max_iter.
    # sag is tested separately in test_ridge_fit_intercept_sparse_sag
    # because it requires more iterations and should raise a warning if default
    # max_iter is used.
    # other solvers raise an exception, as checked in
    # test_ridge_fit_intercept_sparse_error
    #
    # "auto" should switch to "sparse_cg" when X is sparse
    # so the reference we use for both ("auto" and "sparse_cg") is
    # Ridge(solver="sparse_cg"), fitted using the dense representation (note
    # that "sparse_cg" can fit sparse or dense data)
    dense_ridge = Ridge(solver="sparse_cg", tol=1e-12)
    sparse_ridge = Ridge(solver=solver, tol=1e-12, positive=positive)
    dense_ridge.fit(X, y)
    with pytest.warns(None) as record:
        sparse_ridge.fit(X_csr, y)
    assert len(record) == 0
    assert np.allclose(dense_ridge.intercept_, sparse_ridge.intercept_)
    assert np.allclose(dense_ridge.coef_, sparse_ridge.coef_)


@pytest.mark.parametrize("solver", ["saga", "lsqr", "svd", "cholesky"])
def test_ridge_fit_intercept_sparse_error(solver):
    X, y = _make_sparse_offset_regression(n_features=20, random_state=0)
    X_csr = sp.csr_matrix(X)
    sparse_ridge = Ridge(solver=solver)
    err_msg = "solver='{}' does not support".format(solver)
    with pytest.raises(ValueError, match=err_msg):
        sparse_ridge.fit(X_csr, y)


def test_ridge_fit_intercept_sparse_sag():
    X, y = _make_sparse_offset_regression(
        n_features=5, n_samples=20, random_state=0, X_offset=5.0
    )
    X_csr = sp.csr_matrix(X)

    params = dict(
        alpha=1.0, solver="sag", fit_intercept=True, tol=1e-10, max_iter=100000
    )
    dense_ridge = Ridge(**params)
    sparse_ridge = Ridge(**params)
    dense_ridge.fit(X, y)
    with pytest.warns(None) as record:
        sparse_ridge.fit(X_csr, y)
    assert len(record) == 0
    assert np.allclose(dense_ridge.intercept_, sparse_ridge.intercept_, rtol=1e-4)
    assert np.allclose(dense_ridge.coef_, sparse_ridge.coef_, rtol=1e-4)
    with pytest.warns(UserWarning, match='"sag" solver requires.*'):
        Ridge(solver="sag").fit(X_csr, y)


@pytest.mark.parametrize("return_intercept", [False, True])
@pytest.mark.parametrize("sample_weight", [None, np.ones(1000)])
@pytest.mark.parametrize("arr_type", [np.array, sp.csr_matrix])
@pytest.mark.parametrize(
    "solver", ["auto", "sparse_cg", "cholesky", "lsqr", "sag", "saga", "lbfgs"]
)
def test_ridge_regression_check_arguments_validity(
    return_intercept, sample_weight, arr_type, solver
):
    """check if all combinations of arguments give valid estimations"""

    # test excludes 'svd' solver because it raises exception for sparse inputs

    rng = check_random_state(42)
    X = rng.rand(1000, 3)
    true_coefs = [1, 2, 0.1]
    y = np.dot(X, true_coefs)
    true_intercept = 0.0
    if return_intercept:
        true_intercept = 10000.0
    y += true_intercept
    X_testing = arr_type(X)

    alpha, tol = 1e-3, 1e-6
    atol = 1e-3 if _IS_32BIT else 1e-4

    positive = solver == "lbfgs"

    if solver not in ["sag", "auto"] and return_intercept:
        with pytest.raises(ValueError, match="In Ridge, only 'sag' solver"):
            ridge_regression(
                X_testing,
                y,
                alpha=alpha,
                solver=solver,
                sample_weight=sample_weight,
                return_intercept=return_intercept,
                positive=positive,
                tol=tol,
            )
        return

    out = ridge_regression(
        X_testing,
        y,
        alpha=alpha,
        solver=solver,
        sample_weight=sample_weight,
        positive=positive,
        return_intercept=return_intercept,
        tol=tol,
    )

    if return_intercept:
        coef, intercept = out
        assert_allclose(coef, true_coefs, rtol=0, atol=atol)
        assert_allclose(intercept, true_intercept, rtol=0, atol=atol)
    else:
        assert_allclose(out, true_coefs, rtol=0, atol=atol)


@pytest.mark.parametrize(
    "solver", ["svd", "sparse_cg", "cholesky", "lsqr", "sag", "saga", "lbfgs"]
)
def test_dtype_match(solver):
    rng = np.random.RandomState(0)
    alpha = 1.0
    positive = solver == "lbfgs"

    n_samples, n_features = 6, 5
    X_64 = rng.randn(n_samples, n_features)
    y_64 = rng.randn(n_samples)
    X_32 = X_64.astype(np.float32)
    y_32 = y_64.astype(np.float32)

    tol = 2 * np.finfo(np.float32).resolution
    # Check type consistency 32bits
    ridge_32 = Ridge(
        alpha=alpha, solver=solver, max_iter=500, tol=tol, positive=positive
    )
    ridge_32.fit(X_32, y_32)
    coef_32 = ridge_32.coef_

    # Check type consistency 64 bits
    ridge_64 = Ridge(
        alpha=alpha, solver=solver, max_iter=500, tol=tol, positive=positive
    )
    ridge_64.fit(X_64, y_64)
    coef_64 = ridge_64.coef_

    # Do the actual checks at once for easier debug
    assert coef_32.dtype == X_32.dtype
    assert coef_64.dtype == X_64.dtype
    assert ridge_32.predict(X_32).dtype == X_32.dtype
    assert ridge_64.predict(X_64).dtype == X_64.dtype
    assert_allclose(ridge_32.coef_, ridge_64.coef_, rtol=1e-4, atol=5e-4)


def test_dtype_match_cholesky():
    # Test different alphas in cholesky solver to ensure full coverage.
    # This test is separated from test_dtype_match for clarity.
    rng = np.random.RandomState(0)
    alpha = (1.0, 0.5)

    n_samples, n_features, n_target = 6, 7, 2
    X_64 = rng.randn(n_samples, n_features)
    y_64 = rng.randn(n_samples, n_target)
    X_32 = X_64.astype(np.float32)
    y_32 = y_64.astype(np.float32)

    # Check type consistency 32bits
    ridge_32 = Ridge(alpha=alpha, solver="cholesky")
    ridge_32.fit(X_32, y_32)
    coef_32 = ridge_32.coef_

    # Check type consistency 64 bits
    ridge_64 = Ridge(alpha=alpha, solver="cholesky")
    ridge_64.fit(X_64, y_64)
    coef_64 = ridge_64.coef_

    # Do all the checks at once, like this is easier to debug
    assert coef_32.dtype == X_32.dtype
    assert coef_64.dtype == X_64.dtype
    assert ridge_32.predict(X_32).dtype == X_32.dtype
    assert ridge_64.predict(X_64).dtype == X_64.dtype
    assert_almost_equal(ridge_32.coef_, ridge_64.coef_, decimal=5)


@pytest.mark.parametrize(
    "solver", ["svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"]
)
@pytest.mark.parametrize("seed", range(1))
def test_ridge_regression_dtype_stability(solver, seed):
    random_state = np.random.RandomState(seed)
    n_samples, n_features = 6, 5
    X = random_state.randn(n_samples, n_features)
    coef = random_state.randn(n_features)
    y = np.dot(X, coef) + 0.01 * random_state.randn(n_samples)
    alpha = 1.0
    positive = solver == "lbfgs"
    results = dict()
    # XXX: Sparse CG seems to be far less numerically stable than the
    # others, maybe we should not enable float32 for this one.
    atol = 1e-3 if solver == "sparse_cg" else 1e-5
    for current_dtype in (np.float32, np.float64):
        results[current_dtype] = ridge_regression(
            X.astype(current_dtype),
            y.astype(current_dtype),
            alpha=alpha,
            solver=solver,
            random_state=random_state,
            sample_weight=None,
            positive=positive,
            max_iter=500,
            tol=1e-10,
            return_n_iter=False,
            return_intercept=False,
        )

    assert results[np.float32].dtype == np.float32
    assert results[np.float64].dtype == np.float64
    assert_allclose(results[np.float32], results[np.float64], atol=atol)


def test_ridge_sag_with_X_fortran():
    # check that Fortran array are converted when using SAG solver
    X, y = make_regression(random_state=42)
    # for the order of X and y to not be C-ordered arrays
    X = np.asfortranarray(X)
    X = X[::2, :]
    y = y[::2]
    Ridge(solver="sag").fit(X, y)


@pytest.mark.parametrize(
    "Classifier, params",
    [
        (RidgeClassifier, {}),
        (RidgeClassifierCV, {"cv": None}),
        (RidgeClassifierCV, {"cv": 3}),
    ],
)
def test_ridgeclassifier_multilabel(Classifier, params):
    """Check that multilabel classification is supported and give meaningful
    results."""
    X, y = make_multilabel_classification(n_classes=1, random_state=0)
    y = y.reshape(-1, 1)
    Y = np.concatenate([y, y], axis=1)
    clf = Classifier(**params).fit(X, Y)
    Y_pred = clf.predict(X)

    assert Y_pred.shape == Y.shape
    assert_array_equal(Y_pred[:, 0], Y_pred[:, 1])
    Ridge(solver="sag").fit(X, y)


@pytest.mark.parametrize("solver", ["auto", "lbfgs"])
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("alpha", [1e-3, 1e-2, 0.1, 1.0])
def test_ridge_positive_regression_test(solver, fit_intercept, alpha):
    """Test that positive Ridge finds true positive coefficients."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    coef = np.array([1, -10])
    if fit_intercept:
        intercept = 20
        y = X.dot(coef) + intercept
    else:
        y = X.dot(coef)

    model = Ridge(
        alpha=alpha, positive=True, solver=solver, fit_intercept=fit_intercept
    )
    model.fit(X, y)
    assert np.all(model.coef_ >= 0)


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("alpha", [1e-3, 1e-2, 0.1, 1.0])
def test_ridge_ground_truth_positive_test(fit_intercept, alpha):
    """Test that Ridge w/wo positive converges to the same solution.

    Ridge with positive=True and positive=False must give the same
    when the ground truth coefs are all positive.
    """
    rng = np.random.RandomState(42)
    X = rng.randn(300, 100)
    coef = rng.uniform(0.1, 1.0, size=X.shape[1])
    if fit_intercept:
        intercept = 1
        y = X @ coef + intercept
    else:
        y = X @ coef
    y += rng.normal(size=X.shape[0]) * 0.01

    results = []
    for positive in [True, False]:
        model = Ridge(
            alpha=alpha, positive=positive, fit_intercept=fit_intercept, tol=1e-10
        )
        results.append(model.fit(X, y).coef_)
    assert_allclose(*results, atol=1e-6, rtol=0)


@pytest.mark.parametrize(
    "solver", ["svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
)
def test_ridge_positive_error_test(solver):
    """Test input validation for positive argument in Ridge."""
    alpha = 0.1
    X = np.array([[1, 2], [3, 4]])
    coef = np.array([1, -1])
    y = X @ coef

    model = Ridge(alpha=alpha, positive=True, solver=solver, fit_intercept=False)
    with pytest.raises(ValueError, match="does not support positive"):
        model.fit(X, y)

    with pytest.raises(ValueError, match="only 'lbfgs' solver can be used"):
        _, _ = ridge_regression(
            X, y, alpha, positive=True, solver=solver, return_intercept=False
        )


@pytest.mark.parametrize("alpha", [1e-3, 1e-2, 0.1, 1.0])
def test_positive_ridge_loss(alpha):
    """Check ridge loss consistency when positive argument is enabled."""
    X, y = make_regression(n_samples=300, n_features=300, random_state=42)
    alpha = 0.10
    n_checks = 100

    def ridge_loss(model, random_state=None, noise_scale=1e-8):
        intercept = model.intercept_
        if random_state is not None:
            rng = np.random.RandomState(random_state)
            coef = model.coef_ + rng.uniform(0, noise_scale, size=model.coef_.shape)
        else:
            coef = model.coef_

        return 0.5 * np.sum((y - X @ coef - intercept) ** 2) + 0.5 * alpha * np.sum(
            coef ** 2
        )

    model = Ridge(alpha=alpha).fit(X, y)
    model_positive = Ridge(alpha=alpha, positive=True).fit(X, y)

    # Check 1:
    #   Loss for solution found by Ridge(positive=False)
    #   is lower than that for solution found by Ridge(positive=True)
    loss = ridge_loss(model)
    loss_positive = ridge_loss(model_positive)
    assert loss <= loss_positive

    # Check 2:
    #   Loss for solution found by Ridge(positive=True)
    #   is lower than that for small random positive perturbation
    #   of the positive solution.
    for random_state in range(n_checks):
        loss_perturbed = ridge_loss(model_positive, random_state=random_state)
        assert loss_positive <= loss_perturbed


@pytest.mark.parametrize("alpha", [1e-3, 1e-2, 0.1, 1.0])
def test_lbfgs_solver_consistency(alpha):
    """Test that LBGFS gets almost the same coef of svd when positive=False."""
    X, y = make_regression(n_samples=300, n_features=300, random_state=42)
    y = np.expand_dims(y, 1)
    alpha = np.asarray([alpha])
    config = {
        "positive": False,
        "tol": 1e-16,
        "max_iter": 500000,
    }

    coef_lbfgs = _solve_lbfgs(X, y, alpha, **config)
    coef_cholesky = _solve_svd(X, y, alpha)
    assert_allclose(coef_lbfgs, coef_cholesky, atol=1e-4, rtol=0)


def test_lbfgs_solver_error():
    """Test that LBFGS solver raises ConvergenceWarning."""
    X = np.array([[1, -1], [1, 1]])
    y = np.array([-1e10, 1e10])

    model = Ridge(
        alpha=0.01,
        solver="lbfgs",
        fit_intercept=False,
        tol=1e-12,
        positive=True,
        max_iter=1,
    )
    with pytest.warns(ConvergenceWarning, match="lbfgs solver did not converge"):
        model.fit(X, y)


# FIXME: 'normalize' to be removed in 1.2
@pytest.mark.filterwarnings("ignore:'normalize' was deprecated")
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize(
    "solver", ["cholesky", "lsqr", "sparse_cg", "svd", "sag", "saga", "lbfgs"]
)
def test_ridge_sample_weight_invariance(normalize, solver):
    """Test that Ridge fulfils sample weight invariance.

    Note that this test is stricter than the common test
    check_sample_weights_invariance alone.
    """
    params = dict(
        alpha=1.0,
        normalize=normalize,
        solver=solver,
        tol=1e-12,
        positive=(solver == "lbfgs"),
    )
    reg = Ridge(**params)
    name = reg.__class__.__name__
    check_sample_weights_invariance(name, reg, kind="ones")
    check_sample_weights_invariance(name, reg, kind="zeros")

    # Check that duplicating the training dataset is equivalent to multiplying
    # the weights by 2:
    if solver.startswith("sag") and normalize:
        pytest.xfail("sag/saga diverge on the second part of this test")

    rng = np.random.RandomState(42)
    X, y = make_regression(
        n_samples=100,
        n_features=300,
        effective_rank=10,
        n_informative=50,
        random_state=rng,
    )
    sw = rng.uniform(low=0.01, high=2, size=X.shape[0])
    X_dup = np.concatenate([X, X], axis=0)
    y_dup = np.concatenate([y, y], axis=0)
    sw_dup = np.concatenate([sw, sw], axis=0)

    ridge_2sw = Ridge(**params).fit(X, y, sample_weight=2 * sw)
    ridge_dup = Ridge(**params).fit(X_dup, y_dup, sample_weight=sw_dup)

    assert_allclose(ridge_2sw.coef_, ridge_dup.coef_)
    assert_allclose(ridge_2sw.intercept_, ridge_dup.intercept_)
