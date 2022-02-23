# Authors: Olivier Grisel <olivier.grisel@ensta.org>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause

import numpy as np
import pytest
from scipy import interpolate, sparse
from copy import deepcopy
import joblib

from sklearn.base import is_classifier
from sklearn.base import clone
from sklearn.datasets import load_diabetes
from sklearn.datasets import make_regression
from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneGroupOut,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import assert_allclose
from sklearn.utils._testing import assert_almost_equal
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import ignore_warnings
from sklearn.utils._testing import _convert_container

from sklearn.utils._testing import TempMemmap
from sklearn.utils.fixes import parse_version
from sklearn.utils import check_random_state
from sklearn.utils.sparsefuncs import mean_variance_axis

from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    ElasticNet,
    ElasticNetCV,
    enet_path,
    Lars,
    lars_path,
    Lasso,
    LassoCV,
    LassoLars,
    LassoLarsCV,
    LassoLarsIC,
    lasso_path,
    LinearRegression,
    MultiTaskElasticNet,
    MultiTaskElasticNetCV,
    MultiTaskLasso,
    MultiTaskLassoCV,
    OrthogonalMatchingPursuit,
    Ridge,
    RidgeClassifier,
    RidgeClassifierCV,
    RidgeCV,
)

from sklearn.linear_model._coordinate_descent import _set_order
from sklearn.utils import check_array


# FIXME: 'normalize' to be removed in 1.2
filterwarnings_normalize = pytest.mark.filterwarnings(
    "ignore:'normalize' was deprecated in version 1.0"
)


# FIXME: 'normalize' to be removed in 1.2
@pytest.mark.parametrize(
    "CoordinateDescentModel",
    [
        ElasticNet,
        Lasso,
        LassoCV,
        ElasticNetCV,
        MultiTaskElasticNet,
        MultiTaskLasso,
        MultiTaskElasticNetCV,
        MultiTaskLassoCV,
    ],
)
@pytest.mark.parametrize(
    "normalize, n_warnings", [(True, 1), (False, 1), ("deprecated", 0)]
)
def test_assure_warning_when_normalize(CoordinateDescentModel, normalize, n_warnings):
    # check that we issue a FutureWarning when normalize was set
    rng = check_random_state(0)
    n_samples = 200
    n_features = 2
    X = rng.randn(n_samples, n_features)
    X[X < 0.1] = 0.0
    y = rng.rand(n_samples)

    if "MultiTask" in CoordinateDescentModel.__name__:
        y = np.stack((y, y), axis=1)

    model = CoordinateDescentModel(normalize=normalize)
    with pytest.warns(None) as record:
        model.fit(X, y)

    record = [r for r in record if r.category == FutureWarning]
    assert len(record) == n_warnings


@pytest.mark.parametrize("l1_ratio", (-1, 2, None, 10, "something_wrong"))
def test_l1_ratio_param_invalid(l1_ratio):
    # Check that correct error is raised when l1_ratio in ElasticNet
    # is outside the correct range
    X = np.array([[-1.0], [0.0], [1.0]])
    Y = [-1, 0, 1]  # just a straight line

    msg = "l1_ratio must be between 0 and 1; got l1_ratio="
    clf = ElasticNet(alpha=0.1, l1_ratio=l1_ratio)
    with pytest.raises(ValueError, match=msg):
        clf.fit(X, Y)


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("input_order", ["C", "F"])
def test_set_order_dense(order, input_order):
    """Check that _set_order returns arrays with promised order."""
    X = np.array([[0], [0], [0]], order=input_order)
    y = np.array([0, 0, 0], order=input_order)
    X2, y2 = _set_order(X, y, order=order)
    if order == "C":
        assert X2.flags["C_CONTIGUOUS"]
        assert y2.flags["C_CONTIGUOUS"]
    elif order == "F":
        assert X2.flags["F_CONTIGUOUS"]
        assert y2.flags["F_CONTIGUOUS"]

    if order == input_order:
        assert X is X2
        assert y is y2


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("input_order", ["C", "F"])
def test_set_order_sparse(order, input_order):
    """Check that _set_order returns sparse matrices in promised format."""
    X = sparse.coo_matrix(np.array([[0], [0], [0]]))
    y = sparse.coo_matrix(np.array([0, 0, 0]))
    sparse_format = "csc" if input_order == "F" else "csr"
    X = X.asformat(sparse_format)
    y = X.asformat(sparse_format)
    X2, y2 = _set_order(X, y, order=order)
    if order == "C":
        assert sparse.isspmatrix_csr(X2)
        assert sparse.isspmatrix_csr(y2)
    elif order == "F":
        assert sparse.isspmatrix_csc(X2)
        assert sparse.isspmatrix_csc(y2)


def test_lasso_zero():
    # Check that the lasso can handle zero data without crashing
    X = [[0], [0], [0]]
    y = [0, 0, 0]
    clf = Lasso(alpha=0.1).fit(X, y)
    pred = clf.predict([[1], [2], [3]])
    assert_array_almost_equal(clf.coef_, [0])
    assert_array_almost_equal(pred, [0, 0, 0])
    assert_almost_equal(clf.dual_gap_, 0)


def test_lasso_toy():
    # Test Lasso on a toy example for various values of alpha.
    # When validating this against glmnet notice that glmnet divides it
    # against nobs.

    X = [[-1], [0], [1]]
    Y = [-1, 0, 1]  # just a straight line
    T = [[2], [3], [4]]  # test sample

    clf = Lasso(alpha=1e-8)
    clf.fit(X, Y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [1])
    assert_array_almost_equal(pred, [2, 3, 4])
    assert_almost_equal(clf.dual_gap_, 0)

    clf = Lasso(alpha=0.1)
    clf.fit(X, Y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.85])
    assert_array_almost_equal(pred, [1.7, 2.55, 3.4])
    assert_almost_equal(clf.dual_gap_, 0)

    clf = Lasso(alpha=0.5)
    clf.fit(X, Y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.25])
    assert_array_almost_equal(pred, [0.5, 0.75, 1.0])
    assert_almost_equal(clf.dual_gap_, 0)

    clf = Lasso(alpha=1)
    clf.fit(X, Y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.0])
    assert_array_almost_equal(pred, [0, 0, 0])
    assert_almost_equal(clf.dual_gap_, 0)


def test_enet_toy():
    # Test ElasticNet for various parameters of alpha and l1_ratio.
    # Actually, the parameters alpha = 0 should not be allowed. However,
    # we test it as a border case.
    # ElasticNet is tested with and without precomputed Gram matrix

    X = np.array([[-1.0], [0.0], [1.0]])
    Y = [-1, 0, 1]  # just a straight line
    T = [[2.0], [3.0], [4.0]]  # test sample

    # this should be the same as lasso
    clf = ElasticNet(alpha=1e-8, l1_ratio=1.0)
    clf.fit(X, Y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [1])
    assert_array_almost_equal(pred, [2, 3, 4])
    assert_almost_equal(clf.dual_gap_, 0)

    clf = ElasticNet(alpha=0.5, l1_ratio=0.3, max_iter=100, precompute=False)
    clf.fit(X, Y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.50819], decimal=3)
    assert_array_almost_equal(pred, [1.0163, 1.5245, 2.0327], decimal=3)
    assert_almost_equal(clf.dual_gap_, 0)

    clf.set_params(max_iter=100, precompute=True)
    clf.fit(X, Y)  # with Gram
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.50819], decimal=3)
    assert_array_almost_equal(pred, [1.0163, 1.5245, 2.0327], decimal=3)
    assert_almost_equal(clf.dual_gap_, 0)

    clf.set_params(max_iter=100, precompute=np.dot(X.T, X))
    clf.fit(X, Y)  # with Gram
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.50819], decimal=3)
    assert_array_almost_equal(pred, [1.0163, 1.5245, 2.0327], decimal=3)
    assert_almost_equal(clf.dual_gap_, 0)

    clf = ElasticNet(alpha=0.5, l1_ratio=0.5)
    clf.fit(X, Y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.45454], 3)
    assert_array_almost_equal(pred, [0.9090, 1.3636, 1.8181], 3)
    assert_almost_equal(clf.dual_gap_, 0)


def test_lasso_dual_gap():
    """
    Check that Lasso.dual_gap_ matches its objective formulation, with the
    datafit normalized by n_samples
    """
    X, y, _, _ = build_dataset(n_samples=10, n_features=30)
    n_samples = len(y)
    alpha = 0.01 * np.max(np.abs(X.T @ y)) / n_samples
    clf = Lasso(alpha=alpha, fit_intercept=False).fit(X, y)
    w = clf.coef_
    R = y - X @ w
    primal = 0.5 * np.mean(R ** 2) + clf.alpha * np.sum(np.abs(w))
    # dual pt: R / n_samples, dual constraint: norm(X.T @ theta, inf) <= alpha
    R /= np.max(np.abs(X.T @ R) / (n_samples * alpha))
    dual = 0.5 * (np.mean(y ** 2) - np.mean((y - R) ** 2))
    assert_allclose(clf.dual_gap_, primal - dual)


def build_dataset(n_samples=50, n_features=200, n_informative_features=10, n_targets=1):
    """
    build an ill-posed linear regression problem with many noisy features and
    comparatively few samples
    """
    random_state = np.random.RandomState(0)
    if n_targets > 1:
        w = random_state.randn(n_features, n_targets)
    else:
        w = random_state.randn(n_features)
    w[n_informative_features:] = 0.0
    X = random_state.randn(n_samples, n_features)
    y = np.dot(X, w)
    X_test = random_state.randn(n_samples, n_features)
    y_test = np.dot(X_test, w)
    return X, y, X_test, y_test


def test_lasso_cv():
    X, y, X_test, y_test = build_dataset()
    max_iter = 150
    clf = LassoCV(n_alphas=10, eps=1e-3, max_iter=max_iter, cv=3).fit(X, y)
    assert_almost_equal(clf.alpha_, 0.056, 2)

    clf = LassoCV(n_alphas=10, eps=1e-3, max_iter=max_iter, precompute=True, cv=3)
    clf.fit(X, y)
    assert_almost_equal(clf.alpha_, 0.056, 2)

    # Check that the lars and the coordinate descent implementation
    # select a similar alpha
    lars = LassoLarsCV(normalize=False, max_iter=30, cv=3).fit(X, y)
    # for this we check that they don't fall in the grid of
    # clf.alphas further than 1
    assert (
        np.abs(
            np.searchsorted(clf.alphas_[::-1], lars.alpha_)
            - np.searchsorted(clf.alphas_[::-1], clf.alpha_)
        )
        <= 1
    )
    # check that they also give a similar MSE
    mse_lars = interpolate.interp1d(lars.cv_alphas_, lars.mse_path_.T)
    np.testing.assert_approx_equal(
        mse_lars(clf.alphas_[5]).mean(), clf.mse_path_[5].mean(), significant=2
    )

    # test set
    assert clf.score(X_test, y_test) > 0.99


def test_lasso_cv_with_some_model_selection():
    from sklearn.model_selection import ShuffleSplit
    from sklearn import datasets

    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target

    pipe = make_pipeline(StandardScaler(), LassoCV(cv=ShuffleSplit(random_state=0)))
    pipe.fit(X, y)


def test_lasso_cv_positive_constraint():
    X, y, X_test, y_test = build_dataset()
    max_iter = 500

    # Ensure the unconstrained fit has a negative coefficient
    clf_unconstrained = LassoCV(n_alphas=3, eps=1e-1, max_iter=max_iter, cv=2, n_jobs=1)
    clf_unconstrained.fit(X, y)
    assert min(clf_unconstrained.coef_) < 0

    # On same data, constrained fit has non-negative coefficients
    clf_constrained = LassoCV(
        n_alphas=3, eps=1e-1, max_iter=max_iter, positive=True, cv=2, n_jobs=1
    )
    clf_constrained.fit(X, y)
    assert min(clf_constrained.coef_) >= 0


def _scale_alpha_inplace(estimator, n_samples):
    """Rescale the parameter alpha from when the estimator is evoked with
    normalize set to True to when it is evoked in a Pipeline with normalize set
    to False and with a StandardScaler.
    """
    if ("alpha" not in estimator.get_params()) and (
        "alphas" not in estimator.get_params()
    ):
        return

    if isinstance(estimator, (RidgeCV, RidgeClassifierCV)):
        alphas = estimator.alphas * n_samples
        return estimator.set_params(alphas=alphas)
    if isinstance(estimator, (Lasso, LassoLars, MultiTaskLasso)):
        alpha = estimator.alpha * np.sqrt(n_samples)
    if isinstance(estimator, (Ridge, RidgeClassifier)):
        alpha = estimator.alpha * n_samples
    if isinstance(estimator, (ElasticNet, MultiTaskElasticNet)):
        if estimator.l1_ratio == 1:
            alpha = estimator.alpha * np.sqrt(n_samples)
        elif estimator.l1_ratio == 0:
            alpha = estimator.alpha * n_samples
        else:
            # To avoid silent errors in case of refactoring
            raise NotImplementedError

    estimator.set_params(alpha=alpha)


# FIXME: 'normalize' to be removed in 1.2 for all the models excluding:
# OrthogonalMatchingPursuit, Lars, LassoLars, LarsCV, LassoLarsCV
# for which it is to be removed in 1.4
@pytest.mark.filterwarnings("ignore:'normalize' was deprecated")
@pytest.mark.parametrize(
    "LinearModel, params",
    [
        (Lasso, {"tol": 1e-16, "alpha": 0.1}),
        (LassoLars, {"alpha": 0.1}),
        (RidgeClassifier, {"solver": "sparse_cg", "alpha": 0.1}),
        (ElasticNet, {"tol": 1e-16, "l1_ratio": 1, "alpha": 0.1}),
        (ElasticNet, {"tol": 1e-16, "l1_ratio": 0, "alpha": 0.1}),
        (Ridge, {"solver": "sparse_cg", "tol": 1e-12, "alpha": 0.1}),
        (BayesianRidge, {}),
        (ARDRegression, {}),
        (OrthogonalMatchingPursuit, {}),
        (MultiTaskElasticNet, {"tol": 1e-16, "l1_ratio": 1, "alpha": 0.1}),
        (MultiTaskElasticNet, {"tol": 1e-16, "l1_ratio": 0, "alpha": 0.1}),
        (MultiTaskLasso, {"tol": 1e-16, "alpha": 0.1}),
        (Lars, {}),
        (LinearRegression, {}),
        (LassoLarsIC, {}),
        (RidgeCV, {"alphas": [0.1, 0.4]}),
        (RidgeClassifierCV, {"alphas": [0.1, 0.4]}),
    ],
)
def test_model_pipeline_same_as_normalize_true(LinearModel, params):
    # Test that linear models (LinearModel) set with normalize set to True are
    # doing the same as the same linear model preceded by StandardScaler
    # in the pipeline and with normalize set to False

    # normalize is True
    model_normalize = LinearModel(normalize=True, fit_intercept=True, **params)

    pipeline = make_pipeline(
        StandardScaler(), LinearModel(normalize=False, fit_intercept=True, **params)
    )

    is_multitask = model_normalize._get_tags()["multioutput_only"]

    # prepare the data
    n_samples, n_features = 100, 2
    rng = np.random.RandomState(0)
    w = rng.randn(n_features)
    X = rng.randn(n_samples, n_features)
    X += 20  # make features non-zero mean
    y = X.dot(w)

    # make classes out of regression
    if is_classifier(model_normalize):
        y[y > np.mean(y)] = -1
        y[y > 0] = 1
    if is_multitask:
        y = np.stack((y, y), axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    _scale_alpha_inplace(pipeline[1], X_train.shape[0])

    model_normalize.fit(X_train, y_train)
    y_pred_normalize = model_normalize.predict(X_test)

    pipeline.fit(X_train, y_train)
    y_pred_standardize = pipeline.predict(X_test)

    assert_allclose(model_normalize.coef_ * pipeline[0].scale_, pipeline[1].coef_)
    assert pipeline[1].intercept_ == pytest.approx(y_train.mean())
    assert model_normalize.intercept_ == pytest.approx(
        y_train.mean() - model_normalize.coef_.dot(X_train.mean(0))
    )
    assert_allclose(y_pred_normalize, y_pred_standardize)


# FIXME: 'normalize' to be removed in 1.2
@pytest.mark.filterwarnings("ignore:'normalize' was deprecated")
@pytest.mark.parametrize(
    "estimator, params",
    [
        (Lasso, {"tol": 1e-16, "alpha": 0.1}),
        (RidgeClassifier, {"solver": "sparse_cg", "alpha": 0.1}),
        (ElasticNet, {"tol": 1e-16, "l1_ratio": 1, "alpha": 0.1}),
        (ElasticNet, {"tol": 1e-16, "l1_ratio": 0, "alpha": 0.1}),
        (Ridge, {"solver": "sparse_cg", "tol": 1e-12, "alpha": 0.1}),
        (LinearRegression, {}),
        (RidgeCV, {"alphas": [0.1, 0.4]}),
        (RidgeClassifierCV, {"alphas": [0.1, 0.4]}),
    ],
)
@pytest.mark.parametrize(
    "is_sparse, with_mean",
    [
        (False, True),
        (False, False),
        (True, False)
        # No need to test sparse and with_mean=True
    ],
)
def test_linear_model_sample_weights_normalize_in_pipeline(
    is_sparse, with_mean, estimator, params
):
    # Test that the results for running linear model with sample_weight
    # and with normalize set to True gives similar results as the same linear
    # model with normalize set to False in a pipeline with
    # a StandardScaler and sample_weight.
    model_name = estimator.__name__

    if model_name in ["Lasso", "ElasticNet"] and is_sparse:
        pytest.skip(f"{model_name} does not support sample_weight with sparse")

    rng = np.random.RandomState(0)
    X, y = make_regression(n_samples=20, n_features=5, noise=1e-2, random_state=rng)

    if is_classifier(estimator):
        y = np.sign(y)

    # make sure the data is not centered to make the problem more
    # difficult + add 0s for the sparse case
    X[X < 0] = 0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=rng
    )
    if is_sparse:
        X_train = sparse.csr_matrix(X_train)
        X_test = _convert_container(X_train, "sparse")

    sample_weight = rng.uniform(low=0.1, high=100, size=X_train.shape[0])

    # linear estimator with built-in feature normalization
    reg_with_normalize = estimator(normalize=True, fit_intercept=True, **params)
    reg_with_normalize.fit(X_train, y_train, sample_weight=sample_weight)

    # linear estimator in a pipeline with a StandardScaler, normalize=False
    linear_regressor = estimator(normalize=False, fit_intercept=True, **params)

    # rescale alpha
    if model_name in ["Lasso", "ElasticNet"]:
        _scale_alpha_inplace(linear_regressor, y_test.shape[0])
    else:
        _scale_alpha_inplace(linear_regressor, sample_weight.sum())
    reg_with_scaler = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=with_mean)),
            ("linear_regressor", linear_regressor),
        ]
    )

    fit_params = {
        "scaler__sample_weight": sample_weight,
        "linear_regressor__sample_weight": sample_weight,
    }

    reg_with_scaler.fit(X_train, y_train, **fit_params)

    # Check that the 2 regressions models are exactly equivalent in the
    # sense that they predict exactly the same outcome.
    y_pred_normalize = reg_with_normalize.predict(X_test)
    y_pred_scaler = reg_with_scaler.predict(X_test)
    assert_allclose(y_pred_normalize, y_pred_scaler)

    # Check intercept computation when normalize is True
    y_train_mean = np.average(y_train, weights=sample_weight)
    if is_sparse:
        X_train_mean, _ = mean_variance_axis(X_train, axis=0, weights=sample_weight)
    else:
        X_train_mean = np.average(X_train, weights=sample_weight, axis=0)
    assert reg_with_normalize.intercept_ == pytest.approx(
        y_train_mean - reg_with_normalize.coef_.dot(X_train_mean)
    )


# FIXME: 'normalize' to be removed in 1.2
@pytest.mark.filterwarnings("ignore:'normalize' was deprecated")
@pytest.mark.parametrize(
    "LinearModel, params",
    [
        (Lasso, {"tol": 1e-16, "alpha": 0.1}),
        (LassoCV, {"tol": 1e-16}),
        (ElasticNetCV, {}),
        (RidgeClassifier, {"solver": "sparse_cg", "alpha": 0.1}),
        (ElasticNet, {"tol": 1e-16, "l1_ratio": 1, "alpha": 0.01}),
        (ElasticNet, {"tol": 1e-16, "l1_ratio": 0, "alpha": 0.01}),
        (Ridge, {"solver": "sparse_cg", "tol": 1e-12, "alpha": 0.1}),
        (LinearRegression, {}),
        (RidgeCV, {}),
        (RidgeClassifierCV, {}),
    ],
)
def test_model_pipeline_same_dense_and_sparse(LinearModel, params):
    # Test that linear model preceded by StandardScaler in the pipeline and
    # with normalize set to False gives the same y_pred and the same .coef_
    # given X sparse or dense

    model_dense = make_pipeline(
        StandardScaler(with_mean=False), LinearModel(normalize=False, **params)
    )

    model_sparse = make_pipeline(
        StandardScaler(with_mean=False), LinearModel(normalize=False, **params)
    )

    # prepare the data
    rng = np.random.RandomState(0)
    n_samples = 200
    n_features = 2
    X = rng.randn(n_samples, n_features)
    X[X < 0.1] = 0.0

    X_sparse = sparse.csr_matrix(X)
    y = rng.rand(n_samples)

    if is_classifier(model_dense):
        y = np.sign(y)

    model_dense.fit(X, y)
    model_sparse.fit(X_sparse, y)

    assert_allclose(model_sparse[1].coef_, model_dense[1].coef_)
    y_pred_dense = model_dense.predict(X)
    y_pred_sparse = model_sparse.predict(X_sparse)
    assert_allclose(y_pred_dense, y_pred_sparse)

    assert_allclose(model_dense[1].intercept_, model_sparse[1].intercept_)


def test_lasso_path_return_models_vs_new_return_gives_same_coefficients():
    # Test that lasso_path with lars_path style output gives the
    # same result

    # Some toy data
    X = np.array([[1, 2, 3.1], [2.3, 5.4, 4.3]]).T
    y = np.array([1, 2, 3.1])
    alphas = [5.0, 1.0, 0.5]

    # Use lars_path and lasso_path(new output) with 1D linear interpolation
    # to compute the same path
    alphas_lars, _, coef_path_lars = lars_path(X, y, method="lasso")
    coef_path_cont_lars = interpolate.interp1d(
        alphas_lars[::-1], coef_path_lars[:, ::-1]
    )
    alphas_lasso2, coef_path_lasso2, _ = lasso_path(X, y, alphas=alphas)
    coef_path_cont_lasso = interpolate.interp1d(
        alphas_lasso2[::-1], coef_path_lasso2[:, ::-1]
    )

    assert_array_almost_equal(
        coef_path_cont_lasso(alphas), coef_path_cont_lars(alphas), decimal=1
    )


def test_enet_path():
    # We use a large number of samples and of informative features so that
    # the l1_ratio selected is more toward ridge than lasso
    X, y, X_test, y_test = build_dataset(
        n_samples=200, n_features=100, n_informative_features=100
    )
    max_iter = 150

    # Here we have a small number of iterations, and thus the
    # ElasticNet might not converge. This is to speed up tests
    clf = ElasticNetCV(
        alphas=[0.01, 0.05, 0.1], eps=2e-3, l1_ratio=[0.5, 0.7], cv=3, max_iter=max_iter
    )
    ignore_warnings(clf.fit)(X, y)
    # Well-conditioned settings, we should have selected our
    # smallest penalty
    assert_almost_equal(clf.alpha_, min(clf.alphas_))
    # Non-sparse ground truth: we should have selected an elastic-net
    # that is closer to ridge than to lasso
    assert clf.l1_ratio_ == min(clf.l1_ratio)

    clf = ElasticNetCV(
        alphas=[0.01, 0.05, 0.1],
        eps=2e-3,
        l1_ratio=[0.5, 0.7],
        cv=3,
        max_iter=max_iter,
        precompute=True,
    )
    ignore_warnings(clf.fit)(X, y)

    # Well-conditioned settings, we should have selected our
    # smallest penalty
    assert_almost_equal(clf.alpha_, min(clf.alphas_))
    # Non-sparse ground truth: we should have selected an elastic-net
    # that is closer to ridge than to lasso
    assert clf.l1_ratio_ == min(clf.l1_ratio)

    # We are in well-conditioned settings with low noise: we should
    # have a good test-set performance
    assert clf.score(X_test, y_test) > 0.99

    # Multi-output/target case
    X, y, X_test, y_test = build_dataset(n_features=10, n_targets=3)
    clf = MultiTaskElasticNetCV(
        n_alphas=5, eps=2e-3, l1_ratio=[0.5, 0.7], cv=3, max_iter=max_iter
    )
    ignore_warnings(clf.fit)(X, y)
    # We are in well-conditioned settings with low noise: we should
    # have a good test-set performance
    assert clf.score(X_test, y_test) > 0.99
    assert clf.coef_.shape == (3, 10)

    # Mono-output should have same cross-validated alpha_ and l1_ratio_
    # in both cases.
    X, y, _, _ = build_dataset(n_features=10)
    clf1 = ElasticNetCV(n_alphas=5, eps=2e-3, l1_ratio=[0.5, 0.7])
    clf1.fit(X, y)
    clf2 = MultiTaskElasticNetCV(n_alphas=5, eps=2e-3, l1_ratio=[0.5, 0.7])
    clf2.fit(X, y[:, np.newaxis])
    assert_almost_equal(clf1.l1_ratio_, clf2.l1_ratio_)
    assert_almost_equal(clf1.alpha_, clf2.alpha_)


def test_path_parameters():
    X, y, _, _ = build_dataset()
    max_iter = 100

    clf = ElasticNetCV(n_alphas=50, eps=1e-3, max_iter=max_iter, l1_ratio=0.5, tol=1e-3)
    clf.fit(X, y)  # new params
    assert_almost_equal(0.5, clf.l1_ratio)
    assert 50 == clf.n_alphas
    assert 50 == len(clf.alphas_)


def test_warm_start():
    X, y, _, _ = build_dataset()
    clf = ElasticNet(alpha=0.1, max_iter=5, warm_start=True)
    ignore_warnings(clf.fit)(X, y)
    ignore_warnings(clf.fit)(X, y)  # do a second round with 5 iterations

    clf2 = ElasticNet(alpha=0.1, max_iter=10)
    ignore_warnings(clf2.fit)(X, y)
    assert_array_almost_equal(clf2.coef_, clf.coef_)


def test_lasso_alpha_warning():
    X = [[-1], [0], [1]]
    Y = [-1, 0, 1]  # just a straight line

    clf = Lasso(alpha=0)
    warning_message = (
        "With alpha=0, this algorithm does not "
        "converge well. You are advised to use the "
        "LinearRegression estimator"
    )
    with pytest.warns(UserWarning, match=warning_message):
        clf.fit(X, Y)


def test_lasso_positive_constraint():
    X = [[-1], [0], [1]]
    y = [1, 0, -1]  # just a straight line with negative slope

    lasso = Lasso(alpha=0.1, positive=True)
    lasso.fit(X, y)
    assert min(lasso.coef_) >= 0

    lasso = Lasso(alpha=0.1, precompute=True, positive=True)
    lasso.fit(X, y)
    assert min(lasso.coef_) >= 0


def test_enet_positive_constraint():
    X = [[-1], [0], [1]]
    y = [1, 0, -1]  # just a straight line with negative slope

    enet = ElasticNet(alpha=0.1, positive=True)
    enet.fit(X, y)
    assert min(enet.coef_) >= 0


def test_enet_cv_positive_constraint():
    X, y, X_test, y_test = build_dataset()
    max_iter = 500

    # Ensure the unconstrained fit has a negative coefficient
    enetcv_unconstrained = ElasticNetCV(
        n_alphas=3, eps=1e-1, max_iter=max_iter, cv=2, n_jobs=1
    )
    enetcv_unconstrained.fit(X, y)
    assert min(enetcv_unconstrained.coef_) < 0

    # On same data, constrained fit has non-negative coefficients
    enetcv_constrained = ElasticNetCV(
        n_alphas=3, eps=1e-1, max_iter=max_iter, cv=2, positive=True, n_jobs=1
    )
    enetcv_constrained.fit(X, y)
    assert min(enetcv_constrained.coef_) >= 0


def test_uniform_targets():
    enet = ElasticNetCV(n_alphas=3)
    m_enet = MultiTaskElasticNetCV(n_alphas=3)
    lasso = LassoCV(n_alphas=3)
    m_lasso = MultiTaskLassoCV(n_alphas=3)

    models_single_task = (enet, lasso)
    models_multi_task = (m_enet, m_lasso)

    rng = np.random.RandomState(0)

    X_train = rng.random_sample(size=(10, 3))
    X_test = rng.random_sample(size=(10, 3))

    y1 = np.empty(10)
    y2 = np.empty((10, 2))

    for model in models_single_task:
        for y_values in (0, 5):
            y1.fill(y_values)
            assert_array_equal(model.fit(X_train, y1).predict(X_test), y1)
            assert_array_equal(model.alphas_, [np.finfo(float).resolution] * 3)

    for model in models_multi_task:
        for y_values in (0, 5):
            y2[:, 0].fill(y_values)
            y2[:, 1].fill(2 * y_values)
            assert_array_equal(model.fit(X_train, y2).predict(X_test), y2)
            assert_array_equal(model.alphas_, [np.finfo(float).resolution] * 3)


def test_multi_task_lasso_and_enet():
    X, y, X_test, y_test = build_dataset()
    Y = np.c_[y, y]
    # Y_test = np.c_[y_test, y_test]
    clf = MultiTaskLasso(alpha=1, tol=1e-8).fit(X, Y)
    assert 0 < clf.dual_gap_ < 1e-5
    assert_array_almost_equal(clf.coef_[0], clf.coef_[1])

    clf = MultiTaskElasticNet(alpha=1, tol=1e-8).fit(X, Y)
    assert 0 < clf.dual_gap_ < 1e-5
    assert_array_almost_equal(clf.coef_[0], clf.coef_[1])

    clf = MultiTaskElasticNet(alpha=1.0, tol=1e-8, max_iter=1)
    warning_message = (
        "Objective did not converge. You might want to "
        "increase the number of iterations."
    )
    with pytest.warns(ConvergenceWarning, match=warning_message):
        clf.fit(X, Y)


def test_lasso_readonly_data():
    X = np.array([[-1], [0], [1]])
    Y = np.array([-1, 0, 1])  # just a straight line
    T = np.array([[2], [3], [4]])  # test sample
    with TempMemmap((X, Y)) as (X, Y):
        clf = Lasso(alpha=0.5)
        clf.fit(X, Y)
        pred = clf.predict(T)
        assert_array_almost_equal(clf.coef_, [0.25])
        assert_array_almost_equal(pred, [0.5, 0.75, 1.0])
        assert_almost_equal(clf.dual_gap_, 0)


def test_multi_task_lasso_readonly_data():
    X, y, X_test, y_test = build_dataset()
    Y = np.c_[y, y]
    with TempMemmap((X, Y)) as (X, Y):
        Y = np.c_[y, y]
        clf = MultiTaskLasso(alpha=1, tol=1e-8).fit(X, Y)
        assert 0 < clf.dual_gap_ < 1e-5
        assert_array_almost_equal(clf.coef_[0], clf.coef_[1])


def test_enet_multitarget():
    n_targets = 3
    X, y, _, _ = build_dataset(
        n_samples=10, n_features=8, n_informative_features=10, n_targets=n_targets
    )
    estimator = ElasticNet(alpha=0.01)
    estimator.fit(X, y)
    coef, intercept, dual_gap = (
        estimator.coef_,
        estimator.intercept_,
        estimator.dual_gap_,
    )

    for k in range(n_targets):
        estimator.fit(X, y[:, k])
        assert_array_almost_equal(coef[k, :], estimator.coef_)
        assert_array_almost_equal(intercept[k], estimator.intercept_)
        assert_array_almost_equal(dual_gap[k], estimator.dual_gap_)


def test_multioutput_enetcv_error():
    rng = np.random.RandomState(0)
    X = rng.randn(10, 2)
    y = rng.randn(10, 2)
    clf = ElasticNetCV()
    with pytest.raises(ValueError):
        clf.fit(X, y)


def test_multitask_enet_and_lasso_cv():
    X, y, _, _ = build_dataset(n_features=50, n_targets=3)
    clf = MultiTaskElasticNetCV(cv=3).fit(X, y)
    assert_almost_equal(clf.alpha_, 0.00556, 3)
    clf = MultiTaskLassoCV(cv=3).fit(X, y)
    assert_almost_equal(clf.alpha_, 0.00278, 3)

    X, y, _, _ = build_dataset(n_targets=3)
    clf = MultiTaskElasticNetCV(
        n_alphas=10, eps=1e-3, max_iter=100, l1_ratio=[0.3, 0.5], tol=1e-3, cv=3
    )
    clf.fit(X, y)
    assert 0.5 == clf.l1_ratio_
    assert (3, X.shape[1]) == clf.coef_.shape
    assert (3,) == clf.intercept_.shape
    assert (2, 10, 3) == clf.mse_path_.shape
    assert (2, 10) == clf.alphas_.shape

    X, y, _, _ = build_dataset(n_targets=3)
    clf = MultiTaskLassoCV(n_alphas=10, eps=1e-3, max_iter=100, tol=1e-3, cv=3)
    clf.fit(X, y)
    assert (3, X.shape[1]) == clf.coef_.shape
    assert (3,) == clf.intercept_.shape
    assert (10, 3) == clf.mse_path_.shape
    assert 10 == len(clf.alphas_)


def test_1d_multioutput_enet_and_multitask_enet_cv():
    X, y, _, _ = build_dataset(n_features=10)
    y = y[:, np.newaxis]
    clf = ElasticNetCV(n_alphas=5, eps=2e-3, l1_ratio=[0.5, 0.7])
    clf.fit(X, y[:, 0])
    clf1 = MultiTaskElasticNetCV(n_alphas=5, eps=2e-3, l1_ratio=[0.5, 0.7])
    clf1.fit(X, y)
    assert_almost_equal(clf.l1_ratio_, clf1.l1_ratio_)
    assert_almost_equal(clf.alpha_, clf1.alpha_)
    assert_almost_equal(clf.coef_, clf1.coef_[0])
    assert_almost_equal(clf.intercept_, clf1.intercept_[0])


def test_1d_multioutput_lasso_and_multitask_lasso_cv():
    X, y, _, _ = build_dataset(n_features=10)
    y = y[:, np.newaxis]
    clf = LassoCV(n_alphas=5, eps=2e-3)
    clf.fit(X, y[:, 0])
    clf1 = MultiTaskLassoCV(n_alphas=5, eps=2e-3)
    clf1.fit(X, y)
    assert_almost_equal(clf.alpha_, clf1.alpha_)
    assert_almost_equal(clf.coef_, clf1.coef_[0])
    assert_almost_equal(clf.intercept_, clf1.intercept_[0])


def test_sparse_input_dtype_enet_and_lassocv():
    X, y, _, _ = build_dataset(n_features=10)
    clf = ElasticNetCV(n_alphas=5)
    clf.fit(sparse.csr_matrix(X), y)
    clf1 = ElasticNetCV(n_alphas=5)
    clf1.fit(sparse.csr_matrix(X, dtype=np.float32), y)
    assert_almost_equal(clf.alpha_, clf1.alpha_, decimal=6)
    assert_almost_equal(clf.coef_, clf1.coef_, decimal=6)

    clf = LassoCV(n_alphas=5)
    clf.fit(sparse.csr_matrix(X), y)
    clf1 = LassoCV(n_alphas=5)
    clf1.fit(sparse.csr_matrix(X, dtype=np.float32), y)
    assert_almost_equal(clf.alpha_, clf1.alpha_, decimal=6)
    assert_almost_equal(clf.coef_, clf1.coef_, decimal=6)


def test_precompute_invalid_argument():
    X, y, _, _ = build_dataset()
    for clf in [ElasticNetCV(precompute="invalid"), LassoCV(precompute="invalid")]:
        err_msg = ".*should be.*True.*False.*auto.* array-like.*Got 'invalid'"
        with pytest.raises(ValueError, match=err_msg):
            clf.fit(X, y)

    # Precompute = 'auto' is not supported for ElasticNet and Lasso
    err_msg = ".*should be.*True.*False.*array-like.*Got 'auto'"
    with pytest.raises(ValueError, match=err_msg):
        ElasticNet(precompute="auto").fit(X, y)

    err_msg = ".*should be.*True.*False.*array-like.*Got 'auto'"
    with pytest.raises(ValueError, match=err_msg):
        Lasso(precompute="auto").fit(X, y)


def test_elasticnet_precompute_incorrect_gram():
    # check that passing an invalid precomputed Gram matrix will raise an
    # error.
    X, y, _, _ = build_dataset()

    rng = np.random.RandomState(0)

    X_centered = X - np.average(X, axis=0)
    garbage = rng.standard_normal(X.shape)
    precompute = np.dot(garbage.T, garbage)

    clf = ElasticNet(alpha=0.01, precompute=precompute)
    msg = "Gram matrix.*did not pass validation.*"
    with pytest.raises(ValueError, match=msg):
        clf.fit(X_centered, y)


def test_elasticnet_precompute_gram_weighted_samples():
    # check the equivalence between passing a precomputed Gram matrix and
    # internal computation using sample weights.
    X, y, _, _ = build_dataset()

    rng = np.random.RandomState(0)
    sample_weight = rng.lognormal(size=y.shape)

    w_norm = sample_weight * (y.shape / np.sum(sample_weight))
    X_c = X - np.average(X, axis=0, weights=w_norm)
    X_r = X_c * np.sqrt(w_norm)[:, np.newaxis]
    gram = np.dot(X_r.T, X_r)

    clf1 = ElasticNet(alpha=0.01, precompute=gram)
    clf1.fit(X_c, y, sample_weight=sample_weight)

    clf2 = ElasticNet(alpha=0.01, precompute=False)
    clf2.fit(X, y, sample_weight=sample_weight)

    assert_allclose(clf1.coef_, clf2.coef_)


def test_warm_start_convergence():
    X, y, _, _ = build_dataset()
    model = ElasticNet(alpha=1e-3, tol=1e-3).fit(X, y)
    n_iter_reference = model.n_iter_

    # This dataset is not trivial enough for the model to converge in one pass.
    assert n_iter_reference > 2

    # Check that n_iter_ is invariant to multiple calls to fit
    # when warm_start=False, all else being equal.
    model.fit(X, y)
    n_iter_cold_start = model.n_iter_
    assert n_iter_cold_start == n_iter_reference

    # Fit the same model again, using a warm start: the optimizer just performs
    # a single pass before checking that it has already converged
    model.set_params(warm_start=True)
    model.fit(X, y)
    n_iter_warm_start = model.n_iter_
    assert n_iter_warm_start == 1


def test_warm_start_convergence_with_regularizer_decrement():
    X, y = load_diabetes(return_X_y=True)

    # Train a model to converge on a lightly regularized problem
    final_alpha = 1e-5
    low_reg_model = ElasticNet(alpha=final_alpha).fit(X, y)

    # Fitting a new model on a more regularized version of the same problem.
    # Fitting with high regularization is easier it should converge faster
    # in general.
    high_reg_model = ElasticNet(alpha=final_alpha * 10).fit(X, y)
    assert low_reg_model.n_iter_ > high_reg_model.n_iter_

    # Fit the solution to the original, less regularized version of the
    # problem but from the solution of the highly regularized variant of
    # the problem as a better starting point. This should also converge
    # faster than the original model that starts from zero.
    warm_low_reg_model = deepcopy(high_reg_model)
    warm_low_reg_model.set_params(warm_start=True, alpha=final_alpha)
    warm_low_reg_model.fit(X, y)
    assert low_reg_model.n_iter_ > warm_low_reg_model.n_iter_


def test_random_descent():
    # Test that both random and cyclic selection give the same results.
    # Ensure that the test models fully converge and check a wide
    # range of conditions.

    # This uses the coordinate descent algo using the gram trick.
    X, y, _, _ = build_dataset(n_samples=50, n_features=20)
    clf_cyclic = ElasticNet(selection="cyclic", tol=1e-8)
    clf_cyclic.fit(X, y)
    clf_random = ElasticNet(selection="random", tol=1e-8, random_state=42)
    clf_random.fit(X, y)
    assert_array_almost_equal(clf_cyclic.coef_, clf_random.coef_)
    assert_almost_equal(clf_cyclic.intercept_, clf_random.intercept_)

    # This uses the descent algo without the gram trick
    clf_cyclic = ElasticNet(selection="cyclic", tol=1e-8)
    clf_cyclic.fit(X.T, y[:20])
    clf_random = ElasticNet(selection="random", tol=1e-8, random_state=42)
    clf_random.fit(X.T, y[:20])
    assert_array_almost_equal(clf_cyclic.coef_, clf_random.coef_)
    assert_almost_equal(clf_cyclic.intercept_, clf_random.intercept_)

    # Sparse Case
    clf_cyclic = ElasticNet(selection="cyclic", tol=1e-8)
    clf_cyclic.fit(sparse.csr_matrix(X), y)
    clf_random = ElasticNet(selection="random", tol=1e-8, random_state=42)
    clf_random.fit(sparse.csr_matrix(X), y)
    assert_array_almost_equal(clf_cyclic.coef_, clf_random.coef_)
    assert_almost_equal(clf_cyclic.intercept_, clf_random.intercept_)

    # Multioutput case.
    new_y = np.hstack((y[:, np.newaxis], y[:, np.newaxis]))
    clf_cyclic = MultiTaskElasticNet(selection="cyclic", tol=1e-8)
    clf_cyclic.fit(X, new_y)
    clf_random = MultiTaskElasticNet(selection="random", tol=1e-8, random_state=42)
    clf_random.fit(X, new_y)
    assert_array_almost_equal(clf_cyclic.coef_, clf_random.coef_)
    assert_almost_equal(clf_cyclic.intercept_, clf_random.intercept_)

    # Raise error when selection is not in cyclic or random.
    clf_random = ElasticNet(selection="invalid")
    with pytest.raises(ValueError):
        clf_random.fit(X, y)


def test_enet_path_positive():
    # Test positive parameter

    X, Y, _, _ = build_dataset(n_samples=50, n_features=50, n_targets=2)

    # For mono output
    # Test that the coefs returned by positive=True in enet_path are positive
    for path in [enet_path, lasso_path]:
        pos_path_coef = path(X, Y[:, 0], positive=True)[1]
        assert np.all(pos_path_coef >= 0)

    # For multi output, positive parameter is not allowed
    # Test that an error is raised
    for path in [enet_path, lasso_path]:
        with pytest.raises(ValueError):
            path(X, Y, positive=True)


def test_sparse_dense_descent_paths():
    # Test that dense and sparse input give the same input for descent paths.
    X, y, _, _ = build_dataset(n_samples=50, n_features=20)
    csr = sparse.csr_matrix(X)
    for path in [enet_path, lasso_path]:
        _, coefs, _ = path(X, y)
        _, sparse_coefs, _ = path(csr, y)
        assert_array_almost_equal(coefs, sparse_coefs)


@pytest.mark.parametrize("path_func", [enet_path, lasso_path])
def test_path_unknown_parameter(path_func):
    """Check that passing parameter not used by the coordinate descent solver
    will raise an error."""
    X, y, _, _ = build_dataset(n_samples=50, n_features=20)
    err_msg = "Unexpected parameters in params"
    with pytest.raises(ValueError, match=err_msg):
        path_func(X, y, normalize=True, fit_intercept=True)


def test_check_input_false():
    X, y, _, _ = build_dataset(n_samples=20, n_features=10)
    X = check_array(X, order="F", dtype="float64")
    y = check_array(X, order="F", dtype="float64")
    clf = ElasticNet(selection="cyclic", tol=1e-8)
    # Check that no error is raised if data is provided in the right format
    clf.fit(X, y, check_input=False)
    # With check_input=False, an exhaustive check is not made on y but its
    # dtype is still cast in _preprocess_data to X's dtype. So the test should
    # pass anyway
    X = check_array(X, order="F", dtype="float32")
    clf.fit(X, y, check_input=False)
    # With no input checking, providing X in C order should result in false
    # computation
    X = check_array(X, order="C", dtype="float64")
    with pytest.raises(ValueError):
        clf.fit(X, y, check_input=False)


@pytest.mark.parametrize("check_input", [True, False])
def test_enet_copy_X_True(check_input):
    X, y, _, _ = build_dataset()
    X = X.copy(order="F")

    original_X = X.copy()
    enet = ElasticNet(copy_X=True)
    enet.fit(X, y, check_input=check_input)

    assert_array_equal(original_X, X)


def test_enet_copy_X_False_check_input_False():
    X, y, _, _ = build_dataset()
    X = X.copy(order="F")

    original_X = X.copy()
    enet = ElasticNet(copy_X=False)
    enet.fit(X, y, check_input=False)

    # No copying, X is overwritten
    assert np.any(np.not_equal(original_X, X))


def test_overrided_gram_matrix():
    X, y, _, _ = build_dataset(n_samples=20, n_features=10)
    Gram = X.T.dot(X)
    clf = ElasticNet(selection="cyclic", tol=1e-8, precompute=Gram)
    warning_message = (
        "Gram matrix was provided but X was centered"
        " to fit intercept, "
        "or X was normalized : recomputing Gram matrix."
    )
    with pytest.warns(UserWarning, match=warning_message):
        clf.fit(X, y)


@pytest.mark.parametrize("model", [ElasticNet, Lasso])
def test_lasso_non_float_y(model):
    X = [[0, 0], [1, 1], [-1, -1]]
    y = [0, 1, 2]
    y_float = [0.0, 1.0, 2.0]

    clf = model(fit_intercept=False)
    clf.fit(X, y)
    clf_float = model(fit_intercept=False)
    clf_float.fit(X, y_float)
    assert_array_equal(clf.coef_, clf_float.coef_)


# FIXME: 'normalize' to be removed in 1.2
@filterwarnings_normalize
def test_enet_float_precision():
    # Generate dataset
    X, y, X_test, y_test = build_dataset(n_samples=20, n_features=10)
    # Here we have a small number of iterations, and thus the
    # ElasticNet might not converge. This is to speed up tests

    for normalize in [True, False]:
        for fit_intercept in [True, False]:
            coef = {}
            intercept = {}
            for dtype in [np.float64, np.float32]:
                clf = ElasticNet(
                    alpha=0.5,
                    max_iter=100,
                    precompute=False,
                    fit_intercept=fit_intercept,
                    normalize=normalize,
                )

                X = dtype(X)
                y = dtype(y)
                ignore_warnings(clf.fit)(X, y)

                coef[("simple", dtype)] = clf.coef_
                intercept[("simple", dtype)] = clf.intercept_

                assert clf.coef_.dtype == dtype

                # test precompute Gram array
                Gram = X.T.dot(X)
                clf_precompute = ElasticNet(
                    alpha=0.5,
                    max_iter=100,
                    precompute=Gram,
                    fit_intercept=fit_intercept,
                    normalize=normalize,
                )
                ignore_warnings(clf_precompute.fit)(X, y)
                assert_array_almost_equal(clf.coef_, clf_precompute.coef_)
                assert_array_almost_equal(clf.intercept_, clf_precompute.intercept_)

                # test multi task enet
                multi_y = np.hstack((y[:, np.newaxis], y[:, np.newaxis]))
                clf_multioutput = MultiTaskElasticNet(
                    alpha=0.5,
                    max_iter=100,
                    fit_intercept=fit_intercept,
                    normalize=normalize,
                )
                clf_multioutput.fit(X, multi_y)
                coef[("multi", dtype)] = clf_multioutput.coef_
                intercept[("multi", dtype)] = clf_multioutput.intercept_
                assert clf.coef_.dtype == dtype

            for v in ["simple", "multi"]:
                assert_array_almost_equal(
                    coef[(v, np.float32)], coef[(v, np.float64)], decimal=4
                )
                assert_array_almost_equal(
                    intercept[(v, np.float32)], intercept[(v, np.float64)], decimal=4
                )


def test_enet_l1_ratio():
    # Test that an error message is raised if an estimator that
    # uses _alpha_grid is called with l1_ratio=0
    msg = (
        "Automatic alpha grid generation is not supported for l1_ratio=0. "
        "Please supply a grid by providing your estimator with the "
        "appropriate `alphas=` argument."
    )
    X = np.array([[1, 2, 4, 5, 8], [3, 5, 7, 7, 8]]).T
    y = np.array([12, 10, 11, 21, 5])

    with pytest.raises(ValueError, match=msg):
        ElasticNetCV(l1_ratio=0, random_state=42).fit(X, y)

    with pytest.raises(ValueError, match=msg):
        MultiTaskElasticNetCV(l1_ratio=0, random_state=42).fit(X, y[:, None])

    # Test that l1_ratio=0 is allowed if we supply a grid manually
    alphas = [0.1, 10]
    estkwds = {"alphas": alphas, "random_state": 42}
    est_desired = ElasticNetCV(l1_ratio=0.00001, **estkwds)
    est = ElasticNetCV(l1_ratio=0, **estkwds)
    with ignore_warnings():
        est_desired.fit(X, y)
        est.fit(X, y)
    assert_array_almost_equal(est.coef_, est_desired.coef_, decimal=5)

    est_desired = MultiTaskElasticNetCV(l1_ratio=0.00001, **estkwds)
    est = MultiTaskElasticNetCV(l1_ratio=0, **estkwds)
    with ignore_warnings():
        est.fit(X, y[:, None])
        est_desired.fit(X, y[:, None])
    assert_array_almost_equal(est.coef_, est_desired.coef_, decimal=5)


def test_coef_shape_not_zero():
    est_no_intercept = Lasso(fit_intercept=False)
    est_no_intercept.fit(np.c_[np.ones(3)], np.ones(3))
    assert est_no_intercept.coef_.shape == (1,)


def test_warm_start_multitask_lasso():
    X, y, X_test, y_test = build_dataset()
    Y = np.c_[y, y]
    clf = MultiTaskLasso(alpha=0.1, max_iter=5, warm_start=True)
    ignore_warnings(clf.fit)(X, Y)
    ignore_warnings(clf.fit)(X, Y)  # do a second round with 5 iterations

    clf2 = MultiTaskLasso(alpha=0.1, max_iter=10)
    ignore_warnings(clf2.fit)(X, Y)
    assert_array_almost_equal(clf2.coef_, clf.coef_)


@pytest.mark.parametrize(
    "klass, n_classes, kwargs",
    [
        (Lasso, 1, dict(precompute=True)),
        (Lasso, 1, dict(precompute=False)),
        (MultiTaskLasso, 2, dict()),
        (MultiTaskLasso, 2, dict()),
    ],
)
def test_enet_coordinate_descent(klass, n_classes, kwargs):
    """Test that a warning is issued if model does not converge"""
    clf = klass(max_iter=2, **kwargs)
    n_samples = 5
    n_features = 2
    X = np.ones((n_samples, n_features)) * 1e50
    y = np.ones((n_samples, n_classes))
    if klass == Lasso:
        y = y.ravel()
    warning_message = (
        "Objective did not converge. You might want to"
        " increase the number of iterations."
    )
    with pytest.warns(ConvergenceWarning, match=warning_message):
        clf.fit(X, y)


def test_convergence_warnings():
    random_state = np.random.RandomState(0)
    X = random_state.standard_normal((1000, 500))
    y = random_state.standard_normal((1000, 3))

    # check that the model fails to converge (a negative dual gap cannot occur)
    with pytest.warns(ConvergenceWarning):
        MultiTaskElasticNet(max_iter=1, tol=-1).fit(X, y)

    # check that the model converges w/o warnings
    with pytest.warns(None) as record:
        MultiTaskElasticNet().fit(X, y)

    assert not record.list


def test_sparse_input_convergence_warning():
    X, y, _, _ = build_dataset(n_samples=1000, n_features=500)

    with pytest.warns(ConvergenceWarning):
        ElasticNet(max_iter=1, tol=0).fit(sparse.csr_matrix(X, dtype=np.float32), y)

    # check that the model converges w/o warnings
    with pytest.warns(None) as record:
        Lasso().fit(sparse.csr_matrix(X, dtype=np.float32), y)

    assert not record.list


@pytest.mark.parametrize(
    "precompute, inner_precompute",
    [
        (True, True),
        ("auto", False),
        (False, False),
    ],
)
def test_lassoCV_does_not_set_precompute(monkeypatch, precompute, inner_precompute):
    X, y, _, _ = build_dataset()
    calls = 0

    class LassoMock(Lasso):
        def fit(self, X, y):
            super().fit(X, y)
            nonlocal calls
            calls += 1
            assert self.precompute == inner_precompute

    monkeypatch.setattr("sklearn.linear_model._coordinate_descent.Lasso", LassoMock)
    clf = LassoCV(precompute=precompute)
    clf.fit(X, y)
    assert calls > 0


def test_multi_task_lasso_cv_dtype():
    n_samples, n_features = 10, 3
    rng = np.random.RandomState(42)
    X = rng.binomial(1, 0.5, size=(n_samples, n_features))
    X = X.astype(int)  # make it explicit that X is int
    y = X[:, [0, 0]].copy()
    est = MultiTaskLassoCV(n_alphas=5, fit_intercept=True).fit(X, y)
    assert_array_almost_equal(est.coef_, [[1, 0, 0]] * 2, decimal=3)


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("alpha", [0.01])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("precompute", [False, True])
def test_enet_sample_weight_consistency(fit_intercept, alpha, normalize, precompute):
    """Test that the impact of sample_weight is consistent."""
    rng = np.random.RandomState(0)
    n_samples, n_features = 10, 5

    X = rng.rand(n_samples, n_features)
    y = rng.rand(n_samples)
    params = dict(
        alpha=alpha,
        fit_intercept=fit_intercept,
        precompute=precompute,
        tol=1e-6,
        l1_ratio=0.5,
    )

    reg = ElasticNet(**params).fit(X, y)
    coef = reg.coef_.copy()
    if fit_intercept:
        intercept = reg.intercept_

    # sample_weight=np.ones(..) should be equivalent to sample_weight=None
    sample_weight = np.ones_like(y)
    reg.fit(X, y, sample_weight=sample_weight)
    assert_allclose(reg.coef_, coef, rtol=1e-6)
    if fit_intercept:
        assert_allclose(reg.intercept_, intercept)

    # sample_weight=None should be equivalent to sample_weight = number
    sample_weight = 123.0
    reg.fit(X, y, sample_weight=sample_weight)
    assert_allclose(reg.coef_, coef, rtol=1e-6)
    if fit_intercept:
        assert_allclose(reg.intercept_, intercept)

    # scaling of sample_weight should have no effect, cf. np.average()
    sample_weight = 2 * np.ones_like(y)
    reg.fit(X, y, sample_weight=sample_weight)
    assert_allclose(reg.coef_, coef, rtol=1e-6)
    if fit_intercept:
        assert_allclose(reg.intercept_, intercept)

    # setting one element of sample_weight to 0 is equivalent to removing
    # the corresponding sample
    sample_weight = np.ones_like(y)
    sample_weight[-1] = 0
    reg.fit(X, y, sample_weight=sample_weight)
    coef1 = reg.coef_.copy()
    if fit_intercept:
        intercept1 = reg.intercept_
    reg.fit(X[:-1], y[:-1])
    assert_allclose(reg.coef_, coef1, rtol=1e-6)
    if fit_intercept:
        assert_allclose(reg.intercept_, intercept1)

    # check that multiplying sample_weight by 2 is equivalent
    # to repeating corresponding samples twice
    X2 = np.concatenate([X, X[: n_samples // 2]], axis=0)
    y2 = np.concatenate([y, y[: n_samples // 2]])
    sample_weight_1 = np.ones(len(y))
    sample_weight_1[: n_samples // 2] = 2

    reg1 = ElasticNet(**params).fit(X, y, sample_weight=sample_weight_1)

    reg2 = ElasticNet(**params).fit(X2, y2, sample_weight=None)
    assert_allclose(reg1.coef_, reg2.coef_)


@pytest.mark.parametrize("estimator", (Lasso, ElasticNet))
def test_enet_sample_weight_sparse(estimator):
    reg = estimator()
    X = sparse.csc_matrix(np.zeros((3, 2)))
    y = np.array([-1, 0, 1])
    sw = np.array([1, 2, 3])
    with pytest.raises(
        ValueError, match="Sample weights do not.*support sparse matrices"
    ):
        reg.fit(X, y, sample_weight=sw, check_input=True)


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_enet_cv_sample_weight_correctness(fit_intercept):
    """Test that ElasticNetCV with sample weights gives correct results."""
    rng = np.random.RandomState(42)
    n_splits, n_samples, n_features = 3, 10, 5
    X = rng.rand(n_splits * n_samples, n_features)
    beta = rng.rand(n_features)
    beta[0:2] = 0
    y = X @ beta + rng.rand(n_splits * n_samples)
    sw = np.ones_like(y)

    # Set alphas, otherwise the two cv models might use different ones.
    if fit_intercept:
        alphas = np.linspace(0.001, 0.01, num=91)
    else:
        alphas = np.linspace(0.01, 0.1, num=91)

    # We weight the first fold 2 times more.
    sw[:n_samples] = 2
    groups_sw = np.r_[
        np.full(n_samples, 0), np.full(n_samples, 1), np.full(n_samples, 2)
    ]
    splits_sw = list(LeaveOneGroupOut().split(X, groups=groups_sw))
    reg_sw = ElasticNetCV(
        alphas=alphas,
        cv=splits_sw,
        fit_intercept=fit_intercept,
    )
    reg_sw.fit(X, y, sample_weight=sw)

    # We repeat the first fold 2 times and provide splits ourselves
    X = np.r_[X[:n_samples], X]
    y = np.r_[y[:n_samples], y]
    groups = np.r_[
        np.full(2 * n_samples, 0), np.full(n_samples, 1), np.full(n_samples, 2)
    ]
    splits = list(LeaveOneGroupOut().split(X, groups=groups))
    reg = ElasticNetCV(alphas=alphas, cv=splits, fit_intercept=fit_intercept)
    reg.fit(X, y)

    # ensure that we chose meaningful alphas, i.e. not boundaries
    assert alphas[0] < reg.alpha_ < alphas[-1]
    assert reg_sw.alpha_ == reg.alpha_
    assert_allclose(reg_sw.coef_, reg.coef_)
    assert reg_sw.intercept_ == pytest.approx(reg.intercept_)


@pytest.mark.parametrize("sample_weight", [False, True])
def test_enet_cv_grid_search(sample_weight):
    """Test that ElasticNetCV gives same result as GridSearchCV."""
    n_samples, n_features = 200, 10
    cv = 5
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        effective_rank=10,
        n_informative=n_features - 4,
        noise=10,
        random_state=0,
    )
    if sample_weight:
        sample_weight = np.linspace(1, 5, num=n_samples)
    else:
        sample_weight = None

    alphas = np.logspace(np.log10(1e-5), np.log10(1), num=10)
    l1_ratios = [0.1, 0.5, 0.9]
    reg = ElasticNetCV(cv=cv, alphas=alphas, l1_ratio=l1_ratios)
    reg.fit(X, y, sample_weight=sample_weight)

    param = {"alpha": alphas, "l1_ratio": l1_ratios}
    gs = GridSearchCV(
        estimator=ElasticNet(),
        param_grid=param,
        cv=cv,
        scoring="neg_mean_squared_error",
    ).fit(X, y, sample_weight=sample_weight)

    assert reg.l1_ratio_ == pytest.approx(gs.best_params_["l1_ratio"])
    assert reg.alpha_ == pytest.approx(gs.best_params_["alpha"])


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("l1_ratio", [0, 0.5, 1])
@pytest.mark.parametrize("precompute", [False, True])
def test_enet_cv_sample_weight_consistency(fit_intercept, l1_ratio, precompute):
    """Test that the impact of sample_weight is consistent."""
    rng = np.random.RandomState(0)
    n_samples, n_features = 10, 5

    X = rng.rand(n_samples, n_features)
    y = X.sum(axis=1) + rng.rand(n_samples)
    params = dict(
        l1_ratio=l1_ratio,
        fit_intercept=fit_intercept,
        precompute=precompute,
        tol=1e-6,
        cv=3,
    )

    if l1_ratio == 0:
        params.pop("l1_ratio", None)
        reg = LassoCV(**params).fit(X, y)
    else:
        reg = ElasticNetCV(**params).fit(X, y)
    coef = reg.coef_.copy()
    if fit_intercept:
        intercept = reg.intercept_

    # sample_weight=np.ones(..) should be equivalent to sample_weight=None
    sample_weight = np.ones_like(y)
    reg.fit(X, y, sample_weight=sample_weight)
    assert_allclose(reg.coef_, coef, rtol=1e-6)
    if fit_intercept:
        assert_allclose(reg.intercept_, intercept)

    # sample_weight=None should be equivalent to sample_weight = number
    sample_weight = 123.0
    reg.fit(X, y, sample_weight=sample_weight)
    assert_allclose(reg.coef_, coef, rtol=1e-6)
    if fit_intercept:
        assert_allclose(reg.intercept_, intercept)

    # scaling of sample_weight should have no effect, cf. np.average()
    sample_weight = 2 * np.ones_like(y)
    reg.fit(X, y, sample_weight=sample_weight)
    assert_allclose(reg.coef_, coef, rtol=1e-6)
    if fit_intercept:
        assert_allclose(reg.intercept_, intercept)


@pytest.mark.parametrize("estimator", (LassoCV, ElasticNetCV))
def test_enet_cv_sample_weight_sparse(estimator):
    reg = estimator()
    X = sparse.csc_matrix(np.zeros((3, 2)))
    y = np.array([-1, 0, 1])
    sw = np.array([1, 2, 3])
    with pytest.raises(
        ValueError, match="Sample weights do not.*support sparse matrices"
    ):
        reg.fit(X, y, sample_weight=sw)


@pytest.mark.parametrize("estimator", [ElasticNetCV, LassoCV])
def test_linear_models_cv_fit_with_loky(estimator):
    # LinearModelsCV.fit performs inplace operations on fancy-indexed memmapped
    # data when using the loky backend, causing an error due to unexpected
    # behavior of fancy indexing of read-only memmaps (cf. numpy#14132).

    if parse_version(joblib.__version__) < parse_version("0.12"):
        pytest.skip("loky backend does not exist in joblib <0.12")

    # Create a problem sufficiently large to cause memmapping (1MB).
    # Unfortunately the scikit-learn and joblib APIs do not make it possible to
    # change the max_nbyte of the inner Parallel call.
    X, y = make_regression(int(1e6) // 8 + 1, 1)
    assert X.nbytes > 1e6  # 1 MB
    with joblib.parallel_backend("loky"):
        estimator(n_jobs=2, cv=3).fit(X, y)


@pytest.mark.parametrize("check_input", [True, False])
def test_enet_sample_weight_does_not_overwrite_sample_weight(check_input):
    """Check that ElasticNet does not overwrite sample_weights."""

    rng = np.random.RandomState(0)
    n_samples, n_features = 10, 5

    X = rng.rand(n_samples, n_features)
    y = rng.rand(n_samples)

    sample_weight_1_25 = 1.25 * np.ones_like(y)
    sample_weight = sample_weight_1_25.copy()

    reg = ElasticNet()
    reg.fit(X, y, sample_weight=sample_weight, check_input=check_input)

    assert_array_equal(sample_weight, sample_weight_1_25)


# FIXME: 'normalize' to be removed in 1.2
@pytest.mark.filterwarnings("ignore:'normalize' was deprecated")
@pytest.mark.parametrize("ridge_alpha", [1e-1, 1.0, 1e6])
@pytest.mark.parametrize("normalize", [True, False])
def test_enet_ridge_consistency(normalize, ridge_alpha):
    # Check that ElasticNet(l1_ratio=0) converges to the same solution as Ridge
    # provided that the value of alpha is adapted.
    #
    # XXX: this test does not pass for weaker regularization (lower values of
    # ridge_alpha): it could be either a problem of ElasticNet or Ridge (less
    # likely) and depends on the dataset statistics: lower values for
    # effective_rank are more problematic in particular.

    rng = np.random.RandomState(42)
    n_samples = 300
    X, y = make_regression(
        n_samples=n_samples,
        n_features=100,
        effective_rank=10,
        n_informative=50,
        random_state=rng,
    )
    sw = rng.uniform(low=0.01, high=10, size=X.shape[0])
    alpha = 1.0
    common_params = dict(
        normalize=normalize,
        tol=1e-12,
    )
    ridge = Ridge(alpha=alpha, **common_params).fit(X, y, sample_weight=sw)
    if normalize:
        alpha_enet = alpha / n_samples
    else:
        alpha_enet = alpha / sw.sum()
    enet = ElasticNet(alpha=alpha_enet, l1_ratio=0, **common_params).fit(
        X, y, sample_weight=sw
    )
    assert_allclose(ridge.coef_, enet.coef_)
    assert_allclose(ridge.intercept_, enet.intercept_)


@pytest.mark.parametrize(
    "estimator",
    [
        Lasso(alpha=1.0),
        ElasticNet(alpha=1.0, l1_ratio=0.1),
    ],
)
@filterwarnings_normalize
def test_sample_weight_invariance(estimator):
    rng = np.random.RandomState(42)
    X, y = make_regression(
        n_samples=100,
        n_features=300,
        effective_rank=10,
        n_informative=50,
        random_state=rng,
    )
    normalize = False  # These tests don't work for normalize=True.
    sw = rng.uniform(low=0.01, high=2, size=X.shape[0])
    params = dict(normalize=normalize, tol=1e-12)

    # Check that setting some weights to 0 is equivalent to trimming the
    # samples:
    cutoff = X.shape[0] // 3
    sw_with_null = sw.copy()
    sw_with_null[:cutoff] = 0.0
    X_trimmed, y_trimmed = X[cutoff:, :], y[cutoff:]
    sw_trimmed = sw[cutoff:]

    reg_trimmed = (
        clone(estimator)
        .set_params(**params)
        .fit(X_trimmed, y_trimmed, sample_weight=sw_trimmed)
    )
    reg_null_weighted = (
        clone(estimator).set_params(**params).fit(X, y, sample_weight=sw_with_null)
    )
    assert_allclose(reg_null_weighted.coef_, reg_trimmed.coef_)
    assert_allclose(reg_null_weighted.intercept_, reg_trimmed.intercept_)

    # Check that duplicating the training dataset is equivalent to multiplying
    # the weights by 2:
    X_dup = np.concatenate([X, X], axis=0)
    y_dup = np.concatenate([y, y], axis=0)
    sw_dup = np.concatenate([sw, sw], axis=0)

    reg_2sw = clone(estimator).set_params(**params).fit(X, y, sample_weight=2 * sw)
    reg_dup = (
        clone(estimator).set_params(**params).fit(X_dup, y_dup, sample_weight=sw_dup)
    )

    assert_allclose(reg_2sw.coef_, reg_dup.coef_)
    assert_allclose(reg_2sw.intercept_, reg_dup.intercept_)
