import pickle

import joblib
import pytest
import numpy as np
import scipy.sparse as sp

from sklearn.utils._testing import assert_allclose
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_almost_equal
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.fixes import parse_version

from sklearn import linear_model, datasets, metrics
from sklearn.base import clone, is_classifier
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import LabelEncoder, scale, MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.linear_model import _sgd_fast as sgd_fast
from sklearn.model_selection import RandomizedSearchCV


def _update_kwargs(kwargs):
    if "random_state" not in kwargs:
        kwargs["random_state"] = 42

    if "tol" not in kwargs:
        kwargs["tol"] = None
    if "max_iter" not in kwargs:
        kwargs["max_iter"] = 5


class _SparseSGDClassifier(linear_model.SGDClassifier):
    def fit(self, X, y, *args, **kw):
        X = sp.csr_matrix(X)
        return super().fit(X, y, *args, **kw)

    def partial_fit(self, X, y, *args, **kw):
        X = sp.csr_matrix(X)
        return super().partial_fit(X, y, *args, **kw)

    def decision_function(self, X):
        X = sp.csr_matrix(X)
        return super().decision_function(X)

    def predict_proba(self, X):
        X = sp.csr_matrix(X)
        return super().predict_proba(X)


class _SparseSGDRegressor(linear_model.SGDRegressor):
    def fit(self, X, y, *args, **kw):
        X = sp.csr_matrix(X)
        return linear_model.SGDRegressor.fit(self, X, y, *args, **kw)

    def partial_fit(self, X, y, *args, **kw):
        X = sp.csr_matrix(X)
        return linear_model.SGDRegressor.partial_fit(self, X, y, *args, **kw)

    def decision_function(self, X, *args, **kw):
        # XXX untested as of v0.22
        X = sp.csr_matrix(X)
        return linear_model.SGDRegressor.decision_function(self, X, *args, **kw)


class _SparseSGDOneClassSVM(linear_model.SGDOneClassSVM):
    def fit(self, X, *args, **kw):
        X = sp.csr_matrix(X)
        return linear_model.SGDOneClassSVM.fit(self, X, *args, **kw)

    def partial_fit(self, X, *args, **kw):
        X = sp.csr_matrix(X)
        return linear_model.SGDOneClassSVM.partial_fit(self, X, *args, **kw)

    def decision_function(self, X, *args, **kw):
        X = sp.csr_matrix(X)
        return linear_model.SGDOneClassSVM.decision_function(self, X, *args, **kw)


def SGDClassifier(**kwargs):
    _update_kwargs(kwargs)
    return linear_model.SGDClassifier(**kwargs)


def SGDRegressor(**kwargs):
    _update_kwargs(kwargs)
    return linear_model.SGDRegressor(**kwargs)


def SGDOneClassSVM(**kwargs):
    _update_kwargs(kwargs)
    return linear_model.SGDOneClassSVM(**kwargs)


def SparseSGDClassifier(**kwargs):
    _update_kwargs(kwargs)
    return _SparseSGDClassifier(**kwargs)


def SparseSGDRegressor(**kwargs):
    _update_kwargs(kwargs)
    return _SparseSGDRegressor(**kwargs)


def SparseSGDOneClassSVM(**kwargs):
    _update_kwargs(kwargs)
    return _SparseSGDOneClassSVM(**kwargs)


# Test Data

# test sample 1
X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
Y = [1, 1, 1, 2, 2, 2]
T = np.array([[-1, -1], [2, 2], [3, 2]])
true_result = [1, 2, 2]

# test sample 2; string class labels
X2 = np.array(
    [
        [-1, 1],
        [-0.75, 0.5],
        [-1.5, 1.5],
        [1, 1],
        [0.75, 0.5],
        [1.5, 1.5],
        [-1, -1],
        [0, -0.5],
        [1, -1],
    ]
)
Y2 = ["one"] * 3 + ["two"] * 3 + ["three"] * 3
T2 = np.array([[-1.5, 0.5], [1, 2], [0, -2]])
true_result2 = ["one", "two", "three"]

# test sample 3
X3 = np.array(
    [
        [1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0],
    ]
)
Y3 = np.array([1, 1, 1, 1, 2, 2, 2, 2])

# test sample 4 - two more or less redundant feature groups
X4 = np.array(
    [
        [1, 0.9, 0.8, 0, 0, 0],
        [1, 0.84, 0.98, 0, 0, 0],
        [1, 0.96, 0.88, 0, 0, 0],
        [1, 0.91, 0.99, 0, 0, 0],
        [0, 0, 0, 0.89, 0.91, 1],
        [0, 0, 0, 0.79, 0.84, 1],
        [0, 0, 0, 0.91, 0.95, 1],
        [0, 0, 0, 0.93, 1, 1],
    ]
)
Y4 = np.array([1, 1, 1, 1, 2, 2, 2, 2])

iris = datasets.load_iris()

# test sample 5 - test sample 1 as binary classification problem
X5 = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
Y5 = [1, 1, 1, 2, 2, 2]
true_result5 = [0, 1, 1]


###############################################################################
# Common Test Case to classification and regression

# a simple implementation of ASGD to use for testing
# uses squared loss to find the gradient
def asgd(klass, X, y, eta, alpha, weight_init=None, intercept_init=0.0):
    if weight_init is None:
        weights = np.zeros(X.shape[1])
    else:
        weights = weight_init

    average_weights = np.zeros(X.shape[1])
    intercept = intercept_init
    average_intercept = 0.0
    decay = 1.0

    # sparse data has a fixed decay of .01
    if klass in (SparseSGDClassifier, SparseSGDRegressor):
        decay = 0.01

    for i, entry in enumerate(X):
        p = np.dot(entry, weights)
        p += intercept
        gradient = p - y[i]
        weights *= 1.0 - (eta * alpha)
        weights += -(eta * gradient * entry)
        intercept += -(eta * gradient) * decay

        average_weights *= i
        average_weights += weights
        average_weights /= i + 1.0

        average_intercept *= i
        average_intercept += intercept
        average_intercept /= i + 1.0

    return average_weights, average_intercept


@pytest.mark.parametrize(
    "klass",
    [
        SGDClassifier,
        SparseSGDClassifier,
        SGDRegressor,
        SparseSGDRegressor,
        SGDOneClassSVM,
        SparseSGDOneClassSVM,
    ],
)
@pytest.mark.parametrize("fit_method", ["fit", "partial_fit"])
@pytest.mark.parametrize(
    "params, err_msg",
    [
        ({"alpha": -0.1}, "alpha must be >= 0"),
        ({"penalty": "foobar", "l1_ratio": 0.85}, "Penalty foobar is not supported"),
        ({"loss": "foobar"}, "The loss foobar is not supported"),
        ({"l1_ratio": 1.1}, r"l1_ratio must be in \[0, 1\]"),
        ({"learning_rate": "<unknown>"}, "learning rate <unknown> is not supported"),
        ({"nu": -0.5}, r"nu must be in \(0, 1]"),
        ({"nu": 2}, r"nu must be in \(0, 1]"),
        ({"alpha": 0, "learning_rate": "optimal"}, "alpha must be > 0"),
        ({"eta0": 0, "learning_rate": "constant"}, "eta0 must be > 0"),
        ({"max_iter": -1}, "max_iter must be > zero"),
        ({"shuffle": "false"}, "shuffle must be either True or False"),
        ({"early_stopping": "false"}, "early_stopping must be either True or False"),
        (
            {"validation_fraction": -0.1},
            r"validation_fraction must be in range \(0, 1\)",
        ),
        ({"n_iter_no_change": 0}, "n_iter_no_change must be >= 1"),
    ],
    # Avoid long error messages in test names:
    # https://github.com/scikit-learn/scikit-learn/issues/21362
    ids=lambda x: x[:10].replace("]", "") if isinstance(x, str) else x,
)
def test_sgd_estimator_params_validation(klass, fit_method, params, err_msg):
    """Validate parameters in the different SGD estimators."""
    try:
        sgd_estimator = klass(**params)
    except TypeError as err:
        if "unexpected keyword argument" in str(err):
            # skip test if the parameter is not supported by the estimator
            return
        raise err

    with pytest.raises(ValueError, match=err_msg):
        if is_classifier(sgd_estimator) and fit_method == "partial_fit":
            fit_params = {"classes": np.unique(Y)}
        else:
            fit_params = {}
        getattr(sgd_estimator, fit_method)(X, Y, **fit_params)


def _test_warm_start(klass, X, Y, lr):
    # Test that explicit warm restart...
    clf = klass(alpha=0.01, eta0=0.01, shuffle=False, learning_rate=lr)
    clf.fit(X, Y)

    clf2 = klass(alpha=0.001, eta0=0.01, shuffle=False, learning_rate=lr)
    clf2.fit(X, Y, coef_init=clf.coef_.copy(), intercept_init=clf.intercept_.copy())

    # ... and implicit warm restart are equivalent.
    clf3 = klass(
        alpha=0.01, eta0=0.01, shuffle=False, warm_start=True, learning_rate=lr
    )
    clf3.fit(X, Y)

    assert clf3.t_ == clf.t_
    assert_array_almost_equal(clf3.coef_, clf.coef_)

    clf3.set_params(alpha=0.001)
    clf3.fit(X, Y)

    assert clf3.t_ == clf2.t_
    assert_array_almost_equal(clf3.coef_, clf2.coef_)


@pytest.mark.parametrize(
    "klass", [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor]
)
@pytest.mark.parametrize("lr", ["constant", "optimal", "invscaling", "adaptive"])
def test_warm_start(klass, lr):
    _test_warm_start(klass, X, Y, lr)


@pytest.mark.parametrize(
    "klass", [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor]
)
def test_input_format(klass):
    # Input format tests.
    clf = klass(alpha=0.01, shuffle=False)
    clf.fit(X, Y)
    Y_ = np.array(Y)[:, np.newaxis]

    Y_ = np.c_[Y_, Y_]
    with pytest.raises(ValueError):
        clf.fit(X, Y_)


@pytest.mark.parametrize(
    "klass", [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor]
)
def test_clone(klass):
    # Test whether clone works ok.
    clf = klass(alpha=0.01, penalty="l1")
    clf = clone(clf)
    clf.set_params(penalty="l2")
    clf.fit(X, Y)

    clf2 = klass(alpha=0.01, penalty="l2")
    clf2.fit(X, Y)

    assert_array_equal(clf.coef_, clf2.coef_)


@pytest.mark.parametrize(
    "klass",
    [
        SGDClassifier,
        SparseSGDClassifier,
        SGDRegressor,
        SparseSGDRegressor,
        SGDOneClassSVM,
        SparseSGDOneClassSVM,
    ],
)
def test_plain_has_no_average_attr(klass):
    clf = klass(average=True, eta0=0.01)
    clf.fit(X, Y)

    assert hasattr(clf, "_average_coef")
    assert hasattr(clf, "_average_intercept")
    assert hasattr(clf, "_standard_intercept")
    assert hasattr(clf, "_standard_coef")

    clf = klass()
    clf.fit(X, Y)

    assert not hasattr(clf, "_average_coef")
    assert not hasattr(clf, "_average_intercept")
    assert not hasattr(clf, "_standard_intercept")
    assert not hasattr(clf, "_standard_coef")


@pytest.mark.parametrize(
    "klass",
    [
        SGDClassifier,
        SparseSGDClassifier,
        SGDRegressor,
        SparseSGDRegressor,
        SGDOneClassSVM,
        SparseSGDOneClassSVM,
    ],
)
def test_late_onset_averaging_not_reached(klass):
    clf1 = klass(average=600)
    clf2 = klass()
    for _ in range(100):
        if is_classifier(clf1):
            clf1.partial_fit(X, Y, classes=np.unique(Y))
            clf2.partial_fit(X, Y, classes=np.unique(Y))
        else:
            clf1.partial_fit(X, Y)
            clf2.partial_fit(X, Y)

    assert_array_almost_equal(clf1.coef_, clf2.coef_, decimal=16)
    if klass in [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor]:
        assert_almost_equal(clf1.intercept_, clf2.intercept_, decimal=16)
    elif klass in [SGDOneClassSVM, SparseSGDOneClassSVM]:
        assert_allclose(clf1.offset_, clf2.offset_)


@pytest.mark.parametrize(
    "klass", [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor]
)
def test_late_onset_averaging_reached(klass):
    eta0 = 0.001
    alpha = 0.0001
    Y_encode = np.array(Y)
    Y_encode[Y_encode == 1] = -1.0
    Y_encode[Y_encode == 2] = 1.0

    clf1 = klass(
        average=7,
        learning_rate="constant",
        loss="squared_error",
        eta0=eta0,
        alpha=alpha,
        max_iter=2,
        shuffle=False,
    )
    clf2 = klass(
        average=0,
        learning_rate="constant",
        loss="squared_error",
        eta0=eta0,
        alpha=alpha,
        max_iter=1,
        shuffle=False,
    )

    clf1.fit(X, Y_encode)
    clf2.fit(X, Y_encode)

    average_weights, average_intercept = asgd(
        klass,
        X,
        Y_encode,
        eta0,
        alpha,
        weight_init=clf2.coef_.ravel(),
        intercept_init=clf2.intercept_,
    )

    assert_array_almost_equal(clf1.coef_.ravel(), average_weights.ravel(), decimal=16)
    assert_almost_equal(clf1.intercept_, average_intercept, decimal=16)


@pytest.mark.parametrize(
    "klass", [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor]
)
def test_early_stopping(klass):
    X = iris.data[iris.target > 0]
    Y = iris.target[iris.target > 0]
    for early_stopping in [True, False]:
        max_iter = 1000
        clf = klass(early_stopping=early_stopping, tol=1e-3, max_iter=max_iter).fit(
            X, Y
        )
        assert clf.n_iter_ < max_iter


@pytest.mark.parametrize(
    "klass", [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor]
)
def test_adaptive_longer_than_constant(klass):
    clf1 = klass(learning_rate="adaptive", eta0=0.01, tol=1e-3, max_iter=100)
    clf1.fit(iris.data, iris.target)
    clf2 = klass(learning_rate="constant", eta0=0.01, tol=1e-3, max_iter=100)
    clf2.fit(iris.data, iris.target)
    assert clf1.n_iter_ > clf2.n_iter_


@pytest.mark.parametrize(
    "klass", [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor]
)
def test_validation_set_not_used_for_training(klass):
    X, Y = iris.data, iris.target
    validation_fraction = 0.4
    seed = 42
    shuffle = False
    max_iter = 10
    clf1 = klass(
        early_stopping=True,
        random_state=np.random.RandomState(seed),
        validation_fraction=validation_fraction,
        learning_rate="constant",
        eta0=0.01,
        tol=None,
        max_iter=max_iter,
        shuffle=shuffle,
    )
    clf1.fit(X, Y)
    assert clf1.n_iter_ == max_iter

    clf2 = klass(
        early_stopping=False,
        random_state=np.random.RandomState(seed),
        learning_rate="constant",
        eta0=0.01,
        tol=None,
        max_iter=max_iter,
        shuffle=shuffle,
    )

    if is_classifier(clf2):
        cv = StratifiedShuffleSplit(test_size=validation_fraction, random_state=seed)
    else:
        cv = ShuffleSplit(test_size=validation_fraction, random_state=seed)
    idx_train, idx_val = next(cv.split(X, Y))
    idx_train = np.sort(idx_train)  # remove shuffling
    clf2.fit(X[idx_train], Y[idx_train])
    assert clf2.n_iter_ == max_iter

    assert_array_equal(clf1.coef_, clf2.coef_)


@pytest.mark.parametrize(
    "klass", [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor]
)
def test_n_iter_no_change(klass):
    X, Y = iris.data, iris.target
    # test that n_iter_ increases monotonically with n_iter_no_change
    for early_stopping in [True, False]:
        n_iter_list = [
            klass(
                early_stopping=early_stopping,
                n_iter_no_change=n_iter_no_change,
                tol=1e-4,
                max_iter=1000,
            )
            .fit(X, Y)
            .n_iter_
            for n_iter_no_change in [2, 3, 10]
        ]
        assert_array_equal(n_iter_list, sorted(n_iter_list))


@pytest.mark.parametrize(
    "klass", [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor]
)
def test_not_enough_sample_for_early_stopping(klass):
    # test an error is raised if the training or validation set is empty
    clf = klass(early_stopping=True, validation_fraction=0.99)
    with pytest.raises(ValueError):
        clf.fit(X3, Y3)


###############################################################################
# Classification Test Case


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_sgd_clf(klass):
    # Check that SGD gives any results :-)

    for loss in ("hinge", "squared_hinge", "log", "modified_huber"):
        clf = klass(
            penalty="l2",
            alpha=0.01,
            fit_intercept=True,
            loss=loss,
            max_iter=10,
            shuffle=True,
        )
        clf.fit(X, Y)
        # assert_almost_equal(clf.coef_[0], clf.coef_[1], decimal=7)
        assert_array_equal(clf.predict(T), true_result)


@pytest.mark.parametrize(
    "klass", [SGDClassifier, SparseSGDClassifier, SGDOneClassSVM, SparseSGDOneClassSVM]
)
def test_provide_coef(klass):
    """Check that the shape of `coef_init` is validated."""
    with pytest.raises(ValueError, match="Provided coef_init does not match dataset"):
        klass().fit(X, Y, coef_init=np.zeros((3,)))


@pytest.mark.parametrize(
    "klass, fit_params",
    [
        (SGDClassifier, {"intercept_init": np.zeros((3,))}),
        (SparseSGDClassifier, {"intercept_init": np.zeros((3,))}),
        (SGDOneClassSVM, {"offset_init": np.zeros((3,))}),
        (SparseSGDOneClassSVM, {"offset_init": np.zeros((3,))}),
    ],
)
def test_set_intercept_offset(klass, fit_params):
    """Check that `intercept_init` or `offset_init` is validated."""
    sgd_estimator = klass()
    with pytest.raises(ValueError, match="does not match dataset"):
        sgd_estimator.fit(X, Y, **fit_params)


@pytest.mark.parametrize(
    "klass", [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor]
)
def test_sgd_early_stopping_with_partial_fit(klass):
    """Check that we raise an error for `early_stopping` used with
    `partial_fit`.
    """
    err_msg = "early_stopping should be False with partial_fit"
    with pytest.raises(ValueError, match=err_msg):
        klass(early_stopping=True).partial_fit(X, Y)


@pytest.mark.parametrize(
    "klass, fit_params",
    [
        (SGDClassifier, {"intercept_init": 0}),
        (SparseSGDClassifier, {"intercept_init": 0}),
        (SGDOneClassSVM, {"offset_init": 0}),
        (SparseSGDOneClassSVM, {"offset_init": 0}),
    ],
)
def test_set_intercept_offset_binary(klass, fit_params):
    """Check that we can pass a scaler with binary classification to
    `intercept_init` or `offset_init`."""
    klass().fit(X5, Y5, **fit_params)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_average_binary_computed_correctly(klass):
    # Checks the SGDClassifier correctly computes the average weights
    eta = 0.1
    alpha = 2.0
    n_samples = 20
    n_features = 10
    rng = np.random.RandomState(0)
    X = rng.normal(size=(n_samples, n_features))
    w = rng.normal(size=n_features)

    clf = klass(
        loss="squared_error",
        learning_rate="constant",
        eta0=eta,
        alpha=alpha,
        fit_intercept=True,
        max_iter=1,
        average=True,
        shuffle=False,
    )

    # simple linear function without noise
    y = np.dot(X, w)
    y = np.sign(y)

    clf.fit(X, y)

    average_weights, average_intercept = asgd(klass, X, y, eta, alpha)
    average_weights = average_weights.reshape(1, -1)
    assert_array_almost_equal(clf.coef_, average_weights, decimal=14)
    assert_almost_equal(clf.intercept_, average_intercept, decimal=14)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_set_intercept_to_intercept(klass):
    # Checks intercept_ shape consistency for the warm starts
    # Inconsistent intercept_ shape.
    clf = klass().fit(X5, Y5)
    klass().fit(X5, Y5, intercept_init=clf.intercept_)
    clf = klass().fit(X, Y)
    klass().fit(X, Y, intercept_init=clf.intercept_)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_sgd_at_least_two_labels(klass):
    # Target must have at least two labels
    clf = klass(alpha=0.01, max_iter=20)
    with pytest.raises(ValueError):
        clf.fit(X2, np.ones(9))


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_partial_fit_weight_class_balanced(klass):
    # partial_fit with class_weight='balanced' not supported"""
    regex = (
        r"class_weight 'balanced' is not supported for "
        r"partial_fit\. In order to use 'balanced' weights, "
        r"use compute_class_weight\('balanced', classes=classes, y=y\). "
        r"In place of y you can us a large enough sample "
        r"of the full training set target to properly "
        r"estimate the class frequency distributions\. "
        r"Pass the resulting weights as the class_weight "
        r"parameter\."
    )
    with pytest.raises(ValueError, match=regex):
        klass(class_weight="balanced").partial_fit(X, Y, classes=np.unique(Y))


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_sgd_multiclass(klass):
    # Multi-class test case
    clf = klass(alpha=0.01, max_iter=20).fit(X2, Y2)
    assert clf.coef_.shape == (3, 2)
    assert clf.intercept_.shape == (3,)
    assert clf.decision_function([[0, 0]]).shape == (1, 3)
    pred = clf.predict(T2)
    assert_array_equal(pred, true_result2)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_sgd_multiclass_average(klass):
    eta = 0.001
    alpha = 0.01
    # Multi-class average test case
    clf = klass(
        loss="squared_error",
        learning_rate="constant",
        eta0=eta,
        alpha=alpha,
        fit_intercept=True,
        max_iter=1,
        average=True,
        shuffle=False,
    )

    np_Y2 = np.array(Y2)
    clf.fit(X2, np_Y2)
    classes = np.unique(np_Y2)

    for i, cl in enumerate(classes):
        y_i = np.ones(np_Y2.shape[0])
        y_i[np_Y2 != cl] = -1
        average_coef, average_intercept = asgd(klass, X2, y_i, eta, alpha)
        assert_array_almost_equal(average_coef, clf.coef_[i], decimal=16)
        assert_almost_equal(average_intercept, clf.intercept_[i], decimal=16)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_sgd_multiclass_with_init_coef(klass):
    # Multi-class test case
    clf = klass(alpha=0.01, max_iter=20)
    clf.fit(X2, Y2, coef_init=np.zeros((3, 2)), intercept_init=np.zeros(3))
    assert clf.coef_.shape == (3, 2)
    assert clf.intercept_.shape, (3,)
    pred = clf.predict(T2)
    assert_array_equal(pred, true_result2)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_sgd_multiclass_njobs(klass):
    # Multi-class test case with multi-core support
    clf = klass(alpha=0.01, max_iter=20, n_jobs=2).fit(X2, Y2)
    assert clf.coef_.shape == (3, 2)
    assert clf.intercept_.shape == (3,)
    assert clf.decision_function([[0, 0]]).shape == (1, 3)
    pred = clf.predict(T2)
    assert_array_equal(pred, true_result2)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_set_coef_multiclass(klass):
    # Checks coef_init and intercept_init shape for multi-class
    # problems
    # Provided coef_ does not match dataset
    clf = klass()
    with pytest.raises(ValueError):
        clf.fit(X2, Y2, coef_init=np.zeros((2, 2)))

    # Provided coef_ does match dataset
    clf = klass().fit(X2, Y2, coef_init=np.zeros((3, 2)))

    # Provided intercept_ does not match dataset
    clf = klass()
    with pytest.raises(ValueError):
        clf.fit(X2, Y2, intercept_init=np.zeros((1,)))

    # Provided intercept_ does match dataset.
    clf = klass().fit(X2, Y2, intercept_init=np.zeros((3,)))


# TODO: Remove filterwarnings in v1.2.
@pytest.mark.filterwarnings("ignore:.*squared_loss.*:FutureWarning")
@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_sgd_predict_proba_method_access(klass):
    # Checks that SGDClassifier predict_proba and predict_log_proba methods
    # can either be accessed or raise an appropriate error message
    # otherwise. See
    # https://github.com/scikit-learn/scikit-learn/issues/10938 for more
    # details.
    for loss in linear_model.SGDClassifier.loss_functions:
        clf = SGDClassifier(loss=loss)
        if loss in ("log", "modified_huber"):
            assert hasattr(clf, "predict_proba")
            assert hasattr(clf, "predict_log_proba")
        else:
            message = "probability estimates are not available for loss={!r}".format(
                loss
            )
            assert not hasattr(clf, "predict_proba")
            assert not hasattr(clf, "predict_log_proba")
            with pytest.raises(AttributeError, match=message):
                clf.predict_proba
            with pytest.raises(AttributeError, match=message):
                clf.predict_log_proba


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_sgd_proba(klass):
    # Check SGD.predict_proba

    # Hinge loss does not allow for conditional prob estimate.
    # We cannot use the factory here, because it defines predict_proba
    # anyway.
    clf = SGDClassifier(loss="hinge", alpha=0.01, max_iter=10, tol=None).fit(X, Y)
    assert not hasattr(clf, "predict_proba")
    assert not hasattr(clf, "predict_log_proba")

    # log and modified_huber losses can output probability estimates
    # binary case
    for loss in ["log", "modified_huber"]:
        clf = klass(loss=loss, alpha=0.01, max_iter=10)
        clf.fit(X, Y)
        p = clf.predict_proba([[3, 2]])
        assert p[0, 1] > 0.5
        p = clf.predict_proba([[-1, -1]])
        assert p[0, 1] < 0.5

        p = clf.predict_log_proba([[3, 2]])
        assert p[0, 1] > p[0, 0]
        p = clf.predict_log_proba([[-1, -1]])
        assert p[0, 1] < p[0, 0]

    # log loss multiclass probability estimates
    clf = klass(loss="log", alpha=0.01, max_iter=10).fit(X2, Y2)

    d = clf.decision_function([[0.1, -0.1], [0.3, 0.2]])
    p = clf.predict_proba([[0.1, -0.1], [0.3, 0.2]])
    assert_array_equal(np.argmax(p, axis=1), np.argmax(d, axis=1))
    assert_almost_equal(p[0].sum(), 1)
    assert np.all(p[0] >= 0)

    p = clf.predict_proba([[-1, -1]])
    d = clf.decision_function([[-1, -1]])
    assert_array_equal(np.argsort(p[0]), np.argsort(d[0]))

    lp = clf.predict_log_proba([[3, 2]])
    p = clf.predict_proba([[3, 2]])
    assert_array_almost_equal(np.log(p), lp)

    lp = clf.predict_log_proba([[-1, -1]])
    p = clf.predict_proba([[-1, -1]])
    assert_array_almost_equal(np.log(p), lp)

    # Modified Huber multiclass probability estimates; requires a separate
    # test because the hard zero/one probabilities may destroy the
    # ordering present in decision_function output.
    clf = klass(loss="modified_huber", alpha=0.01, max_iter=10)
    clf.fit(X2, Y2)
    d = clf.decision_function([[3, 2]])
    p = clf.predict_proba([[3, 2]])
    if klass != SparseSGDClassifier:
        assert np.argmax(d, axis=1) == np.argmax(p, axis=1)
    else:  # XXX the sparse test gets a different X2 (?)
        assert np.argmin(d, axis=1) == np.argmin(p, axis=1)

    # the following sample produces decision_function values < -1,
    # which would cause naive normalization to fail (see comment
    # in SGDClassifier.predict_proba)
    x = X.mean(axis=0)
    d = clf.decision_function([x])
    if np.all(d < -1):  # XXX not true in sparse test case (why?)
        p = clf.predict_proba([x])
        assert_array_almost_equal(p[0], [1 / 3.0] * 3)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_sgd_l1(klass):
    # Test L1 regularization
    n = len(X4)
    rng = np.random.RandomState(13)
    idx = np.arange(n)
    rng.shuffle(idx)

    X = X4[idx, :]
    Y = Y4[idx]

    clf = klass(
        penalty="l1",
        alpha=0.2,
        fit_intercept=False,
        max_iter=2000,
        tol=None,
        shuffle=False,
    )
    clf.fit(X, Y)
    assert_array_equal(clf.coef_[0, 1:-1], np.zeros((4,)))
    pred = clf.predict(X)
    assert_array_equal(pred, Y)

    # test sparsify with dense inputs
    clf.sparsify()
    assert sp.issparse(clf.coef_)
    pred = clf.predict(X)
    assert_array_equal(pred, Y)

    # pickle and unpickle with sparse coef_
    clf = pickle.loads(pickle.dumps(clf))
    assert sp.issparse(clf.coef_)
    pred = clf.predict(X)
    assert_array_equal(pred, Y)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_class_weights(klass):
    # Test class weights.
    X = np.array([[-1.0, -1.0], [-1.0, 0], [-0.8, -1.0], [1.0, 1.0], [1.0, 0.0]])
    y = [1, 1, 1, -1, -1]

    clf = klass(alpha=0.1, max_iter=1000, fit_intercept=False, class_weight=None)
    clf.fit(X, y)
    assert_array_equal(clf.predict([[0.2, -1.0]]), np.array([1]))

    # we give a small weights to class 1
    clf = klass(alpha=0.1, max_iter=1000, fit_intercept=False, class_weight={1: 0.001})
    clf.fit(X, y)

    # now the hyperplane should rotate clock-wise and
    # the prediction on this point should shift
    assert_array_equal(clf.predict([[0.2, -1.0]]), np.array([-1]))


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_equal_class_weight(klass):
    # Test if equal class weights approx. equals no class weights.
    X = [[1, 0], [1, 0], [0, 1], [0, 1]]
    y = [0, 0, 1, 1]
    clf = klass(alpha=0.1, max_iter=1000, class_weight=None)
    clf.fit(X, y)

    X = [[1, 0], [0, 1]]
    y = [0, 1]
    clf_weighted = klass(alpha=0.1, max_iter=1000, class_weight={0: 0.5, 1: 0.5})
    clf_weighted.fit(X, y)

    # should be similar up to some epsilon due to learning rate schedule
    assert_almost_equal(clf.coef_, clf_weighted.coef_, decimal=2)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_wrong_class_weight_label(klass):
    # ValueError due to not existing class label.
    clf = klass(alpha=0.1, max_iter=1000, class_weight={0: 0.5})
    with pytest.raises(ValueError):
        clf.fit(X, Y)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_wrong_class_weight_format(klass):
    # ValueError due to wrong class_weight argument type.
    clf = klass(alpha=0.1, max_iter=1000, class_weight=[0.5])
    with pytest.raises(ValueError):
        clf.fit(X, Y)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_weights_multiplied(klass):
    # Tests that class_weight and sample_weight are multiplicative
    class_weights = {1: 0.6, 2: 0.3}
    rng = np.random.RandomState(0)
    sample_weights = rng.random_sample(Y4.shape[0])
    multiplied_together = np.copy(sample_weights)
    multiplied_together[Y4 == 1] *= class_weights[1]
    multiplied_together[Y4 == 2] *= class_weights[2]

    clf1 = klass(alpha=0.1, max_iter=20, class_weight=class_weights)
    clf2 = klass(alpha=0.1, max_iter=20)

    clf1.fit(X4, Y4, sample_weight=sample_weights)
    clf2.fit(X4, Y4, sample_weight=multiplied_together)

    assert_almost_equal(clf1.coef_, clf2.coef_)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_balanced_weight(klass):
    # Test class weights for imbalanced data"""
    # compute reference metrics on iris dataset that is quite balanced by
    # default
    X, y = iris.data, iris.target
    X = scale(X)
    idx = np.arange(X.shape[0])
    rng = np.random.RandomState(6)
    rng.shuffle(idx)
    X = X[idx]
    y = y[idx]
    clf = klass(alpha=0.0001, max_iter=1000, class_weight=None, shuffle=False).fit(X, y)
    f1 = metrics.f1_score(y, clf.predict(X), average="weighted")
    assert_almost_equal(f1, 0.96, decimal=1)

    # make the same prediction using balanced class_weight
    clf_balanced = klass(
        alpha=0.0001, max_iter=1000, class_weight="balanced", shuffle=False
    ).fit(X, y)
    f1 = metrics.f1_score(y, clf_balanced.predict(X), average="weighted")
    assert_almost_equal(f1, 0.96, decimal=1)

    # Make sure that in the balanced case it does not change anything
    # to use "balanced"
    assert_array_almost_equal(clf.coef_, clf_balanced.coef_, 6)

    # build an very very imbalanced dataset out of iris data
    X_0 = X[y == 0, :]
    y_0 = y[y == 0]

    X_imbalanced = np.vstack([X] + [X_0] * 10)
    y_imbalanced = np.concatenate([y] + [y_0] * 10)

    # fit a model on the imbalanced data without class weight info
    clf = klass(max_iter=1000, class_weight=None, shuffle=False)
    clf.fit(X_imbalanced, y_imbalanced)
    y_pred = clf.predict(X)
    assert metrics.f1_score(y, y_pred, average="weighted") < 0.96

    # fit a model with balanced class_weight enabled
    clf = klass(max_iter=1000, class_weight="balanced", shuffle=False)
    clf.fit(X_imbalanced, y_imbalanced)
    y_pred = clf.predict(X)
    assert metrics.f1_score(y, y_pred, average="weighted") > 0.96


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_sample_weights(klass):
    # Test weights on individual samples
    X = np.array([[-1.0, -1.0], [-1.0, 0], [-0.8, -1.0], [1.0, 1.0], [1.0, 0.0]])
    y = [1, 1, 1, -1, -1]

    clf = klass(alpha=0.1, max_iter=1000, fit_intercept=False)
    clf.fit(X, y)
    assert_array_equal(clf.predict([[0.2, -1.0]]), np.array([1]))

    # we give a small weights to class 1
    clf.fit(X, y, sample_weight=[0.001] * 3 + [1] * 2)

    # now the hyperplane should rotate clock-wise and
    # the prediction on this point should shift
    assert_array_equal(clf.predict([[0.2, -1.0]]), np.array([-1]))


@pytest.mark.parametrize(
    "klass", [SGDClassifier, SparseSGDClassifier, SGDOneClassSVM, SparseSGDOneClassSVM]
)
def test_wrong_sample_weights(klass):
    # Test if ValueError is raised if sample_weight has wrong shape
    if klass in [SGDClassifier, SparseSGDClassifier]:
        clf = klass(alpha=0.1, max_iter=1000, fit_intercept=False)
    elif klass in [SGDOneClassSVM, SparseSGDOneClassSVM]:
        clf = klass(nu=0.1, max_iter=1000, fit_intercept=False)
    # provided sample_weight too long
    with pytest.raises(ValueError):
        clf.fit(X, Y, sample_weight=np.arange(7))


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_partial_fit_exception(klass):
    clf = klass(alpha=0.01)
    # classes was not specified
    with pytest.raises(ValueError):
        clf.partial_fit(X3, Y3)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_partial_fit_binary(klass):
    third = X.shape[0] // 3
    clf = klass(alpha=0.01)
    classes = np.unique(Y)

    clf.partial_fit(X[:third], Y[:third], classes=classes)
    assert clf.coef_.shape == (1, X.shape[1])
    assert clf.intercept_.shape == (1,)
    assert clf.decision_function([[0, 0]]).shape == (1,)
    id1 = id(clf.coef_.data)

    clf.partial_fit(X[third:], Y[third:])
    id2 = id(clf.coef_.data)
    # check that coef_ haven't been re-allocated
    assert id1, id2

    y_pred = clf.predict(T)
    assert_array_equal(y_pred, true_result)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_partial_fit_multiclass(klass):
    third = X2.shape[0] // 3
    clf = klass(alpha=0.01)
    classes = np.unique(Y2)

    clf.partial_fit(X2[:third], Y2[:third], classes=classes)
    assert clf.coef_.shape == (3, X2.shape[1])
    assert clf.intercept_.shape == (3,)
    assert clf.decision_function([[0, 0]]).shape == (1, 3)
    id1 = id(clf.coef_.data)

    clf.partial_fit(X2[third:], Y2[third:])
    id2 = id(clf.coef_.data)
    # check that coef_ haven't been re-allocated
    assert id1, id2


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_partial_fit_multiclass_average(klass):
    third = X2.shape[0] // 3
    clf = klass(alpha=0.01, average=X2.shape[0])
    classes = np.unique(Y2)

    clf.partial_fit(X2[:third], Y2[:third], classes=classes)
    assert clf.coef_.shape == (3, X2.shape[1])
    assert clf.intercept_.shape == (3,)

    clf.partial_fit(X2[third:], Y2[third:])
    assert clf.coef_.shape == (3, X2.shape[1])
    assert clf.intercept_.shape == (3,)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_fit_then_partial_fit(klass):
    # Partial_fit should work after initial fit in the multiclass case.
    # Non-regression test for #2496; fit would previously produce a
    # Fortran-ordered coef_ that subsequent partial_fit couldn't handle.
    clf = klass()
    clf.fit(X2, Y2)
    clf.partial_fit(X2, Y2)  # no exception here


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
@pytest.mark.parametrize("lr", ["constant", "optimal", "invscaling", "adaptive"])
def test_partial_fit_equal_fit_classif(klass, lr):
    for X_, Y_, T_ in ((X, Y, T), (X2, Y2, T2)):
        clf = klass(alpha=0.01, eta0=0.01, max_iter=2, learning_rate=lr, shuffle=False)
        clf.fit(X_, Y_)
        y_pred = clf.decision_function(T_)
        t = clf.t_

        classes = np.unique(Y_)
        clf = klass(alpha=0.01, eta0=0.01, learning_rate=lr, shuffle=False)
        for i in range(2):
            clf.partial_fit(X_, Y_, classes=classes)
        y_pred2 = clf.decision_function(T_)

        assert clf.t_ == t
        assert_array_almost_equal(y_pred, y_pred2, decimal=2)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_regression_losses(klass):
    random_state = np.random.RandomState(1)
    clf = klass(
        alpha=0.01,
        learning_rate="constant",
        eta0=0.1,
        loss="epsilon_insensitive",
        random_state=random_state,
    )
    clf.fit(X, Y)
    assert 1.0 == np.mean(clf.predict(X) == Y)

    clf = klass(
        alpha=0.01,
        learning_rate="constant",
        eta0=0.1,
        loss="squared_epsilon_insensitive",
        random_state=random_state,
    )
    clf.fit(X, Y)
    assert 1.0 == np.mean(clf.predict(X) == Y)

    clf = klass(alpha=0.01, loss="huber", random_state=random_state)
    clf.fit(X, Y)
    assert 1.0 == np.mean(clf.predict(X) == Y)

    clf = klass(
        alpha=0.01,
        learning_rate="constant",
        eta0=0.01,
        loss="squared_error",
        random_state=random_state,
    )
    clf.fit(X, Y)
    assert 1.0 == np.mean(clf.predict(X) == Y)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_warm_start_multiclass(klass):
    _test_warm_start(klass, X2, Y2, "optimal")


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_multiple_fit(klass):
    # Test multiple calls of fit w/ different shaped inputs.
    clf = klass(alpha=0.01, shuffle=False)
    clf.fit(X, Y)
    assert hasattr(clf, "coef_")

    # Non-regression test: try fitting with a different label set.
    y = [["ham", "spam"][i] for i in LabelEncoder().fit_transform(Y)]
    clf.fit(X[:, :-1], y)


###############################################################################
# Regression Test Case


@pytest.mark.parametrize("klass", [SGDRegressor, SparseSGDRegressor])
def test_sgd_reg(klass):
    # Check that SGD gives any results.
    clf = klass(alpha=0.1, max_iter=2, fit_intercept=False)
    clf.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
    assert clf.coef_[0] == clf.coef_[1]


@pytest.mark.parametrize("klass", [SGDRegressor, SparseSGDRegressor])
def test_sgd_averaged_computed_correctly(klass):
    # Tests the average regressor matches the naive implementation

    eta = 0.001
    alpha = 0.01
    n_samples = 20
    n_features = 10
    rng = np.random.RandomState(0)
    X = rng.normal(size=(n_samples, n_features))
    w = rng.normal(size=n_features)

    # simple linear function without noise
    y = np.dot(X, w)

    clf = klass(
        loss="squared_error",
        learning_rate="constant",
        eta0=eta,
        alpha=alpha,
        fit_intercept=True,
        max_iter=1,
        average=True,
        shuffle=False,
    )

    clf.fit(X, y)
    average_weights, average_intercept = asgd(klass, X, y, eta, alpha)

    assert_array_almost_equal(clf.coef_, average_weights, decimal=16)
    assert_almost_equal(clf.intercept_, average_intercept, decimal=16)


@pytest.mark.parametrize("klass", [SGDRegressor, SparseSGDRegressor])
def test_sgd_averaged_partial_fit(klass):
    # Tests whether the partial fit yields the same average as the fit
    eta = 0.001
    alpha = 0.01
    n_samples = 20
    n_features = 10
    rng = np.random.RandomState(0)
    X = rng.normal(size=(n_samples, n_features))
    w = rng.normal(size=n_features)

    # simple linear function without noise
    y = np.dot(X, w)

    clf = klass(
        loss="squared_error",
        learning_rate="constant",
        eta0=eta,
        alpha=alpha,
        fit_intercept=True,
        max_iter=1,
        average=True,
        shuffle=False,
    )

    clf.partial_fit(X[: int(n_samples / 2)][:], y[: int(n_samples / 2)])
    clf.partial_fit(X[int(n_samples / 2) :][:], y[int(n_samples / 2) :])
    average_weights, average_intercept = asgd(klass, X, y, eta, alpha)

    assert_array_almost_equal(clf.coef_, average_weights, decimal=16)
    assert_almost_equal(clf.intercept_[0], average_intercept, decimal=16)


@pytest.mark.parametrize("klass", [SGDRegressor, SparseSGDRegressor])
def test_average_sparse(klass):
    # Checks the average weights on data with 0s

    eta = 0.001
    alpha = 0.01
    clf = klass(
        loss="squared_error",
        learning_rate="constant",
        eta0=eta,
        alpha=alpha,
        fit_intercept=True,
        max_iter=1,
        average=True,
        shuffle=False,
    )

    n_samples = Y3.shape[0]

    clf.partial_fit(X3[: int(n_samples / 2)][:], Y3[: int(n_samples / 2)])
    clf.partial_fit(X3[int(n_samples / 2) :][:], Y3[int(n_samples / 2) :])
    average_weights, average_intercept = asgd(klass, X3, Y3, eta, alpha)

    assert_array_almost_equal(clf.coef_, average_weights, decimal=16)
    assert_almost_equal(clf.intercept_, average_intercept, decimal=16)


@pytest.mark.parametrize("klass", [SGDRegressor, SparseSGDRegressor])
def test_sgd_least_squares_fit(klass):
    xmin, xmax = -5, 5
    n_samples = 100
    rng = np.random.RandomState(0)
    X = np.linspace(xmin, xmax, n_samples).reshape(n_samples, 1)

    # simple linear function without noise
    y = 0.5 * X.ravel()

    clf = klass(loss="squared_error", alpha=0.1, max_iter=20, fit_intercept=False)
    clf.fit(X, y)
    score = clf.score(X, y)
    assert score > 0.99

    # simple linear function with noise
    y = 0.5 * X.ravel() + rng.randn(n_samples, 1).ravel()

    clf = klass(loss="squared_error", alpha=0.1, max_iter=20, fit_intercept=False)
    clf.fit(X, y)
    score = clf.score(X, y)
    assert score > 0.5


@pytest.mark.parametrize("klass", [SGDRegressor, SparseSGDRegressor])
def test_sgd_epsilon_insensitive(klass):
    xmin, xmax = -5, 5
    n_samples = 100
    rng = np.random.RandomState(0)
    X = np.linspace(xmin, xmax, n_samples).reshape(n_samples, 1)

    # simple linear function without noise
    y = 0.5 * X.ravel()

    clf = klass(
        loss="epsilon_insensitive",
        epsilon=0.01,
        alpha=0.1,
        max_iter=20,
        fit_intercept=False,
    )
    clf.fit(X, y)
    score = clf.score(X, y)
    assert score > 0.99

    # simple linear function with noise
    y = 0.5 * X.ravel() + rng.randn(n_samples, 1).ravel()

    clf = klass(
        loss="epsilon_insensitive",
        epsilon=0.01,
        alpha=0.1,
        max_iter=20,
        fit_intercept=False,
    )
    clf.fit(X, y)
    score = clf.score(X, y)
    assert score > 0.5


@pytest.mark.parametrize("klass", [SGDRegressor, SparseSGDRegressor])
def test_sgd_huber_fit(klass):
    xmin, xmax = -5, 5
    n_samples = 100
    rng = np.random.RandomState(0)
    X = np.linspace(xmin, xmax, n_samples).reshape(n_samples, 1)

    # simple linear function without noise
    y = 0.5 * X.ravel()

    clf = klass(loss="huber", epsilon=0.1, alpha=0.1, max_iter=20, fit_intercept=False)
    clf.fit(X, y)
    score = clf.score(X, y)
    assert score > 0.99

    # simple linear function with noise
    y = 0.5 * X.ravel() + rng.randn(n_samples, 1).ravel()

    clf = klass(loss="huber", epsilon=0.1, alpha=0.1, max_iter=20, fit_intercept=False)
    clf.fit(X, y)
    score = clf.score(X, y)
    assert score > 0.5


@pytest.mark.parametrize("klass", [SGDRegressor, SparseSGDRegressor])
def test_elasticnet_convergence(klass):
    # Check that the SGD output is consistent with coordinate descent

    n_samples, n_features = 1000, 5
    rng = np.random.RandomState(0)
    X = rng.randn(n_samples, n_features)
    # ground_truth linear model that generate y from X and to which the
    # models should converge if the regularizer would be set to 0.0
    ground_truth_coef = rng.randn(n_features)
    y = np.dot(X, ground_truth_coef)

    # XXX: alpha = 0.1 seems to cause convergence problems
    for alpha in [0.01, 0.001]:
        for l1_ratio in [0.5, 0.8, 1.0]:
            cd = linear_model.ElasticNet(
                alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False
            )
            cd.fit(X, y)
            sgd = klass(
                penalty="elasticnet",
                max_iter=50,
                alpha=alpha,
                l1_ratio=l1_ratio,
                fit_intercept=False,
            )
            sgd.fit(X, y)
            err_msg = (
                "cd and sgd did not converge to comparable "
                "results for alpha=%f and l1_ratio=%f" % (alpha, l1_ratio)
            )
            assert_almost_equal(cd.coef_, sgd.coef_, decimal=2, err_msg=err_msg)


@ignore_warnings
@pytest.mark.parametrize("klass", [SGDRegressor, SparseSGDRegressor])
def test_partial_fit(klass):
    third = X.shape[0] // 3
    clf = klass(alpha=0.01)

    clf.partial_fit(X[:third], Y[:third])
    assert clf.coef_.shape == (X.shape[1],)
    assert clf.intercept_.shape == (1,)
    assert clf.predict([[0, 0]]).shape == (1,)
    id1 = id(clf.coef_.data)

    clf.partial_fit(X[third:], Y[third:])
    id2 = id(clf.coef_.data)
    # check that coef_ haven't been re-allocated
    assert id1, id2


@pytest.mark.parametrize("klass", [SGDRegressor, SparseSGDRegressor])
@pytest.mark.parametrize("lr", ["constant", "optimal", "invscaling", "adaptive"])
def test_partial_fit_equal_fit(klass, lr):
    clf = klass(alpha=0.01, max_iter=2, eta0=0.01, learning_rate=lr, shuffle=False)
    clf.fit(X, Y)
    y_pred = clf.predict(T)
    t = clf.t_

    clf = klass(alpha=0.01, eta0=0.01, learning_rate=lr, shuffle=False)
    for i in range(2):
        clf.partial_fit(X, Y)
    y_pred2 = clf.predict(T)

    assert clf.t_ == t
    assert_array_almost_equal(y_pred, y_pred2, decimal=2)


@pytest.mark.parametrize("klass", [SGDRegressor, SparseSGDRegressor])
def test_loss_function_epsilon(klass):
    clf = klass(epsilon=0.9)
    clf.set_params(epsilon=0.1)
    assert clf.loss_functions["huber"][1] == 0.1


###############################################################################
# SGD One Class SVM Test Case

# a simple implementation of ASGD to use for testing SGDOneClassSVM
def asgd_oneclass(klass, X, eta, nu, coef_init=None, offset_init=0.0):
    if coef_init is None:
        coef = np.zeros(X.shape[1])
    else:
        coef = coef_init

    average_coef = np.zeros(X.shape[1])
    offset = offset_init
    intercept = 1 - offset
    average_intercept = 0.0
    decay = 1.0

    # sparse data has a fixed decay of .01
    if klass == SparseSGDOneClassSVM:
        decay = 0.01

    for i, entry in enumerate(X):
        p = np.dot(entry, coef)
        p += intercept
        if p <= 1.0:
            gradient = -1
        else:
            gradient = 0
        coef *= max(0, 1.0 - (eta * nu / 2))
        coef += -(eta * gradient * entry)
        intercept += -(eta * (nu + gradient)) * decay

        average_coef *= i
        average_coef += coef
        average_coef /= i + 1.0

        average_intercept *= i
        average_intercept += intercept
        average_intercept /= i + 1.0

    return average_coef, 1 - average_intercept


@pytest.mark.parametrize("klass", [SGDOneClassSVM, SparseSGDOneClassSVM])
def _test_warm_start_oneclass(klass, X, lr):
    # Test that explicit warm restart...
    clf = klass(nu=0.5, eta0=0.01, shuffle=False, learning_rate=lr)
    clf.fit(X)

    clf2 = klass(nu=0.1, eta0=0.01, shuffle=False, learning_rate=lr)
    clf2.fit(X, coef_init=clf.coef_.copy(), offset_init=clf.offset_.copy())

    # ... and implicit warm restart are equivalent.
    clf3 = klass(nu=0.5, eta0=0.01, shuffle=False, warm_start=True, learning_rate=lr)
    clf3.fit(X)

    assert clf3.t_ == clf.t_
    assert_allclose(clf3.coef_, clf.coef_)

    clf3.set_params(nu=0.1)
    clf3.fit(X)

    assert clf3.t_ == clf2.t_
    assert_allclose(clf3.coef_, clf2.coef_)


@pytest.mark.parametrize("klass", [SGDOneClassSVM, SparseSGDOneClassSVM])
@pytest.mark.parametrize("lr", ["constant", "optimal", "invscaling", "adaptive"])
def test_warm_start_oneclass(klass, lr):
    _test_warm_start_oneclass(klass, X, lr)


@pytest.mark.parametrize("klass", [SGDOneClassSVM, SparseSGDOneClassSVM])
def test_clone_oneclass(klass):
    # Test whether clone works ok.
    clf = klass(nu=0.5)
    clf = clone(clf)
    clf.set_params(nu=0.1)
    clf.fit(X)

    clf2 = klass(nu=0.1)
    clf2.fit(X)

    assert_array_equal(clf.coef_, clf2.coef_)


@pytest.mark.parametrize("klass", [SGDOneClassSVM, SparseSGDOneClassSVM])
def test_partial_fit_oneclass(klass):
    third = X.shape[0] // 3
    clf = klass(nu=0.1)

    clf.partial_fit(X[:third])
    assert clf.coef_.shape == (X.shape[1],)
    assert clf.offset_.shape == (1,)
    assert clf.predict([[0, 0]]).shape == (1,)
    previous_coefs = clf.coef_

    clf.partial_fit(X[third:])
    # check that coef_ haven't been re-allocated
    assert clf.coef_ is previous_coefs

    # raises ValueError if number of features does not match previous data
    with pytest.raises(ValueError):
        clf.partial_fit(X[:, 1])


@pytest.mark.parametrize("klass", [SGDOneClassSVM, SparseSGDOneClassSVM])
@pytest.mark.parametrize("lr", ["constant", "optimal", "invscaling", "adaptive"])
def test_partial_fit_equal_fit_oneclass(klass, lr):
    clf = klass(nu=0.05, max_iter=2, eta0=0.01, learning_rate=lr, shuffle=False)
    clf.fit(X)
    y_scores = clf.decision_function(T)
    t = clf.t_
    coef = clf.coef_
    offset = clf.offset_

    clf = klass(nu=0.05, eta0=0.01, max_iter=1, learning_rate=lr, shuffle=False)
    for _ in range(2):
        clf.partial_fit(X)
    y_scores2 = clf.decision_function(T)

    assert clf.t_ == t
    assert_allclose(y_scores, y_scores2)
    assert_allclose(clf.coef_, coef)
    assert_allclose(clf.offset_, offset)


@pytest.mark.parametrize("klass", [SGDOneClassSVM, SparseSGDOneClassSVM])
def test_late_onset_averaging_reached_oneclass(klass):
    # Test average
    eta0 = 0.001
    nu = 0.05

    # 2 passes over the training set but average only at second pass
    clf1 = klass(
        average=7, learning_rate="constant", eta0=eta0, nu=nu, max_iter=2, shuffle=False
    )
    # 1 pass over the training set with no averaging
    clf2 = klass(
        average=0, learning_rate="constant", eta0=eta0, nu=nu, max_iter=1, shuffle=False
    )

    clf1.fit(X)
    clf2.fit(X)

    # Start from clf2 solution, compute averaging using asgd function and
    # compare with clf1 solution
    average_coef, average_offset = asgd_oneclass(
        klass, X, eta0, nu, coef_init=clf2.coef_.ravel(), offset_init=clf2.offset_
    )

    assert_allclose(clf1.coef_.ravel(), average_coef.ravel())
    assert_allclose(clf1.offset_, average_offset)


@pytest.mark.parametrize("klass", [SGDOneClassSVM, SparseSGDOneClassSVM])
def test_sgd_averaged_computed_correctly_oneclass(klass):
    # Tests the average SGD One-Class SVM matches the naive implementation
    eta = 0.001
    nu = 0.05
    n_samples = 20
    n_features = 10
    rng = np.random.RandomState(0)
    X = rng.normal(size=(n_samples, n_features))

    clf = klass(
        learning_rate="constant",
        eta0=eta,
        nu=nu,
        fit_intercept=True,
        max_iter=1,
        average=True,
        shuffle=False,
    )

    clf.fit(X)
    average_coef, average_offset = asgd_oneclass(klass, X, eta, nu)

    assert_allclose(clf.coef_, average_coef)
    assert_allclose(clf.offset_, average_offset)


@pytest.mark.parametrize("klass", [SGDOneClassSVM, SparseSGDOneClassSVM])
def test_sgd_averaged_partial_fit_oneclass(klass):
    # Tests whether the partial fit yields the same average as the fit
    eta = 0.001
    nu = 0.05
    n_samples = 20
    n_features = 10
    rng = np.random.RandomState(0)
    X = rng.normal(size=(n_samples, n_features))

    clf = klass(
        learning_rate="constant",
        eta0=eta,
        nu=nu,
        fit_intercept=True,
        max_iter=1,
        average=True,
        shuffle=False,
    )

    clf.partial_fit(X[: int(n_samples / 2)][:])
    clf.partial_fit(X[int(n_samples / 2) :][:])
    average_coef, average_offset = asgd_oneclass(klass, X, eta, nu)

    assert_allclose(clf.coef_, average_coef)
    assert_allclose(clf.offset_, average_offset)


@pytest.mark.parametrize("klass", [SGDOneClassSVM, SparseSGDOneClassSVM])
def test_average_sparse_oneclass(klass):
    # Checks the average coef on data with 0s
    eta = 0.001
    nu = 0.01
    clf = klass(
        learning_rate="constant",
        eta0=eta,
        nu=nu,
        fit_intercept=True,
        max_iter=1,
        average=True,
        shuffle=False,
    )

    n_samples = X3.shape[0]

    clf.partial_fit(X3[: int(n_samples / 2)])
    clf.partial_fit(X3[int(n_samples / 2) :])
    average_coef, average_offset = asgd_oneclass(klass, X3, eta, nu)

    assert_allclose(clf.coef_, average_coef)
    assert_allclose(clf.offset_, average_offset)


def test_sgd_oneclass():
    # Test fit, decision_function, predict and score_samples on a toy
    # dataset
    X_train = np.array([[-2, -1], [-1, -1], [1, 1]])
    X_test = np.array([[0.5, -2], [2, 2]])
    clf = SGDOneClassSVM(
        nu=0.5, eta0=1, learning_rate="constant", shuffle=False, max_iter=1
    )
    clf.fit(X_train)
    assert_allclose(clf.coef_, np.array([-0.125, 0.4375]))
    assert clf.offset_[0] == -0.5

    scores = clf.score_samples(X_test)
    assert_allclose(scores, np.array([-0.9375, 0.625]))

    dec = clf.score_samples(X_test) - clf.offset_
    assert_allclose(clf.decision_function(X_test), dec)

    pred = clf.predict(X_test)
    assert_array_equal(pred, np.array([-1, 1]))


def test_ocsvm_vs_sgdocsvm():
    # Checks SGDOneClass SVM gives a good approximation of kernelized
    # One-Class SVM
    nu = 0.05
    gamma = 2.0
    random_state = 42

    # Generate train and test data
    rng = np.random.RandomState(random_state)
    X = 0.3 * rng.randn(500, 2)
    X_train = np.r_[X + 2, X - 2]
    X = 0.3 * rng.randn(100, 2)
    X_test = np.r_[X + 2, X - 2]

    # One-Class SVM
    clf = OneClassSVM(gamma=gamma, kernel="rbf", nu=nu)
    clf.fit(X_train)
    y_pred_ocsvm = clf.predict(X_test)
    dec_ocsvm = clf.decision_function(X_test).reshape(1, -1)

    # SGDOneClassSVM using kernel approximation
    max_iter = 15
    transform = Nystroem(gamma=gamma, random_state=random_state)
    clf_sgd = SGDOneClassSVM(
        nu=nu,
        shuffle=True,
        fit_intercept=True,
        max_iter=max_iter,
        random_state=random_state,
        tol=-np.inf,
    )
    pipe_sgd = make_pipeline(transform, clf_sgd)
    pipe_sgd.fit(X_train)
    y_pred_sgdocsvm = pipe_sgd.predict(X_test)
    dec_sgdocsvm = pipe_sgd.decision_function(X_test).reshape(1, -1)

    assert np.mean(y_pred_sgdocsvm == y_pred_ocsvm) >= 0.99
    corrcoef = np.corrcoef(np.concatenate((dec_ocsvm, dec_sgdocsvm)))[0, 1]
    assert corrcoef >= 0.9


def test_l1_ratio():
    # Test if l1 ratio extremes match L1 and L2 penalty settings.
    X, y = datasets.make_classification(
        n_samples=1000, n_features=100, n_informative=20, random_state=1234
    )

    # test if elasticnet with l1_ratio near 1 gives same result as pure l1
    est_en = SGDClassifier(
        alpha=0.001,
        penalty="elasticnet",
        tol=None,
        max_iter=6,
        l1_ratio=0.9999999999,
        random_state=42,
    ).fit(X, y)
    est_l1 = SGDClassifier(
        alpha=0.001, penalty="l1", max_iter=6, random_state=42, tol=None
    ).fit(X, y)
    assert_array_almost_equal(est_en.coef_, est_l1.coef_)

    # test if elasticnet with l1_ratio near 0 gives same result as pure l2
    est_en = SGDClassifier(
        alpha=0.001,
        penalty="elasticnet",
        tol=None,
        max_iter=6,
        l1_ratio=0.0000000001,
        random_state=42,
    ).fit(X, y)
    est_l2 = SGDClassifier(
        alpha=0.001, penalty="l2", max_iter=6, random_state=42, tol=None
    ).fit(X, y)
    assert_array_almost_equal(est_en.coef_, est_l2.coef_)


def test_underflow_or_overlow():
    with np.errstate(all="raise"):
        # Generate some weird data with hugely unscaled features
        rng = np.random.RandomState(0)
        n_samples = 100
        n_features = 10

        X = rng.normal(size=(n_samples, n_features))
        X[:, :2] *= 1e300
        assert np.isfinite(X).all()

        # Use MinMaxScaler to scale the data without introducing a numerical
        # instability (computing the standard deviation naively is not possible
        # on this data)
        X_scaled = MinMaxScaler().fit_transform(X)
        assert np.isfinite(X_scaled).all()

        # Define a ground truth on the scaled data
        ground_truth = rng.normal(size=n_features)
        y = (np.dot(X_scaled, ground_truth) > 0.0).astype(np.int32)
        assert_array_equal(np.unique(y), [0, 1])

        model = SGDClassifier(alpha=0.1, loss="squared_hinge", max_iter=500)

        # smoke test: model is stable on scaled data
        model.fit(X_scaled, y)
        assert np.isfinite(model.coef_).all()

        # model is numerically unstable on unscaled data
        msg_regxp = (
            r"Floating-point under-/overflow occurred at epoch #.*"
            " Scaling input data with StandardScaler or MinMaxScaler"
            " might help."
        )
        with pytest.raises(ValueError, match=msg_regxp):
            model.fit(X, y)


def test_numerical_stability_large_gradient():
    # Non regression test case for numerical stability on scaled problems
    # where the gradient can still explode with some losses
    model = SGDClassifier(
        loss="squared_hinge",
        max_iter=10,
        shuffle=True,
        penalty="elasticnet",
        l1_ratio=0.3,
        alpha=0.01,
        eta0=0.001,
        random_state=0,
        tol=None,
    )
    with np.errstate(all="raise"):
        model.fit(iris.data, iris.target)
    assert np.isfinite(model.coef_).all()


@pytest.mark.parametrize("penalty", ["l2", "l1", "elasticnet"])
def test_large_regularization(penalty):
    # Non regression tests for numerical stability issues caused by large
    # regularization parameters
    model = SGDClassifier(
        alpha=1e5,
        learning_rate="constant",
        eta0=0.1,
        penalty=penalty,
        shuffle=False,
        tol=None,
        max_iter=6,
    )
    with np.errstate(all="raise"):
        model.fit(iris.data, iris.target)
    assert_array_almost_equal(model.coef_, np.zeros_like(model.coef_))


def test_tol_parameter():
    # Test that the tol parameter behaves as expected
    X = StandardScaler().fit_transform(iris.data)
    y = iris.target == 1

    # With tol is None, the number of iteration should be equal to max_iter
    max_iter = 42
    model_0 = SGDClassifier(tol=None, random_state=0, max_iter=max_iter)
    model_0.fit(X, y)
    assert max_iter == model_0.n_iter_

    # If tol is not None, the number of iteration should be less than max_iter
    max_iter = 2000
    model_1 = SGDClassifier(tol=0, random_state=0, max_iter=max_iter)
    model_1.fit(X, y)
    assert max_iter > model_1.n_iter_
    assert model_1.n_iter_ > 5

    # A larger tol should yield a smaller number of iteration
    model_2 = SGDClassifier(tol=0.1, random_state=0, max_iter=max_iter)
    model_2.fit(X, y)
    assert model_1.n_iter_ > model_2.n_iter_
    assert model_2.n_iter_ > 3

    # Strict tolerance and small max_iter should trigger a warning
    model_3 = SGDClassifier(max_iter=3, tol=1e-3, random_state=0)
    warning_message = (
        "Maximum number of iteration reached before "
        "convergence. Consider increasing max_iter to "
        "improve the fit."
    )
    with pytest.warns(ConvergenceWarning, match=warning_message):
        model_3.fit(X, y)
    assert model_3.n_iter_ == 3


def _test_loss_common(loss_function, cases):
    # Test the different loss functions
    # cases is a list of (p, y, expected)
    for p, y, expected_loss, expected_dloss in cases:
        assert_almost_equal(loss_function.py_loss(p, y), expected_loss)
        assert_almost_equal(loss_function.py_dloss(p, y), expected_dloss)


def test_loss_hinge():
    # Test Hinge (hinge / perceptron)
    # hinge
    loss = sgd_fast.Hinge(1.0)
    cases = [
        # (p, y, expected_loss, expected_dloss)
        (1.1, 1.0, 0.0, 0.0),
        (-2.0, -1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0, -1.0),
        (-1.0, -1.0, 0.0, 1.0),
        (0.5, 1.0, 0.5, -1.0),
        (2.0, -1.0, 3.0, 1.0),
        (-0.5, -1.0, 0.5, 1.0),
        (0.0, 1.0, 1, -1.0),
    ]
    _test_loss_common(loss, cases)

    # perceptron
    loss = sgd_fast.Hinge(0.0)
    cases = [
        # (p, y, expected_loss, expected_dloss)
        (1.0, 1.0, 0.0, 0.0),
        (-0.1, -1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, -1.0),
        (0.0, -1.0, 0.0, 1.0),
        (0.5, -1.0, 0.5, 1.0),
        (2.0, -1.0, 2.0, 1.0),
        (-0.5, 1.0, 0.5, -1.0),
        (-1.0, 1.0, 1.0, -1.0),
    ]
    _test_loss_common(loss, cases)


def test_gradient_squared_hinge():
    # Test SquaredHinge
    loss = sgd_fast.SquaredHinge(1.0)
    cases = [
        # (p, y, expected_loss, expected_dloss)
        (1.0, 1.0, 0.0, 0.0),
        (-2.0, -1.0, 0.0, 0.0),
        (1.0, -1.0, 4.0, 4.0),
        (-1.0, 1.0, 4.0, -4.0),
        (0.5, 1.0, 0.25, -1.0),
        (0.5, -1.0, 2.25, 3.0),
    ]
    _test_loss_common(loss, cases)


def test_loss_log():
    # Test Log (logistic loss)
    loss = sgd_fast.Log()
    cases = [
        # (p, y, expected_loss, expected_dloss)
        (1.0, 1.0, np.log(1.0 + np.exp(-1.0)), -1.0 / (np.exp(1.0) + 1.0)),
        (1.0, -1.0, np.log(1.0 + np.exp(1.0)), 1.0 / (np.exp(-1.0) + 1.0)),
        (-1.0, -1.0, np.log(1.0 + np.exp(-1.0)), 1.0 / (np.exp(1.0) + 1.0)),
        (-1.0, 1.0, np.log(1.0 + np.exp(1.0)), -1.0 / (np.exp(-1.0) + 1.0)),
        (0.0, 1.0, np.log(2), -0.5),
        (0.0, -1.0, np.log(2), 0.5),
        (17.9, -1.0, 17.9, 1.0),
        (-17.9, 1.0, 17.9, -1.0),
    ]
    _test_loss_common(loss, cases)
    assert_almost_equal(loss.py_dloss(18.1, 1.0), np.exp(-18.1) * -1.0, 16)
    assert_almost_equal(loss.py_loss(18.1, 1.0), np.exp(-18.1), 16)
    assert_almost_equal(loss.py_dloss(-18.1, -1.0), np.exp(-18.1) * 1.0, 16)
    assert_almost_equal(loss.py_loss(-18.1, 1.0), 18.1, 16)


def test_loss_squared_loss():
    # Test SquaredLoss
    loss = sgd_fast.SquaredLoss()
    cases = [
        # (p, y, expected_loss, expected_dloss)
        (0.0, 0.0, 0.0, 0.0),
        (1.0, 1.0, 0.0, 0.0),
        (1.0, 0.0, 0.5, 1.0),
        (0.5, -1.0, 1.125, 1.5),
        (-2.5, 2.0, 10.125, -4.5),
    ]
    _test_loss_common(loss, cases)


def test_loss_huber():
    # Test Huber
    loss = sgd_fast.Huber(0.1)
    cases = [
        # (p, y, expected_loss, expected_dloss)
        (0.0, 0.0, 0.0, 0.0),
        (0.1, 0.0, 0.005, 0.1),
        (0.0, 0.1, 0.005, -0.1),
        (3.95, 4.0, 0.00125, -0.05),
        (5.0, 2.0, 0.295, 0.1),
        (-1.0, 5.0, 0.595, -0.1),
    ]
    _test_loss_common(loss, cases)


def test_loss_modified_huber():
    # (p, y, expected_loss, expected_dloss)
    loss = sgd_fast.ModifiedHuber()
    cases = [
        # (p, y, expected_loss, expected_dloss)
        (1.0, 1.0, 0.0, 0.0),
        (-1.0, -1.0, 0.0, 0.0),
        (2.0, 1.0, 0.0, 0.0),
        (0.0, 1.0, 1.0, -2.0),
        (-1.0, 1.0, 4.0, -4.0),
        (0.5, -1.0, 2.25, 3.0),
        (-2.0, 1.0, 8, -4.0),
        (-3.0, 1.0, 12, -4.0),
    ]
    _test_loss_common(loss, cases)


def test_loss_epsilon_insensitive():
    # Test EpsilonInsensitive
    loss = sgd_fast.EpsilonInsensitive(0.1)
    cases = [
        # (p, y, expected_loss, expected_dloss)
        (0.0, 0.0, 0.0, 0.0),
        (0.1, 0.0, 0.0, 0.0),
        (-2.05, -2.0, 0.0, 0.0),
        (3.05, 3.0, 0.0, 0.0),
        (2.2, 2.0, 0.1, 1.0),
        (2.0, -1.0, 2.9, 1.0),
        (2.0, 2.2, 0.1, -1.0),
        (-2.0, 1.0, 2.9, -1.0),
    ]
    _test_loss_common(loss, cases)


def test_loss_squared_epsilon_insensitive():
    # Test SquaredEpsilonInsensitive
    loss = sgd_fast.SquaredEpsilonInsensitive(0.1)
    cases = [
        # (p, y, expected_loss, expected_dloss)
        (0.0, 0.0, 0.0, 0.0),
        (0.1, 0.0, 0.0, 0.0),
        (-2.05, -2.0, 0.0, 0.0),
        (3.05, 3.0, 0.0, 0.0),
        (2.2, 2.0, 0.01, 0.2),
        (2.0, -1.0, 8.41, 5.8),
        (2.0, 2.2, 0.01, -0.2),
        (-2.0, 1.0, 8.41, -5.8),
    ]
    _test_loss_common(loss, cases)


def test_multi_thread_multi_class_and_early_stopping():
    # This is a non-regression test for a bad interaction between
    # early stopping internal attribute and thread-based parallelism.
    clf = SGDClassifier(
        alpha=1e-3,
        tol=1e-3,
        max_iter=1000,
        early_stopping=True,
        n_iter_no_change=100,
        random_state=0,
        n_jobs=2,
    )
    clf.fit(iris.data, iris.target)
    assert clf.n_iter_ > clf.n_iter_no_change
    assert clf.n_iter_ < clf.n_iter_no_change + 20
    assert clf.score(iris.data, iris.target) > 0.8


def test_multi_core_gridsearch_and_early_stopping():
    # This is a non-regression test for a bad interaction between
    # early stopping internal attribute and process-based multi-core
    # parallelism.
    param_grid = {
        "alpha": np.logspace(-4, 4, 9),
        "n_iter_no_change": [5, 10, 50],
    }

    clf = SGDClassifier(tol=1e-2, max_iter=1000, early_stopping=True, random_state=0)
    search = RandomizedSearchCV(clf, param_grid, n_iter=3, n_jobs=2, random_state=0)
    search.fit(iris.data, iris.target)
    assert search.best_score_ > 0.8


@pytest.mark.parametrize("backend", ["loky", "multiprocessing", "threading"])
def test_SGDClassifier_fit_for_all_backends(backend):
    # This is a non-regression smoke test. In the multi-class case,
    # SGDClassifier.fit fits each class in a one-versus-all fashion using
    # joblib.Parallel.  However, each OvA step updates the coef_ attribute of
    # the estimator in-place. Internally, SGDClassifier calls Parallel using
    # require='sharedmem'. This test makes sure SGDClassifier.fit works
    # consistently even when the user asks for a backend that does not provide
    # sharedmem semantics.

    # We further test a case where memmapping would have been used if
    # SGDClassifier.fit was called from a loky or multiprocessing backend. In
    # this specific case, in-place modification of clf.coef_ would have caused
    # a segmentation fault when trying to write in a readonly memory mapped
    # buffer.

    if parse_version(joblib.__version__) < parse_version("0.12") and backend == "loky":
        pytest.skip("loky backend does not exist in joblib <0.12")

    random_state = np.random.RandomState(42)

    # Create a classification problem with 50000 features and 20 classes. Using
    # loky or multiprocessing this make the clf.coef_ exceed the threshold
    # above which memmaping is used in joblib and loky (1MB as of 2018/11/1).
    X = sp.random(500, 2000, density=0.02, format="csr", random_state=random_state)
    y = random_state.choice(20, 500)

    # Begin by fitting a SGD classifier sequentially
    clf_sequential = SGDClassifier(max_iter=1000, n_jobs=1, random_state=42)
    clf_sequential.fit(X, y)

    # Fit a SGDClassifier using the specified backend, and make sure the
    # coefficients are equal to those obtained using a sequential fit
    clf_parallel = SGDClassifier(max_iter=1000, n_jobs=4, random_state=42)
    with joblib.parallel_backend(backend=backend):
        clf_parallel.fit(X, y)
    assert_array_almost_equal(clf_sequential.coef_, clf_parallel.coef_)


# TODO: Remove in v1.2
@pytest.mark.parametrize(
    "Estimator", [linear_model.SGDClassifier, linear_model.SGDRegressor]
)
def test_loss_squared_loss_deprecated(Estimator):

    # Note: class BaseSGD calls self._validate_params() in __init__, therefore
    # even instatiation of class raises FutureWarning for squared_loss.
    with pytest.warns(FutureWarning, match="The loss 'squared_loss' was deprecated"):
        est1 = Estimator(loss="squared_loss", random_state=0)
        est1.fit(X, Y)

    est2 = Estimator(loss="squared_error", random_state=0)
    est2.fit(X, Y)
    if hasattr(est1, "predict_proba"):
        assert_allclose(est1.predict_proba(X), est2.predict_proba(X))
    else:
        assert_allclose(est1.predict(X), est2.predict(X))
