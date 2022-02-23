import numpy as np
import scipy.sparse as sp

import pytest

from sklearn.base import is_classifier
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_almost_equal
from sklearn.base import ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.datasets import load_iris
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import PassiveAggressiveRegressor

iris = load_iris()
random_state = check_random_state(12)
indices = np.arange(iris.data.shape[0])
random_state.shuffle(indices)
X = iris.data[indices]
y = iris.target[indices]
X_csr = sp.csr_matrix(X)


class MyPassiveAggressive(ClassifierMixin):
    def __init__(
        self,
        C=1.0,
        epsilon=0.01,
        loss="hinge",
        fit_intercept=True,
        n_iter=1,
        random_state=None,
    ):
        self.C = C
        self.epsilon = epsilon
        self.loss = loss
        self.fit_intercept = fit_intercept
        self.n_iter = n_iter

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features, dtype=np.float64)
        self.b = 0.0

        for t in range(self.n_iter):
            for i in range(n_samples):
                p = self.project(X[i])
                if self.loss in ("hinge", "squared_hinge"):
                    loss = max(1 - y[i] * p, 0)
                else:
                    loss = max(np.abs(p - y[i]) - self.epsilon, 0)

                sqnorm = np.dot(X[i], X[i])

                if self.loss in ("hinge", "epsilon_insensitive"):
                    step = min(self.C, loss / sqnorm)
                elif self.loss in ("squared_hinge", "squared_epsilon_insensitive"):
                    step = loss / (sqnorm + 1.0 / (2 * self.C))

                if self.loss in ("hinge", "squared_hinge"):
                    step *= y[i]
                else:
                    step *= np.sign(y[i] - p)

                self.w += step * X[i]
                if self.fit_intercept:
                    self.b += step

    def project(self, X):
        return np.dot(X, self.w) + self.b


def test_classifier_accuracy():
    for data in (X, X_csr):
        for fit_intercept in (True, False):
            for average in (False, True):
                clf = PassiveAggressiveClassifier(
                    C=1.0,
                    max_iter=30,
                    fit_intercept=fit_intercept,
                    random_state=1,
                    average=average,
                    tol=None,
                )
                clf.fit(data, y)
                score = clf.score(data, y)
                assert score > 0.79
                if average:
                    assert hasattr(clf, "_average_coef")
                    assert hasattr(clf, "_average_intercept")
                    assert hasattr(clf, "_standard_intercept")
                    assert hasattr(clf, "_standard_coef")


def test_classifier_partial_fit():
    classes = np.unique(y)
    for data in (X, X_csr):
        for average in (False, True):
            clf = PassiveAggressiveClassifier(
                random_state=0, average=average, max_iter=5
            )
            for t in range(30):
                clf.partial_fit(data, y, classes)
            score = clf.score(data, y)
            assert score > 0.79
            if average:
                assert hasattr(clf, "_average_coef")
                assert hasattr(clf, "_average_intercept")
                assert hasattr(clf, "_standard_intercept")
                assert hasattr(clf, "_standard_coef")


def test_classifier_refit():
    # Classifier can be retrained on different labels and features.
    clf = PassiveAggressiveClassifier(max_iter=5).fit(X, y)
    assert_array_equal(clf.classes_, np.unique(y))

    clf.fit(X[:, :-1], iris.target_names[y])
    assert_array_equal(clf.classes_, iris.target_names)


@pytest.mark.parametrize("loss", ("hinge", "squared_hinge"))
def test_classifier_correctness(loss):
    y_bin = y.copy()
    y_bin[y != 1] = -1

    clf1 = MyPassiveAggressive(loss=loss, n_iter=2)
    clf1.fit(X, y_bin)

    for data in (X, X_csr):
        clf2 = PassiveAggressiveClassifier(
            loss=loss, max_iter=2, shuffle=False, tol=None
        )
        clf2.fit(data, y_bin)

        assert_array_almost_equal(clf1.w, clf2.coef_.ravel(), decimal=2)


@pytest.mark.parametrize(
    "response_method", ["predict_proba", "predict_log_proba", "transform"]
)
def test_classifier_undefined_methods(response_method):
    clf = PassiveAggressiveClassifier(max_iter=100)
    with pytest.raises(AttributeError):
        getattr(clf, response_method)


def test_class_weights():
    # Test class weights.
    X2 = np.array([[-1.0, -1.0], [-1.0, 0], [-0.8, -1.0], [1.0, 1.0], [1.0, 0.0]])
    y2 = [1, 1, 1, -1, -1]

    clf = PassiveAggressiveClassifier(
        C=0.1, max_iter=100, class_weight=None, random_state=100
    )
    clf.fit(X2, y2)
    assert_array_equal(clf.predict([[0.2, -1.0]]), np.array([1]))

    # we give a small weights to class 1
    clf = PassiveAggressiveClassifier(
        C=0.1, max_iter=100, class_weight={1: 0.001}, random_state=100
    )
    clf.fit(X2, y2)

    # now the hyperplane should rotate clock-wise and
    # the prediction on this point should shift
    assert_array_equal(clf.predict([[0.2, -1.0]]), np.array([-1]))


def test_partial_fit_weight_class_balanced():
    # partial_fit with class_weight='balanced' not supported
    clf = PassiveAggressiveClassifier(class_weight="balanced", max_iter=100)
    with pytest.raises(ValueError):
        clf.partial_fit(X, y, classes=np.unique(y))


def test_equal_class_weight():
    X2 = [[1, 0], [1, 0], [0, 1], [0, 1]]
    y2 = [0, 0, 1, 1]
    clf = PassiveAggressiveClassifier(C=0.1, tol=None, class_weight=None)
    clf.fit(X2, y2)

    # Already balanced, so "balanced" weights should have no effect
    clf_balanced = PassiveAggressiveClassifier(C=0.1, tol=None, class_weight="balanced")
    clf_balanced.fit(X2, y2)

    clf_weighted = PassiveAggressiveClassifier(
        C=0.1, tol=None, class_weight={0: 0.5, 1: 0.5}
    )
    clf_weighted.fit(X2, y2)

    # should be similar up to some epsilon due to learning rate schedule
    assert_almost_equal(clf.coef_, clf_weighted.coef_, decimal=2)
    assert_almost_equal(clf.coef_, clf_balanced.coef_, decimal=2)


def test_wrong_class_weight_label():
    # ValueError due to wrong class_weight label.
    X2 = np.array([[-1.0, -1.0], [-1.0, 0], [-0.8, -1.0], [1.0, 1.0], [1.0, 0.0]])
    y2 = [1, 1, 1, -1, -1]

    clf = PassiveAggressiveClassifier(class_weight={0: 0.5}, max_iter=100)
    with pytest.raises(ValueError):
        clf.fit(X2, y2)


def test_wrong_class_weight_format():
    # ValueError due to wrong class_weight argument type.
    X2 = np.array([[-1.0, -1.0], [-1.0, 0], [-0.8, -1.0], [1.0, 1.0], [1.0, 0.0]])
    y2 = [1, 1, 1, -1, -1]

    clf = PassiveAggressiveClassifier(class_weight=[0.5], max_iter=100)
    with pytest.raises(ValueError):
        clf.fit(X2, y2)

    clf = PassiveAggressiveClassifier(class_weight="the larch", max_iter=100)
    with pytest.raises(ValueError):
        clf.fit(X2, y2)


def test_regressor_mse():
    y_bin = y.copy()
    y_bin[y != 1] = -1

    for data in (X, X_csr):
        for fit_intercept in (True, False):
            for average in (False, True):
                reg = PassiveAggressiveRegressor(
                    C=1.0,
                    fit_intercept=fit_intercept,
                    random_state=0,
                    average=average,
                    max_iter=5,
                )
                reg.fit(data, y_bin)
                pred = reg.predict(data)
                assert np.mean((pred - y_bin) ** 2) < 1.7
                if average:
                    assert hasattr(reg, "_average_coef")
                    assert hasattr(reg, "_average_intercept")
                    assert hasattr(reg, "_standard_intercept")
                    assert hasattr(reg, "_standard_coef")


def test_regressor_partial_fit():
    y_bin = y.copy()
    y_bin[y != 1] = -1

    for data in (X, X_csr):
        for average in (False, True):
            reg = PassiveAggressiveRegressor(
                random_state=0, average=average, max_iter=100
            )
            for t in range(50):
                reg.partial_fit(data, y_bin)
            pred = reg.predict(data)
            assert np.mean((pred - y_bin) ** 2) < 1.7
            if average:
                assert hasattr(reg, "_average_coef")
                assert hasattr(reg, "_average_intercept")
                assert hasattr(reg, "_standard_intercept")
                assert hasattr(reg, "_standard_coef")


@pytest.mark.parametrize("loss", ("epsilon_insensitive", "squared_epsilon_insensitive"))
def test_regressor_correctness(loss):
    y_bin = y.copy()
    y_bin[y != 1] = -1

    reg1 = MyPassiveAggressive(loss=loss, n_iter=2)
    reg1.fit(X, y_bin)

    for data in (X, X_csr):
        reg2 = PassiveAggressiveRegressor(
            tol=None, loss=loss, max_iter=2, shuffle=False
        )
        reg2.fit(data, y_bin)

        assert_array_almost_equal(reg1.w, reg2.coef_.ravel(), decimal=2)


def test_regressor_undefined_methods():
    reg = PassiveAggressiveRegressor(max_iter=100)
    with pytest.raises(AttributeError):
        reg.transform(X)


@pytest.mark.parametrize(
    "klass", [PassiveAggressiveClassifier, PassiveAggressiveRegressor]
)
@pytest.mark.parametrize("fit_method", ["fit", "partial_fit"])
@pytest.mark.parametrize(
    "params, err_msg",
    [
        ({"loss": "foobar"}, "The loss foobar is not supported"),
        ({"max_iter": -1}, "max_iter must be > zero"),
        ({"shuffle": "false"}, "shuffle must be either True or False"),
        ({"early_stopping": "false"}, "early_stopping must be either True or False"),
        (
            {"validation_fraction": -0.1},
            r"validation_fraction must be in range \(0, 1\)",
        ),
        ({"n_iter_no_change": 0}, "n_iter_no_change must be >= 1"),
    ],
)
def test_passive_aggressive_estimator_params_validation(
    klass, fit_method, params, err_msg
):
    """Validate parameters in the different PassiveAggressive estimators."""
    sgd_estimator = klass(**params)

    with pytest.raises(ValueError, match=err_msg):
        if is_classifier(sgd_estimator) and fit_method == "partial_fit":
            fit_params = {"classes": np.unique(y)}
        else:
            fit_params = {}
        getattr(sgd_estimator, fit_method)(X, y, **fit_params)
