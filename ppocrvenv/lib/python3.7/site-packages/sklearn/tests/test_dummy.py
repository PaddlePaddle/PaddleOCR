import pytest

import numpy as np
import scipy.sparse as sp

from sklearn.base import clone
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_almost_equal
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.stats import _weighted_percentile

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError


@ignore_warnings
def _check_predict_proba(clf, X, y):
    proba = clf.predict_proba(X)
    # We know that we can have division by zero
    log_proba = clf.predict_log_proba(X)

    y = np.atleast_1d(y)
    if y.ndim == 1:
        y = np.reshape(y, (-1, 1))

    n_outputs = y.shape[1]
    n_samples = len(X)

    if n_outputs == 1:
        proba = [proba]
        log_proba = [log_proba]

    for k in range(n_outputs):
        assert proba[k].shape[0] == n_samples
        assert proba[k].shape[1] == len(np.unique(y[:, k]))
        assert_array_almost_equal(proba[k].sum(axis=1), np.ones(len(X)))
        # We know that we can have division by zero
        assert_array_almost_equal(np.log(proba[k]), log_proba[k])


def _check_behavior_2d(clf):
    # 1d case
    X = np.array([[0], [0], [0], [0]])  # ignored
    y = np.array([1, 2, 1, 1])
    est = clone(clf)
    est.fit(X, y)
    y_pred = est.predict(X)
    assert y.shape == y_pred.shape

    # 2d case
    y = np.array([[1, 0], [2, 0], [1, 0], [1, 3]])
    est = clone(clf)
    est.fit(X, y)
    y_pred = est.predict(X)
    assert y.shape == y_pred.shape


def _check_behavior_2d_for_constant(clf):
    # 2d case only
    X = np.array([[0], [0], [0], [0]])  # ignored
    y = np.array([[1, 0, 5, 4, 3], [2, 0, 1, 2, 5], [1, 0, 4, 5, 2], [1, 3, 3, 2, 0]])
    est = clone(clf)
    est.fit(X, y)
    y_pred = est.predict(X)
    assert y.shape == y_pred.shape


def _check_equality_regressor(statistic, y_learn, y_pred_learn, y_test, y_pred_test):
    assert_array_almost_equal(np.tile(statistic, (y_learn.shape[0], 1)), y_pred_learn)
    assert_array_almost_equal(np.tile(statistic, (y_test.shape[0], 1)), y_pred_test)


def test_most_frequent_and_prior_strategy():
    X = [[0], [0], [0], [0]]  # ignored
    y = [1, 2, 1, 1]

    for strategy in ("most_frequent", "prior"):
        clf = DummyClassifier(strategy=strategy, random_state=0)
        clf.fit(X, y)
        assert_array_equal(clf.predict(X), np.ones(len(X)))
        _check_predict_proba(clf, X, y)

        if strategy == "prior":
            assert_array_almost_equal(
                clf.predict_proba([X[0]]), clf.class_prior_.reshape((1, -1))
            )
        else:
            assert_array_almost_equal(
                clf.predict_proba([X[0]]), clf.class_prior_.reshape((1, -1)) > 0.5
            )


def test_most_frequent_and_prior_strategy_with_2d_column_y():
    # non-regression test added in
    # https://github.com/scikit-learn/scikit-learn/pull/13545
    X = [[0], [0], [0], [0]]
    y_1d = [1, 2, 1, 1]
    y_2d = [[1], [2], [1], [1]]

    for strategy in ("most_frequent", "prior"):
        clf_1d = DummyClassifier(strategy=strategy, random_state=0)
        clf_2d = DummyClassifier(strategy=strategy, random_state=0)

        clf_1d.fit(X, y_1d)
        clf_2d.fit(X, y_2d)
        assert_array_equal(clf_1d.predict(X), clf_2d.predict(X))


def test_most_frequent_and_prior_strategy_multioutput():
    X = [[0], [0], [0], [0]]  # ignored
    y = np.array([[1, 0], [2, 0], [1, 0], [1, 3]])

    n_samples = len(X)

    for strategy in ("prior", "most_frequent"):
        clf = DummyClassifier(strategy=strategy, random_state=0)
        clf.fit(X, y)
        assert_array_equal(
            clf.predict(X),
            np.hstack([np.ones((n_samples, 1)), np.zeros((n_samples, 1))]),
        )
        _check_predict_proba(clf, X, y)
        _check_behavior_2d(clf)


def test_stratified_strategy():
    X = [[0]] * 5  # ignored
    y = [1, 2, 1, 1, 2]
    clf = DummyClassifier(strategy="stratified", random_state=0)
    clf.fit(X, y)

    X = [[0]] * 500
    y_pred = clf.predict(X)
    p = np.bincount(y_pred) / float(len(X))
    assert_almost_equal(p[1], 3.0 / 5, decimal=1)
    assert_almost_equal(p[2], 2.0 / 5, decimal=1)
    _check_predict_proba(clf, X, y)


def test_stratified_strategy_multioutput():
    X = [[0]] * 5  # ignored
    y = np.array([[2, 1], [2, 2], [1, 1], [1, 2], [1, 1]])

    clf = DummyClassifier(strategy="stratified", random_state=0)
    clf.fit(X, y)

    X = [[0]] * 500
    y_pred = clf.predict(X)

    for k in range(y.shape[1]):
        p = np.bincount(y_pred[:, k]) / float(len(X))
        assert_almost_equal(p[1], 3.0 / 5, decimal=1)
        assert_almost_equal(p[2], 2.0 / 5, decimal=1)
        _check_predict_proba(clf, X, y)

    _check_behavior_2d(clf)


def test_uniform_strategy():
    X = [[0]] * 4  # ignored
    y = [1, 2, 1, 1]
    clf = DummyClassifier(strategy="uniform", random_state=0)
    clf.fit(X, y)

    X = [[0]] * 500
    y_pred = clf.predict(X)
    p = np.bincount(y_pred) / float(len(X))
    assert_almost_equal(p[1], 0.5, decimal=1)
    assert_almost_equal(p[2], 0.5, decimal=1)
    _check_predict_proba(clf, X, y)


def test_uniform_strategy_multioutput():
    X = [[0]] * 4  # ignored
    y = np.array([[2, 1], [2, 2], [1, 2], [1, 1]])
    clf = DummyClassifier(strategy="uniform", random_state=0)
    clf.fit(X, y)

    X = [[0]] * 500
    y_pred = clf.predict(X)

    for k in range(y.shape[1]):
        p = np.bincount(y_pred[:, k]) / float(len(X))
        assert_almost_equal(p[1], 0.5, decimal=1)
        assert_almost_equal(p[2], 0.5, decimal=1)
        _check_predict_proba(clf, X, y)

    _check_behavior_2d(clf)


def test_string_labels():
    X = [[0]] * 5
    y = ["paris", "paris", "tokyo", "amsterdam", "berlin"]
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(X, y)
    assert_array_equal(clf.predict(X), ["paris"] * 5)


@pytest.mark.parametrize(
    "y,y_test",
    [
        ([2, 1, 1, 1], [2, 2, 1, 1]),
        (
            np.array([[2, 2], [1, 1], [1, 1], [1, 1]]),
            np.array([[2, 2], [2, 2], [1, 1], [1, 1]]),
        ),
    ],
)
def test_classifier_score_with_None(y, y_test):
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(None, y)
    assert clf.score(None, y_test) == 0.5


@pytest.mark.parametrize(
    "strategy", ["stratified", "most_frequent", "prior", "uniform", "constant"]
)
def test_classifier_prediction_independent_of_X(strategy):
    y = [0, 2, 1, 1]
    X1 = [[0]] * 4
    clf1 = DummyClassifier(strategy=strategy, random_state=0, constant=0)
    clf1.fit(X1, y)
    predictions1 = clf1.predict(X1)

    X2 = [[1]] * 4
    clf2 = DummyClassifier(strategy=strategy, random_state=0, constant=0)
    clf2.fit(X2, y)
    predictions2 = clf2.predict(X2)

    assert_array_equal(predictions1, predictions2)


def test_classifier_exceptions():
    clf = DummyClassifier(strategy="unknown")
    with pytest.raises(ValueError):
        clf.fit([], [])

    with pytest.raises(NotFittedError):
        clf.predict([])
    with pytest.raises(NotFittedError):
        clf.predict_proba([])


def test_mean_strategy_regressor():

    random_state = np.random.RandomState(seed=1)

    X = [[0]] * 4  # ignored
    y = random_state.randn(4)

    reg = DummyRegressor()
    reg.fit(X, y)
    assert_array_equal(reg.predict(X), [np.mean(y)] * len(X))


def test_mean_strategy_multioutput_regressor():

    random_state = np.random.RandomState(seed=1)

    X_learn = random_state.randn(10, 10)
    y_learn = random_state.randn(10, 5)

    mean = np.mean(y_learn, axis=0).reshape((1, -1))

    X_test = random_state.randn(20, 10)
    y_test = random_state.randn(20, 5)

    # Correctness oracle
    est = DummyRegressor()
    est.fit(X_learn, y_learn)
    y_pred_learn = est.predict(X_learn)
    y_pred_test = est.predict(X_test)

    _check_equality_regressor(mean, y_learn, y_pred_learn, y_test, y_pred_test)
    _check_behavior_2d(est)


def test_regressor_exceptions():
    reg = DummyRegressor()
    with pytest.raises(NotFittedError):
        reg.predict([])


def test_median_strategy_regressor():

    random_state = np.random.RandomState(seed=1)

    X = [[0]] * 5  # ignored
    y = random_state.randn(5)

    reg = DummyRegressor(strategy="median")
    reg.fit(X, y)
    assert_array_equal(reg.predict(X), [np.median(y)] * len(X))


def test_median_strategy_multioutput_regressor():

    random_state = np.random.RandomState(seed=1)

    X_learn = random_state.randn(10, 10)
    y_learn = random_state.randn(10, 5)

    median = np.median(y_learn, axis=0).reshape((1, -1))

    X_test = random_state.randn(20, 10)
    y_test = random_state.randn(20, 5)

    # Correctness oracle
    est = DummyRegressor(strategy="median")
    est.fit(X_learn, y_learn)
    y_pred_learn = est.predict(X_learn)
    y_pred_test = est.predict(X_test)

    _check_equality_regressor(median, y_learn, y_pred_learn, y_test, y_pred_test)
    _check_behavior_2d(est)


def test_quantile_strategy_regressor():

    random_state = np.random.RandomState(seed=1)

    X = [[0]] * 5  # ignored
    y = random_state.randn(5)

    reg = DummyRegressor(strategy="quantile", quantile=0.5)
    reg.fit(X, y)
    assert_array_equal(reg.predict(X), [np.median(y)] * len(X))

    reg = DummyRegressor(strategy="quantile", quantile=0)
    reg.fit(X, y)
    assert_array_equal(reg.predict(X), [np.min(y)] * len(X))

    reg = DummyRegressor(strategy="quantile", quantile=1)
    reg.fit(X, y)
    assert_array_equal(reg.predict(X), [np.max(y)] * len(X))

    reg = DummyRegressor(strategy="quantile", quantile=0.3)
    reg.fit(X, y)
    assert_array_equal(reg.predict(X), [np.percentile(y, q=30)] * len(X))


def test_quantile_strategy_multioutput_regressor():

    random_state = np.random.RandomState(seed=1)

    X_learn = random_state.randn(10, 10)
    y_learn = random_state.randn(10, 5)

    median = np.median(y_learn, axis=0).reshape((1, -1))
    quantile_values = np.percentile(y_learn, axis=0, q=80).reshape((1, -1))

    X_test = random_state.randn(20, 10)
    y_test = random_state.randn(20, 5)

    # Correctness oracle
    est = DummyRegressor(strategy="quantile", quantile=0.5)
    est.fit(X_learn, y_learn)
    y_pred_learn = est.predict(X_learn)
    y_pred_test = est.predict(X_test)

    _check_equality_regressor(median, y_learn, y_pred_learn, y_test, y_pred_test)
    _check_behavior_2d(est)

    # Correctness oracle
    est = DummyRegressor(strategy="quantile", quantile=0.8)
    est.fit(X_learn, y_learn)
    y_pred_learn = est.predict(X_learn)
    y_pred_test = est.predict(X_test)

    _check_equality_regressor(
        quantile_values, y_learn, y_pred_learn, y_test, y_pred_test
    )
    _check_behavior_2d(est)


def test_quantile_invalid():

    X = [[0]] * 5  # ignored
    y = [0] * 5  # ignored

    est = DummyRegressor(strategy="quantile")
    with pytest.raises(ValueError):
        est.fit(X, y)

    est = DummyRegressor(strategy="quantile", quantile=None)
    with pytest.raises(ValueError):
        est.fit(X, y)

    est = DummyRegressor(strategy="quantile", quantile=[0])
    with pytest.raises(ValueError):
        est.fit(X, y)

    est = DummyRegressor(strategy="quantile", quantile=-0.1)
    with pytest.raises(ValueError):
        est.fit(X, y)

    est = DummyRegressor(strategy="quantile", quantile=1.1)
    with pytest.raises(ValueError):
        est.fit(X, y)

    est = DummyRegressor(strategy="quantile", quantile="abc")
    with pytest.raises(TypeError):
        est.fit(X, y)


def test_quantile_strategy_empty_train():
    est = DummyRegressor(strategy="quantile", quantile=0.4)
    with pytest.raises(ValueError):
        est.fit([], [])


def test_constant_strategy_regressor():

    random_state = np.random.RandomState(seed=1)

    X = [[0]] * 5  # ignored
    y = random_state.randn(5)

    reg = DummyRegressor(strategy="constant", constant=[43])
    reg.fit(X, y)
    assert_array_equal(reg.predict(X), [43] * len(X))

    reg = DummyRegressor(strategy="constant", constant=43)
    reg.fit(X, y)
    assert_array_equal(reg.predict(X), [43] * len(X))


def test_constant_strategy_multioutput_regressor():

    random_state = np.random.RandomState(seed=1)

    X_learn = random_state.randn(10, 10)
    y_learn = random_state.randn(10, 5)

    # test with 2d array
    constants = random_state.randn(5)

    X_test = random_state.randn(20, 10)
    y_test = random_state.randn(20, 5)

    # Correctness oracle
    est = DummyRegressor(strategy="constant", constant=constants)
    est.fit(X_learn, y_learn)
    y_pred_learn = est.predict(X_learn)
    y_pred_test = est.predict(X_test)

    _check_equality_regressor(constants, y_learn, y_pred_learn, y_test, y_pred_test)
    _check_behavior_2d_for_constant(est)


def test_y_mean_attribute_regressor():
    X = [[0]] * 5
    y = [1, 2, 4, 6, 8]
    # when strategy = 'mean'
    est = DummyRegressor(strategy="mean")
    est.fit(X, y)

    assert est.constant_ == np.mean(y)


def test_unknown_strategey_regressor():
    X = [[0]] * 5
    y = [1, 2, 4, 6, 8]

    est = DummyRegressor(strategy="gona")
    with pytest.raises(ValueError):
        est.fit(X, y)


def test_constants_not_specified_regressor():
    X = [[0]] * 5
    y = [1, 2, 4, 6, 8]

    est = DummyRegressor(strategy="constant")
    with pytest.raises(TypeError):
        est.fit(X, y)


def test_constant_size_multioutput_regressor():
    random_state = np.random.RandomState(seed=1)
    X = random_state.randn(10, 10)
    y = random_state.randn(10, 5)

    est = DummyRegressor(strategy="constant", constant=[1, 2, 3, 4])
    with pytest.raises(ValueError):
        est.fit(X, y)


def test_constant_strategy():
    X = [[0], [0], [0], [0]]  # ignored
    y = [2, 1, 2, 2]

    clf = DummyClassifier(strategy="constant", random_state=0, constant=1)
    clf.fit(X, y)
    assert_array_equal(clf.predict(X), np.ones(len(X)))
    _check_predict_proba(clf, X, y)

    X = [[0], [0], [0], [0]]  # ignored
    y = ["two", "one", "two", "two"]
    clf = DummyClassifier(strategy="constant", random_state=0, constant="one")
    clf.fit(X, y)
    assert_array_equal(clf.predict(X), np.array(["one"] * 4))
    _check_predict_proba(clf, X, y)


def test_constant_strategy_multioutput():
    X = [[0], [0], [0], [0]]  # ignored
    y = np.array([[2, 3], [1, 3], [2, 3], [2, 0]])

    n_samples = len(X)

    clf = DummyClassifier(strategy="constant", random_state=0, constant=[1, 0])
    clf.fit(X, y)
    assert_array_equal(
        clf.predict(X), np.hstack([np.ones((n_samples, 1)), np.zeros((n_samples, 1))])
    )
    _check_predict_proba(clf, X, y)


@pytest.mark.parametrize(
    "y, params, err_msg",
    [
        ([2, 1, 2, 2], {"random_state": 0}, "Constant.*has to be specified"),
        ([2, 1, 2, 2], {"constant": [2, 0]}, "Constant.*should have shape"),
        (
            np.transpose([[2, 1, 2, 2], [2, 1, 2, 2]]),
            {"constant": 2},
            "Constant.*should have shape",
        ),
        (
            [2, 1, 2, 2],
            {"constant": "my-constant"},
            "constant=my-constant.*Possible values.*\\[1, 2]",
        ),
        (
            np.transpose([[2, 1, 2, 2], [2, 1, 2, 2]]),
            {"constant": [2, "unknown"]},
            "constant=\\[2, 'unknown'].*Possible values.*\\[1, 2]",
        ),
    ],
    ids=[
        "no-constant",
        "too-many-constant",
        "not-enough-output",
        "single-output",
        "multi-output",
    ],
)
def test_constant_strategy_exceptions(y, params, err_msg):
    X = [[0], [0], [0], [0]]

    clf = DummyClassifier(strategy="constant", **params)

    with pytest.raises(ValueError, match=err_msg):
        clf.fit(X, y)


def test_classification_sample_weight():
    X = [[0], [0], [1]]
    y = [0, 1, 0]
    sample_weight = [0.1, 1.0, 0.1]

    clf = DummyClassifier(strategy="stratified").fit(X, y, sample_weight)
    assert_array_almost_equal(clf.class_prior_, [0.2 / 1.2, 1.0 / 1.2])


def test_constant_strategy_sparse_target():
    X = [[0]] * 5  # ignored
    y = sp.csc_matrix(np.array([[0, 1], [4, 0], [1, 1], [1, 4], [1, 1]]))

    n_samples = len(X)

    clf = DummyClassifier(strategy="constant", random_state=0, constant=[1, 0])
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert sp.issparse(y_pred)
    assert_array_equal(
        y_pred.toarray(), np.hstack([np.ones((n_samples, 1)), np.zeros((n_samples, 1))])
    )


def test_uniform_strategy_sparse_target_warning():
    X = [[0]] * 5  # ignored
    y = sp.csc_matrix(np.array([[2, 1], [2, 2], [1, 4], [4, 2], [1, 1]]))

    clf = DummyClassifier(strategy="uniform", random_state=0)
    with pytest.warns(UserWarning, match="the uniform strategy would not save memory"):
        clf.fit(X, y)

    X = [[0]] * 500
    y_pred = clf.predict(X)

    for k in range(y.shape[1]):
        p = np.bincount(y_pred[:, k]) / float(len(X))
        assert_almost_equal(p[1], 1 / 3, decimal=1)
        assert_almost_equal(p[2], 1 / 3, decimal=1)
        assert_almost_equal(p[4], 1 / 3, decimal=1)


def test_stratified_strategy_sparse_target():
    X = [[0]] * 5  # ignored
    y = sp.csc_matrix(np.array([[4, 1], [0, 0], [1, 1], [1, 4], [1, 1]]))

    clf = DummyClassifier(strategy="stratified", random_state=0)
    clf.fit(X, y)

    X = [[0]] * 500
    y_pred = clf.predict(X)
    assert sp.issparse(y_pred)
    y_pred = y_pred.toarray()

    for k in range(y.shape[1]):
        p = np.bincount(y_pred[:, k]) / float(len(X))
        assert_almost_equal(p[1], 3.0 / 5, decimal=1)
        assert_almost_equal(p[0], 1.0 / 5, decimal=1)
        assert_almost_equal(p[4], 1.0 / 5, decimal=1)


def test_most_frequent_and_prior_strategy_sparse_target():
    X = [[0]] * 5  # ignored
    y = sp.csc_matrix(np.array([[1, 0], [1, 3], [4, 0], [0, 1], [1, 0]]))

    n_samples = len(X)
    y_expected = np.hstack([np.ones((n_samples, 1)), np.zeros((n_samples, 1))])
    for strategy in ("most_frequent", "prior"):
        clf = DummyClassifier(strategy=strategy, random_state=0)
        clf.fit(X, y)

        y_pred = clf.predict(X)
        assert sp.issparse(y_pred)
        assert_array_equal(y_pred.toarray(), y_expected)


def test_dummy_regressor_sample_weight(n_samples=10):
    random_state = np.random.RandomState(seed=1)

    X = [[0]] * n_samples
    y = random_state.rand(n_samples)
    sample_weight = random_state.rand(n_samples)

    est = DummyRegressor(strategy="mean").fit(X, y, sample_weight)
    assert est.constant_ == np.average(y, weights=sample_weight)

    est = DummyRegressor(strategy="median").fit(X, y, sample_weight)
    assert est.constant_ == _weighted_percentile(y, sample_weight, 50.0)

    est = DummyRegressor(strategy="quantile", quantile=0.95).fit(X, y, sample_weight)
    assert est.constant_ == _weighted_percentile(y, sample_weight, 95.0)


def test_dummy_regressor_on_3D_array():
    X = np.array([[["foo"]], [["bar"]], [["baz"]]])
    y = np.array([2, 2, 2])
    y_expected = np.array([2, 2, 2])
    cls = DummyRegressor()
    cls.fit(X, y)
    y_pred = cls.predict(X)
    assert_array_equal(y_pred, y_expected)


def test_dummy_classifier_on_3D_array():
    X = np.array([[["foo"]], [["bar"]], [["baz"]]])
    y = [2, 2, 2]
    y_expected = [2, 2, 2]
    y_proba_expected = [[1], [1], [1]]
    cls = DummyClassifier(strategy="stratified")
    cls.fit(X, y)
    y_pred = cls.predict(X)
    y_pred_proba = cls.predict_proba(X)
    assert_array_equal(y_pred, y_expected)
    assert_array_equal(y_pred_proba, y_proba_expected)


def test_dummy_regressor_return_std():
    X = [[0]] * 3  # ignored
    y = np.array([2, 2, 2])
    y_std_expected = np.array([0, 0, 0])
    cls = DummyRegressor()
    cls.fit(X, y)
    y_pred_list = cls.predict(X, return_std=True)
    # there should be two elements when return_std is True
    assert len(y_pred_list) == 2
    # the second element should be all zeros
    assert_array_equal(y_pred_list[1], y_std_expected)


@pytest.mark.parametrize(
    "y,y_test",
    [
        ([1, 1, 1, 2], [1.25] * 4),
        (np.array([[2, 2], [1, 1], [1, 1], [1, 1]]), [[1.25, 1.25]] * 4),
    ],
)
def test_regressor_score_with_None(y, y_test):
    reg = DummyRegressor()
    reg.fit(None, y)
    assert reg.score(None, y_test) == 1.0


@pytest.mark.parametrize("strategy", ["mean", "median", "quantile", "constant"])
def test_regressor_prediction_independent_of_X(strategy):
    y = [0, 2, 1, 1]
    X1 = [[0]] * 4
    reg1 = DummyRegressor(strategy=strategy, constant=0, quantile=0.7)
    reg1.fit(X1, y)
    predictions1 = reg1.predict(X1)

    X2 = [[1]] * 4
    reg2 = DummyRegressor(strategy=strategy, constant=0, quantile=0.7)
    reg2.fit(X2, y)
    predictions2 = reg2.predict(X2)

    assert_array_equal(predictions1, predictions2)


@pytest.mark.parametrize(
    "strategy", ["stratified", "most_frequent", "prior", "uniform", "constant"]
)
def test_dtype_of_classifier_probas(strategy):
    y = [0, 2, 1, 1]
    X = np.zeros(4)
    model = DummyClassifier(strategy=strategy, random_state=0, constant=0)
    probas = model.fit(X, y).predict_proba(X)

    assert probas.dtype == np.float64


# TODO: remove in 1.2
@pytest.mark.filterwarnings("ignore:`n_features_in_` is deprecated")
@pytest.mark.parametrize("Dummy", (DummyRegressor, DummyClassifier))
def test_n_features_in_(Dummy):
    X = [[1, 2]]
    y = [0]
    d = Dummy()
    assert not hasattr(d, "n_features_in_")
    d.fit(X, y)

    with pytest.warns(FutureWarning, match="`n_features_in_` is deprecated"):
        n_features_in = d.n_features_in_
    assert n_features_in is None
