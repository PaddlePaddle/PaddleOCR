"""Test the search module"""

from collections.abc import Iterable, Sized
from io import StringIO
from itertools import chain, product
from functools import partial
import pickle
import sys
from types import GeneratorType
import re

import numpy as np
import scipy.sparse as sp
import pytest

from sklearn.utils._testing import (
    assert_array_equal,
    assert_array_almost_equal,
    assert_allclose,
    assert_almost_equal,
    ignore_warnings,
    MinimalClassifier,
    MinimalRegressor,
    MinimalTransformer,
)
from sklearn.utils._mocking import CheckingClassifier, MockDataFrame

from scipy.stats import bernoulli, expon, uniform

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import is_classifier
from sklearn.exceptions import NotFittedError
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_multilabel_classification

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import LeavePGroupsOut
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import ParameterSampler
from sklearn.model_selection._search import BaseSearchCV

from sklearn.model_selection._validation import FitFailedWarning

from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, SGDClassifier, LinearRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.model_selection.tests.common import OneTimeSplitter


# Neither of the following two estimators inherit from BaseEstimator,
# to test hyperparameter search on user-defined classifiers.
class MockClassifier:
    """Dummy classifier to test the parameter search algorithms"""

    def __init__(self, foo_param=0):
        self.foo_param = foo_param

    def fit(self, X, Y):
        assert len(X) == len(Y)
        self.classes_ = np.unique(Y)
        return self

    def predict(self, T):
        return T.shape[0]

    def transform(self, X):
        return X + self.foo_param

    def inverse_transform(self, X):
        return X - self.foo_param

    predict_proba = predict
    predict_log_proba = predict
    decision_function = predict

    def score(self, X=None, Y=None):
        if self.foo_param > 1:
            score = 1.0
        else:
            score = 0.0
        return score

    def get_params(self, deep=False):
        return {"foo_param": self.foo_param}

    def set_params(self, **params):
        self.foo_param = params["foo_param"]
        return self


class LinearSVCNoScore(LinearSVC):
    """An LinearSVC classifier that has no score method."""

    @property
    def score(self):
        raise AttributeError


X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])


def assert_grid_iter_equals_getitem(grid):
    assert list(grid) == [grid[i] for i in range(len(grid))]


@pytest.mark.parametrize("klass", [ParameterGrid, partial(ParameterSampler, n_iter=10)])
@pytest.mark.parametrize(
    "input, error_type, error_message",
    [
        (0, TypeError, r"Parameter .* is not a dict or a list \(0\)"),
        ([{"foo": [0]}, 0], TypeError, r"Parameter .* is not a dict \(0\)"),
        (
            {"foo": 0},
            TypeError,
            "Parameter.* value is not iterable .*" r"\(key='foo', value=0\)",
        ),
    ],
)
def test_validate_parameter_input(klass, input, error_type, error_message):
    with pytest.raises(error_type, match=error_message):
        klass(input)


def test_parameter_grid():

    # Test basic properties of ParameterGrid.
    params1 = {"foo": [1, 2, 3]}
    grid1 = ParameterGrid(params1)
    assert isinstance(grid1, Iterable)
    assert isinstance(grid1, Sized)
    assert len(grid1) == 3
    assert_grid_iter_equals_getitem(grid1)

    params2 = {"foo": [4, 2], "bar": ["ham", "spam", "eggs"]}
    grid2 = ParameterGrid(params2)
    assert len(grid2) == 6

    # loop to assert we can iterate over the grid multiple times
    for i in range(2):
        # tuple + chain transforms {"a": 1, "b": 2} to ("a", 1, "b", 2)
        points = set(tuple(chain(*(sorted(p.items())))) for p in grid2)
        assert points == set(
            ("bar", x, "foo", y) for x, y in product(params2["bar"], params2["foo"])
        )
    assert_grid_iter_equals_getitem(grid2)

    # Special case: empty grid (useful to get default estimator settings)
    empty = ParameterGrid({})
    assert len(empty) == 1
    assert list(empty) == [{}]
    assert_grid_iter_equals_getitem(empty)
    with pytest.raises(IndexError):
        empty[1]

    has_empty = ParameterGrid([{"C": [1, 10]}, {}, {"C": [0.5]}])
    assert len(has_empty) == 4
    assert list(has_empty) == [{"C": 1}, {"C": 10}, {}, {"C": 0.5}]
    assert_grid_iter_equals_getitem(has_empty)


def test_grid_search():
    # Test that the best estimator contains the right value for foo_param
    clf = MockClassifier()
    grid_search = GridSearchCV(clf, {"foo_param": [1, 2, 3]}, cv=3, verbose=3)
    # make sure it selects the smallest parameter in case of ties
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    grid_search.fit(X, y)
    sys.stdout = old_stdout
    assert grid_search.best_estimator_.foo_param == 2

    assert_array_equal(grid_search.cv_results_["param_foo_param"].data, [1, 2, 3])

    # Smoke test the score etc:
    grid_search.score(X, y)
    grid_search.predict_proba(X)
    grid_search.decision_function(X)
    grid_search.transform(X)

    # Test exception handling on scoring
    grid_search.scoring = "sklearn"
    with pytest.raises(ValueError):
        grid_search.fit(X, y)


def test_grid_search_pipeline_steps():
    # check that parameters that are estimators are cloned before fitting
    pipe = Pipeline([("regressor", LinearRegression())])
    param_grid = {"regressor": [LinearRegression(), Ridge()]}
    grid_search = GridSearchCV(pipe, param_grid, cv=2)
    grid_search.fit(X, y)
    regressor_results = grid_search.cv_results_["param_regressor"]
    assert isinstance(regressor_results[0], LinearRegression)
    assert isinstance(regressor_results[1], Ridge)
    assert not hasattr(regressor_results[0], "coef_")
    assert not hasattr(regressor_results[1], "coef_")
    assert regressor_results[0] is not grid_search.best_estimator_
    assert regressor_results[1] is not grid_search.best_estimator_
    # check that we didn't modify the parameter grid that was passed
    assert not hasattr(param_grid["regressor"][0], "coef_")
    assert not hasattr(param_grid["regressor"][1], "coef_")


@pytest.mark.parametrize("SearchCV", [GridSearchCV, RandomizedSearchCV])
def test_SearchCV_with_fit_params(SearchCV):
    X = np.arange(100).reshape(10, 10)
    y = np.array([0] * 5 + [1] * 5)
    clf = CheckingClassifier(expected_fit_params=["spam", "eggs"])
    searcher = SearchCV(clf, {"foo_param": [1, 2, 3]}, cv=2, error_score="raise")

    # The CheckingClassifier generates an assertion error if
    # a parameter is missing or has length != len(X).
    err_msg = r"Expected fit parameter\(s\) \['eggs'\] not seen."
    with pytest.raises(AssertionError, match=err_msg):
        searcher.fit(X, y, spam=np.ones(10))

    err_msg = "Fit parameter spam has length 1; expected"
    with pytest.raises(AssertionError, match=err_msg):
        searcher.fit(X, y, spam=np.ones(1), eggs=np.zeros(10))
    searcher.fit(X, y, spam=np.ones(10), eggs=np.zeros(10))


@ignore_warnings
def test_grid_search_no_score():
    # Test grid-search on classifier that has no score function.
    clf = LinearSVC(random_state=0)
    X, y = make_blobs(random_state=0, centers=2)
    Cs = [0.1, 1, 10]
    clf_no_score = LinearSVCNoScore(random_state=0)
    grid_search = GridSearchCV(clf, {"C": Cs}, scoring="accuracy")
    grid_search.fit(X, y)

    grid_search_no_score = GridSearchCV(clf_no_score, {"C": Cs}, scoring="accuracy")
    # smoketest grid search
    grid_search_no_score.fit(X, y)

    # check that best params are equal
    assert grid_search_no_score.best_params_ == grid_search.best_params_
    # check that we can call score and that it gives the correct result
    assert grid_search.score(X, y) == grid_search_no_score.score(X, y)

    # giving no scoring function raises an error
    grid_search_no_score = GridSearchCV(clf_no_score, {"C": Cs})
    with pytest.raises(TypeError, match="no scoring"):
        grid_search_no_score.fit([[1]])


def test_grid_search_score_method():
    X, y = make_classification(n_samples=100, n_classes=2, flip_y=0.2, random_state=0)
    clf = LinearSVC(random_state=0)
    grid = {"C": [0.1]}

    search_no_scoring = GridSearchCV(clf, grid, scoring=None).fit(X, y)
    search_accuracy = GridSearchCV(clf, grid, scoring="accuracy").fit(X, y)
    search_no_score_method_auc = GridSearchCV(
        LinearSVCNoScore(), grid, scoring="roc_auc"
    ).fit(X, y)
    search_auc = GridSearchCV(clf, grid, scoring="roc_auc").fit(X, y)

    # Check warning only occurs in situation where behavior changed:
    # estimator requires score method to compete with scoring parameter
    score_no_scoring = search_no_scoring.score(X, y)
    score_accuracy = search_accuracy.score(X, y)
    score_no_score_auc = search_no_score_method_auc.score(X, y)
    score_auc = search_auc.score(X, y)

    # ensure the test is sane
    assert score_auc < 1.0
    assert score_accuracy < 1.0
    assert score_auc != score_accuracy

    assert_almost_equal(score_accuracy, score_no_scoring)
    assert_almost_equal(score_auc, score_no_score_auc)


def test_grid_search_groups():
    # Check if ValueError (when groups is None) propagates to GridSearchCV
    # And also check if groups is correctly passed to the cv object
    rng = np.random.RandomState(0)

    X, y = make_classification(n_samples=15, n_classes=2, random_state=0)
    groups = rng.randint(0, 3, 15)

    clf = LinearSVC(random_state=0)
    grid = {"C": [1]}

    group_cvs = [
        LeaveOneGroupOut(),
        LeavePGroupsOut(2),
        GroupKFold(n_splits=3),
        GroupShuffleSplit(),
    ]
    error_msg = "The 'groups' parameter should not be None."
    for cv in group_cvs:
        gs = GridSearchCV(clf, grid, cv=cv)
        with pytest.raises(ValueError, match=error_msg):
            gs.fit(X, y)
        gs.fit(X, y, groups=groups)

    non_group_cvs = [StratifiedKFold(), StratifiedShuffleSplit()]
    for cv in non_group_cvs:
        gs = GridSearchCV(clf, grid, cv=cv)
        # Should not raise an error
        gs.fit(X, y)


def test_classes__property():
    # Test that classes_ property matches best_estimator_.classes_
    X = np.arange(100).reshape(10, 10)
    y = np.array([0] * 5 + [1] * 5)
    Cs = [0.1, 1, 10]

    grid_search = GridSearchCV(LinearSVC(random_state=0), {"C": Cs})
    grid_search.fit(X, y)
    assert_array_equal(grid_search.best_estimator_.classes_, grid_search.classes_)

    # Test that regressors do not have a classes_ attribute
    grid_search = GridSearchCV(Ridge(), {"alpha": [1.0, 2.0]})
    grid_search.fit(X, y)
    assert not hasattr(grid_search, "classes_")

    # Test that the grid searcher has no classes_ attribute before it's fit
    grid_search = GridSearchCV(LinearSVC(random_state=0), {"C": Cs})
    assert not hasattr(grid_search, "classes_")

    # Test that the grid searcher has no classes_ attribute without a refit
    grid_search = GridSearchCV(LinearSVC(random_state=0), {"C": Cs}, refit=False)
    grid_search.fit(X, y)
    assert not hasattr(grid_search, "classes_")


def test_trivial_cv_results_attr():
    # Test search over a "grid" with only one point.
    clf = MockClassifier()
    grid_search = GridSearchCV(clf, {"foo_param": [1]}, cv=3)
    grid_search.fit(X, y)
    assert hasattr(grid_search, "cv_results_")

    random_search = RandomizedSearchCV(clf, {"foo_param": [0]}, n_iter=1, cv=3)
    random_search.fit(X, y)
    assert hasattr(grid_search, "cv_results_")


def test_no_refit():
    # Test that GSCV can be used for model selection alone without refitting
    clf = MockClassifier()
    for scoring in [None, ["accuracy", "precision"]]:
        grid_search = GridSearchCV(clf, {"foo_param": [1, 2, 3]}, refit=False, cv=3)
        grid_search.fit(X, y)
        assert (
            not hasattr(grid_search, "best_estimator_")
            and hasattr(grid_search, "best_index_")
            and hasattr(grid_search, "best_params_")
        )

        # Make sure the functions predict/transform etc raise meaningful
        # error messages
        for fn_name in (
            "predict",
            "predict_proba",
            "predict_log_proba",
            "transform",
            "inverse_transform",
        ):
            error_msg = (
                f"`refit=False`. {fn_name} is available only after "
                "refitting on the best parameters"
            )
            with pytest.raises(AttributeError, match=error_msg):
                getattr(grid_search, fn_name)(X)

    # Test that an invalid refit param raises appropriate error messages
    error_msg = (
        "For multi-metric scoring, the parameter refit must be set to a scorer key"
    )
    for refit in ["", 5, True, "recall", "accuracy"]:
        with pytest.raises(ValueError, match=error_msg):
            GridSearchCV(
                clf, {}, refit=refit, scoring={"acc": "accuracy", "prec": "precision"}
            ).fit(X, y)


def test_grid_search_error():
    # Test that grid search will capture errors on data with different length
    X_, y_ = make_classification(n_samples=200, n_features=100, random_state=0)

    clf = LinearSVC()
    cv = GridSearchCV(clf, {"C": [0.1, 1.0]})
    with pytest.raises(ValueError):
        cv.fit(X_[:180], y_)


def test_grid_search_one_grid_point():
    X_, y_ = make_classification(n_samples=200, n_features=100, random_state=0)
    param_dict = {"C": [1.0], "kernel": ["rbf"], "gamma": [0.1]}

    clf = SVC(gamma="auto")
    cv = GridSearchCV(clf, param_dict)
    cv.fit(X_, y_)

    clf = SVC(C=1.0, kernel="rbf", gamma=0.1)
    clf.fit(X_, y_)

    assert_array_equal(clf.dual_coef_, cv.best_estimator_.dual_coef_)


def test_grid_search_when_param_grid_includes_range():
    # Test that the best estimator contains the right value for foo_param
    clf = MockClassifier()
    grid_search = None
    grid_search = GridSearchCV(clf, {"foo_param": range(1, 4)}, cv=3)
    grid_search.fit(X, y)
    assert grid_search.best_estimator_.foo_param == 2


def test_grid_search_bad_param_grid():
    param_dict = {"C": 1}
    clf = SVC(gamma="auto")
    error_msg = re.escape(
        "Parameter grid for parameter (C) needs to"
        " be a list or numpy array, but got (<class 'int'>)."
        " Single values need to be wrapped in a list"
        " with one element."
    )
    with pytest.raises(ValueError, match=error_msg):
        GridSearchCV(clf, param_dict)

    param_dict = {"C": []}
    clf = SVC()
    error_msg = re.escape(
        "Parameter values for parameter (C) need to be a non-empty sequence."
    )
    with pytest.raises(ValueError, match=error_msg):
        GridSearchCV(clf, param_dict)

    param_dict = {"C": "1,2,3"}
    clf = SVC(gamma="auto")
    error_msg = re.escape(
        "Parameter grid for parameter (C) needs to"
        " be a list or numpy array, but got (<class 'str'>)."
        " Single values need to be wrapped in a list"
        " with one element."
    )
    with pytest.raises(ValueError, match=error_msg):
        GridSearchCV(clf, param_dict)

    param_dict = {"C": np.ones((3, 2))}
    clf = SVC()
    with pytest.raises(ValueError):
        GridSearchCV(clf, param_dict)


def test_grid_search_sparse():
    # Test that grid search works with both dense and sparse matrices
    X_, y_ = make_classification(n_samples=200, n_features=100, random_state=0)

    clf = LinearSVC()
    cv = GridSearchCV(clf, {"C": [0.1, 1.0]})
    cv.fit(X_[:180], y_[:180])
    y_pred = cv.predict(X_[180:])
    C = cv.best_estimator_.C

    X_ = sp.csr_matrix(X_)
    clf = LinearSVC()
    cv = GridSearchCV(clf, {"C": [0.1, 1.0]})
    cv.fit(X_[:180].tocoo(), y_[:180])
    y_pred2 = cv.predict(X_[180:])
    C2 = cv.best_estimator_.C

    assert np.mean(y_pred == y_pred2) >= 0.9
    assert C == C2


def test_grid_search_sparse_scoring():
    X_, y_ = make_classification(n_samples=200, n_features=100, random_state=0)

    clf = LinearSVC()
    cv = GridSearchCV(clf, {"C": [0.1, 1.0]}, scoring="f1")
    cv.fit(X_[:180], y_[:180])
    y_pred = cv.predict(X_[180:])
    C = cv.best_estimator_.C

    X_ = sp.csr_matrix(X_)
    clf = LinearSVC()
    cv = GridSearchCV(clf, {"C": [0.1, 1.0]}, scoring="f1")
    cv.fit(X_[:180], y_[:180])
    y_pred2 = cv.predict(X_[180:])
    C2 = cv.best_estimator_.C

    assert_array_equal(y_pred, y_pred2)
    assert C == C2
    # Smoke test the score
    # np.testing.assert_allclose(f1_score(cv.predict(X_[:180]), y[:180]),
    #                            cv.score(X_[:180], y[:180]))

    # test loss where greater is worse
    def f1_loss(y_true_, y_pred_):
        return -f1_score(y_true_, y_pred_)

    F1Loss = make_scorer(f1_loss, greater_is_better=False)
    cv = GridSearchCV(clf, {"C": [0.1, 1.0]}, scoring=F1Loss)
    cv.fit(X_[:180], y_[:180])
    y_pred3 = cv.predict(X_[180:])
    C3 = cv.best_estimator_.C

    assert C == C3
    assert_array_equal(y_pred, y_pred3)


def test_grid_search_precomputed_kernel():
    # Test that grid search works when the input features are given in the
    # form of a precomputed kernel matrix
    X_, y_ = make_classification(n_samples=200, n_features=100, random_state=0)

    # compute the training kernel matrix corresponding to the linear kernel
    K_train = np.dot(X_[:180], X_[:180].T)
    y_train = y_[:180]

    clf = SVC(kernel="precomputed")
    cv = GridSearchCV(clf, {"C": [0.1, 1.0]})
    cv.fit(K_train, y_train)

    assert cv.best_score_ >= 0

    # compute the test kernel matrix
    K_test = np.dot(X_[180:], X_[:180].T)
    y_test = y_[180:]

    y_pred = cv.predict(K_test)

    assert np.mean(y_pred == y_test) >= 0

    # test error is raised when the precomputed kernel is not array-like
    # or sparse
    with pytest.raises(ValueError):
        cv.fit(K_train.tolist(), y_train)


def test_grid_search_precomputed_kernel_error_nonsquare():
    # Test that grid search returns an error with a non-square precomputed
    # training kernel matrix
    K_train = np.zeros((10, 20))
    y_train = np.ones((10,))
    clf = SVC(kernel="precomputed")
    cv = GridSearchCV(clf, {"C": [0.1, 1.0]})
    with pytest.raises(ValueError):
        cv.fit(K_train, y_train)


class BrokenClassifier(BaseEstimator):
    """Broken classifier that cannot be fit twice"""

    def __init__(self, parameter=None):
        self.parameter = parameter

    def fit(self, X, y):
        assert not hasattr(self, "has_been_fit_")
        self.has_been_fit_ = True

    def predict(self, X):
        return np.zeros(X.shape[0])


@ignore_warnings
def test_refit():
    # Regression test for bug in refitting
    # Simulates re-fitting a broken estimator; this used to break with
    # sparse SVMs.
    X = np.arange(100).reshape(10, 10)
    y = np.array([0] * 5 + [1] * 5)

    clf = GridSearchCV(
        BrokenClassifier(), [{"parameter": [0, 1]}], scoring="precision", refit=True
    )
    clf.fit(X, y)


def test_refit_callable():
    """
    Test refit=callable, which adds flexibility in identifying the
    "best" estimator.
    """

    def refit_callable(cv_results):
        """
        A dummy function tests `refit=callable` interface.
        Return the index of a model that has the least
        `mean_test_score`.
        """
        # Fit a dummy clf with `refit=True` to get a list of keys in
        # clf.cv_results_.
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        clf = GridSearchCV(
            LinearSVC(random_state=42),
            {"C": [0.01, 0.1, 1]},
            scoring="precision",
            refit=True,
        )
        clf.fit(X, y)
        # Ensure that `best_index_ != 0` for this dummy clf
        assert clf.best_index_ != 0

        # Assert every key matches those in `cv_results`
        for key in clf.cv_results_.keys():
            assert key in cv_results

        return cv_results["mean_test_score"].argmin()

    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    clf = GridSearchCV(
        LinearSVC(random_state=42),
        {"C": [0.01, 0.1, 1]},
        scoring="precision",
        refit=refit_callable,
    )
    clf.fit(X, y)

    assert clf.best_index_ == 0
    # Ensure `best_score_` is disabled when using `refit=callable`
    assert not hasattr(clf, "best_score_")


def test_refit_callable_invalid_type():
    """
    Test implementation catches the errors when 'best_index_' returns an
    invalid result.
    """

    def refit_callable_invalid_type(cv_results):
        """
        A dummy function tests when returned 'best_index_' is not integer.
        """
        return None

    X, y = make_classification(n_samples=100, n_features=4, random_state=42)

    clf = GridSearchCV(
        LinearSVC(random_state=42),
        {"C": [0.1, 1]},
        scoring="precision",
        refit=refit_callable_invalid_type,
    )
    with pytest.raises(TypeError, match="best_index_ returned is not an integer"):
        clf.fit(X, y)


@pytest.mark.parametrize("out_bound_value", [-1, 2])
@pytest.mark.parametrize("search_cv", [RandomizedSearchCV, GridSearchCV])
def test_refit_callable_out_bound(out_bound_value, search_cv):
    """
    Test implementation catches the errors when 'best_index_' returns an
    out of bound result.
    """

    def refit_callable_out_bound(cv_results):
        """
        A dummy function tests when returned 'best_index_' is out of bounds.
        """
        return out_bound_value

    X, y = make_classification(n_samples=100, n_features=4, random_state=42)

    clf = search_cv(
        LinearSVC(random_state=42),
        {"C": [0.1, 1]},
        scoring="precision",
        refit=refit_callable_out_bound,
    )
    with pytest.raises(IndexError, match="best_index_ index out of range"):
        clf.fit(X, y)


def test_refit_callable_multi_metric():
    """
    Test refit=callable in multiple metric evaluation setting
    """

    def refit_callable(cv_results):
        """
        A dummy function tests `refit=callable` interface.
        Return the index of a model that has the least
        `mean_test_prec`.
        """
        assert "mean_test_prec" in cv_results
        return cv_results["mean_test_prec"].argmin()

    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    scoring = {"Accuracy": make_scorer(accuracy_score), "prec": "precision"}
    clf = GridSearchCV(
        LinearSVC(random_state=42),
        {"C": [0.01, 0.1, 1]},
        scoring=scoring,
        refit=refit_callable,
    )
    clf.fit(X, y)

    assert clf.best_index_ == 0
    # Ensure `best_score_` is disabled when using `refit=callable`
    assert not hasattr(clf, "best_score_")


def test_gridsearch_nd():
    # Pass X as list in GridSearchCV
    X_4d = np.arange(10 * 5 * 3 * 2).reshape(10, 5, 3, 2)
    y_3d = np.arange(10 * 7 * 11).reshape(10, 7, 11)

    def check_X(x):
        return x.shape[1:] == (5, 3, 2)

    def check_y(x):
        return x.shape[1:] == (7, 11)

    clf = CheckingClassifier(
        check_X=check_X,
        check_y=check_y,
        methods_to_check=["fit"],
    )
    grid_search = GridSearchCV(clf, {"foo_param": [1, 2, 3]})
    grid_search.fit(X_4d, y_3d).score(X, y)
    assert hasattr(grid_search, "cv_results_")


def test_X_as_list():
    # Pass X as list in GridSearchCV
    X = np.arange(100).reshape(10, 10)
    y = np.array([0] * 5 + [1] * 5)

    clf = CheckingClassifier(
        check_X=lambda x: isinstance(x, list),
        methods_to_check=["fit"],
    )
    cv = KFold(n_splits=3)
    grid_search = GridSearchCV(clf, {"foo_param": [1, 2, 3]}, cv=cv)
    grid_search.fit(X.tolist(), y).score(X, y)
    assert hasattr(grid_search, "cv_results_")


def test_y_as_list():
    # Pass y as list in GridSearchCV
    X = np.arange(100).reshape(10, 10)
    y = np.array([0] * 5 + [1] * 5)

    clf = CheckingClassifier(
        check_y=lambda x: isinstance(x, list),
        methods_to_check=["fit"],
    )
    cv = KFold(n_splits=3)
    grid_search = GridSearchCV(clf, {"foo_param": [1, 2, 3]}, cv=cv)
    grid_search.fit(X, y.tolist()).score(X, y)
    assert hasattr(grid_search, "cv_results_")


@ignore_warnings
def test_pandas_input():
    # check cross_val_score doesn't destroy pandas dataframe
    types = [(MockDataFrame, MockDataFrame)]
    try:
        from pandas import Series, DataFrame

        types.append((DataFrame, Series))
    except ImportError:
        pass

    X = np.arange(100).reshape(10, 10)
    y = np.array([0] * 5 + [1] * 5)

    for InputFeatureType, TargetType in types:
        # X dataframe, y series
        X_df, y_ser = InputFeatureType(X), TargetType(y)

        def check_df(x):
            return isinstance(x, InputFeatureType)

        def check_series(x):
            return isinstance(x, TargetType)

        clf = CheckingClassifier(check_X=check_df, check_y=check_series)

        grid_search = GridSearchCV(clf, {"foo_param": [1, 2, 3]})
        grid_search.fit(X_df, y_ser).score(X_df, y_ser)
        grid_search.predict(X_df)
        assert hasattr(grid_search, "cv_results_")


def test_unsupervised_grid_search():
    # test grid-search with unsupervised estimator
    X, y = make_blobs(n_samples=50, random_state=0)
    km = KMeans(random_state=0, init="random", n_init=1)

    # Multi-metric evaluation unsupervised
    scoring = ["adjusted_rand_score", "fowlkes_mallows_score"]
    for refit in ["adjusted_rand_score", "fowlkes_mallows_score"]:
        grid_search = GridSearchCV(
            km, param_grid=dict(n_clusters=[2, 3, 4]), scoring=scoring, refit=refit
        )
        grid_search.fit(X, y)
        # Both ARI and FMS can find the right number :)
        assert grid_search.best_params_["n_clusters"] == 3

    # Single metric evaluation unsupervised
    grid_search = GridSearchCV(
        km, param_grid=dict(n_clusters=[2, 3, 4]), scoring="fowlkes_mallows_score"
    )
    grid_search.fit(X, y)
    assert grid_search.best_params_["n_clusters"] == 3

    # Now without a score, and without y
    grid_search = GridSearchCV(km, param_grid=dict(n_clusters=[2, 3, 4]))
    grid_search.fit(X)
    assert grid_search.best_params_["n_clusters"] == 4


def test_gridsearch_no_predict():
    # test grid-search with an estimator without predict.
    # slight duplication of a test from KDE
    def custom_scoring(estimator, X):
        return 42 if estimator.bandwidth == 0.1 else 0

    X, _ = make_blobs(cluster_std=0.1, random_state=1, centers=[[0, 1], [1, 0], [0, 0]])
    search = GridSearchCV(
        KernelDensity(),
        param_grid=dict(bandwidth=[0.01, 0.1, 1]),
        scoring=custom_scoring,
    )
    search.fit(X)
    assert search.best_params_["bandwidth"] == 0.1
    assert search.best_score_ == 42


def test_param_sampler():
    # test basic properties of param sampler
    param_distributions = {"kernel": ["rbf", "linear"], "C": uniform(0, 1)}
    sampler = ParameterSampler(
        param_distributions=param_distributions, n_iter=10, random_state=0
    )
    samples = [x for x in sampler]
    assert len(samples) == 10
    for sample in samples:
        assert sample["kernel"] in ["rbf", "linear"]
        assert 0 <= sample["C"] <= 1

    # test that repeated calls yield identical parameters
    param_distributions = {"C": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    sampler = ParameterSampler(
        param_distributions=param_distributions, n_iter=3, random_state=0
    )
    assert [x for x in sampler] == [x for x in sampler]

    param_distributions = {"C": uniform(0, 1)}
    sampler = ParameterSampler(
        param_distributions=param_distributions, n_iter=10, random_state=0
    )
    assert [x for x in sampler] == [x for x in sampler]


def check_cv_results_array_types(search, param_keys, score_keys):
    # Check if the search `cv_results`'s array are of correct types
    cv_results = search.cv_results_
    assert all(isinstance(cv_results[param], np.ma.MaskedArray) for param in param_keys)
    assert all(cv_results[key].dtype == object for key in param_keys)
    assert not any(isinstance(cv_results[key], np.ma.MaskedArray) for key in score_keys)
    assert all(
        cv_results[key].dtype == np.float64
        for key in score_keys
        if not key.startswith("rank")
    )

    scorer_keys = search.scorer_.keys() if search.multimetric_ else ["score"]

    for key in scorer_keys:
        assert cv_results["rank_test_%s" % key].dtype == np.int32


def check_cv_results_keys(cv_results, param_keys, score_keys, n_cand):
    # Test the search.cv_results_ contains all the required results
    assert_array_equal(
        sorted(cv_results.keys()), sorted(param_keys + score_keys + ("params",))
    )
    assert all(cv_results[key].shape == (n_cand,) for key in param_keys + score_keys)


def test_grid_search_cv_results():
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)

    n_splits = 3
    n_grid_points = 6
    params = [
        dict(
            kernel=[
                "rbf",
            ],
            C=[1, 10],
            gamma=[0.1, 1],
        ),
        dict(
            kernel=[
                "poly",
            ],
            degree=[1, 2],
        ),
    ]

    param_keys = ("param_C", "param_degree", "param_gamma", "param_kernel")
    score_keys = (
        "mean_test_score",
        "mean_train_score",
        "rank_test_score",
        "split0_test_score",
        "split1_test_score",
        "split2_test_score",
        "split0_train_score",
        "split1_train_score",
        "split2_train_score",
        "std_test_score",
        "std_train_score",
        "mean_fit_time",
        "std_fit_time",
        "mean_score_time",
        "std_score_time",
    )
    n_candidates = n_grid_points

    search = GridSearchCV(
        SVC(), cv=n_splits, param_grid=params, return_train_score=True
    )
    search.fit(X, y)
    cv_results = search.cv_results_
    # Check if score and timing are reasonable
    assert all(cv_results["rank_test_score"] >= 1)
    assert (all(cv_results[k] >= 0) for k in score_keys if k != "rank_test_score")
    assert (
        all(cv_results[k] <= 1)
        for k in score_keys
        if "time" not in k and k != "rank_test_score"
    )
    # Check cv_results structure
    check_cv_results_array_types(search, param_keys, score_keys)
    check_cv_results_keys(cv_results, param_keys, score_keys, n_candidates)
    # Check masking
    cv_results = search.cv_results_
    n_candidates = len(search.cv_results_["params"])
    assert all(
        (
            cv_results["param_C"].mask[i]
            and cv_results["param_gamma"].mask[i]
            and not cv_results["param_degree"].mask[i]
        )
        for i in range(n_candidates)
        if cv_results["param_kernel"][i] == "linear"
    )
    assert all(
        (
            not cv_results["param_C"].mask[i]
            and not cv_results["param_gamma"].mask[i]
            and cv_results["param_degree"].mask[i]
        )
        for i in range(n_candidates)
        if cv_results["param_kernel"][i] == "rbf"
    )


def test_random_search_cv_results():
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)

    n_splits = 3
    n_search_iter = 30

    params = [
        {"kernel": ["rbf"], "C": expon(scale=10), "gamma": expon(scale=0.1)},
        {"kernel": ["poly"], "degree": [2, 3]},
    ]
    param_keys = ("param_C", "param_degree", "param_gamma", "param_kernel")
    score_keys = (
        "mean_test_score",
        "mean_train_score",
        "rank_test_score",
        "split0_test_score",
        "split1_test_score",
        "split2_test_score",
        "split0_train_score",
        "split1_train_score",
        "split2_train_score",
        "std_test_score",
        "std_train_score",
        "mean_fit_time",
        "std_fit_time",
        "mean_score_time",
        "std_score_time",
    )
    n_cand = n_search_iter

    search = RandomizedSearchCV(
        SVC(),
        n_iter=n_search_iter,
        cv=n_splits,
        param_distributions=params,
        return_train_score=True,
    )
    search.fit(X, y)
    cv_results = search.cv_results_
    # Check results structure
    check_cv_results_array_types(search, param_keys, score_keys)
    check_cv_results_keys(cv_results, param_keys, score_keys, n_cand)
    n_candidates = len(search.cv_results_["params"])
    assert all(
        (
            cv_results["param_C"].mask[i]
            and cv_results["param_gamma"].mask[i]
            and not cv_results["param_degree"].mask[i]
        )
        for i in range(n_candidates)
        if cv_results["param_kernel"][i] == "linear"
    )
    assert all(
        (
            not cv_results["param_C"].mask[i]
            and not cv_results["param_gamma"].mask[i]
            and cv_results["param_degree"].mask[i]
        )
        for i in range(n_candidates)
        if cv_results["param_kernel"][i] == "rbf"
    )


@pytest.mark.parametrize(
    "SearchCV, specialized_params",
    [
        (GridSearchCV, {"param_grid": {"C": [1, 10]}}),
        (RandomizedSearchCV, {"param_distributions": {"C": [1, 10]}, "n_iter": 2}),
    ],
)
def test_search_default_iid(SearchCV, specialized_params):
    # Test the IID parameter  TODO: Clearly this test does something else???
    # noise-free simple 2d-data
    X, y = make_blobs(
        centers=[[0, 0], [1, 0], [0, 1], [1, 1]],
        random_state=0,
        cluster_std=0.1,
        shuffle=False,
        n_samples=80,
    )
    # split dataset into two folds that are not iid
    # first one contains data of all 4 blobs, second only from two.
    mask = np.ones(X.shape[0], dtype=bool)
    mask[np.where(y == 1)[0][::2]] = 0
    mask[np.where(y == 2)[0][::2]] = 0
    # this leads to perfect classification on one fold and a score of 1/3 on
    # the other
    # create "cv" for splits
    cv = [[mask, ~mask], [~mask, mask]]

    common_params = {"estimator": SVC(), "cv": cv, "return_train_score": True}
    search = SearchCV(**common_params, **specialized_params)
    search.fit(X, y)

    test_cv_scores = np.array(
        [
            search.cv_results_["split%d_test_score" % s][0]
            for s in range(search.n_splits_)
        ]
    )
    test_mean = search.cv_results_["mean_test_score"][0]
    test_std = search.cv_results_["std_test_score"][0]

    train_cv_scores = np.array(
        [
            search.cv_results_["split%d_train_score" % s][0]
            for s in range(search.n_splits_)
        ]
    )
    train_mean = search.cv_results_["mean_train_score"][0]
    train_std = search.cv_results_["std_train_score"][0]

    assert search.cv_results_["param_C"][0] == 1
    # scores are the same as above
    assert_allclose(test_cv_scores, [1, 1.0 / 3.0])
    assert_allclose(train_cv_scores, [1, 1])
    # Unweighted mean/std is used
    assert test_mean == pytest.approx(np.mean(test_cv_scores))
    assert test_std == pytest.approx(np.std(test_cv_scores))

    # For the train scores, we do not take a weighted mean irrespective of
    # i.i.d. or not
    assert train_mean == pytest.approx(1)
    assert train_std == pytest.approx(0)


def test_grid_search_cv_results_multimetric():
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)

    n_splits = 3
    params = [
        dict(
            kernel=[
                "rbf",
            ],
            C=[1, 10],
            gamma=[0.1, 1],
        ),
        dict(
            kernel=[
                "poly",
            ],
            degree=[1, 2],
        ),
    ]

    grid_searches = []
    for scoring in (
        {"accuracy": make_scorer(accuracy_score), "recall": make_scorer(recall_score)},
        "accuracy",
        "recall",
    ):
        grid_search = GridSearchCV(
            SVC(), cv=n_splits, param_grid=params, scoring=scoring, refit=False
        )
        grid_search.fit(X, y)
        grid_searches.append(grid_search)

    compare_cv_results_multimetric_with_single(*grid_searches)


def test_random_search_cv_results_multimetric():
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)

    n_splits = 3
    n_search_iter = 30

    # Scipy 0.12's stats dists do not accept seed, hence we use param grid
    params = dict(C=np.logspace(-4, 1, 3), gamma=np.logspace(-5, 0, 3, base=0.1))
    for refit in (True, False):
        random_searches = []
        for scoring in (("accuracy", "recall"), "accuracy", "recall"):
            # If True, for multi-metric pass refit='accuracy'
            if refit:
                probability = True
                refit = "accuracy" if isinstance(scoring, tuple) else refit
            else:
                probability = False
            clf = SVC(probability=probability, random_state=42)
            random_search = RandomizedSearchCV(
                clf,
                n_iter=n_search_iter,
                cv=n_splits,
                param_distributions=params,
                scoring=scoring,
                refit=refit,
                random_state=0,
            )
            random_search.fit(X, y)
            random_searches.append(random_search)

        compare_cv_results_multimetric_with_single(*random_searches)
        compare_refit_methods_when_refit_with_acc(
            random_searches[0], random_searches[1], refit
        )


def compare_cv_results_multimetric_with_single(search_multi, search_acc, search_rec):
    """Compare multi-metric cv_results with the ensemble of multiple
    single metric cv_results from single metric grid/random search"""

    assert search_multi.multimetric_
    assert_array_equal(sorted(search_multi.scorer_), ("accuracy", "recall"))

    cv_results_multi = search_multi.cv_results_
    cv_results_acc_rec = {
        re.sub("_score$", "_accuracy", k): v for k, v in search_acc.cv_results_.items()
    }
    cv_results_acc_rec.update(
        {re.sub("_score$", "_recall", k): v for k, v in search_rec.cv_results_.items()}
    )

    # Check if score and timing are reasonable, also checks if the keys
    # are present
    assert all(
        (
            np.all(cv_results_multi[k] <= 1)
            for k in (
                "mean_score_time",
                "std_score_time",
                "mean_fit_time",
                "std_fit_time",
            )
        )
    )

    # Compare the keys, other than time keys, among multi-metric and
    # single metric grid search results. np.testing.assert_equal performs a
    # deep nested comparison of the two cv_results dicts
    np.testing.assert_equal(
        {k: v for k, v in cv_results_multi.items() if not k.endswith("_time")},
        {k: v for k, v in cv_results_acc_rec.items() if not k.endswith("_time")},
    )


def compare_refit_methods_when_refit_with_acc(search_multi, search_acc, refit):
    """Compare refit multi-metric search methods with single metric methods"""
    assert search_acc.refit == refit
    if refit:
        assert search_multi.refit == "accuracy"
    else:
        assert not search_multi.refit
        return  # search cannot predict/score without refit

    X, y = make_blobs(n_samples=100, n_features=4, random_state=42)
    for method in ("predict", "predict_proba", "predict_log_proba"):
        assert_almost_equal(
            getattr(search_multi, method)(X), getattr(search_acc, method)(X)
        )
    assert_almost_equal(search_multi.score(X, y), search_acc.score(X, y))
    for key in ("best_index_", "best_score_", "best_params_"):
        assert getattr(search_multi, key) == getattr(search_acc, key)


@pytest.mark.parametrize(
    "search_cv",
    [
        RandomizedSearchCV(
            estimator=DecisionTreeClassifier(),
            param_distributions={"max_depth": [5, 10]},
        ),
        GridSearchCV(
            estimator=DecisionTreeClassifier(), param_grid={"max_depth": [5, 10]}
        ),
    ],
)
def test_search_cv_score_samples_error(search_cv):
    X, y = make_blobs(n_samples=100, n_features=4, random_state=42)
    search_cv.fit(X, y)

    # Make sure to error out when underlying estimator does not implement
    # the method `score_samples`
    err_msg = "'DecisionTreeClassifier' object has no attribute 'score_samples'"

    with pytest.raises(AttributeError, match=err_msg):
        search_cv.score_samples(X)


@pytest.mark.parametrize(
    "search_cv",
    [
        RandomizedSearchCV(
            estimator=LocalOutlierFactor(novelty=True),
            param_distributions={"n_neighbors": [5, 10]},
            scoring="precision",
        ),
        GridSearchCV(
            estimator=LocalOutlierFactor(novelty=True),
            param_grid={"n_neighbors": [5, 10]},
            scoring="precision",
        ),
    ],
)
def test_search_cv_score_samples_method(search_cv):
    # Set parameters
    rng = np.random.RandomState(42)
    n_samples = 300
    outliers_fraction = 0.15
    n_outliers = int(outliers_fraction * n_samples)
    n_inliers = n_samples - n_outliers

    # Create dataset
    X = make_blobs(
        n_samples=n_inliers,
        n_features=2,
        centers=[[0, 0], [0, 0]],
        cluster_std=0.5,
        random_state=0,
    )[0]
    # Add some noisy points
    X = np.concatenate([X, rng.uniform(low=-6, high=6, size=(n_outliers, 2))], axis=0)

    # Define labels to be able to score the estimator with `search_cv`
    y_true = np.array([1] * n_samples)
    y_true[-n_outliers:] = -1

    # Fit on data
    search_cv.fit(X, y_true)

    # Verify that the stand alone estimator yields the same results
    # as the ones obtained with *SearchCV
    assert_allclose(
        search_cv.score_samples(X), search_cv.best_estimator_.score_samples(X)
    )


def test_search_cv_results_rank_tie_breaking():
    X, y = make_blobs(n_samples=50, random_state=42)

    # The two C values are close enough to give similar models
    # which would result in a tie of their mean cv-scores
    param_grid = {"C": [1, 1.001, 0.001]}

    grid_search = GridSearchCV(SVC(), param_grid=param_grid, return_train_score=True)
    random_search = RandomizedSearchCV(
        SVC(), n_iter=3, param_distributions=param_grid, return_train_score=True
    )

    for search in (grid_search, random_search):
        search.fit(X, y)
        cv_results = search.cv_results_
        # Check tie breaking strategy -
        # Check that there is a tie in the mean scores between
        # candidates 1 and 2 alone
        assert_almost_equal(
            cv_results["mean_test_score"][0], cv_results["mean_test_score"][1]
        )
        assert_almost_equal(
            cv_results["mean_train_score"][0], cv_results["mean_train_score"][1]
        )
        assert not np.allclose(
            cv_results["mean_test_score"][1], cv_results["mean_test_score"][2]
        )
        assert not np.allclose(
            cv_results["mean_train_score"][1], cv_results["mean_train_score"][2]
        )
        # 'min' rank should be assigned to the tied candidates
        assert_almost_equal(search.cv_results_["rank_test_score"], [1, 1, 3])


def test_search_cv_results_none_param():
    X, y = [[1], [2], [3], [4], [5]], [0, 0, 0, 0, 1]
    estimators = (DecisionTreeRegressor(), DecisionTreeClassifier())
    est_parameters = {"random_state": [0, None]}
    cv = KFold()

    for est in estimators:
        grid_search = GridSearchCV(
            est,
            est_parameters,
            cv=cv,
        ).fit(X, y)
        assert_array_equal(grid_search.cv_results_["param_random_state"], [0, None])


@ignore_warnings()
def test_search_cv_timing():
    svc = LinearSVC(random_state=0)

    X = [
        [
            1,
        ],
        [
            2,
        ],
        [
            3,
        ],
        [
            4,
        ],
    ]
    y = [0, 1, 1, 0]

    gs = GridSearchCV(svc, {"C": [0, 1]}, cv=2, error_score=0)
    rs = RandomizedSearchCV(svc, {"C": [0, 1]}, cv=2, error_score=0, n_iter=2)

    for search in (gs, rs):
        search.fit(X, y)
        for key in ["mean_fit_time", "std_fit_time"]:
            # NOTE The precision of time.time in windows is not high
            # enough for the fit/score times to be non-zero for trivial X and y
            assert np.all(search.cv_results_[key] >= 0)
            assert np.all(search.cv_results_[key] < 1)

        for key in ["mean_score_time", "std_score_time"]:
            assert search.cv_results_[key][1] >= 0
            assert search.cv_results_[key][0] == 0.0
            assert np.all(search.cv_results_[key] < 1)

        assert hasattr(search, "refit_time_")
        assert isinstance(search.refit_time_, float)
        assert search.refit_time_ >= 0


def test_grid_search_correct_score_results():
    # test that correct scores are used
    n_splits = 3
    clf = LinearSVC(random_state=0)
    X, y = make_blobs(random_state=0, centers=2)
    Cs = [0.1, 1, 10]
    for score in ["f1", "roc_auc"]:
        grid_search = GridSearchCV(clf, {"C": Cs}, scoring=score, cv=n_splits)
        cv_results = grid_search.fit(X, y).cv_results_

        # Test scorer names
        result_keys = list(cv_results.keys())
        expected_keys = ("mean_test_score", "rank_test_score") + tuple(
            "split%d_test_score" % cv_i for cv_i in range(n_splits)
        )
        assert all(np.in1d(expected_keys, result_keys))

        cv = StratifiedKFold(n_splits=n_splits)
        n_splits = grid_search.n_splits_
        for candidate_i, C in enumerate(Cs):
            clf.set_params(C=C)
            cv_scores = np.array(
                list(
                    grid_search.cv_results_["split%d_test_score" % s][candidate_i]
                    for s in range(n_splits)
                )
            )
            for i, (train, test) in enumerate(cv.split(X, y)):
                clf.fit(X[train], y[train])
                if score == "f1":
                    correct_score = f1_score(y[test], clf.predict(X[test]))
                elif score == "roc_auc":
                    dec = clf.decision_function(X[test])
                    correct_score = roc_auc_score(y[test], dec)
                assert_almost_equal(correct_score, cv_scores[i])


def test_pickle():
    # Test that a fit search can be pickled
    clf = MockClassifier()
    grid_search = GridSearchCV(clf, {"foo_param": [1, 2, 3]}, refit=True, cv=3)
    grid_search.fit(X, y)
    grid_search_pickled = pickle.loads(pickle.dumps(grid_search))
    assert_array_almost_equal(grid_search.predict(X), grid_search_pickled.predict(X))

    random_search = RandomizedSearchCV(
        clf, {"foo_param": [1, 2, 3]}, refit=True, n_iter=3, cv=3
    )
    random_search.fit(X, y)
    random_search_pickled = pickle.loads(pickle.dumps(random_search))
    assert_array_almost_equal(
        random_search.predict(X), random_search_pickled.predict(X)
    )


def test_grid_search_with_multioutput_data():
    # Test search with multi-output estimator

    X, y = make_multilabel_classification(return_indicator=True, random_state=0)

    est_parameters = {"max_depth": [1, 2, 3, 4]}
    cv = KFold()

    estimators = [
        DecisionTreeRegressor(random_state=0),
        DecisionTreeClassifier(random_state=0),
    ]

    # Test with grid search cv
    for est in estimators:
        grid_search = GridSearchCV(est, est_parameters, cv=cv)
        grid_search.fit(X, y)
        res_params = grid_search.cv_results_["params"]
        for cand_i in range(len(res_params)):
            est.set_params(**res_params[cand_i])

            for i, (train, test) in enumerate(cv.split(X, y)):
                est.fit(X[train], y[train])
                correct_score = est.score(X[test], y[test])
                assert_almost_equal(
                    correct_score,
                    grid_search.cv_results_["split%d_test_score" % i][cand_i],
                )

    # Test with a randomized search
    for est in estimators:
        random_search = RandomizedSearchCV(est, est_parameters, cv=cv, n_iter=3)
        random_search.fit(X, y)
        res_params = random_search.cv_results_["params"]
        for cand_i in range(len(res_params)):
            est.set_params(**res_params[cand_i])

            for i, (train, test) in enumerate(cv.split(X, y)):
                est.fit(X[train], y[train])
                correct_score = est.score(X[test], y[test])
                assert_almost_equal(
                    correct_score,
                    random_search.cv_results_["split%d_test_score" % i][cand_i],
                )


def test_predict_proba_disabled():
    # Test predict_proba when disabled on estimator.
    X = np.arange(20).reshape(5, -1)
    y = [0, 0, 1, 1, 1]
    clf = SVC(probability=False)
    gs = GridSearchCV(clf, {}, cv=2).fit(X, y)
    assert not hasattr(gs, "predict_proba")


def test_grid_search_allows_nans():
    # Test GridSearchCV with SimpleImputer
    X = np.arange(20, dtype=np.float64).reshape(5, -1)
    X[2, :] = np.nan
    y = [0, 0, 1, 1, 1]
    p = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean", missing_values=np.nan)),
            ("classifier", MockClassifier()),
        ]
    )
    GridSearchCV(p, {"classifier__foo_param": [1, 2, 3]}, cv=2).fit(X, y)


class FailingClassifier(BaseEstimator):
    """Classifier that raises a ValueError on fit()"""

    FAILING_PARAMETER = 2

    def __init__(self, parameter=None):
        self.parameter = parameter

    def fit(self, X, y=None):
        if self.parameter == FailingClassifier.FAILING_PARAMETER:
            raise ValueError("Failing classifier failed as required")

    def predict(self, X):
        return np.zeros(X.shape[0])

    def score(self, X=None, Y=None):
        return 0.0


def test_grid_search_failing_classifier():
    # GridSearchCV with on_error != 'raise'
    # Ensures that a warning is raised and score reset where appropriate.

    X, y = make_classification(n_samples=20, n_features=10, random_state=0)

    clf = FailingClassifier()

    # refit=False because we only want to check that errors caused by fits
    # to individual folds will be caught and warnings raised instead. If
    # refit was done, then an exception would be raised on refit and not
    # caught by grid_search (expected behavior), and this would cause an
    # error in this test.
    gs = GridSearchCV(
        clf,
        [{"parameter": [0, 1, 2]}],
        scoring="accuracy",
        refit=False,
        error_score=0.0,
    )

    warning_message = re.compile(
        "5 fits failed.+total of 15.+The score on these"
        r" train-test partitions for these parameters will be set to 0\.0.+"
        "5 fits failed with the following error.+ValueError.+Failing classifier failed"
        " as required",
        flags=re.DOTALL,
    )
    with pytest.warns(FitFailedWarning, match=warning_message):
        gs.fit(X, y)
    n_candidates = len(gs.cv_results_["params"])

    # Ensure that grid scores were set to zero as required for those fits
    # that are expected to fail.
    def get_cand_scores(i):
        return np.array(
            list(
                gs.cv_results_["split%d_test_score" % s][i] for s in range(gs.n_splits_)
            )
        )

    assert all(
        (
            np.all(get_cand_scores(cand_i) == 0.0)
            for cand_i in range(n_candidates)
            if gs.cv_results_["param_parameter"][cand_i]
            == FailingClassifier.FAILING_PARAMETER
        )
    )

    gs = GridSearchCV(
        clf,
        [{"parameter": [0, 1, 2]}],
        scoring="accuracy",
        refit=False,
        error_score=float("nan"),
    )
    warning_message = re.compile(
        "5 fits failed.+total of 15.+The score on these"
        r" train-test partitions for these parameters will be set to nan.+"
        "5 fits failed with the following error.+ValueError.+Failing classifier failed"
        " as required",
        flags=re.DOTALL,
    )
    with pytest.warns(FitFailedWarning, match=warning_message):
        gs.fit(X, y)
    n_candidates = len(gs.cv_results_["params"])
    assert all(
        np.all(np.isnan(get_cand_scores(cand_i)))
        for cand_i in range(n_candidates)
        if gs.cv_results_["param_parameter"][cand_i]
        == FailingClassifier.FAILING_PARAMETER
    )

    ranks = gs.cv_results_["rank_test_score"]

    # Check that succeeded estimators have lower ranks
    assert ranks[0] <= 2 and ranks[1] <= 2
    # Check that failed estimator has the highest rank
    assert ranks[clf.FAILING_PARAMETER] == 3
    assert gs.best_index_ != clf.FAILING_PARAMETER


def test_grid_search_failing_classifier_raise():
    # GridSearchCV with on_error == 'raise' raises the error

    X, y = make_classification(n_samples=20, n_features=10, random_state=0)

    clf = FailingClassifier()

    # refit=False because we want to test the behaviour of the grid search part
    gs = GridSearchCV(
        clf,
        [{"parameter": [0, 1, 2]}],
        scoring="accuracy",
        refit=False,
        error_score="raise",
    )

    # FailingClassifier issues a ValueError so this is what we look for.
    with pytest.raises(ValueError):
        gs.fit(X, y)


def test_parameters_sampler_replacement():
    # raise warning if n_iter is bigger than total parameter space
    params = [
        {"first": [0, 1], "second": ["a", "b", "c"]},
        {"third": ["two", "values"]},
    ]
    sampler = ParameterSampler(params, n_iter=9)
    n_iter = 9
    grid_size = 8
    expected_warning = (
        "The total space of parameters %d is smaller "
        "than n_iter=%d. Running %d iterations. For "
        "exhaustive searches, use GridSearchCV." % (grid_size, n_iter, grid_size)
    )
    with pytest.warns(UserWarning, match=expected_warning):
        list(sampler)

    # degenerates to GridSearchCV if n_iter the same as grid_size
    sampler = ParameterSampler(params, n_iter=8)
    samples = list(sampler)
    assert len(samples) == 8
    for values in ParameterGrid(params):
        assert values in samples
    assert len(ParameterSampler(params, n_iter=1000)) == 8

    # test sampling without replacement in a large grid
    params = {"a": range(10), "b": range(10), "c": range(10)}
    sampler = ParameterSampler(params, n_iter=99, random_state=42)
    samples = list(sampler)
    assert len(samples) == 99
    hashable_samples = ["a%db%dc%d" % (p["a"], p["b"], p["c"]) for p in samples]
    assert len(set(hashable_samples)) == 99

    # doesn't go into infinite loops
    params_distribution = {"first": bernoulli(0.5), "second": ["a", "b", "c"]}
    sampler = ParameterSampler(params_distribution, n_iter=7)
    samples = list(sampler)
    assert len(samples) == 7


def test_stochastic_gradient_loss_param():
    # Make sure the predict_proba works when loss is specified
    # as one of the parameters in the param_grid.
    param_grid = {
        "loss": ["log"],
    }
    X = np.arange(24).reshape(6, -1)
    y = [0, 0, 0, 1, 1, 1]
    clf = GridSearchCV(
        estimator=SGDClassifier(loss="hinge"), param_grid=param_grid, cv=3
    )

    # When the estimator is not fitted, `predict_proba` is not available as the
    # loss is 'hinge'.
    assert not hasattr(clf, "predict_proba")
    clf.fit(X, y)
    clf.predict_proba(X)
    clf.predict_log_proba(X)

    # Make sure `predict_proba` is not available when setting loss=['hinge']
    # in param_grid
    param_grid = {
        "loss": ["hinge"],
    }
    clf = GridSearchCV(
        estimator=SGDClassifier(loss="hinge"), param_grid=param_grid, cv=3
    )
    assert not hasattr(clf, "predict_proba")
    clf.fit(X, y)
    assert not hasattr(clf, "predict_proba")


def test_search_train_scores_set_to_false():
    X = np.arange(6).reshape(6, -1)
    y = [0, 0, 0, 1, 1, 1]
    clf = LinearSVC(random_state=0)

    gs = GridSearchCV(clf, param_grid={"C": [0.1, 0.2]}, cv=3)
    gs.fit(X, y)


def test_grid_search_cv_splits_consistency():
    # Check if a one time iterable is accepted as a cv parameter.
    n_samples = 100
    n_splits = 5
    X, y = make_classification(n_samples=n_samples, random_state=0)

    gs = GridSearchCV(
        LinearSVC(random_state=0),
        param_grid={"C": [0.1, 0.2, 0.3]},
        cv=OneTimeSplitter(n_splits=n_splits, n_samples=n_samples),
        return_train_score=True,
    )
    gs.fit(X, y)

    gs2 = GridSearchCV(
        LinearSVC(random_state=0),
        param_grid={"C": [0.1, 0.2, 0.3]},
        cv=KFold(n_splits=n_splits),
        return_train_score=True,
    )
    gs2.fit(X, y)

    # Give generator as a cv parameter
    assert isinstance(
        KFold(n_splits=n_splits, shuffle=True, random_state=0).split(X, y),
        GeneratorType,
    )
    gs3 = GridSearchCV(
        LinearSVC(random_state=0),
        param_grid={"C": [0.1, 0.2, 0.3]},
        cv=KFold(n_splits=n_splits, shuffle=True, random_state=0).split(X, y),
        return_train_score=True,
    )
    gs3.fit(X, y)

    gs4 = GridSearchCV(
        LinearSVC(random_state=0),
        param_grid={"C": [0.1, 0.2, 0.3]},
        cv=KFold(n_splits=n_splits, shuffle=True, random_state=0),
        return_train_score=True,
    )
    gs4.fit(X, y)

    def _pop_time_keys(cv_results):
        for key in (
            "mean_fit_time",
            "std_fit_time",
            "mean_score_time",
            "std_score_time",
        ):
            cv_results.pop(key)
        return cv_results

    # Check if generators are supported as cv and
    # that the splits are consistent
    np.testing.assert_equal(
        _pop_time_keys(gs3.cv_results_), _pop_time_keys(gs4.cv_results_)
    )

    # OneTimeSplitter is a non-re-entrant cv where split can be called only
    # once if ``cv.split`` is called once per param setting in GridSearchCV.fit
    # the 2nd and 3rd parameter will not be evaluated as no train/test indices
    # will be generated for the 2nd and subsequent cv.split calls.
    # This is a check to make sure cv.split is not called once per param
    # setting.
    np.testing.assert_equal(
        {k: v for k, v in gs.cv_results_.items() if not k.endswith("_time")},
        {k: v for k, v in gs2.cv_results_.items() if not k.endswith("_time")},
    )

    # Check consistency of folds across the parameters
    gs = GridSearchCV(
        LinearSVC(random_state=0),
        param_grid={"C": [0.1, 0.1, 0.2, 0.2]},
        cv=KFold(n_splits=n_splits, shuffle=True),
        return_train_score=True,
    )
    gs.fit(X, y)

    # As the first two param settings (C=0.1) and the next two param
    # settings (C=0.2) are same, the test and train scores must also be
    # same as long as the same train/test indices are generated for all
    # the cv splits, for both param setting
    for score_type in ("train", "test"):
        per_param_scores = {}
        for param_i in range(4):
            per_param_scores[param_i] = list(
                gs.cv_results_["split%d_%s_score" % (s, score_type)][param_i]
                for s in range(5)
            )

        assert_array_almost_equal(per_param_scores[0], per_param_scores[1])
        assert_array_almost_equal(per_param_scores[2], per_param_scores[3])


def test_transform_inverse_transform_round_trip():
    clf = MockClassifier()
    grid_search = GridSearchCV(clf, {"foo_param": [1, 2, 3]}, cv=3, verbose=3)

    grid_search.fit(X, y)
    X_round_trip = grid_search.inverse_transform(grid_search.transform(X))
    assert_array_equal(X, X_round_trip)


def test_custom_run_search():
    def check_results(results, gscv):
        exp_results = gscv.cv_results_
        assert sorted(results.keys()) == sorted(exp_results)
        for k in results:
            if not k.endswith("_time"):
                # XXX: results['params'] is a list :|
                results[k] = np.asanyarray(results[k])
                if results[k].dtype.kind == "O":
                    assert_array_equal(
                        exp_results[k], results[k], err_msg="Checking " + k
                    )
                else:
                    assert_allclose(exp_results[k], results[k], err_msg="Checking " + k)

    def fit_grid(param_grid):
        return GridSearchCV(clf, param_grid, return_train_score=True).fit(X, y)

    class CustomSearchCV(BaseSearchCV):
        def __init__(self, estimator, **kwargs):
            super().__init__(estimator, **kwargs)

        def _run_search(self, evaluate):
            results = evaluate([{"max_depth": 1}, {"max_depth": 2}])
            check_results(results, fit_grid({"max_depth": [1, 2]}))
            results = evaluate([{"min_samples_split": 5}, {"min_samples_split": 10}])
            check_results(
                results,
                fit_grid([{"max_depth": [1, 2]}, {"min_samples_split": [5, 10]}]),
            )

    # Using regressor to make sure each score differs
    clf = DecisionTreeRegressor(random_state=0)
    X, y = make_classification(n_samples=100, n_informative=4, random_state=0)
    mycv = CustomSearchCV(clf, return_train_score=True).fit(X, y)
    gscv = fit_grid([{"max_depth": [1, 2]}, {"min_samples_split": [5, 10]}])

    results = mycv.cv_results_
    check_results(results, gscv)
    for attr in dir(gscv):
        if (
            attr[0].islower()
            and attr[-1:] == "_"
            and attr
            not in {"cv_results_", "best_estimator_", "refit_time_", "classes_"}
        ):
            assert getattr(gscv, attr) == getattr(mycv, attr), (
                "Attribute %s not equal" % attr
            )


def test__custom_fit_no_run_search():
    class NoRunSearchSearchCV(BaseSearchCV):
        def __init__(self, estimator, **kwargs):
            super().__init__(estimator, **kwargs)

        def fit(self, X, y=None, groups=None, **fit_params):
            return self

    # this should not raise any exceptions
    NoRunSearchSearchCV(SVC()).fit(X, y)

    class BadSearchCV(BaseSearchCV):
        def __init__(self, estimator, **kwargs):
            super().__init__(estimator, **kwargs)

    with pytest.raises(NotImplementedError, match="_run_search not implemented."):
        # this should raise a NotImplementedError
        BadSearchCV(SVC()).fit(X, y)


def test_empty_cv_iterator_error():
    # Use global X, y

    # create cv
    cv = KFold(n_splits=3).split(X)

    # pop all of it, this should cause the expected ValueError
    [u for u in cv]
    # cv is empty now

    train_size = 100
    ridge = RandomizedSearchCV(Ridge(), {"alpha": [1e-3, 1e-2, 1e-1]}, cv=cv, n_jobs=4)

    # assert that this raises an error
    with pytest.raises(
        ValueError,
        match=(
            "No fits were performed. "
            "Was the CV iterator empty\\? "
            "Were there no candidates\\?"
        ),
    ):
        ridge.fit(X[:train_size], y[:train_size])


def test_random_search_bad_cv():
    # Use global X, y

    class BrokenKFold(KFold):
        def get_n_splits(self, *args, **kw):
            return 1

    # create bad cv
    cv = BrokenKFold(n_splits=3)

    train_size = 100
    ridge = RandomizedSearchCV(Ridge(), {"alpha": [1e-3, 1e-2, 1e-1]}, cv=cv, n_jobs=4)

    # assert that this raises an error
    with pytest.raises(
        ValueError,
        match=(
            "cv.split and cv.get_n_splits returned "
            "inconsistent results. Expected \\d+ "
            "splits, got \\d+"
        ),
    ):
        ridge.fit(X[:train_size], y[:train_size])


@pytest.mark.parametrize("return_train_score", [False, True])
@pytest.mark.parametrize(
    "SearchCV, specialized_params",
    [
        (GridSearchCV, {"param_grid": {"max_depth": [2, 3]}}),
        (
            RandomizedSearchCV,
            {"param_distributions": {"max_depth": [2, 3]}, "n_iter": 2},
        ),
    ],
)
def test_searchcv_raise_warning_with_non_finite_score(
    SearchCV, specialized_params, return_train_score
):
    # Non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/10529
    # Check that we raise a UserWarning when a non-finite score is
    # computed in the SearchCV
    X, y = make_classification(n_classes=2, random_state=0)

    class FailingScorer:
        """Scorer that will fail for some split but not all."""

        def __init__(self):
            self.n_counts = 0

        def __call__(self, estimator, X, y):
            self.n_counts += 1
            if self.n_counts % 5 == 0:
                return np.nan
            return 1

    grid = SearchCV(
        DecisionTreeClassifier(),
        scoring=FailingScorer(),
        cv=3,
        return_train_score=return_train_score,
        **specialized_params,
    )

    with pytest.warns(UserWarning) as warn_msg:
        grid.fit(X, y)

    set_with_warning = ["test", "train"] if return_train_score else ["test"]
    assert len(warn_msg) == len(set_with_warning)
    for msg, dataset in zip(warn_msg, set_with_warning):
        assert f"One or more of the {dataset} scores are non-finite" in str(msg.message)


def test_callable_multimetric_confusion_matrix():
    # Test callable with many metrics inserts the correct names and metrics
    # into the search cv object
    def custom_scorer(clf, X, y):
        y_pred = clf.predict(X)
        cm = confusion_matrix(y, y_pred)
        return {"tn": cm[0, 0], "fp": cm[0, 1], "fn": cm[1, 0], "tp": cm[1, 1]}

    X, y = make_classification(n_samples=40, n_features=4, random_state=42)
    est = LinearSVC(random_state=42)
    search = GridSearchCV(est, {"C": [0.1, 1]}, scoring=custom_scorer, refit="fp")

    search.fit(X, y)

    score_names = ["tn", "fp", "fn", "tp"]
    for name in score_names:
        assert "mean_test_{}".format(name) in search.cv_results_

    y_pred = search.predict(X)
    cm = confusion_matrix(y, y_pred)
    assert search.score(X, y) == pytest.approx(cm[0, 1])


def test_callable_multimetric_same_as_list_of_strings():
    # Test callable multimetric is the same as a list of strings
    def custom_scorer(est, X, y):
        y_pred = est.predict(X)
        return {
            "recall": recall_score(y, y_pred),
            "accuracy": accuracy_score(y, y_pred),
        }

    X, y = make_classification(n_samples=40, n_features=4, random_state=42)
    est = LinearSVC(random_state=42)
    search_callable = GridSearchCV(
        est, {"C": [0.1, 1]}, scoring=custom_scorer, refit="recall"
    )
    search_str = GridSearchCV(
        est, {"C": [0.1, 1]}, scoring=["recall", "accuracy"], refit="recall"
    )

    search_callable.fit(X, y)
    search_str.fit(X, y)

    assert search_callable.best_score_ == pytest.approx(search_str.best_score_)
    assert search_callable.best_index_ == search_str.best_index_
    assert search_callable.score(X, y) == pytest.approx(search_str.score(X, y))


def test_callable_single_metric_same_as_single_string():
    # Tests callable scorer is the same as scoring with a single string
    def custom_scorer(est, X, y):
        y_pred = est.predict(X)
        return recall_score(y, y_pred)

    X, y = make_classification(n_samples=40, n_features=4, random_state=42)
    est = LinearSVC(random_state=42)
    search_callable = GridSearchCV(
        est, {"C": [0.1, 1]}, scoring=custom_scorer, refit=True
    )
    search_str = GridSearchCV(est, {"C": [0.1, 1]}, scoring="recall", refit="recall")
    search_list_str = GridSearchCV(
        est, {"C": [0.1, 1]}, scoring=["recall"], refit="recall"
    )
    search_callable.fit(X, y)
    search_str.fit(X, y)
    search_list_str.fit(X, y)

    assert search_callable.best_score_ == pytest.approx(search_str.best_score_)
    assert search_callable.best_index_ == search_str.best_index_
    assert search_callable.score(X, y) == pytest.approx(search_str.score(X, y))

    assert search_list_str.best_score_ == pytest.approx(search_str.best_score_)
    assert search_list_str.best_index_ == search_str.best_index_
    assert search_list_str.score(X, y) == pytest.approx(search_str.score(X, y))


def test_callable_multimetric_error_on_invalid_key():
    # Raises when the callable scorer does not return a dict with `refit` key.
    def bad_scorer(est, X, y):
        return {"bad_name": 1}

    X, y = make_classification(n_samples=40, n_features=4, random_state=42)
    clf = GridSearchCV(
        LinearSVC(random_state=42),
        {"C": [0.1, 1]},
        scoring=bad_scorer,
        refit="good_name",
    )

    msg = (
        "For multi-metric scoring, the parameter refit must be set to a "
        "scorer key or a callable to refit"
    )
    with pytest.raises(ValueError, match=msg):
        clf.fit(X, y)


def test_callable_multimetric_error_failing_clf():
    # Warns when there is an estimator the fails to fit with a float
    # error_score
    def custom_scorer(est, X, y):
        return {"acc": 1}

    X, y = make_classification(n_samples=20, n_features=10, random_state=0)

    clf = FailingClassifier()
    gs = GridSearchCV(
        clf,
        [{"parameter": [0, 1, 2]}],
        scoring=custom_scorer,
        refit=False,
        error_score=0.1,
    )

    warning_message = re.compile(
        "5 fits failed.+total of 15.+The score on these"
        r" train-test partitions for these parameters will be set to 0\.1",
        flags=re.DOTALL,
    )
    with pytest.warns(FitFailedWarning, match=warning_message):
        gs.fit(X, y)

    assert_allclose(gs.cv_results_["mean_test_acc"], [1, 1, 0.1])


def test_callable_multimetric_clf_all_fails():
    # Warns and raises when all estimator fails to fit.
    def custom_scorer(est, X, y):
        return {"acc": 1}

    X, y = make_classification(n_samples=20, n_features=10, random_state=0)

    clf = FailingClassifier()

    gs = GridSearchCV(
        clf,
        [{"parameter": [2, 2, 2]}],
        scoring=custom_scorer,
        refit=False,
        error_score=0.1,
    )

    with pytest.warns(
        FitFailedWarning,
        match="15 fits failed.+total of 15",
    ), pytest.raises(NotFittedError, match="All estimators failed to fit"):
        gs.fit(X, y)


def test_n_features_in():
    # make sure grid search and random search delegate n_features_in to the
    # best estimator
    n_features = 4
    X, y = make_classification(n_features=n_features)
    gbdt = HistGradientBoostingClassifier()
    param_grid = {"max_iter": [3, 4]}
    gs = GridSearchCV(gbdt, param_grid)
    rs = RandomizedSearchCV(gbdt, param_grid, n_iter=1)
    assert not hasattr(gs, "n_features_in_")
    assert not hasattr(rs, "n_features_in_")
    gs.fit(X, y)
    rs.fit(X, y)
    assert gs.n_features_in_ == n_features
    assert rs.n_features_in_ == n_features


@pytest.mark.parametrize("pairwise", [True, False])
def test_search_cv_pairwise_property_delegated_to_base_estimator(pairwise):
    """
    Test implementation of BaseSearchCV has the pairwise tag
    which matches the pairwise tag of its estimator.
    This test make sure pairwise tag is delegated to the base estimator.

    Non-regression test for issue #13920.
    """

    class TestEstimator(BaseEstimator):
        def _more_tags(self):
            return {"pairwise": pairwise}

    est = TestEstimator()
    attr_message = "BaseSearchCV pairwise tag must match estimator"
    cv = GridSearchCV(est, {"n_neighbors": [10]})
    assert pairwise == cv._get_tags()["pairwise"], attr_message


# TODO: Remove in 1.1
@ignore_warnings(category=FutureWarning)
def test_search_cv__pairwise_property_delegated_to_base_estimator():
    """
    Test implementation of BaseSearchCV has the _pairwise property
    which matches the _pairwise property of its estimator.
    This test make sure _pairwise is delegated to the base estimator.

    Non-regression test for issue #13920.
    """
    est = BaseEstimator()
    attr_message = "BaseSearchCV _pairwise property must match estimator"

    for _pairwise_setting in [True, False]:
        setattr(est, "_pairwise", _pairwise_setting)
        cv = GridSearchCV(est, {"n_neighbors": [10]})
        assert _pairwise_setting == cv._pairwise, attr_message


def test_search_cv_pairwise_property_equivalence_of_precomputed():
    """
    Test implementation of BaseSearchCV has the pairwise tag
    which matches the pairwise tag of its estimator.
    This test ensures the equivalence of 'precomputed'.

    Non-regression test for issue #13920.
    """
    n_samples = 50
    n_splits = 2
    X, y = make_classification(n_samples=n_samples, random_state=0)
    grid_params = {"n_neighbors": [10]}

    # defaults to euclidean metric (minkowski p = 2)
    clf = KNeighborsClassifier()
    cv = GridSearchCV(clf, grid_params, cv=n_splits)
    cv.fit(X, y)
    preds_original = cv.predict(X)

    # precompute euclidean metric to validate pairwise is working
    X_precomputed = euclidean_distances(X)
    clf = KNeighborsClassifier(metric="precomputed")
    cv = GridSearchCV(clf, grid_params, cv=n_splits)
    cv.fit(X_precomputed, y)
    preds_precomputed = cv.predict(X_precomputed)

    attr_message = "GridSearchCV not identical with precomputed metric"
    assert (preds_original == preds_precomputed).all(), attr_message


@pytest.mark.parametrize(
    "SearchCV, param_search",
    [(GridSearchCV, {"a": [0.1, 0.01]}), (RandomizedSearchCV, {"a": uniform(1, 3)})],
)
def test_scalar_fit_param(SearchCV, param_search):
    # unofficially sanctioned tolerance for scalar values in fit_params
    # non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/15805
    class TestEstimator(ClassifierMixin, BaseEstimator):
        def __init__(self, a=None):
            self.a = a

        def fit(self, X, y, r=None):
            self.r_ = r

        def predict(self, X):
            return np.zeros(shape=(len(X)))

    model = SearchCV(TestEstimator(), param_search)
    X, y = make_classification(random_state=42)
    model.fit(X, y, r=42)
    assert model.best_estimator_.r_ == 42


@pytest.mark.parametrize(
    "SearchCV, param_search",
    [
        (GridSearchCV, {"alpha": [0.1, 0.01]}),
        (RandomizedSearchCV, {"alpha": uniform(0.01, 0.1)}),
    ],
)
def test_scalar_fit_param_compat(SearchCV, param_search):
    # check support for scalar values in fit_params, for instance in LightGBM
    # that do not exactly respect the scikit-learn API contract but that we do
    # not want to break without an explicit deprecation cycle and API
    # recommendations for implementing early stopping with a user provided
    # validation set. non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/15805
    X_train, X_valid, y_train, y_valid = train_test_split(
        *make_classification(random_state=42), random_state=42
    )

    class _FitParamClassifier(SGDClassifier):
        def fit(
            self,
            X,
            y,
            sample_weight=None,
            tuple_of_arrays=None,
            scalar_param=None,
            callable_param=None,
        ):
            super().fit(X, y, sample_weight=sample_weight)
            assert scalar_param > 0
            assert callable(callable_param)

            # The tuple of arrays should be preserved as tuple.
            assert isinstance(tuple_of_arrays, tuple)
            assert tuple_of_arrays[0].ndim == 2
            assert tuple_of_arrays[1].ndim == 1
            return self

    def _fit_param_callable():
        pass

    model = SearchCV(_FitParamClassifier(), param_search)

    # NOTE: `fit_params` should be data dependent (e.g. `sample_weight`) which
    # is not the case for the following parameters. But this abuse is common in
    # popular third-party libraries and we should tolerate this behavior for
    # now and be careful not to break support for those without following
    # proper deprecation cycle.
    fit_params = {
        "tuple_of_arrays": (X_valid, y_valid),
        "callable_param": _fit_param_callable,
        "scalar_param": 42,
    }
    model.fit(X_train, y_train, **fit_params)


# FIXME: Replace this test with a full `check_estimator` once we have API only
# checks.
@pytest.mark.filterwarnings("ignore:The total space of parameters 4 is")
@pytest.mark.parametrize("SearchCV", [GridSearchCV, RandomizedSearchCV])
@pytest.mark.parametrize("Predictor", [MinimalRegressor, MinimalClassifier])
def test_search_cv_using_minimal_compatible_estimator(SearchCV, Predictor):
    # Check that third-party library can run tests without inheriting from
    # BaseEstimator.
    rng = np.random.RandomState(0)
    X, y = rng.randn(25, 2), np.array([0] * 5 + [1] * 20)

    model = Pipeline(
        [("transformer", MinimalTransformer()), ("predictor", Predictor())]
    )

    params = {
        "transformer__param": [1, 10],
        "predictor__parama": [1, 10],
    }
    search = SearchCV(model, params, error_score="raise")
    search.fit(X, y)

    assert search.best_params_.keys() == params.keys()

    y_pred = search.predict(X)
    if is_classifier(search):
        assert_array_equal(y_pred, 1)
        assert search.score(X, y) == pytest.approx(accuracy_score(y, y_pred))
    else:
        assert_allclose(y_pred, y.mean())
        assert search.score(X, y) == pytest.approx(r2_score(y, y_pred))


@pytest.mark.parametrize("return_train_score", [True, False])
def test_search_cv_verbose_3(capsys, return_train_score):
    """Check that search cv with verbose>2 shows the score for single
    metrics. non-regression test for #19658."""
    X, y = make_classification(n_samples=100, n_classes=2, flip_y=0.2, random_state=0)
    clf = LinearSVC(random_state=0)
    grid = {"C": [0.1]}

    GridSearchCV(
        clf,
        grid,
        scoring="accuracy",
        verbose=3,
        cv=3,
        return_train_score=return_train_score,
    ).fit(X, y)
    captured = capsys.readouterr().out
    if return_train_score:
        match = re.findall(r"score=\(train=[\d\.]+, test=[\d.]+\)", captured)
    else:
        match = re.findall(r"score=[\d\.]+", captured)
    assert len(match) == 3
