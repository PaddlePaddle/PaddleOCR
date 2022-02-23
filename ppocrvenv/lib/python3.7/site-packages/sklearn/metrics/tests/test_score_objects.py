from copy import deepcopy
import pickle
import tempfile
import shutil
import os
import numbers
from unittest.mock import Mock
from functools import partial

import numpy as np
import pytest
import joblib

from numpy.testing import assert_allclose
from sklearn.utils._testing import assert_almost_equal
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import ignore_warnings

from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    jaccard_score,
    log_loss,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.metrics import cluster as cluster_module
from sklearn.metrics import check_scoring
from sklearn.metrics._scorer import (
    _PredictScorer,
    _passthrough_scorer,
    _MultimetricScorer,
    _check_multimetric_scoring,
)
from sklearn.metrics import make_scorer, get_scorer, SCORERS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge, LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import make_blobs
from sklearn.datasets import make_classification, make_regression
from sklearn.datasets import make_multilabel_classification
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier


REGRESSION_SCORERS = [
    "explained_variance",
    "r2",
    "neg_mean_absolute_error",
    "neg_mean_squared_error",
    "neg_mean_absolute_percentage_error",
    "neg_mean_squared_log_error",
    "neg_median_absolute_error",
    "neg_root_mean_squared_error",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "mean_squared_error",
    "median_absolute_error",
    "max_error",
    "neg_mean_poisson_deviance",
    "neg_mean_gamma_deviance",
]

CLF_SCORERS = [
    "accuracy",
    "balanced_accuracy",
    "top_k_accuracy",
    "f1",
    "f1_weighted",
    "f1_macro",
    "f1_micro",
    "roc_auc",
    "average_precision",
    "precision",
    "precision_weighted",
    "precision_macro",
    "precision_micro",
    "recall",
    "recall_weighted",
    "recall_macro",
    "recall_micro",
    "neg_log_loss",
    "neg_brier_score",
    "jaccard",
    "jaccard_weighted",
    "jaccard_macro",
    "jaccard_micro",
    "roc_auc_ovr",
    "roc_auc_ovo",
    "roc_auc_ovr_weighted",
    "roc_auc_ovo_weighted",
]

# All supervised cluster scorers (They behave like classification metric)
CLUSTER_SCORERS = [
    "adjusted_rand_score",
    "rand_score",
    "homogeneity_score",
    "completeness_score",
    "v_measure_score",
    "mutual_info_score",
    "adjusted_mutual_info_score",
    "normalized_mutual_info_score",
    "fowlkes_mallows_score",
]

MULTILABEL_ONLY_SCORERS = [
    "precision_samples",
    "recall_samples",
    "f1_samples",
    "jaccard_samples",
]

REQUIRE_POSITIVE_Y_SCORERS = ["neg_mean_poisson_deviance", "neg_mean_gamma_deviance"]


def _require_positive_y(y):
    """Make targets strictly positive"""
    offset = abs(y.min()) + 1
    y = y + offset
    return y


def _make_estimators(X_train, y_train, y_ml_train):
    # Make estimators that make sense to test various scoring methods
    sensible_regr = DecisionTreeRegressor(random_state=0)
    # some of the regressions scorers require strictly positive input.
    sensible_regr.fit(X_train, _require_positive_y(y_train))
    sensible_clf = DecisionTreeClassifier(random_state=0)
    sensible_clf.fit(X_train, y_train)
    sensible_ml_clf = DecisionTreeClassifier(random_state=0)
    sensible_ml_clf.fit(X_train, y_ml_train)
    return dict(
        [(name, sensible_regr) for name in REGRESSION_SCORERS]
        + [(name, sensible_clf) for name in CLF_SCORERS]
        + [(name, sensible_clf) for name in CLUSTER_SCORERS]
        + [(name, sensible_ml_clf) for name in MULTILABEL_ONLY_SCORERS]
    )


X_mm, y_mm, y_ml_mm = None, None, None
ESTIMATORS = None
TEMP_FOLDER = None


def setup_module():
    # Create some memory mapped data
    global X_mm, y_mm, y_ml_mm, TEMP_FOLDER, ESTIMATORS
    TEMP_FOLDER = tempfile.mkdtemp(prefix="sklearn_test_score_objects_")
    X, y = make_classification(n_samples=30, n_features=5, random_state=0)
    _, y_ml = make_multilabel_classification(n_samples=X.shape[0], random_state=0)
    filename = os.path.join(TEMP_FOLDER, "test_data.pkl")
    joblib.dump((X, y, y_ml), filename)
    X_mm, y_mm, y_ml_mm = joblib.load(filename, mmap_mode="r")
    ESTIMATORS = _make_estimators(X_mm, y_mm, y_ml_mm)


def teardown_module():
    global X_mm, y_mm, y_ml_mm, TEMP_FOLDER, ESTIMATORS
    # GC closes the mmap file descriptors
    X_mm, y_mm, y_ml_mm, ESTIMATORS = None, None, None, None
    shutil.rmtree(TEMP_FOLDER)


class EstimatorWithoutFit:
    """Dummy estimator to test scoring validators"""

    pass


class EstimatorWithFit(BaseEstimator):
    """Dummy estimator to test scoring validators"""

    def fit(self, X, y):
        return self


class EstimatorWithFitAndScore:
    """Dummy estimator to test scoring validators"""

    def fit(self, X, y):
        return self

    def score(self, X, y):
        return 1.0


class EstimatorWithFitAndPredict:
    """Dummy estimator to test scoring validators"""

    def fit(self, X, y):
        self.y = y
        return self

    def predict(self, X):
        return self.y


class DummyScorer:
    """Dummy scorer that always returns 1."""

    def __call__(self, est, X, y):
        return 1


def test_all_scorers_repr():
    # Test that all scorers have a working repr
    for name, scorer in SCORERS.items():
        repr(scorer)


def check_scoring_validator_for_single_metric_usecases(scoring_validator):
    # Test all branches of single metric usecases
    estimator = EstimatorWithoutFit()
    pattern = (
        r"estimator should be an estimator implementing 'fit' method," r" .* was passed"
    )
    with pytest.raises(TypeError, match=pattern):
        scoring_validator(estimator)

    estimator = EstimatorWithFitAndScore()
    estimator.fit([[1]], [1])
    scorer = scoring_validator(estimator)
    assert scorer is _passthrough_scorer
    assert_almost_equal(scorer(estimator, [[1]], [1]), 1.0)

    estimator = EstimatorWithFitAndPredict()
    estimator.fit([[1]], [1])
    pattern = (
        r"If no scoring is specified, the estimator passed should have"
        r" a 'score' method\. The estimator .* does not\."
    )
    with pytest.raises(TypeError, match=pattern):
        scoring_validator(estimator)

    scorer = scoring_validator(estimator, scoring="accuracy")
    assert_almost_equal(scorer(estimator, [[1]], [1]), 1.0)

    estimator = EstimatorWithFit()
    scorer = scoring_validator(estimator, scoring="accuracy")
    assert isinstance(scorer, _PredictScorer)

    # Test the allow_none parameter for check_scoring alone
    if scoring_validator is check_scoring:
        estimator = EstimatorWithFit()
        scorer = scoring_validator(estimator, allow_none=True)
        assert scorer is None


@pytest.mark.parametrize(
    "scoring",
    (
        ("accuracy",),
        ["precision"],
        {"acc": "accuracy", "precision": "precision"},
        ("accuracy", "precision"),
        ["precision", "accuracy"],
        {
            "accuracy": make_scorer(accuracy_score),
            "precision": make_scorer(precision_score),
        },
    ),
    ids=[
        "single_tuple",
        "single_list",
        "dict_str",
        "multi_tuple",
        "multi_list",
        "dict_callable",
    ],
)
def test_check_scoring_and_check_multimetric_scoring(scoring):
    check_scoring_validator_for_single_metric_usecases(check_scoring)
    # To make sure the check_scoring is correctly applied to the constituent
    # scorers

    estimator = LinearSVC(random_state=0)
    estimator.fit([[1], [2], [3]], [1, 1, 0])

    scorers = _check_multimetric_scoring(estimator, scoring)
    assert isinstance(scorers, dict)
    assert sorted(scorers.keys()) == sorted(list(scoring))
    assert all(
        [isinstance(scorer, _PredictScorer) for scorer in list(scorers.values())]
    )

    if "acc" in scoring:
        assert_almost_equal(
            scorers["acc"](estimator, [[1], [2], [3]], [1, 0, 0]), 2.0 / 3.0
        )
    if "accuracy" in scoring:
        assert_almost_equal(
            scorers["accuracy"](estimator, [[1], [2], [3]], [1, 0, 0]), 2.0 / 3.0
        )
    if "precision" in scoring:
        assert_almost_equal(
            scorers["precision"](estimator, [[1], [2], [3]], [1, 0, 0]), 0.5
        )


@pytest.mark.parametrize(
    "scoring, msg",
    [
        (
            (make_scorer(precision_score), make_scorer(accuracy_score)),
            "One or more of the elements were callables",
        ),
        ([5], "Non-string types were found"),
        ((make_scorer(precision_score),), "One or more of the elements were callables"),
        ((), "Empty list was given"),
        (("f1", "f1"), "Duplicate elements were found"),
        ({4: "accuracy"}, "Non-string types were found in the keys"),
        ({}, "An empty dict was passed"),
    ],
    ids=[
        "tuple of callables",
        "list of int",
        "tuple of one callable",
        "empty tuple",
        "non-unique str",
        "non-string key dict",
        "empty dict",
    ],
)
def test_check_scoring_and_check_multimetric_scoring_errors(scoring, msg):
    # Make sure it raises errors when scoring parameter is not valid.
    # More weird corner cases are tested at test_validation.py
    estimator = EstimatorWithFitAndPredict()
    estimator.fit([[1]], [1])

    with pytest.raises(ValueError, match=msg):
        _check_multimetric_scoring(estimator, scoring=scoring)


def test_check_scoring_gridsearchcv():
    # test that check_scoring works on GridSearchCV and pipeline.
    # slightly redundant non-regression test.

    grid = GridSearchCV(LinearSVC(), param_grid={"C": [0.1, 1]}, cv=3)
    scorer = check_scoring(grid, scoring="f1")
    assert isinstance(scorer, _PredictScorer)

    pipe = make_pipeline(LinearSVC())
    scorer = check_scoring(pipe, scoring="f1")
    assert isinstance(scorer, _PredictScorer)

    # check that cross_val_score definitely calls the scorer
    # and doesn't make any assumptions about the estimator apart from having a
    # fit.
    scores = cross_val_score(
        EstimatorWithFit(), [[1], [2], [3]], [1, 0, 1], scoring=DummyScorer(), cv=3
    )
    assert_array_equal(scores, 1)


def test_make_scorer():
    # Sanity check on the make_scorer factory function.
    f = lambda *args: 0
    with pytest.raises(ValueError):
        make_scorer(f, needs_threshold=True, needs_proba=True)


@pytest.mark.parametrize(
    "scorer_name, metric",
    [
        ("f1", f1_score),
        ("f1_weighted", partial(f1_score, average="weighted")),
        ("f1_macro", partial(f1_score, average="macro")),
        ("f1_micro", partial(f1_score, average="micro")),
        ("precision", precision_score),
        ("precision_weighted", partial(precision_score, average="weighted")),
        ("precision_macro", partial(precision_score, average="macro")),
        ("precision_micro", partial(precision_score, average="micro")),
        ("recall", recall_score),
        ("recall_weighted", partial(recall_score, average="weighted")),
        ("recall_macro", partial(recall_score, average="macro")),
        ("recall_micro", partial(recall_score, average="micro")),
        ("jaccard", jaccard_score),
        ("jaccard_weighted", partial(jaccard_score, average="weighted")),
        ("jaccard_macro", partial(jaccard_score, average="macro")),
        ("jaccard_micro", partial(jaccard_score, average="micro")),
        ("top_k_accuracy", top_k_accuracy_score),
    ],
)
def test_classification_binary_scores(scorer_name, metric):
    # check consistency between score and scorer for scores supporting
    # binary classification.
    X, y = make_blobs(random_state=0, centers=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = LinearSVC(random_state=0)
    clf.fit(X_train, y_train)

    score = SCORERS[scorer_name](clf, X_test, y_test)
    expected_score = metric(y_test, clf.predict(X_test))
    assert_almost_equal(score, expected_score)


@pytest.mark.parametrize(
    "scorer_name, metric",
    [
        ("accuracy", accuracy_score),
        ("balanced_accuracy", balanced_accuracy_score),
        ("f1_weighted", partial(f1_score, average="weighted")),
        ("f1_macro", partial(f1_score, average="macro")),
        ("f1_micro", partial(f1_score, average="micro")),
        ("precision_weighted", partial(precision_score, average="weighted")),
        ("precision_macro", partial(precision_score, average="macro")),
        ("precision_micro", partial(precision_score, average="micro")),
        ("recall_weighted", partial(recall_score, average="weighted")),
        ("recall_macro", partial(recall_score, average="macro")),
        ("recall_micro", partial(recall_score, average="micro")),
        ("jaccard_weighted", partial(jaccard_score, average="weighted")),
        ("jaccard_macro", partial(jaccard_score, average="macro")),
        ("jaccard_micro", partial(jaccard_score, average="micro")),
    ],
)
def test_classification_multiclass_scores(scorer_name, metric):
    # check consistency between score and scorer for scores supporting
    # multiclass classification.
    X, y = make_classification(
        n_classes=3, n_informative=3, n_samples=30, random_state=0
    )

    # use `stratify` = y to ensure train and test sets capture all classes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0, stratify=y
    )

    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    score = SCORERS[scorer_name](clf, X_test, y_test)
    expected_score = metric(y_test, clf.predict(X_test))
    assert score == pytest.approx(expected_score)


def test_custom_scorer_pickling():
    # test that custom scorer can be pickled
    X, y = make_blobs(random_state=0, centers=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = LinearSVC(random_state=0)
    clf.fit(X_train, y_train)

    scorer = make_scorer(fbeta_score, beta=2)
    score1 = scorer(clf, X_test, y_test)
    unpickled_scorer = pickle.loads(pickle.dumps(scorer))
    score2 = unpickled_scorer(clf, X_test, y_test)
    assert score1 == pytest.approx(score2)

    # smoke test the repr:
    repr(fbeta_score)


def test_regression_scorers():
    # Test regression scorers.
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = Ridge()
    clf.fit(X_train, y_train)
    score1 = get_scorer("r2")(clf, X_test, y_test)
    score2 = r2_score(y_test, clf.predict(X_test))
    assert_almost_equal(score1, score2)


def test_thresholded_scorers():
    # Test scorers that take thresholds.
    X, y = make_blobs(random_state=0, centers=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = LogisticRegression(random_state=0)
    clf.fit(X_train, y_train)
    score1 = get_scorer("roc_auc")(clf, X_test, y_test)
    score2 = roc_auc_score(y_test, clf.decision_function(X_test))
    score3 = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    assert_almost_equal(score1, score2)
    assert_almost_equal(score1, score3)

    logscore = get_scorer("neg_log_loss")(clf, X_test, y_test)
    logloss = log_loss(y_test, clf.predict_proba(X_test))
    assert_almost_equal(-logscore, logloss)

    # same for an estimator without decision_function
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    score1 = get_scorer("roc_auc")(clf, X_test, y_test)
    score2 = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    assert_almost_equal(score1, score2)

    # test with a regressor (no decision_function)
    reg = DecisionTreeRegressor()
    reg.fit(X_train, y_train)
    score1 = get_scorer("roc_auc")(reg, X_test, y_test)
    score2 = roc_auc_score(y_test, reg.predict(X_test))
    assert_almost_equal(score1, score2)

    # Test that an exception is raised on more than two classes
    X, y = make_blobs(random_state=0, centers=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf.fit(X_train, y_train)
    with pytest.raises(ValueError, match="multiclass format is not supported"):
        get_scorer("roc_auc")(clf, X_test, y_test)

    # test error is raised with a single class present in model
    # (predict_proba shape is not suitable for binary auc)
    X, y = make_blobs(random_state=0, centers=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, np.zeros_like(y_train))
    with pytest.raises(ValueError, match="need classifier with two classes"):
        get_scorer("roc_auc")(clf, X_test, y_test)

    # for proba scorers
    with pytest.raises(ValueError, match="need classifier with two classes"):
        get_scorer("neg_log_loss")(clf, X_test, y_test)


def test_thresholded_scorers_multilabel_indicator_data():
    # Test that the scorer work with multilabel-indicator format
    # for multilabel and multi-output multi-class classifier
    X, y = make_multilabel_classification(allow_unlabeled=False, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # Multi-output multi-class predict_proba
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)
    score1 = get_scorer("roc_auc")(clf, X_test, y_test)
    score2 = roc_auc_score(y_test, np.vstack([p[:, -1] for p in y_proba]).T)
    assert_almost_equal(score1, score2)

    # Multi-output multi-class decision_function
    # TODO Is there any yet?
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    clf._predict_proba = clf.predict_proba
    clf.predict_proba = None
    clf.decision_function = lambda X: [p[:, 1] for p in clf._predict_proba(X)]

    y_proba = clf.decision_function(X_test)
    score1 = get_scorer("roc_auc")(clf, X_test, y_test)
    score2 = roc_auc_score(y_test, np.vstack([p for p in y_proba]).T)
    assert_almost_equal(score1, score2)

    # Multilabel predict_proba
    clf = OneVsRestClassifier(DecisionTreeClassifier())
    clf.fit(X_train, y_train)
    score1 = get_scorer("roc_auc")(clf, X_test, y_test)
    score2 = roc_auc_score(y_test, clf.predict_proba(X_test))
    assert_almost_equal(score1, score2)

    # Multilabel decision function
    clf = OneVsRestClassifier(LinearSVC(random_state=0))
    clf.fit(X_train, y_train)
    score1 = get_scorer("roc_auc")(clf, X_test, y_test)
    score2 = roc_auc_score(y_test, clf.decision_function(X_test))
    assert_almost_equal(score1, score2)


def test_supervised_cluster_scorers():
    # Test clustering scorers against gold standard labeling.
    X, y = make_blobs(random_state=0, centers=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    km = KMeans(n_clusters=3)
    km.fit(X_train)
    for name in CLUSTER_SCORERS:
        score1 = get_scorer(name)(km, X_test, y_test)
        score2 = getattr(cluster_module, name)(y_test, km.predict(X_test))
        assert_almost_equal(score1, score2)


@ignore_warnings
def test_raises_on_score_list():
    # Test that when a list of scores is returned, we raise proper errors.
    X, y = make_blobs(random_state=0)
    f1_scorer_no_average = make_scorer(f1_score, average=None)
    clf = DecisionTreeClassifier()
    with pytest.raises(ValueError):
        cross_val_score(clf, X, y, scoring=f1_scorer_no_average)
    grid_search = GridSearchCV(
        clf, scoring=f1_scorer_no_average, param_grid={"max_depth": [1, 2]}
    )
    with pytest.raises(ValueError):
        grid_search.fit(X, y)


@ignore_warnings
def test_classification_scorer_sample_weight():
    # Test that classification scorers support sample_weight or raise sensible
    # errors

    # Unlike the metrics invariance test, in the scorer case it's harder
    # to ensure that, on the classifier output, weighted and unweighted
    # scores really should be unequal.
    X, y = make_classification(random_state=0)
    _, y_ml = make_multilabel_classification(n_samples=X.shape[0], random_state=0)
    split = train_test_split(X, y, y_ml, random_state=0)
    X_train, X_test, y_train, y_test, y_ml_train, y_ml_test = split

    sample_weight = np.ones_like(y_test)
    sample_weight[:10] = 0

    # get sensible estimators for each metric
    estimator = _make_estimators(X_train, y_train, y_ml_train)

    for name, scorer in SCORERS.items():
        if name in REGRESSION_SCORERS:
            # skip the regression scores
            continue
        if name == "top_k_accuracy":
            # in the binary case k > 1 will always lead to a perfect score
            scorer._kwargs = {"k": 1}
        if name in MULTILABEL_ONLY_SCORERS:
            target = y_ml_test
        else:
            target = y_test
        try:
            weighted = scorer(
                estimator[name], X_test, target, sample_weight=sample_weight
            )
            ignored = scorer(estimator[name], X_test[10:], target[10:])
            unweighted = scorer(estimator[name], X_test, target)
            assert weighted != unweighted, (
                f"scorer {name} behaves identically when called with "
                f"sample weights: {weighted} vs {unweighted}"
            )
            assert_almost_equal(
                weighted,
                ignored,
                err_msg=(
                    f"scorer {name} behaves differently "
                    "when ignoring samples and setting "
                    f"sample_weight to 0: {weighted} vs {ignored}"
                ),
            )

        except TypeError as e:
            assert "sample_weight" in str(e), (
                f"scorer {name} raises unhelpful exception when called "
                f"with sample weights: {str(e)}"
            )


@ignore_warnings
def test_regression_scorer_sample_weight():
    # Test that regression scorers support sample_weight or raise sensible
    # errors

    # Odd number of test samples req for neg_median_absolute_error
    X, y = make_regression(n_samples=101, n_features=20, random_state=0)
    y = _require_positive_y(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    sample_weight = np.ones_like(y_test)
    # Odd number req for neg_median_absolute_error
    sample_weight[:11] = 0

    reg = DecisionTreeRegressor(random_state=0)
    reg.fit(X_train, y_train)

    for name, scorer in SCORERS.items():
        if name not in REGRESSION_SCORERS:
            # skip classification scorers
            continue
        try:
            weighted = scorer(reg, X_test, y_test, sample_weight=sample_weight)
            ignored = scorer(reg, X_test[11:], y_test[11:])
            unweighted = scorer(reg, X_test, y_test)
            assert weighted != unweighted, (
                f"scorer {name} behaves identically when called with "
                f"sample weights: {weighted} vs {unweighted}"
            )
            assert_almost_equal(
                weighted,
                ignored,
                err_msg=(
                    f"scorer {name} behaves differently "
                    "when ignoring samples and setting "
                    f"sample_weight to 0: {weighted} vs {ignored}"
                ),
            )

        except TypeError as e:
            assert "sample_weight" in str(e), (
                f"scorer {name} raises unhelpful exception when called "
                f"with sample weights: {str(e)}"
            )


@pytest.mark.parametrize("name", SCORERS)
def test_scorer_memmap_input(name):
    # Non-regression test for #6147: some score functions would
    # return singleton memmap when computed on memmap data instead of scalar
    # float values.

    if name in REQUIRE_POSITIVE_Y_SCORERS:
        y_mm_1 = _require_positive_y(y_mm)
        y_ml_mm_1 = _require_positive_y(y_ml_mm)
    else:
        y_mm_1, y_ml_mm_1 = y_mm, y_ml_mm

    # UndefinedMetricWarning for P / R scores
    with ignore_warnings():
        scorer, estimator = SCORERS[name], ESTIMATORS[name]
        if name in MULTILABEL_ONLY_SCORERS:
            score = scorer(estimator, X_mm, y_ml_mm_1)
        else:
            score = scorer(estimator, X_mm, y_mm_1)
        assert isinstance(score, numbers.Number), name


def test_scoring_is_not_metric():
    with pytest.raises(ValueError, match="make_scorer"):
        check_scoring(LogisticRegression(), scoring=f1_score)
    with pytest.raises(ValueError, match="make_scorer"):
        check_scoring(LogisticRegression(), scoring=roc_auc_score)
    with pytest.raises(ValueError, match="make_scorer"):
        check_scoring(Ridge(), scoring=r2_score)
    with pytest.raises(ValueError, match="make_scorer"):
        check_scoring(KMeans(), scoring=cluster_module.adjusted_rand_score)
    with pytest.raises(ValueError, match="make_scorer"):
        check_scoring(KMeans(), scoring=cluster_module.rand_score)


@pytest.mark.parametrize(
    "scorers,expected_predict_count,"
    "expected_predict_proba_count,expected_decision_func_count",
    [
        (
            {
                "a1": "accuracy",
                "a2": "accuracy",
                "ll1": "neg_log_loss",
                "ll2": "neg_log_loss",
                "ra1": "roc_auc",
                "ra2": "roc_auc",
            },
            1,
            1,
            1,
        ),
        (["roc_auc", "accuracy"], 1, 0, 1),
        (["neg_log_loss", "accuracy"], 1, 1, 0),
    ],
)
def test_multimetric_scorer_calls_method_once(
    scorers,
    expected_predict_count,
    expected_predict_proba_count,
    expected_decision_func_count,
):
    X, y = np.array([[1], [1], [0], [0], [0]]), np.array([0, 1, 1, 1, 0])

    mock_est = Mock()
    fit_func = Mock(return_value=mock_est)
    predict_func = Mock(return_value=y)

    pos_proba = np.random.rand(X.shape[0])
    proba = np.c_[1 - pos_proba, pos_proba]
    predict_proba_func = Mock(return_value=proba)
    decision_function_func = Mock(return_value=pos_proba)

    mock_est.fit = fit_func
    mock_est.predict = predict_func
    mock_est.predict_proba = predict_proba_func
    mock_est.decision_function = decision_function_func
    # add the classes that would be found during fit
    mock_est.classes_ = np.array([0, 1])

    scorer_dict = _check_multimetric_scoring(LogisticRegression(), scorers)
    multi_scorer = _MultimetricScorer(**scorer_dict)
    results = multi_scorer(mock_est, X, y)

    assert set(scorers) == set(results)  # compare dict keys

    assert predict_func.call_count == expected_predict_count
    assert predict_proba_func.call_count == expected_predict_proba_count
    assert decision_function_func.call_count == expected_decision_func_count


def test_multimetric_scorer_calls_method_once_classifier_no_decision():
    predict_proba_call_cnt = 0

    class MockKNeighborsClassifier(KNeighborsClassifier):
        def predict_proba(self, X):
            nonlocal predict_proba_call_cnt
            predict_proba_call_cnt += 1
            return super().predict_proba(X)

    X, y = np.array([[1], [1], [0], [0], [0]]), np.array([0, 1, 1, 1, 0])

    # no decision function
    clf = MockKNeighborsClassifier(n_neighbors=1)
    clf.fit(X, y)

    scorers = ["roc_auc", "neg_log_loss"]
    scorer_dict = _check_multimetric_scoring(clf, scorers)
    scorer = _MultimetricScorer(**scorer_dict)
    scorer(clf, X, y)

    assert predict_proba_call_cnt == 1


def test_multimetric_scorer_calls_method_once_regressor_threshold():
    predict_called_cnt = 0

    class MockDecisionTreeRegressor(DecisionTreeRegressor):
        def predict(self, X):
            nonlocal predict_called_cnt
            predict_called_cnt += 1
            return super().predict(X)

    X, y = np.array([[1], [1], [0], [0], [0]]), np.array([0, 1, 1, 1, 0])

    # no decision function
    clf = MockDecisionTreeRegressor()
    clf.fit(X, y)

    scorers = {"neg_mse": "neg_mean_squared_error", "r2": "roc_auc"}
    scorer_dict = _check_multimetric_scoring(clf, scorers)
    scorer = _MultimetricScorer(**scorer_dict)
    scorer(clf, X, y)

    assert predict_called_cnt == 1


def test_multimetric_scorer_sanity_check():
    # scoring dictionary returned is the same as calling each scorer separately
    scorers = {
        "a1": "accuracy",
        "a2": "accuracy",
        "ll1": "neg_log_loss",
        "ll2": "neg_log_loss",
        "ra1": "roc_auc",
        "ra2": "roc_auc",
    }

    X, y = make_classification(random_state=0)

    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    scorer_dict = _check_multimetric_scoring(clf, scorers)
    multi_scorer = _MultimetricScorer(**scorer_dict)

    result = multi_scorer(clf, X, y)

    separate_scores = {
        name: get_scorer(name)(clf, X, y)
        for name in ["accuracy", "neg_log_loss", "roc_auc"]
    }

    for key, value in result.items():
        score_name = scorers[key]
        assert_allclose(value, separate_scores[score_name])


@pytest.mark.parametrize(
    "scorer_name, metric",
    [
        ("roc_auc_ovr", partial(roc_auc_score, multi_class="ovr")),
        ("roc_auc_ovo", partial(roc_auc_score, multi_class="ovo")),
        (
            "roc_auc_ovr_weighted",
            partial(roc_auc_score, multi_class="ovr", average="weighted"),
        ),
        (
            "roc_auc_ovo_weighted",
            partial(roc_auc_score, multi_class="ovo", average="weighted"),
        ),
    ],
)
def test_multiclass_roc_proba_scorer(scorer_name, metric):
    scorer = get_scorer(scorer_name)
    X, y = make_classification(
        n_classes=3, n_informative=3, n_samples=20, random_state=0
    )
    lr = LogisticRegression(multi_class="multinomial").fit(X, y)
    y_proba = lr.predict_proba(X)
    expected_score = metric(y, y_proba)

    assert scorer(lr, X, y) == pytest.approx(expected_score)


def test_multiclass_roc_proba_scorer_label():
    scorer = make_scorer(
        roc_auc_score, multi_class="ovo", labels=[0, 1, 2], needs_proba=True
    )
    X, y = make_classification(
        n_classes=3, n_informative=3, n_samples=20, random_state=0
    )
    lr = LogisticRegression(multi_class="multinomial").fit(X, y)
    y_proba = lr.predict_proba(X)

    y_binary = y == 0
    expected_score = roc_auc_score(
        y_binary, y_proba, multi_class="ovo", labels=[0, 1, 2]
    )

    assert scorer(lr, X, y_binary) == pytest.approx(expected_score)


@pytest.mark.parametrize(
    "scorer_name",
    ["roc_auc_ovr", "roc_auc_ovo", "roc_auc_ovr_weighted", "roc_auc_ovo_weighted"],
)
def test_multiclass_roc_no_proba_scorer_errors(scorer_name):
    # Perceptron has no predict_proba
    scorer = get_scorer(scorer_name)
    X, y = make_classification(
        n_classes=3, n_informative=3, n_samples=20, random_state=0
    )
    lr = Perceptron().fit(X, y)
    msg = "'Perceptron' object has no attribute 'predict_proba'"
    with pytest.raises(AttributeError, match=msg):
        scorer(lr, X, y)


@pytest.fixture
def string_labeled_classification_problem():
    """Train a classifier on binary problem with string target.

    The classifier is trained on a binary classification problem where the
    minority class of interest has a string label that is intentionally not the
    greatest class label using the lexicographic order. In this case, "cancer"
    is the positive label, and `classifier.classes_` is
    `["cancer", "not cancer"]`.

    In addition, the dataset is imbalanced to better identify problems when
    using non-symmetric performance metrics such as f1-score, average precision
    and so on.

    Returns
    -------
    classifier : estimator object
        Trained classifier on the binary problem.
    X_test : ndarray of shape (n_samples, n_features)
        Data to be used as testing set in tests.
    y_test : ndarray of shape (n_samples,), dtype=object
        Binary target where labels are strings.
    y_pred : ndarray of shape (n_samples,), dtype=object
        Prediction of `classifier` when predicting for `X_test`.
    y_pred_proba : ndarray of shape (n_samples, 2), dtype=np.float64
        Probabilities of `classifier` when predicting for `X_test`.
    y_pred_decision : ndarray of shape (n_samples,), dtype=np.float64
        Decision function values of `classifier` when predicting on `X_test`.
    """
    from sklearn.datasets import load_breast_cancer
    from sklearn.utils import shuffle

    X, y = load_breast_cancer(return_X_y=True)
    # create an highly imbalanced classification task
    idx_positive = np.flatnonzero(y == 1)
    idx_negative = np.flatnonzero(y == 0)
    idx_selected = np.hstack([idx_negative, idx_positive[:25]])
    X, y = X[idx_selected], y[idx_selected]
    X, y = shuffle(X, y, random_state=42)
    # only use 2 features to make the problem even harder
    X = X[:, :2]
    y = np.array(["cancer" if c == 1 else "not cancer" for c in y], dtype=object)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        random_state=0,
    )
    classifier = LogisticRegression().fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)
    y_pred_decision = classifier.decision_function(X_test)

    return classifier, X_test, y_test, y_pred, y_pred_proba, y_pred_decision


def test_average_precision_pos_label(string_labeled_classification_problem):
    # check that _ThresholdScorer will lead to the right score when passing
    # `pos_label`. Currently, only `average_precision_score` is defined to
    # be such a scorer.
    (
        clf,
        X_test,
        y_test,
        _,
        y_pred_proba,
        y_pred_decision,
    ) = string_labeled_classification_problem

    pos_label = "cancer"
    # we need to select the positive column or reverse the decision values
    y_pred_proba = y_pred_proba[:, 0]
    y_pred_decision = y_pred_decision * -1
    assert clf.classes_[0] == pos_label

    # check that when calling the scoring function, probability estimates and
    # decision values lead to the same results
    ap_proba = average_precision_score(y_test, y_pred_proba, pos_label=pos_label)
    ap_decision_function = average_precision_score(
        y_test, y_pred_decision, pos_label=pos_label
    )
    assert ap_proba == pytest.approx(ap_decision_function)

    # create a scorer which would require to pass a `pos_label`
    # check that it fails if `pos_label` is not provided
    average_precision_scorer = make_scorer(
        average_precision_score,
        needs_threshold=True,
    )
    err_msg = "pos_label=1 is not a valid label. It should be one of "
    with pytest.raises(ValueError, match=err_msg):
        average_precision_scorer(clf, X_test, y_test)

    # otherwise, the scorer should give the same results than calling the
    # scoring function
    average_precision_scorer = make_scorer(
        average_precision_score, needs_threshold=True, pos_label=pos_label
    )
    ap_scorer = average_precision_scorer(clf, X_test, y_test)

    assert ap_scorer == pytest.approx(ap_proba)

    # The above scorer call is using `clf.decision_function`. We will force
    # it to use `clf.predict_proba`.
    clf_without_predict_proba = deepcopy(clf)

    def _predict_proba(self, X):
        raise NotImplementedError

    clf_without_predict_proba.predict_proba = partial(
        _predict_proba, clf_without_predict_proba
    )
    # sanity check
    with pytest.raises(NotImplementedError):
        clf_without_predict_proba.predict_proba(X_test)

    ap_scorer = average_precision_scorer(clf_without_predict_proba, X_test, y_test)
    assert ap_scorer == pytest.approx(ap_proba)


def test_brier_score_loss_pos_label(string_labeled_classification_problem):
    # check that _ProbaScorer leads to the right score when `pos_label` is
    # provided. Currently only the `brier_score_loss` is defined to be such
    # a scorer.
    clf, X_test, y_test, _, y_pred_proba, _ = string_labeled_classification_problem

    pos_label = "cancer"
    assert clf.classes_[0] == pos_label

    # brier score loss is symmetric
    brier_pos_cancer = brier_score_loss(y_test, y_pred_proba[:, 0], pos_label="cancer")
    brier_pos_not_cancer = brier_score_loss(
        y_test, y_pred_proba[:, 1], pos_label="not cancer"
    )
    assert brier_pos_cancer == pytest.approx(brier_pos_not_cancer)

    brier_scorer = make_scorer(
        brier_score_loss,
        needs_proba=True,
        pos_label=pos_label,
    )
    assert brier_scorer(clf, X_test, y_test) == pytest.approx(brier_pos_cancer)


@pytest.mark.parametrize(
    "score_func", [f1_score, precision_score, recall_score, jaccard_score]
)
def test_non_symmetric_metric_pos_label(
    score_func, string_labeled_classification_problem
):
    # check that _PredictScorer leads to the right score when `pos_label` is
    # provided. We check for all possible metric supported.
    # Note: At some point we may end up having "scorer tags".
    clf, X_test, y_test, y_pred, _, _ = string_labeled_classification_problem

    pos_label = "cancer"
    assert clf.classes_[0] == pos_label

    score_pos_cancer = score_func(y_test, y_pred, pos_label="cancer")
    score_pos_not_cancer = score_func(y_test, y_pred, pos_label="not cancer")

    assert score_pos_cancer != pytest.approx(score_pos_not_cancer)

    scorer = make_scorer(score_func, pos_label=pos_label)
    assert scorer(clf, X_test, y_test) == pytest.approx(score_pos_cancer)


@pytest.mark.parametrize(
    "scorer",
    [
        make_scorer(average_precision_score, needs_threshold=True, pos_label="xxx"),
        make_scorer(brier_score_loss, needs_proba=True, pos_label="xxx"),
        make_scorer(f1_score, pos_label="xxx"),
    ],
    ids=["ThresholdScorer", "ProbaScorer", "PredictScorer"],
)
def test_scorer_select_proba_error(scorer):
    # check that we raise the the proper error when passing an unknown
    # pos_label
    X, y = make_classification(
        n_classes=2, n_informative=3, n_samples=20, random_state=0
    )
    lr = LogisticRegression().fit(X, y)
    assert scorer._kwargs["pos_label"] not in np.unique(y).tolist()

    err_msg = "is not a valid label"
    with pytest.raises(ValueError, match=err_msg):
        scorer(lr, X, y)


def test_scorer_no_op_multiclass_select_proba():
    # check that calling a ProbaScorer on a multiclass problem do not raise
    # even if `y_true` would be binary during the scoring.
    # `_select_proba_binary` should not be called in this case.
    X, y = make_classification(
        n_classes=3, n_informative=3, n_samples=20, random_state=0
    )
    lr = LogisticRegression().fit(X, y)

    mask_last_class = y == lr.classes_[-1]
    X_test, y_test = X[~mask_last_class], y[~mask_last_class]
    assert_array_equal(np.unique(y_test), lr.classes_[:-1])

    scorer = make_scorer(
        roc_auc_score,
        needs_proba=True,
        multi_class="ovo",
        labels=lr.classes_,
    )
    scorer(lr, X_test, y_test)
