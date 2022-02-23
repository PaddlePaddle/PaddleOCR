import numpy as np
import pytest

from sklearn.base import clone
from sklearn.base import ClassifierMixin
from sklearn.base import is_classifier

from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn.datasets import load_iris, load_diabetes
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor

X, y = load_iris(return_X_y=True)

X_r, y_r = load_diabetes(return_X_y=True)


@pytest.mark.parametrize(
    "X, y, estimator",
    [
        (
            *make_classification(n_samples=10),
            StackingClassifier(
                estimators=[
                    ("lr", LogisticRegression()),
                    ("svm", LinearSVC()),
                    ("rf", RandomForestClassifier(n_estimators=5, max_depth=3)),
                ],
                cv=2,
            ),
        ),
        (
            *make_classification(n_samples=10),
            VotingClassifier(
                estimators=[
                    ("lr", LogisticRegression()),
                    ("svm", LinearSVC()),
                    ("rf", RandomForestClassifier(n_estimators=5, max_depth=3)),
                ]
            ),
        ),
        (
            *make_regression(n_samples=10),
            StackingRegressor(
                estimators=[
                    ("lr", LinearRegression()),
                    ("svm", LinearSVR()),
                    ("rf", RandomForestRegressor(n_estimators=5, max_depth=3)),
                ],
                cv=2,
            ),
        ),
        (
            *make_regression(n_samples=10),
            VotingRegressor(
                estimators=[
                    ("lr", LinearRegression()),
                    ("svm", LinearSVR()),
                    ("rf", RandomForestRegressor(n_estimators=5, max_depth=3)),
                ]
            ),
        ),
    ],
    ids=[
        "stacking-classifier",
        "voting-classifier",
        "stacking-regressor",
        "voting-regressor",
    ],
)
def test_ensemble_heterogeneous_estimators_behavior(X, y, estimator):
    # check that the behavior of `estimators`, `estimators_`,
    # `named_estimators`, `named_estimators_` is consistent across all
    # ensemble classes and when using `set_params()`.

    # before fit
    assert "svm" in estimator.named_estimators
    assert estimator.named_estimators.svm is estimator.estimators[1][1]
    assert estimator.named_estimators.svm is estimator.named_estimators["svm"]

    # check fitted attributes
    estimator.fit(X, y)
    assert len(estimator.named_estimators) == 3
    assert len(estimator.named_estimators_) == 3
    assert sorted(list(estimator.named_estimators_.keys())) == sorted(
        ["lr", "svm", "rf"]
    )

    # check that set_params() does not add a new attribute
    estimator_new_params = clone(estimator)
    svm_estimator = SVC() if is_classifier(estimator) else SVR()
    estimator_new_params.set_params(svm=svm_estimator).fit(X, y)
    assert not hasattr(estimator_new_params, "svm")
    assert (
        estimator_new_params.named_estimators.lr.get_params()
        == estimator.named_estimators.lr.get_params()
    )
    assert (
        estimator_new_params.named_estimators.rf.get_params()
        == estimator.named_estimators.rf.get_params()
    )

    # check the behavior when setting an dropping an estimator
    estimator_dropped = clone(estimator)
    estimator_dropped.set_params(svm="drop")
    estimator_dropped.fit(X, y)
    assert len(estimator_dropped.named_estimators) == 3
    assert estimator_dropped.named_estimators.svm == "drop"
    assert len(estimator_dropped.named_estimators_) == 3
    assert sorted(list(estimator_dropped.named_estimators_.keys())) == sorted(
        ["lr", "svm", "rf"]
    )
    for sub_est in estimator_dropped.named_estimators_:
        # check that the correspondence is correct
        assert not isinstance(sub_est, type(estimator.named_estimators.svm))

    # check that we can set the parameters of the underlying classifier
    estimator.set_params(svm__C=10.0)
    estimator.set_params(rf__max_depth=5)
    assert (
        estimator.get_params()["svm__C"]
        == estimator.get_params()["svm"].get_params()["C"]
    )
    assert (
        estimator.get_params()["rf__max_depth"]
        == estimator.get_params()["rf"].get_params()["max_depth"]
    )


@pytest.mark.parametrize(
    "Ensemble",
    [StackingClassifier, VotingClassifier, StackingRegressor, VotingRegressor],
)
def test_ensemble_heterogeneous_estimators_type(Ensemble):
    # check that ensemble will fail during validation if the underlying
    # estimators are not of the same type (i.e. classifier or regressor)
    if issubclass(Ensemble, ClassifierMixin):
        X, y = make_classification(n_samples=10)
        estimators = [("lr", LinearRegression())]
        ensemble_type = "classifier"
    else:
        X, y = make_regression(n_samples=10)
        estimators = [("lr", LogisticRegression())]
        ensemble_type = "regressor"
    ensemble = Ensemble(estimators=estimators)

    err_msg = "should be a {}".format(ensemble_type)
    with pytest.raises(ValueError, match=err_msg):
        ensemble.fit(X, y)


@pytest.mark.parametrize(
    "X, y, Ensemble",
    [
        (*make_classification(n_samples=10), StackingClassifier),
        (*make_classification(n_samples=10), VotingClassifier),
        (*make_regression(n_samples=10), StackingRegressor),
        (*make_regression(n_samples=10), VotingRegressor),
    ],
)
def test_ensemble_heterogeneous_estimators_name_validation(X, y, Ensemble):
    # raise an error when the name contains dunder
    if issubclass(Ensemble, ClassifierMixin):
        estimators = [("lr__", LogisticRegression())]
    else:
        estimators = [("lr__", LinearRegression())]
    ensemble = Ensemble(estimators=estimators)

    err_msg = r"Estimator names must not contain __: got \['lr__'\]"
    with pytest.raises(ValueError, match=err_msg):
        ensemble.fit(X, y)

    # raise an error when the name is not unique
    if issubclass(Ensemble, ClassifierMixin):
        estimators = [("lr", LogisticRegression()), ("lr", LogisticRegression())]
    else:
        estimators = [("lr", LinearRegression()), ("lr", LinearRegression())]
    ensemble = Ensemble(estimators=estimators)

    err_msg = r"Names provided are not unique: \['lr', 'lr'\]"
    with pytest.raises(ValueError, match=err_msg):
        ensemble.fit(X, y)

    # raise an error when the name conflicts with the parameters
    if issubclass(Ensemble, ClassifierMixin):
        estimators = [("estimators", LogisticRegression())]
    else:
        estimators = [("estimators", LinearRegression())]
    ensemble = Ensemble(estimators=estimators)

    err_msg = "Estimator names conflict with constructor arguments"
    with pytest.raises(ValueError, match=err_msg):
        ensemble.fit(X, y)


@pytest.mark.parametrize(
    "X, y, estimator",
    [
        (
            *make_classification(n_samples=10),
            StackingClassifier(estimators=[("lr", LogisticRegression())]),
        ),
        (
            *make_classification(n_samples=10),
            VotingClassifier(estimators=[("lr", LogisticRegression())]),
        ),
        (
            *make_regression(n_samples=10),
            StackingRegressor(estimators=[("lr", LinearRegression())]),
        ),
        (
            *make_regression(n_samples=10),
            VotingRegressor(estimators=[("lr", LinearRegression())]),
        ),
    ],
    ids=[
        "stacking-classifier",
        "voting-classifier",
        "stacking-regressor",
        "voting-regressor",
    ],
)
def test_ensemble_heterogeneous_estimators_all_dropped(X, y, estimator):
    # check that we raise a consistent error when all estimators are
    # dropped
    estimator.set_params(lr="drop")
    with pytest.raises(ValueError, match="All estimators are dropped."):
        estimator.fit(X, y)


@pytest.mark.parametrize(
    "Ensemble, Estimator, X, y",
    [
        (StackingClassifier, LogisticRegression, X, y),
        (StackingRegressor, LinearRegression, X_r, y_r),
        (VotingClassifier, LogisticRegression, X, y),
        (VotingRegressor, LinearRegression, X_r, y_r),
    ],
)
# FIXME: we should move this test in `estimator_checks` once we are able
# to construct meta-estimator instances
def test_heterogeneous_ensemble_support_missing_values(Ensemble, Estimator, X, y):
    # check that Voting and Stacking predictor delegate the missing values
    # validation to the underlying estimator.
    X = X.copy()
    mask = np.random.choice([1, 0], X.shape, p=[0.1, 0.9]).astype(bool)
    X[mask] = np.nan
    pipe = make_pipeline(SimpleImputer(), Estimator())
    ensemble = Ensemble(estimators=[("pipe1", pipe), ("pipe2", pipe)])
    ensemble.fit(X, y).score(X, y)
