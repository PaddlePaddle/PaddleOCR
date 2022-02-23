"""Testing for the VotingClassifier and VotingRegressor"""

import warnings
import pytest
import re
import numpy as np

from sklearn.utils._testing import assert_almost_equal, assert_array_equal
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.datasets import make_multilabel_classification
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyRegressor


# Load datasets
iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target

X_r, y_r = datasets.load_diabetes(return_X_y=True)


@pytest.mark.parametrize(
    "params, err_msg",
    [
        (
            {"estimators": []},
            "Invalid 'estimators' attribute, 'estimators' should be a list of",
        ),
        (
            {"estimators": [("lr", LogisticRegression())], "voting": "error"},
            r"Voting must be 'soft' or 'hard'; got \(voting='error'\)",
        ),
        (
            {"estimators": [("lr", LogisticRegression())], "weights": [1, 2]},
            "Number of `estimators` and weights must be equal",
        ),
    ],
)
def test_voting_classifier_estimator_init(params, err_msg):
    ensemble = VotingClassifier(**params)
    with pytest.raises(ValueError, match=err_msg):
        ensemble.fit(X, y)


def test_predictproba_hardvoting():
    eclf = VotingClassifier(
        estimators=[("lr1", LogisticRegression()), ("lr2", LogisticRegression())],
        voting="hard",
    )
    msg = "predict_proba is not available when voting='hard'"
    with pytest.raises(AttributeError, match=msg):
        eclf.predict_proba

    assert not hasattr(eclf, "predict_proba")
    eclf.fit(X, y)
    assert not hasattr(eclf, "predict_proba")


def test_notfitted():
    eclf = VotingClassifier(
        estimators=[("lr1", LogisticRegression()), ("lr2", LogisticRegression())],
        voting="soft",
    )
    ereg = VotingRegressor([("dr", DummyRegressor())])
    msg = (
        "This %s instance is not fitted yet. Call 'fit'"
        " with appropriate arguments before using this estimator."
    )
    with pytest.raises(NotFittedError, match=msg % "VotingClassifier"):
        eclf.predict(X)
    with pytest.raises(NotFittedError, match=msg % "VotingClassifier"):
        eclf.predict_proba(X)
    with pytest.raises(NotFittedError, match=msg % "VotingClassifier"):
        eclf.transform(X)
    with pytest.raises(NotFittedError, match=msg % "VotingRegressor"):
        ereg.predict(X_r)
    with pytest.raises(NotFittedError, match=msg % "VotingRegressor"):
        ereg.transform(X_r)


def test_majority_label_iris():
    """Check classification by majority label on dataset iris."""
    clf1 = LogisticRegression(solver="liblinear", random_state=123)
    clf2 = RandomForestClassifier(n_estimators=10, random_state=123)
    clf3 = GaussianNB()
    eclf = VotingClassifier(
        estimators=[("lr", clf1), ("rf", clf2), ("gnb", clf3)], voting="hard"
    )
    scores = cross_val_score(eclf, X, y, scoring="accuracy")
    assert_almost_equal(scores.mean(), 0.95, decimal=2)


def test_tie_situation():
    """Check voting classifier selects smaller class label in tie situation."""
    clf1 = LogisticRegression(random_state=123, solver="liblinear")
    clf2 = RandomForestClassifier(random_state=123)
    eclf = VotingClassifier(estimators=[("lr", clf1), ("rf", clf2)], voting="hard")
    assert clf1.fit(X, y).predict(X)[73] == 2
    assert clf2.fit(X, y).predict(X)[73] == 1
    assert eclf.fit(X, y).predict(X)[73] == 1


def test_weights_iris():
    """Check classification by average probabilities on dataset iris."""
    clf1 = LogisticRegression(random_state=123)
    clf2 = RandomForestClassifier(random_state=123)
    clf3 = GaussianNB()
    eclf = VotingClassifier(
        estimators=[("lr", clf1), ("rf", clf2), ("gnb", clf3)],
        voting="soft",
        weights=[1, 2, 10],
    )
    scores = cross_val_score(eclf, X, y, scoring="accuracy")
    assert_almost_equal(scores.mean(), 0.93, decimal=2)


def test_weights_regressor():
    """Check weighted average regression prediction on diabetes dataset."""
    reg1 = DummyRegressor(strategy="mean")
    reg2 = DummyRegressor(strategy="median")
    reg3 = DummyRegressor(strategy="quantile", quantile=0.2)
    ereg = VotingRegressor(
        [("mean", reg1), ("median", reg2), ("quantile", reg3)], weights=[1, 2, 10]
    )

    X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(
        X_r, y_r, test_size=0.25
    )

    reg1_pred = reg1.fit(X_r_train, y_r_train).predict(X_r_test)
    reg2_pred = reg2.fit(X_r_train, y_r_train).predict(X_r_test)
    reg3_pred = reg3.fit(X_r_train, y_r_train).predict(X_r_test)
    ereg_pred = ereg.fit(X_r_train, y_r_train).predict(X_r_test)

    avg = np.average(
        np.asarray([reg1_pred, reg2_pred, reg3_pred]), axis=0, weights=[1, 2, 10]
    )
    assert_almost_equal(ereg_pred, avg, decimal=2)

    ereg_weights_none = VotingRegressor(
        [("mean", reg1), ("median", reg2), ("quantile", reg3)], weights=None
    )
    ereg_weights_equal = VotingRegressor(
        [("mean", reg1), ("median", reg2), ("quantile", reg3)], weights=[1, 1, 1]
    )
    ereg_weights_none.fit(X_r_train, y_r_train)
    ereg_weights_equal.fit(X_r_train, y_r_train)
    ereg_none_pred = ereg_weights_none.predict(X_r_test)
    ereg_equal_pred = ereg_weights_equal.predict(X_r_test)
    assert_almost_equal(ereg_none_pred, ereg_equal_pred, decimal=2)


def test_predict_on_toy_problem():
    """Manually check predicted class labels for toy dataset."""
    clf1 = LogisticRegression(random_state=123)
    clf2 = RandomForestClassifier(random_state=123)
    clf3 = GaussianNB()

    X = np.array(
        [[-1.1, -1.5], [-1.2, -1.4], [-3.4, -2.2], [1.1, 1.2], [2.1, 1.4], [3.1, 2.3]]
    )

    y = np.array([1, 1, 1, 2, 2, 2])

    assert_array_equal(clf1.fit(X, y).predict(X), [1, 1, 1, 2, 2, 2])
    assert_array_equal(clf2.fit(X, y).predict(X), [1, 1, 1, 2, 2, 2])
    assert_array_equal(clf3.fit(X, y).predict(X), [1, 1, 1, 2, 2, 2])

    eclf = VotingClassifier(
        estimators=[("lr", clf1), ("rf", clf2), ("gnb", clf3)],
        voting="hard",
        weights=[1, 1, 1],
    )
    assert_array_equal(eclf.fit(X, y).predict(X), [1, 1, 1, 2, 2, 2])

    eclf = VotingClassifier(
        estimators=[("lr", clf1), ("rf", clf2), ("gnb", clf3)],
        voting="soft",
        weights=[1, 1, 1],
    )
    assert_array_equal(eclf.fit(X, y).predict(X), [1, 1, 1, 2, 2, 2])


def test_predict_proba_on_toy_problem():
    """Calculate predicted probabilities on toy dataset."""
    clf1 = LogisticRegression(random_state=123)
    clf2 = RandomForestClassifier(random_state=123)
    clf3 = GaussianNB()
    X = np.array([[-1.1, -1.5], [-1.2, -1.4], [-3.4, -2.2], [1.1, 1.2]])
    y = np.array([1, 1, 2, 2])

    clf1_res = np.array(
        [
            [0.59790391, 0.40209609],
            [0.57622162, 0.42377838],
            [0.50728456, 0.49271544],
            [0.40241774, 0.59758226],
        ]
    )

    clf2_res = np.array([[0.8, 0.2], [0.8, 0.2], [0.2, 0.8], [0.3, 0.7]])

    clf3_res = np.array(
        [[0.9985082, 0.0014918], [0.99845843, 0.00154157], [0.0, 1.0], [0.0, 1.0]]
    )

    t00 = (2 * clf1_res[0][0] + clf2_res[0][0] + clf3_res[0][0]) / 4
    t11 = (2 * clf1_res[1][1] + clf2_res[1][1] + clf3_res[1][1]) / 4
    t21 = (2 * clf1_res[2][1] + clf2_res[2][1] + clf3_res[2][1]) / 4
    t31 = (2 * clf1_res[3][1] + clf2_res[3][1] + clf3_res[3][1]) / 4

    eclf = VotingClassifier(
        estimators=[("lr", clf1), ("rf", clf2), ("gnb", clf3)],
        voting="soft",
        weights=[2, 1, 1],
    )
    eclf_res = eclf.fit(X, y).predict_proba(X)

    assert_almost_equal(t00, eclf_res[0][0], decimal=1)
    assert_almost_equal(t11, eclf_res[1][1], decimal=1)
    assert_almost_equal(t21, eclf_res[2][1], decimal=1)
    assert_almost_equal(t31, eclf_res[3][1], decimal=1)

    with pytest.raises(
        AttributeError, match="predict_proba is not available when voting='hard'"
    ):
        eclf = VotingClassifier(
            estimators=[("lr", clf1), ("rf", clf2), ("gnb", clf3)], voting="hard"
        )
        eclf.fit(X, y).predict_proba(X)


def test_multilabel():
    """Check if error is raised for multilabel classification."""
    X, y = make_multilabel_classification(
        n_classes=2, n_labels=1, allow_unlabeled=False, random_state=123
    )
    clf = OneVsRestClassifier(SVC(kernel="linear"))

    eclf = VotingClassifier(estimators=[("ovr", clf)], voting="hard")

    try:
        eclf.fit(X, y)
    except NotImplementedError:
        return


def test_gridsearch():
    """Check GridSearch support."""
    clf1 = LogisticRegression(random_state=1)
    clf2 = RandomForestClassifier(random_state=1, n_estimators=3)
    clf3 = GaussianNB()
    eclf = VotingClassifier(
        estimators=[("lr", clf1), ("rf", clf2), ("gnb", clf3)], voting="soft"
    )

    params = {
        "lr__C": [1.0, 100.0],
        "voting": ["soft", "hard"],
        "weights": [[0.5, 0.5, 0.5], [1.0, 0.5, 0.5]],
    }

    grid = GridSearchCV(estimator=eclf, param_grid=params, cv=2)
    grid.fit(iris.data, iris.target)


def test_parallel_fit():
    """Check parallel backend of VotingClassifier on toy dataset."""
    clf1 = LogisticRegression(random_state=123)
    clf2 = RandomForestClassifier(random_state=123)
    clf3 = GaussianNB()
    X = np.array([[-1.1, -1.5], [-1.2, -1.4], [-3.4, -2.2], [1.1, 1.2]])
    y = np.array([1, 1, 2, 2])

    eclf1 = VotingClassifier(
        estimators=[("lr", clf1), ("rf", clf2), ("gnb", clf3)], voting="soft", n_jobs=1
    ).fit(X, y)
    eclf2 = VotingClassifier(
        estimators=[("lr", clf1), ("rf", clf2), ("gnb", clf3)], voting="soft", n_jobs=2
    ).fit(X, y)

    assert_array_equal(eclf1.predict(X), eclf2.predict(X))
    assert_array_almost_equal(eclf1.predict_proba(X), eclf2.predict_proba(X))


def test_sample_weight():
    """Tests sample_weight parameter of VotingClassifier"""
    clf1 = LogisticRegression(random_state=123)
    clf2 = RandomForestClassifier(random_state=123)
    clf3 = SVC(probability=True, random_state=123)
    eclf1 = VotingClassifier(
        estimators=[("lr", clf1), ("rf", clf2), ("svc", clf3)], voting="soft"
    ).fit(X, y, sample_weight=np.ones((len(y),)))
    eclf2 = VotingClassifier(
        estimators=[("lr", clf1), ("rf", clf2), ("svc", clf3)], voting="soft"
    ).fit(X, y)
    assert_array_equal(eclf1.predict(X), eclf2.predict(X))
    assert_array_almost_equal(eclf1.predict_proba(X), eclf2.predict_proba(X))

    sample_weight = np.random.RandomState(123).uniform(size=(len(y),))
    eclf3 = VotingClassifier(estimators=[("lr", clf1)], voting="soft")
    eclf3.fit(X, y, sample_weight)
    clf1.fit(X, y, sample_weight)
    assert_array_equal(eclf3.predict(X), clf1.predict(X))
    assert_array_almost_equal(eclf3.predict_proba(X), clf1.predict_proba(X))

    # check that an error is raised and indicative if sample_weight is not
    # supported.
    clf4 = KNeighborsClassifier()
    eclf3 = VotingClassifier(
        estimators=[("lr", clf1), ("svc", clf3), ("knn", clf4)], voting="soft"
    )
    msg = "Underlying estimator KNeighborsClassifier does not support sample weights."
    with pytest.raises(TypeError, match=msg):
        eclf3.fit(X, y, sample_weight)

    # check that _fit_single_estimator will raise the right error
    # it should raise the original error if this is not linked to sample_weight
    class ClassifierErrorFit(ClassifierMixin, BaseEstimator):
        def fit(self, X, y, sample_weight):
            raise TypeError("Error unrelated to sample_weight.")

    clf = ClassifierErrorFit()
    with pytest.raises(TypeError, match="Error unrelated to sample_weight"):
        clf.fit(X, y, sample_weight=sample_weight)


def test_sample_weight_kwargs():
    """Check that VotingClassifier passes sample_weight as kwargs"""

    class MockClassifier(ClassifierMixin, BaseEstimator):
        """Mock Classifier to check that sample_weight is received as kwargs"""

        def fit(self, X, y, *args, **sample_weight):
            assert "sample_weight" in sample_weight

    clf = MockClassifier()
    eclf = VotingClassifier(estimators=[("mock", clf)], voting="soft")

    # Should not raise an error.
    eclf.fit(X, y, sample_weight=np.ones((len(y),)))


def test_voting_classifier_set_params():
    # check equivalence in the output when setting underlying estimators
    clf1 = LogisticRegression(random_state=123, C=1.0)
    clf2 = RandomForestClassifier(random_state=123, max_depth=None)
    clf3 = GaussianNB()

    eclf1 = VotingClassifier(
        [("lr", clf1), ("rf", clf2)], voting="soft", weights=[1, 2]
    ).fit(X, y)
    eclf2 = VotingClassifier(
        [("lr", clf1), ("nb", clf3)], voting="soft", weights=[1, 2]
    )
    eclf2.set_params(nb=clf2).fit(X, y)

    assert_array_equal(eclf1.predict(X), eclf2.predict(X))
    assert_array_almost_equal(eclf1.predict_proba(X), eclf2.predict_proba(X))
    assert eclf2.estimators[0][1].get_params() == clf1.get_params()
    assert eclf2.estimators[1][1].get_params() == clf2.get_params()


def test_set_estimator_drop():
    # VotingClassifier set_params should be able to set estimators as drop
    # Test predict
    clf1 = LogisticRegression(random_state=123)
    clf2 = RandomForestClassifier(n_estimators=10, random_state=123)
    clf3 = GaussianNB()
    eclf1 = VotingClassifier(
        estimators=[("lr", clf1), ("rf", clf2), ("nb", clf3)],
        voting="hard",
        weights=[1, 0, 0.5],
    ).fit(X, y)

    eclf2 = VotingClassifier(
        estimators=[("lr", clf1), ("rf", clf2), ("nb", clf3)],
        voting="hard",
        weights=[1, 1, 0.5],
    )
    with pytest.warns(None) as record:
        with warnings.catch_warnings():
            # scipy 1.3.0 uses tostring which is deprecated in numpy
            warnings.filterwarnings("ignore", "tostring", DeprecationWarning)
            eclf2.set_params(rf="drop").fit(X, y)

    assert not record
    assert_array_equal(eclf1.predict(X), eclf2.predict(X))

    assert dict(eclf2.estimators)["rf"] == "drop"
    assert len(eclf2.estimators_) == 2
    assert all(
        isinstance(est, (LogisticRegression, GaussianNB)) for est in eclf2.estimators_
    )
    assert eclf2.get_params()["rf"] == "drop"

    eclf1.set_params(voting="soft").fit(X, y)
    with pytest.warns(None) as record:
        with warnings.catch_warnings():
            # scipy 1.3.0 uses tostring which is deprecated in numpy
            warnings.filterwarnings("ignore", "tostring", DeprecationWarning)
            eclf2.set_params(voting="soft").fit(X, y)

    assert not record
    assert_array_equal(eclf1.predict(X), eclf2.predict(X))
    assert_array_almost_equal(eclf1.predict_proba(X), eclf2.predict_proba(X))
    msg = "All estimators are dropped. At least one is required"
    with pytest.warns(None) as record:
        with pytest.raises(ValueError, match=msg):
            eclf2.set_params(lr="drop", rf="drop", nb="drop").fit(X, y)
    assert not record

    # Test soft voting transform
    X1 = np.array([[1], [2]])
    y1 = np.array([1, 2])
    eclf1 = VotingClassifier(
        estimators=[("rf", clf2), ("nb", clf3)],
        voting="soft",
        weights=[0, 0.5],
        flatten_transform=False,
    ).fit(X1, y1)

    eclf2 = VotingClassifier(
        estimators=[("rf", clf2), ("nb", clf3)],
        voting="soft",
        weights=[1, 0.5],
        flatten_transform=False,
    )
    with pytest.warns(None) as record:
        with warnings.catch_warnings():
            # scipy 1.3.0 uses tostring which is deprecated in numpy
            warnings.filterwarnings("ignore", "tostring", DeprecationWarning)
            eclf2.set_params(rf="drop").fit(X1, y1)
    assert not record
    assert_array_almost_equal(
        eclf1.transform(X1),
        np.array([[[0.7, 0.3], [0.3, 0.7]], [[1.0, 0.0], [0.0, 1.0]]]),
    )
    assert_array_almost_equal(eclf2.transform(X1), np.array([[[1.0, 0.0], [0.0, 1.0]]]))
    eclf1.set_params(voting="hard")
    eclf2.set_params(voting="hard")
    assert_array_equal(eclf1.transform(X1), np.array([[0, 0], [1, 1]]))
    assert_array_equal(eclf2.transform(X1), np.array([[0], [1]]))


def test_estimator_weights_format():
    # Test estimator weights inputs as list and array
    clf1 = LogisticRegression(random_state=123)
    clf2 = RandomForestClassifier(random_state=123)
    eclf1 = VotingClassifier(
        estimators=[("lr", clf1), ("rf", clf2)], weights=[1, 2], voting="soft"
    )
    eclf2 = VotingClassifier(
        estimators=[("lr", clf1), ("rf", clf2)], weights=np.array((1, 2)), voting="soft"
    )
    eclf1.fit(X, y)
    eclf2.fit(X, y)
    assert_array_almost_equal(eclf1.predict_proba(X), eclf2.predict_proba(X))


def test_transform():
    """Check transform method of VotingClassifier on toy dataset."""
    clf1 = LogisticRegression(random_state=123)
    clf2 = RandomForestClassifier(random_state=123)
    clf3 = GaussianNB()
    X = np.array([[-1.1, -1.5], [-1.2, -1.4], [-3.4, -2.2], [1.1, 1.2]])
    y = np.array([1, 1, 2, 2])

    eclf1 = VotingClassifier(
        estimators=[("lr", clf1), ("rf", clf2), ("gnb", clf3)], voting="soft"
    ).fit(X, y)
    eclf2 = VotingClassifier(
        estimators=[("lr", clf1), ("rf", clf2), ("gnb", clf3)],
        voting="soft",
        flatten_transform=True,
    ).fit(X, y)
    eclf3 = VotingClassifier(
        estimators=[("lr", clf1), ("rf", clf2), ("gnb", clf3)],
        voting="soft",
        flatten_transform=False,
    ).fit(X, y)

    assert_array_equal(eclf1.transform(X).shape, (4, 6))
    assert_array_equal(eclf2.transform(X).shape, (4, 6))
    assert_array_equal(eclf3.transform(X).shape, (3, 4, 2))
    assert_array_almost_equal(eclf1.transform(X), eclf2.transform(X))
    assert_array_almost_equal(
        eclf3.transform(X).swapaxes(0, 1).reshape((4, 6)), eclf2.transform(X)
    )


@pytest.mark.parametrize(
    "X, y, voter",
    [
        (
            X,
            y,
            VotingClassifier(
                [
                    ("lr", LogisticRegression()),
                    ("rf", RandomForestClassifier(n_estimators=5)),
                ]
            ),
        ),
        (
            X_r,
            y_r,
            VotingRegressor(
                [
                    ("lr", LinearRegression()),
                    ("rf", RandomForestRegressor(n_estimators=5)),
                ]
            ),
        ),
    ],
)
def test_none_estimator_with_weights(X, y, voter):
    # check that an estimator can be set to 'drop' and passing some weight
    # regression test for
    # https://github.com/scikit-learn/scikit-learn/issues/13777
    voter = clone(voter)
    voter.fit(X, y, sample_weight=np.ones(y.shape))
    voter.set_params(lr="drop")
    with pytest.warns(None) as record:
        voter.fit(X, y, sample_weight=np.ones(y.shape))
    assert not record
    y_pred = voter.predict(X)
    assert y_pred.shape == y.shape


@pytest.mark.parametrize(
    "est",
    [
        VotingRegressor(
            estimators=[
                ("lr", LinearRegression()),
                ("tree", DecisionTreeRegressor(random_state=0)),
            ]
        ),
        VotingClassifier(
            estimators=[
                ("lr", LogisticRegression(random_state=0)),
                ("tree", DecisionTreeClassifier(random_state=0)),
            ]
        ),
    ],
    ids=["VotingRegressor", "VotingClassifier"],
)
def test_n_features_in(est):

    X = [[1, 2], [3, 4], [5, 6]]
    y = [0, 1, 2]

    assert not hasattr(est, "n_features_in_")
    est.fit(X, y)
    assert est.n_features_in_ == 2


@pytest.mark.parametrize(
    "estimator",
    [
        VotingRegressor(
            estimators=[
                ("lr", LinearRegression()),
                ("rf", RandomForestRegressor(random_state=123)),
            ],
            verbose=True,
        ),
        VotingClassifier(
            estimators=[
                ("lr", LogisticRegression(random_state=123)),
                ("rf", RandomForestClassifier(random_state=123)),
            ],
            verbose=True,
        ),
    ],
)
def test_voting_verbose(estimator, capsys):

    X = np.array([[-1.1, -1.5], [-1.2, -1.4], [-3.4, -2.2], [1.1, 1.2]])
    y = np.array([1, 1, 2, 2])

    pattern = (
        r"\[Voting\].*\(1 of 2\) Processing lr, total=.*\n"
        r"\[Voting\].*\(2 of 2\) Processing rf, total=.*\n$"
    )

    estimator.fit(X, y)
    assert re.match(pattern, capsys.readouterr()[0])
