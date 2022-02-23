"""
Testing for the forest module (sklearn.ensemble.forest).
"""

# Authors: Gilles Louppe,
#          Brian Holt,
#          Andreas Mueller,
#          Arnaud Joly
# License: BSD 3 clause

import pickle
import math
from collections import defaultdict
import itertools
from itertools import combinations
from itertools import product
from typing import Dict, Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import coo_matrix
from scipy.special import comb

import pytest

import joblib
from numpy.testing import assert_allclose

from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.utils._testing import assert_almost_equal
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import _convert_container
from sklearn.utils._testing import ignore_warnings
from sklearn.utils._testing import skip_if_no_parallel
from sklearn.utils.fixes import parse_version

from sklearn.exceptions import NotFittedError

from sklearn import datasets
from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_random_state

from sklearn.metrics import mean_squared_error

from sklearn.tree._classes import SPARSE_SPLITTERS


# toy sample
X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
y = [-1, -1, -1, 1, 1, 1]
T = [[-1, -1], [2, 2], [3, 2]]
true_result = [-1, 1, 1]

# Larger classification sample used for testing feature importances
X_large, y_large = datasets.make_classification(
    n_samples=500,
    n_features=10,
    n_informative=3,
    n_redundant=0,
    n_repeated=0,
    shuffle=False,
    random_state=0,
)

# also load the iris dataset
# and randomly permute it
iris = datasets.load_iris()
rng = check_random_state(0)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

# Make regression dataset
X_reg, y_reg = datasets.make_regression(n_samples=500, n_features=10, random_state=1)

# also make a hastie_10_2 dataset
hastie_X, hastie_y = datasets.make_hastie_10_2(n_samples=20, random_state=1)
hastie_X = hastie_X.astype(np.float32)

# Get the default backend in joblib to test parallelism and interaction with
# different backends
DEFAULT_JOBLIB_BACKEND = joblib.parallel.get_active_backend()[0].__class__

FOREST_CLASSIFIERS = {
    "ExtraTreesClassifier": ExtraTreesClassifier,
    "RandomForestClassifier": RandomForestClassifier,
}

FOREST_REGRESSORS = {
    "ExtraTreesRegressor": ExtraTreesRegressor,
    "RandomForestRegressor": RandomForestRegressor,
}

FOREST_TRANSFORMERS = {
    "RandomTreesEmbedding": RandomTreesEmbedding,
}

FOREST_ESTIMATORS: Dict[str, Any] = dict()
FOREST_ESTIMATORS.update(FOREST_CLASSIFIERS)
FOREST_ESTIMATORS.update(FOREST_REGRESSORS)
FOREST_ESTIMATORS.update(FOREST_TRANSFORMERS)

FOREST_CLASSIFIERS_REGRESSORS: Dict[str, Any] = FOREST_CLASSIFIERS.copy()
FOREST_CLASSIFIERS_REGRESSORS.update(FOREST_REGRESSORS)


def check_classification_toy(name):
    """Check classification on a toy dataset."""
    ForestClassifier = FOREST_CLASSIFIERS[name]

    clf = ForestClassifier(n_estimators=10, random_state=1)
    clf.fit(X, y)
    assert_array_equal(clf.predict(T), true_result)
    assert 10 == len(clf)

    clf = ForestClassifier(n_estimators=10, max_features=1, random_state=1)
    clf.fit(X, y)
    assert_array_equal(clf.predict(T), true_result)
    assert 10 == len(clf)

    # also test apply
    leaf_indices = clf.apply(X)
    assert leaf_indices.shape == (len(X), clf.n_estimators)


@pytest.mark.parametrize("name", FOREST_CLASSIFIERS)
def test_classification_toy(name):
    check_classification_toy(name)


def check_iris_criterion(name, criterion):
    # Check consistency on dataset iris.
    ForestClassifier = FOREST_CLASSIFIERS[name]

    clf = ForestClassifier(n_estimators=10, criterion=criterion, random_state=1)
    clf.fit(iris.data, iris.target)
    score = clf.score(iris.data, iris.target)
    assert score > 0.9, "Failed with criterion %s and score = %f" % (criterion, score)

    clf = ForestClassifier(
        n_estimators=10, criterion=criterion, max_features=2, random_state=1
    )
    clf.fit(iris.data, iris.target)
    score = clf.score(iris.data, iris.target)
    assert score > 0.5, "Failed with criterion %s and score = %f" % (criterion, score)


@pytest.mark.parametrize("name", FOREST_CLASSIFIERS)
@pytest.mark.parametrize("criterion", ("gini", "entropy"))
def test_iris(name, criterion):
    check_iris_criterion(name, criterion)


def check_regression_criterion(name, criterion):
    # Check consistency on regression dataset.
    ForestRegressor = FOREST_REGRESSORS[name]

    reg = ForestRegressor(n_estimators=5, criterion=criterion, random_state=1)
    reg.fit(X_reg, y_reg)
    score = reg.score(X_reg, y_reg)
    assert (
        score > 0.93
    ), "Failed with max_features=None, criterion %s and score = %f" % (
        criterion,
        score,
    )

    reg = ForestRegressor(
        n_estimators=5, criterion=criterion, max_features=6, random_state=1
    )
    reg.fit(X_reg, y_reg)
    score = reg.score(X_reg, y_reg)
    assert score > 0.92, "Failed with max_features=6, criterion %s and score = %f" % (
        criterion,
        score,
    )


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
@pytest.mark.parametrize(
    "criterion", ("squared_error", "absolute_error", "friedman_mse")
)
def test_regression(name, criterion):
    check_regression_criterion(name, criterion)


def test_poisson_vs_mse():
    """Test that random forest with poisson criterion performs better than
    mse for a poisson target."""
    rng = np.random.RandomState(42)
    n_train, n_test, n_features = 500, 500, 10
    X = datasets.make_low_rank_matrix(
        n_samples=n_train + n_test, n_features=n_features, random_state=rng
    )
    X = np.abs(X)
    X /= np.max(np.abs(X), axis=0)
    # We create a log-linear Poisson model
    coef = rng.uniform(low=-4, high=1, size=n_features)
    y = rng.poisson(lam=np.exp(X @ coef))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=n_test, random_state=rng
    )

    forest_poi = RandomForestRegressor(
        criterion="poisson", min_samples_leaf=10, max_features="sqrt", random_state=rng
    )
    forest_mse = RandomForestRegressor(
        criterion="squared_error",
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=rng,
    )

    forest_poi.fit(X_train, y_train)
    forest_mse.fit(X_train, y_train)
    dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)

    for X, y, val in [(X_train, y_train, "train"), (X_test, y_test, "test")]:
        metric_poi = mean_poisson_deviance(y, forest_poi.predict(X))
        # squared_error forest might produce non-positive predictions => clip
        # If y = 0 for those, the poisson deviance gets too good.
        # If we drew more samples, we would eventually get y > 0 and the
        # poisson deviance would explode, i.e. be undefined. Therefore, we do
        # not clip to a tiny value like 1e-15, but to 0.1. This acts like a
        # mild penalty to the non-positive predictions.
        metric_mse = mean_poisson_deviance(
            y, np.clip(forest_mse.predict(X), 1e-6, None)
        )
        metric_dummy = mean_poisson_deviance(y, dummy.predict(X))
        # As squared_error might correctly predict 0 in train set, its train
        # score can be better than Poisson. This is no longer the case for the
        # test set. But keep the above comment for clipping in mind.
        if val == "test":
            assert metric_poi < metric_mse
        assert metric_poi < metric_dummy


@pytest.mark.parametrize("criterion", ("poisson", "squared_error"))
def test_balance_property_random_forest(criterion):
    """ "Test that sum(y_pred)==sum(y_true) on the training set."""
    rng = np.random.RandomState(42)
    n_train, n_test, n_features = 500, 500, 10
    X = datasets.make_low_rank_matrix(
        n_samples=n_train + n_test, n_features=n_features, random_state=rng
    )

    coef = rng.uniform(low=-2, high=2, size=n_features) / np.max(X, axis=0)
    y = rng.poisson(lam=np.exp(X @ coef))

    reg = RandomForestRegressor(
        criterion=criterion, n_estimators=10, bootstrap=False, random_state=rng
    )
    reg.fit(X, y)

    assert np.sum(reg.predict(X)) == pytest.approx(np.sum(y))


def check_regressor_attributes(name):
    # Regression models should not have a classes_ attribute.
    r = FOREST_REGRESSORS[name](random_state=0)
    assert not hasattr(r, "classes_")
    assert not hasattr(r, "n_classes_")

    r.fit([[1, 2, 3], [4, 5, 6]], [1, 2])
    assert not hasattr(r, "classes_")
    assert not hasattr(r, "n_classes_")


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
def test_regressor_attributes(name):
    check_regressor_attributes(name)


def check_probability(name):
    # Predict probabilities.
    ForestClassifier = FOREST_CLASSIFIERS[name]
    with np.errstate(divide="ignore"):
        clf = ForestClassifier(
            n_estimators=10, random_state=1, max_features=1, max_depth=1
        )
        clf.fit(iris.data, iris.target)
        assert_array_almost_equal(
            np.sum(clf.predict_proba(iris.data), axis=1), np.ones(iris.data.shape[0])
        )
        assert_array_almost_equal(
            clf.predict_proba(iris.data), np.exp(clf.predict_log_proba(iris.data))
        )


@pytest.mark.parametrize("name", FOREST_CLASSIFIERS)
def test_probability(name):
    check_probability(name)


def check_importances(name, criterion, dtype, tolerance):
    # cast as dype
    X = X_large.astype(dtype, copy=False)
    y = y_large.astype(dtype, copy=False)

    ForestEstimator = FOREST_ESTIMATORS[name]

    est = ForestEstimator(n_estimators=10, criterion=criterion, random_state=0)
    est.fit(X, y)
    importances = est.feature_importances_

    # The forest estimator can detect that only the first 3 features of the
    # dataset are informative:
    n_important = np.sum(importances > 0.1)
    assert importances.shape[0] == 10
    assert n_important == 3
    assert np.all(importances[:3] > 0.1)

    # Check with parallel
    importances = est.feature_importances_
    est.set_params(n_jobs=2)
    importances_parallel = est.feature_importances_
    assert_array_almost_equal(importances, importances_parallel)

    # Check with sample weights
    sample_weight = check_random_state(0).randint(1, 10, len(X))
    est = ForestEstimator(n_estimators=10, random_state=0, criterion=criterion)
    est.fit(X, y, sample_weight=sample_weight)
    importances = est.feature_importances_
    assert np.all(importances >= 0.0)

    for scale in [0.5, 100]:
        est = ForestEstimator(n_estimators=10, random_state=0, criterion=criterion)
        est.fit(X, y, sample_weight=scale * sample_weight)
        importances_bis = est.feature_importances_
        assert np.abs(importances - importances_bis).mean() < tolerance


@pytest.mark.parametrize("dtype", (np.float64, np.float32))
@pytest.mark.parametrize(
    "name, criterion",
    itertools.chain(
        product(FOREST_CLASSIFIERS, ["gini", "entropy"]),
        product(FOREST_REGRESSORS, ["squared_error", "friedman_mse", "absolute_error"]),
    ),
)
def test_importances(dtype, name, criterion):
    tolerance = 0.01
    if name in FOREST_REGRESSORS and criterion == "absolute_error":
        tolerance = 0.05
    check_importances(name, criterion, dtype, tolerance)


def test_importances_asymptotic():
    # Check whether variable importances of totally randomized trees
    # converge towards their theoretical values (See Louppe et al,
    # Understanding variable importances in forests of randomized trees, 2013).

    def binomial(k, n):
        return 0 if k < 0 or k > n else comb(int(n), int(k), exact=True)

    def entropy(samples):
        n_samples = len(samples)
        entropy = 0.0

        for count in np.bincount(samples):
            p = 1.0 * count / n_samples
            if p > 0:
                entropy -= p * np.log2(p)

        return entropy

    def mdi_importance(X_m, X, y):
        n_samples, n_features = X.shape

        features = list(range(n_features))
        features.pop(X_m)
        values = [np.unique(X[:, i]) for i in range(n_features)]

        imp = 0.0

        for k in range(n_features):
            # Weight of each B of size k
            coef = 1.0 / (binomial(k, n_features) * (n_features - k))

            # For all B of size k
            for B in combinations(features, k):
                # For all values B=b
                for b in product(*[values[B[j]] for j in range(k)]):
                    mask_b = np.ones(n_samples, dtype=bool)

                    for j in range(k):
                        mask_b &= X[:, B[j]] == b[j]

                    X_, y_ = X[mask_b, :], y[mask_b]
                    n_samples_b = len(X_)

                    if n_samples_b > 0:
                        children = []

                        for xi in values[X_m]:
                            mask_xi = X_[:, X_m] == xi
                            children.append(y_[mask_xi])

                        imp += (
                            coef
                            * (1.0 * n_samples_b / n_samples)  # P(B=b)
                            * (
                                entropy(y_)
                                - sum(
                                    [
                                        entropy(c) * len(c) / n_samples_b
                                        for c in children
                                    ]
                                )
                            )
                        )

        return imp

    data = np.array(
        [
            [0, 0, 1, 0, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 0, 1, 2],
            [1, 0, 1, 1, 0, 1, 1, 3],
            [0, 1, 1, 1, 0, 1, 0, 4],
            [1, 1, 0, 1, 0, 1, 1, 5],
            [1, 1, 0, 1, 1, 1, 1, 6],
            [1, 0, 1, 0, 0, 1, 0, 7],
            [1, 1, 1, 1, 1, 1, 1, 8],
            [1, 1, 1, 1, 0, 1, 1, 9],
            [1, 1, 1, 0, 1, 1, 1, 0],
        ]
    )

    X, y = np.array(data[:, :7], dtype=bool), data[:, 7]
    n_features = X.shape[1]

    # Compute true importances
    true_importances = np.zeros(n_features)

    for i in range(n_features):
        true_importances[i] = mdi_importance(i, X, y)

    # Estimate importances with totally randomized trees
    clf = ExtraTreesClassifier(
        n_estimators=500, max_features=1, criterion="entropy", random_state=0
    ).fit(X, y)

    importances = (
        sum(
            tree.tree_.compute_feature_importances(normalize=False)
            for tree in clf.estimators_
        )
        / clf.n_estimators
    )

    # Check correctness
    assert_almost_equal(entropy(y), sum(importances))
    assert np.abs(true_importances - importances).mean() < 0.01


@pytest.mark.parametrize("name", FOREST_ESTIMATORS)
def test_unfitted_feature_importances(name):
    err_msg = (
        "This {} instance is not fitted yet. Call 'fit' with "
        "appropriate arguments before using this estimator.".format(name)
    )
    with pytest.raises(NotFittedError, match=err_msg):
        getattr(FOREST_ESTIMATORS[name](), "feature_importances_")


@pytest.mark.parametrize("ForestClassifier", FOREST_CLASSIFIERS.values())
@pytest.mark.parametrize("X_type", ["array", "sparse_csr", "sparse_csc"])
@pytest.mark.parametrize(
    "X, y, lower_bound_accuracy",
    [
        (
            *datasets.make_classification(n_samples=300, n_classes=2, random_state=0),
            0.9,
        ),
        (
            *datasets.make_classification(
                n_samples=1000, n_classes=3, n_informative=6, random_state=0
            ),
            0.65,
        ),
        (
            iris.data,
            iris.target * 2 + 1,
            0.65,
        ),
        (
            *datasets.make_multilabel_classification(n_samples=300, random_state=0),
            0.18,
        ),
    ],
)
def test_forest_classifier_oob(ForestClassifier, X, y, X_type, lower_bound_accuracy):
    """Check that OOB score is close to score on a test set."""
    X = _convert_container(X, constructor_name=X_type)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.5,
        random_state=0,
    )
    classifier = ForestClassifier(
        n_estimators=40,
        bootstrap=True,
        oob_score=True,
        random_state=0,
    )

    assert not hasattr(classifier, "oob_score_")
    assert not hasattr(classifier, "oob_decision_function_")

    classifier.fit(X_train, y_train)
    test_score = classifier.score(X_test, y_test)

    assert abs(test_score - classifier.oob_score_) <= 0.1
    assert classifier.oob_score_ >= lower_bound_accuracy

    assert hasattr(classifier, "oob_score_")
    assert not hasattr(classifier, "oob_prediction_")
    assert hasattr(classifier, "oob_decision_function_")

    if y.ndim == 1:
        expected_shape = (X_train.shape[0], len(set(y)))
    else:
        expected_shape = (X_train.shape[0], len(set(y[:, 0])), y.shape[1])
    assert classifier.oob_decision_function_.shape == expected_shape


@pytest.mark.parametrize("ForestRegressor", FOREST_REGRESSORS.values())
@pytest.mark.parametrize("X_type", ["array", "sparse_csr", "sparse_csc"])
@pytest.mark.parametrize(
    "X, y, lower_bound_r2",
    [
        (
            *datasets.make_regression(
                n_samples=500, n_features=10, n_targets=1, random_state=0
            ),
            0.7,
        ),
        (
            *datasets.make_regression(
                n_samples=500, n_features=10, n_targets=2, random_state=0
            ),
            0.55,
        ),
    ],
)
def test_forest_regressor_oob(ForestRegressor, X, y, X_type, lower_bound_r2):
    """Check that forest-based regressor provide an OOB score close to the
    score on a test set."""
    X = _convert_container(X, constructor_name=X_type)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.5,
        random_state=0,
    )
    regressor = ForestRegressor(
        n_estimators=50,
        bootstrap=True,
        oob_score=True,
        random_state=0,
    )

    assert not hasattr(regressor, "oob_score_")
    assert not hasattr(regressor, "oob_prediction_")

    regressor.fit(X_train, y_train)
    test_score = regressor.score(X_test, y_test)

    assert abs(test_score - regressor.oob_score_) <= 0.1
    assert regressor.oob_score_ >= lower_bound_r2

    assert hasattr(regressor, "oob_score_")
    assert hasattr(regressor, "oob_prediction_")
    assert not hasattr(regressor, "oob_decision_function_")

    if y.ndim == 1:
        expected_shape = (X_train.shape[0],)
    else:
        expected_shape = (X_train.shape[0], y.ndim)
    assert regressor.oob_prediction_.shape == expected_shape


@pytest.mark.parametrize("ForestEstimator", FOREST_CLASSIFIERS_REGRESSORS.values())
def test_forest_oob_warning(ForestEstimator):
    """Check that a warning is raised when not enough estimator and the OOB
    estimates will be inaccurate."""
    estimator = ForestEstimator(
        n_estimators=1,
        oob_score=True,
        bootstrap=True,
        random_state=0,
    )
    with pytest.warns(UserWarning, match="Some inputs do not have OOB scores"):
        estimator.fit(iris.data, iris.target)


@pytest.mark.parametrize("ForestEstimator", FOREST_CLASSIFIERS_REGRESSORS.values())
@pytest.mark.parametrize(
    "X, y, params, err_msg",
    [
        (
            iris.data,
            iris.target,
            {"oob_score": True, "bootstrap": False},
            "Out of bag estimation only available if bootstrap=True",
        ),
        (
            iris.data,
            rng.randint(low=0, high=5, size=(iris.data.shape[0], 2)),
            {"oob_score": True, "bootstrap": True},
            "The type of target cannot be used to compute OOB estimates",
        ),
    ],
)
def test_forest_oob_error(ForestEstimator, X, y, params, err_msg):
    estimator = ForestEstimator(**params)
    with pytest.raises(ValueError, match=err_msg):
        estimator.fit(X, y)


@pytest.mark.parametrize("oob_score", [True, False])
def test_random_trees_embedding_raise_error_oob(oob_score):
    with pytest.raises(TypeError, match="got an unexpected keyword argument"):
        RandomTreesEmbedding(oob_score=oob_score)
    with pytest.raises(NotImplementedError, match="OOB score not supported"):
        RandomTreesEmbedding()._set_oob_score_and_attributes(X, y)


def check_gridsearch(name):
    forest = FOREST_CLASSIFIERS[name]()
    clf = GridSearchCV(forest, {"n_estimators": (1, 2), "max_depth": (1, 2)})
    clf.fit(iris.data, iris.target)


@pytest.mark.parametrize("name", FOREST_CLASSIFIERS)
def test_gridsearch(name):
    # Check that base trees can be grid-searched.
    check_gridsearch(name)


def check_parallel(name, X, y):
    """Check parallel computations in classification"""
    ForestEstimator = FOREST_ESTIMATORS[name]
    forest = ForestEstimator(n_estimators=10, n_jobs=3, random_state=0)

    forest.fit(X, y)
    assert len(forest) == 10

    forest.set_params(n_jobs=1)
    y1 = forest.predict(X)
    forest.set_params(n_jobs=2)
    y2 = forest.predict(X)
    assert_array_almost_equal(y1, y2, 3)


@pytest.mark.parametrize("name", FOREST_CLASSIFIERS_REGRESSORS)
def test_parallel(name):
    if name in FOREST_CLASSIFIERS:
        X = iris.data
        y = iris.target
    elif name in FOREST_REGRESSORS:
        X = X_reg
        y = y_reg

    check_parallel(name, X, y)


def check_pickle(name, X, y):
    # Check pickability.

    ForestEstimator = FOREST_ESTIMATORS[name]
    obj = ForestEstimator(random_state=0)
    obj.fit(X, y)
    score = obj.score(X, y)
    pickle_object = pickle.dumps(obj)

    obj2 = pickle.loads(pickle_object)
    assert type(obj2) == obj.__class__
    score2 = obj2.score(X, y)
    assert score == score2


@pytest.mark.parametrize("name", FOREST_CLASSIFIERS_REGRESSORS)
def test_pickle(name):
    if name in FOREST_CLASSIFIERS:
        X = iris.data
        y = iris.target
    elif name in FOREST_REGRESSORS:
        X = X_reg
        y = y_reg

    check_pickle(name, X[::2], y[::2])


def check_multioutput(name):
    # Check estimators on multi-output problems.

    X_train = [
        [-2, -1],
        [-1, -1],
        [-1, -2],
        [1, 1],
        [1, 2],
        [2, 1],
        [-2, 1],
        [-1, 1],
        [-1, 2],
        [2, -1],
        [1, -1],
        [1, -2],
    ]
    y_train = [
        [-1, 0],
        [-1, 0],
        [-1, 0],
        [1, 1],
        [1, 1],
        [1, 1],
        [-1, 2],
        [-1, 2],
        [-1, 2],
        [1, 3],
        [1, 3],
        [1, 3],
    ]
    X_test = [[-1, -1], [1, 1], [-1, 1], [1, -1]]
    y_test = [[-1, 0], [1, 1], [-1, 2], [1, 3]]

    est = FOREST_ESTIMATORS[name](random_state=0, bootstrap=False)
    y_pred = est.fit(X_train, y_train).predict(X_test)
    assert_array_almost_equal(y_pred, y_test)

    if name in FOREST_CLASSIFIERS:
        with np.errstate(divide="ignore"):
            proba = est.predict_proba(X_test)
            assert len(proba) == 2
            assert proba[0].shape == (4, 2)
            assert proba[1].shape == (4, 4)

            log_proba = est.predict_log_proba(X_test)
            assert len(log_proba) == 2
            assert log_proba[0].shape == (4, 2)
            assert log_proba[1].shape == (4, 4)


@pytest.mark.parametrize("name", FOREST_CLASSIFIERS_REGRESSORS)
def test_multioutput(name):
    check_multioutput(name)


@pytest.mark.parametrize("name", FOREST_CLASSIFIERS)
def test_multioutput_string(name):
    # Check estimators on multi-output problems with string outputs.

    X_train = [
        [-2, -1],
        [-1, -1],
        [-1, -2],
        [1, 1],
        [1, 2],
        [2, 1],
        [-2, 1],
        [-1, 1],
        [-1, 2],
        [2, -1],
        [1, -1],
        [1, -2],
    ]
    y_train = [
        ["red", "blue"],
        ["red", "blue"],
        ["red", "blue"],
        ["green", "green"],
        ["green", "green"],
        ["green", "green"],
        ["red", "purple"],
        ["red", "purple"],
        ["red", "purple"],
        ["green", "yellow"],
        ["green", "yellow"],
        ["green", "yellow"],
    ]
    X_test = [[-1, -1], [1, 1], [-1, 1], [1, -1]]
    y_test = [
        ["red", "blue"],
        ["green", "green"],
        ["red", "purple"],
        ["green", "yellow"],
    ]

    est = FOREST_ESTIMATORS[name](random_state=0, bootstrap=False)
    y_pred = est.fit(X_train, y_train).predict(X_test)
    assert_array_equal(y_pred, y_test)

    with np.errstate(divide="ignore"):
        proba = est.predict_proba(X_test)
        assert len(proba) == 2
        assert proba[0].shape == (4, 2)
        assert proba[1].shape == (4, 4)

        log_proba = est.predict_log_proba(X_test)
        assert len(log_proba) == 2
        assert log_proba[0].shape == (4, 2)
        assert log_proba[1].shape == (4, 4)


def check_classes_shape(name):
    # Test that n_classes_ and classes_ have proper shape.
    ForestClassifier = FOREST_CLASSIFIERS[name]

    # Classification, single output
    clf = ForestClassifier(random_state=0).fit(X, y)

    assert clf.n_classes_ == 2
    assert_array_equal(clf.classes_, [-1, 1])

    # Classification, multi-output
    _y = np.vstack((y, np.array(y) * 2)).T
    clf = ForestClassifier(random_state=0).fit(X, _y)

    assert_array_equal(clf.n_classes_, [2, 2])
    assert_array_equal(clf.classes_, [[-1, 1], [-2, 2]])


@pytest.mark.parametrize("name", FOREST_CLASSIFIERS)
def test_classes_shape(name):
    check_classes_shape(name)


def test_random_trees_dense_type():
    # Test that the `sparse_output` parameter of RandomTreesEmbedding
    # works by returning a dense array.

    # Create the RTE with sparse=False
    hasher = RandomTreesEmbedding(n_estimators=10, sparse_output=False)
    X, y = datasets.make_circles(factor=0.5)
    X_transformed = hasher.fit_transform(X)

    # Assert that type is ndarray, not scipy.sparse.csr.csr_matrix
    assert type(X_transformed) == np.ndarray


def test_random_trees_dense_equal():
    # Test that the `sparse_output` parameter of RandomTreesEmbedding
    # works by returning the same array for both argument values.

    # Create the RTEs
    hasher_dense = RandomTreesEmbedding(
        n_estimators=10, sparse_output=False, random_state=0
    )
    hasher_sparse = RandomTreesEmbedding(
        n_estimators=10, sparse_output=True, random_state=0
    )
    X, y = datasets.make_circles(factor=0.5)
    X_transformed_dense = hasher_dense.fit_transform(X)
    X_transformed_sparse = hasher_sparse.fit_transform(X)

    # Assert that dense and sparse hashers have same array.
    assert_array_equal(X_transformed_sparse.toarray(), X_transformed_dense)


# Ignore warnings from switching to more power iterations in randomized_svd
@ignore_warnings
def test_random_hasher():
    # test random forest hashing on circles dataset
    # make sure that it is linearly separable.
    # even after projected to two SVD dimensions
    # Note: Not all random_states produce perfect results.
    hasher = RandomTreesEmbedding(n_estimators=30, random_state=1)
    X, y = datasets.make_circles(factor=0.5)
    X_transformed = hasher.fit_transform(X)

    # test fit and transform:
    hasher = RandomTreesEmbedding(n_estimators=30, random_state=1)
    assert_array_equal(hasher.fit(X).transform(X).toarray(), X_transformed.toarray())

    # one leaf active per data point per forest
    assert X_transformed.shape[0] == X.shape[0]
    assert_array_equal(X_transformed.sum(axis=1), hasher.n_estimators)
    svd = TruncatedSVD(n_components=2)
    X_reduced = svd.fit_transform(X_transformed)
    linear_clf = LinearSVC()
    linear_clf.fit(X_reduced, y)
    assert linear_clf.score(X_reduced, y) == 1.0


def test_random_hasher_sparse_data():
    X, y = datasets.make_multilabel_classification(random_state=0)
    hasher = RandomTreesEmbedding(n_estimators=30, random_state=1)
    X_transformed = hasher.fit_transform(X)
    X_transformed_sparse = hasher.fit_transform(csc_matrix(X))
    assert_array_equal(X_transformed_sparse.toarray(), X_transformed.toarray())


def test_parallel_train():
    rng = check_random_state(12321)
    n_samples, n_features = 80, 30
    X_train = rng.randn(n_samples, n_features)
    y_train = rng.randint(0, 2, n_samples)

    clfs = [
        RandomForestClassifier(n_estimators=20, n_jobs=n_jobs, random_state=12345).fit(
            X_train, y_train
        )
        for n_jobs in [1, 2, 3, 8, 16, 32]
    ]

    X_test = rng.randn(n_samples, n_features)
    probas = [clf.predict_proba(X_test) for clf in clfs]
    for proba1, proba2 in zip(probas, probas[1:]):
        assert_array_almost_equal(proba1, proba2)


def test_distribution():
    rng = check_random_state(12321)

    # Single variable with 4 values
    X = rng.randint(0, 4, size=(1000, 1))
    y = rng.rand(1000)
    n_trees = 500

    reg = ExtraTreesRegressor(n_estimators=n_trees, random_state=42).fit(X, y)

    uniques = defaultdict(int)
    for tree in reg.estimators_:
        tree = "".join(
            ("%d,%d/" % (f, int(t)) if f >= 0 else "-")
            for f, t in zip(tree.tree_.feature, tree.tree_.threshold)
        )

        uniques[tree] += 1

    uniques = sorted([(1.0 * count / n_trees, tree) for tree, count in uniques.items()])

    # On a single variable problem where X_0 has 4 equiprobable values, there
    # are 5 ways to build a random tree. The more compact (0,1/0,0/--0,2/--) of
    # them has probability 1/3 while the 4 others have probability 1/6.

    assert len(uniques) == 5
    assert 0.20 > uniques[0][0]  # Rough approximation of 1/6.
    assert 0.20 > uniques[1][0]
    assert 0.20 > uniques[2][0]
    assert 0.20 > uniques[3][0]
    assert uniques[4][0] > 0.3
    assert uniques[4][1] == "0,1/0,0/--0,2/--"

    # Two variables, one with 2 values, one with 3 values
    X = np.empty((1000, 2))
    X[:, 0] = np.random.randint(0, 2, 1000)
    X[:, 1] = np.random.randint(0, 3, 1000)
    y = rng.rand(1000)

    reg = ExtraTreesRegressor(max_features=1, random_state=1).fit(X, y)

    uniques = defaultdict(int)
    for tree in reg.estimators_:
        tree = "".join(
            ("%d,%d/" % (f, int(t)) if f >= 0 else "-")
            for f, t in zip(tree.tree_.feature, tree.tree_.threshold)
        )

        uniques[tree] += 1

    uniques = [(count, tree) for tree, count in uniques.items()]
    assert len(uniques) == 8


def check_max_leaf_nodes_max_depth(name):
    X, y = hastie_X, hastie_y

    # Test precedence of max_leaf_nodes over max_depth.
    ForestEstimator = FOREST_ESTIMATORS[name]
    est = ForestEstimator(
        max_depth=1, max_leaf_nodes=4, n_estimators=1, random_state=0
    ).fit(X, y)
    assert est.estimators_[0].get_depth() == 1

    est = ForestEstimator(max_depth=1, n_estimators=1, random_state=0).fit(X, y)
    assert est.estimators_[0].get_depth() == 1


@pytest.mark.parametrize("name", FOREST_ESTIMATORS)
def test_max_leaf_nodes_max_depth(name):
    check_max_leaf_nodes_max_depth(name)


def check_min_samples_split(name):
    X, y = hastie_X, hastie_y
    ForestEstimator = FOREST_ESTIMATORS[name]

    # test boundary value
    with pytest.raises(ValueError):
        ForestEstimator(min_samples_split=-1).fit(X, y)
    with pytest.raises(ValueError):
        ForestEstimator(min_samples_split=0).fit(X, y)
    with pytest.raises(ValueError):
        ForestEstimator(min_samples_split=1.1).fit(X, y)

    est = ForestEstimator(min_samples_split=10, n_estimators=1, random_state=0)
    est.fit(X, y)
    node_idx = est.estimators_[0].tree_.children_left != -1
    node_samples = est.estimators_[0].tree_.n_node_samples[node_idx]

    assert np.min(node_samples) > len(X) * 0.5 - 1, "Failed with {0}".format(name)

    est = ForestEstimator(min_samples_split=0.5, n_estimators=1, random_state=0)
    est.fit(X, y)
    node_idx = est.estimators_[0].tree_.children_left != -1
    node_samples = est.estimators_[0].tree_.n_node_samples[node_idx]

    assert np.min(node_samples) > len(X) * 0.5 - 1, "Failed with {0}".format(name)


@pytest.mark.parametrize("name", FOREST_ESTIMATORS)
def test_min_samples_split(name):
    check_min_samples_split(name)


def check_min_samples_leaf(name):
    X, y = hastie_X, hastie_y

    # Test if leaves contain more than leaf_count training examples
    ForestEstimator = FOREST_ESTIMATORS[name]

    # test boundary value
    with pytest.raises(ValueError):
        ForestEstimator(min_samples_leaf=-1).fit(X, y)
    with pytest.raises(ValueError):
        ForestEstimator(min_samples_leaf=0).fit(X, y)

    est = ForestEstimator(min_samples_leaf=5, n_estimators=1, random_state=0)
    est.fit(X, y)
    out = est.estimators_[0].tree_.apply(X)
    node_counts = np.bincount(out)
    # drop inner nodes
    leaf_count = node_counts[node_counts != 0]
    assert np.min(leaf_count) > 4, "Failed with {0}".format(name)

    est = ForestEstimator(min_samples_leaf=0.25, n_estimators=1, random_state=0)
    est.fit(X, y)
    out = est.estimators_[0].tree_.apply(X)
    node_counts = np.bincount(out)
    # drop inner nodes
    leaf_count = node_counts[node_counts != 0]
    assert np.min(leaf_count) > len(X) * 0.25 - 1, "Failed with {0}".format(name)


@pytest.mark.parametrize("name", FOREST_ESTIMATORS)
def test_min_samples_leaf(name):
    check_min_samples_leaf(name)


def check_min_weight_fraction_leaf(name):
    X, y = hastie_X, hastie_y

    # Test if leaves contain at least min_weight_fraction_leaf of the
    # training set
    ForestEstimator = FOREST_ESTIMATORS[name]
    rng = np.random.RandomState(0)
    weights = rng.rand(X.shape[0])
    total_weight = np.sum(weights)

    # test both DepthFirstTreeBuilder and BestFirstTreeBuilder
    # by setting max_leaf_nodes
    for frac in np.linspace(0, 0.5, 6):
        est = ForestEstimator(
            min_weight_fraction_leaf=frac, n_estimators=1, random_state=0
        )
        if "RandomForest" in name:
            est.bootstrap = False

        est.fit(X, y, sample_weight=weights)
        out = est.estimators_[0].tree_.apply(X)
        node_weights = np.bincount(out, weights=weights)
        # drop inner nodes
        leaf_weights = node_weights[node_weights != 0]
        assert (
            np.min(leaf_weights) >= total_weight * est.min_weight_fraction_leaf
        ), "Failed with {0} min_weight_fraction_leaf={1}".format(
            name, est.min_weight_fraction_leaf
        )


@pytest.mark.parametrize("name", FOREST_ESTIMATORS)
def test_min_weight_fraction_leaf(name):
    check_min_weight_fraction_leaf(name)


def check_sparse_input(name, X, X_sparse, y):
    ForestEstimator = FOREST_ESTIMATORS[name]

    dense = ForestEstimator(random_state=0, max_depth=2).fit(X, y)
    sparse = ForestEstimator(random_state=0, max_depth=2).fit(X_sparse, y)

    assert_array_almost_equal(sparse.apply(X), dense.apply(X))

    if name in FOREST_CLASSIFIERS or name in FOREST_REGRESSORS:
        assert_array_almost_equal(sparse.predict(X), dense.predict(X))
        assert_array_almost_equal(
            sparse.feature_importances_, dense.feature_importances_
        )

    if name in FOREST_CLASSIFIERS:
        assert_array_almost_equal(sparse.predict_proba(X), dense.predict_proba(X))
        assert_array_almost_equal(
            sparse.predict_log_proba(X), dense.predict_log_proba(X)
        )

    if name in FOREST_TRANSFORMERS:
        assert_array_almost_equal(
            sparse.transform(X).toarray(), dense.transform(X).toarray()
        )
        assert_array_almost_equal(
            sparse.fit_transform(X).toarray(), dense.fit_transform(X).toarray()
        )


@pytest.mark.parametrize("name", FOREST_ESTIMATORS)
@pytest.mark.parametrize("sparse_matrix", (csr_matrix, csc_matrix, coo_matrix))
def test_sparse_input(name, sparse_matrix):
    X, y = datasets.make_multilabel_classification(random_state=0, n_samples=50)

    check_sparse_input(name, X, sparse_matrix(X), y)


def check_memory_layout(name, dtype):
    # Check that it works no matter the memory layout

    est = FOREST_ESTIMATORS[name](random_state=0, bootstrap=False)

    # Nothing
    X = np.asarray(iris.data, dtype=dtype)
    y = iris.target
    assert_array_almost_equal(est.fit(X, y).predict(X), y)

    # C-order
    X = np.asarray(iris.data, order="C", dtype=dtype)
    y = iris.target
    assert_array_almost_equal(est.fit(X, y).predict(X), y)

    # F-order
    X = np.asarray(iris.data, order="F", dtype=dtype)
    y = iris.target
    assert_array_almost_equal(est.fit(X, y).predict(X), y)

    # Contiguous
    X = np.ascontiguousarray(iris.data, dtype=dtype)
    y = iris.target
    assert_array_almost_equal(est.fit(X, y).predict(X), y)

    if est.base_estimator.splitter in SPARSE_SPLITTERS:
        # csr matrix
        X = csr_matrix(iris.data, dtype=dtype)
        y = iris.target
        assert_array_almost_equal(est.fit(X, y).predict(X), y)

        # csc_matrix
        X = csc_matrix(iris.data, dtype=dtype)
        y = iris.target
        assert_array_almost_equal(est.fit(X, y).predict(X), y)

        # coo_matrix
        X = coo_matrix(iris.data, dtype=dtype)
        y = iris.target
        assert_array_almost_equal(est.fit(X, y).predict(X), y)

    # Strided
    X = np.asarray(iris.data[::3], dtype=dtype)
    y = iris.target[::3]
    assert_array_almost_equal(est.fit(X, y).predict(X), y)


@pytest.mark.parametrize("name", FOREST_CLASSIFIERS_REGRESSORS)
@pytest.mark.parametrize("dtype", (np.float64, np.float32))
def test_memory_layout(name, dtype):
    check_memory_layout(name, dtype)


@ignore_warnings
def check_1d_input(name, X, X_2d, y):
    ForestEstimator = FOREST_ESTIMATORS[name]
    with pytest.raises(ValueError):
        ForestEstimator(n_estimators=1, random_state=0).fit(X, y)

    est = ForestEstimator(random_state=0)
    est.fit(X_2d, y)

    if name in FOREST_CLASSIFIERS or name in FOREST_REGRESSORS:
        with pytest.raises(ValueError):
            est.predict(X)


@pytest.mark.parametrize("name", FOREST_ESTIMATORS)
def test_1d_input(name):
    X = iris.data[:, 0]
    X_2d = iris.data[:, 0].reshape((-1, 1))
    y = iris.target

    with ignore_warnings():
        check_1d_input(name, X, X_2d, y)


def check_class_weights(name):
    # Check class_weights resemble sample_weights behavior.
    ForestClassifier = FOREST_CLASSIFIERS[name]

    # Iris is balanced, so no effect expected for using 'balanced' weights
    clf1 = ForestClassifier(random_state=0)
    clf1.fit(iris.data, iris.target)
    clf2 = ForestClassifier(class_weight="balanced", random_state=0)
    clf2.fit(iris.data, iris.target)
    assert_almost_equal(clf1.feature_importances_, clf2.feature_importances_)

    # Make a multi-output problem with three copies of Iris
    iris_multi = np.vstack((iris.target, iris.target, iris.target)).T
    # Create user-defined weights that should balance over the outputs
    clf3 = ForestClassifier(
        class_weight=[
            {0: 2.0, 1: 2.0, 2: 1.0},
            {0: 2.0, 1: 1.0, 2: 2.0},
            {0: 1.0, 1: 2.0, 2: 2.0},
        ],
        random_state=0,
    )
    clf3.fit(iris.data, iris_multi)
    assert_almost_equal(clf2.feature_importances_, clf3.feature_importances_)
    # Check against multi-output "balanced" which should also have no effect
    clf4 = ForestClassifier(class_weight="balanced", random_state=0)
    clf4.fit(iris.data, iris_multi)
    assert_almost_equal(clf3.feature_importances_, clf4.feature_importances_)

    # Inflate importance of class 1, check against user-defined weights
    sample_weight = np.ones(iris.target.shape)
    sample_weight[iris.target == 1] *= 100
    class_weight = {0: 1.0, 1: 100.0, 2: 1.0}
    clf1 = ForestClassifier(random_state=0)
    clf1.fit(iris.data, iris.target, sample_weight)
    clf2 = ForestClassifier(class_weight=class_weight, random_state=0)
    clf2.fit(iris.data, iris.target)
    assert_almost_equal(clf1.feature_importances_, clf2.feature_importances_)

    # Check that sample_weight and class_weight are multiplicative
    clf1 = ForestClassifier(random_state=0)
    clf1.fit(iris.data, iris.target, sample_weight ** 2)
    clf2 = ForestClassifier(class_weight=class_weight, random_state=0)
    clf2.fit(iris.data, iris.target, sample_weight)
    assert_almost_equal(clf1.feature_importances_, clf2.feature_importances_)


@pytest.mark.parametrize("name", FOREST_CLASSIFIERS)
def test_class_weights(name):
    check_class_weights(name)


def check_class_weight_balanced_and_bootstrap_multi_output(name):
    # Test class_weight works for multi-output"""
    ForestClassifier = FOREST_CLASSIFIERS[name]
    _y = np.vstack((y, np.array(y) * 2)).T
    clf = ForestClassifier(class_weight="balanced", random_state=0)
    clf.fit(X, _y)
    clf = ForestClassifier(
        class_weight=[{-1: 0.5, 1: 1.0}, {-2: 1.0, 2: 1.0}], random_state=0
    )
    clf.fit(X, _y)
    # smoke test for balanced subsample
    clf = ForestClassifier(class_weight="balanced_subsample", random_state=0)
    clf.fit(X, _y)


@pytest.mark.parametrize("name", FOREST_CLASSIFIERS)
def test_class_weight_balanced_and_bootstrap_multi_output(name):
    check_class_weight_balanced_and_bootstrap_multi_output(name)


def check_class_weight_errors(name):
    # Test if class_weight raises errors and warnings when expected.
    ForestClassifier = FOREST_CLASSIFIERS[name]
    _y = np.vstack((y, np.array(y) * 2)).T

    # Invalid preset string
    clf = ForestClassifier(class_weight="the larch", random_state=0)
    with pytest.raises(ValueError):
        clf.fit(X, y)
    with pytest.raises(ValueError):
        clf.fit(X, _y)

    # Warning warm_start with preset
    clf = ForestClassifier(class_weight="balanced", warm_start=True, random_state=0)
    clf.fit(X, y)

    warn_msg = (
        "Warm-start fitting without increasing n_estimators does not fit new trees."
    )
    with pytest.warns(UserWarning, match=warn_msg):
        clf.fit(X, _y)

    # Not a list or preset for multi-output
    clf = ForestClassifier(class_weight=1, random_state=0)
    with pytest.raises(ValueError):
        clf.fit(X, _y)

    # Incorrect length list for multi-output
    clf = ForestClassifier(class_weight=[{-1: 0.5, 1: 1.0}], random_state=0)
    with pytest.raises(ValueError):
        clf.fit(X, _y)


@pytest.mark.parametrize("name", FOREST_CLASSIFIERS)
def test_class_weight_errors(name):
    check_class_weight_errors(name)


def check_warm_start(name, random_state=42):
    # Test if fitting incrementally with warm start gives a forest of the
    # right size and the same results as a normal fit.
    X, y = hastie_X, hastie_y
    ForestEstimator = FOREST_ESTIMATORS[name]
    est_ws = None
    for n_estimators in [5, 10]:
        if est_ws is None:
            est_ws = ForestEstimator(
                n_estimators=n_estimators, random_state=random_state, warm_start=True
            )
        else:
            est_ws.set_params(n_estimators=n_estimators)
        est_ws.fit(X, y)
        assert len(est_ws) == n_estimators

    est_no_ws = ForestEstimator(
        n_estimators=10, random_state=random_state, warm_start=False
    )
    est_no_ws.fit(X, y)

    assert set([tree.random_state for tree in est_ws]) == set(
        [tree.random_state for tree in est_no_ws]
    )

    assert_array_equal(
        est_ws.apply(X), est_no_ws.apply(X), err_msg="Failed with {0}".format(name)
    )


@pytest.mark.parametrize("name", FOREST_ESTIMATORS)
def test_warm_start(name):
    check_warm_start(name)


def check_warm_start_clear(name):
    # Test if fit clears state and grows a new forest when warm_start==False.
    X, y = hastie_X, hastie_y
    ForestEstimator = FOREST_ESTIMATORS[name]
    est = ForestEstimator(n_estimators=5, max_depth=1, warm_start=False, random_state=1)
    est.fit(X, y)

    est_2 = ForestEstimator(
        n_estimators=5, max_depth=1, warm_start=True, random_state=2
    )
    est_2.fit(X, y)  # inits state
    est_2.set_params(warm_start=False, random_state=1)
    est_2.fit(X, y)  # clears old state and equals est

    assert_array_almost_equal(est_2.apply(X), est.apply(X))


@pytest.mark.parametrize("name", FOREST_ESTIMATORS)
def test_warm_start_clear(name):
    check_warm_start_clear(name)


def check_warm_start_smaller_n_estimators(name):
    # Test if warm start second fit with smaller n_estimators raises error.
    X, y = hastie_X, hastie_y
    ForestEstimator = FOREST_ESTIMATORS[name]
    est = ForestEstimator(n_estimators=5, max_depth=1, warm_start=True)
    est.fit(X, y)
    est.set_params(n_estimators=4)
    with pytest.raises(ValueError):
        est.fit(X, y)


@pytest.mark.parametrize("name", FOREST_ESTIMATORS)
def test_warm_start_smaller_n_estimators(name):
    check_warm_start_smaller_n_estimators(name)


def check_warm_start_equal_n_estimators(name):
    # Test if warm start with equal n_estimators does nothing and returns the
    # same forest and raises a warning.
    X, y = hastie_X, hastie_y
    ForestEstimator = FOREST_ESTIMATORS[name]
    est = ForestEstimator(n_estimators=5, max_depth=3, warm_start=True, random_state=1)
    est.fit(X, y)

    est_2 = ForestEstimator(
        n_estimators=5, max_depth=3, warm_start=True, random_state=1
    )
    est_2.fit(X, y)
    # Now est_2 equals est.

    est_2.set_params(random_state=2)
    warn_msg = (
        "Warm-start fitting without increasing n_estimators does not fit new trees."
    )
    with pytest.warns(UserWarning, match=warn_msg):
        est_2.fit(X, y)
    # If we had fit the trees again we would have got a different forest as we
    # changed the random state.
    assert_array_equal(est.apply(X), est_2.apply(X))


@pytest.mark.parametrize("name", FOREST_ESTIMATORS)
def test_warm_start_equal_n_estimators(name):
    check_warm_start_equal_n_estimators(name)


def check_warm_start_oob(name):
    # Test that the warm start computes oob score when asked.
    X, y = hastie_X, hastie_y
    ForestEstimator = FOREST_ESTIMATORS[name]
    # Use 15 estimators to avoid 'some inputs do not have OOB scores' warning.
    est = ForestEstimator(
        n_estimators=15,
        max_depth=3,
        warm_start=False,
        random_state=1,
        bootstrap=True,
        oob_score=True,
    )
    est.fit(X, y)

    est_2 = ForestEstimator(
        n_estimators=5,
        max_depth=3,
        warm_start=False,
        random_state=1,
        bootstrap=True,
        oob_score=False,
    )
    est_2.fit(X, y)

    est_2.set_params(warm_start=True, oob_score=True, n_estimators=15)
    est_2.fit(X, y)

    assert hasattr(est_2, "oob_score_")
    assert est.oob_score_ == est_2.oob_score_

    # Test that oob_score is computed even if we don't need to train
    # additional trees.
    est_3 = ForestEstimator(
        n_estimators=15,
        max_depth=3,
        warm_start=True,
        random_state=1,
        bootstrap=True,
        oob_score=False,
    )
    est_3.fit(X, y)
    assert not hasattr(est_3, "oob_score_")

    est_3.set_params(oob_score=True)
    ignore_warnings(est_3.fit)(X, y)

    assert est.oob_score_ == est_3.oob_score_


@pytest.mark.parametrize("name", FOREST_CLASSIFIERS_REGRESSORS)
def test_warm_start_oob(name):
    check_warm_start_oob(name)


def test_dtype_convert(n_classes=15):
    classifier = RandomForestClassifier(random_state=0, bootstrap=False)

    X = np.eye(n_classes)
    y = [ch for ch in "ABCDEFGHIJKLMNOPQRSTU"[:n_classes]]

    result = classifier.fit(X, y).predict(X)
    assert_array_equal(classifier.classes_, y)
    assert_array_equal(result, y)


def check_decision_path(name):
    X, y = hastie_X, hastie_y
    n_samples = X.shape[0]
    ForestEstimator = FOREST_ESTIMATORS[name]
    est = ForestEstimator(n_estimators=5, max_depth=1, warm_start=False, random_state=1)
    est.fit(X, y)
    indicator, n_nodes_ptr = est.decision_path(X)

    assert indicator.shape[1] == n_nodes_ptr[-1]
    assert indicator.shape[0] == n_samples
    assert_array_equal(
        np.diff(n_nodes_ptr), [e.tree_.node_count for e in est.estimators_]
    )

    # Assert that leaves index are correct
    leaves = est.apply(X)
    for est_id in range(leaves.shape[1]):
        leave_indicator = [
            indicator[i, n_nodes_ptr[est_id] + j]
            for i, j in enumerate(leaves[:, est_id])
        ]
        assert_array_almost_equal(leave_indicator, np.ones(shape=n_samples))


@pytest.mark.parametrize("name", FOREST_CLASSIFIERS_REGRESSORS)
def test_decision_path(name):
    check_decision_path(name)


def test_min_impurity_decrease():
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)
    all_estimators = [
        RandomForestClassifier,
        RandomForestRegressor,
        ExtraTreesClassifier,
        ExtraTreesRegressor,
    ]

    for Estimator in all_estimators:
        est = Estimator(min_impurity_decrease=0.1)
        est.fit(X, y)
        for tree in est.estimators_:
            # Simply check if the parameter is passed on correctly. Tree tests
            # will suffice for the actual working of this param
            assert tree.min_impurity_decrease == 0.1


def test_poisson_y_positive_check():
    est = RandomForestRegressor(criterion="poisson")
    X = np.zeros((3, 3))

    y = [-1, 1, 3]
    err_msg = (
        r"Some value\(s\) of y are negative which is "
        r"not allowed for Poisson regression."
    )
    with pytest.raises(ValueError, match=err_msg):
        est.fit(X, y)

    y = [0, 0, 0]
    err_msg = (
        r"Sum of y is not strictly positive which "
        r"is necessary for Poisson regression."
    )
    with pytest.raises(ValueError, match=err_msg):
        est.fit(X, y)


# mypy error: Variable "DEFAULT_JOBLIB_BACKEND" is not valid type
class MyBackend(DEFAULT_JOBLIB_BACKEND):  # type: ignore
    def __init__(self, *args, **kwargs):
        self.count = 0
        super().__init__(*args, **kwargs)

    def start_call(self):
        self.count += 1
        return super().start_call()


joblib.register_parallel_backend("testing", MyBackend)


@pytest.mark.skipif(
    parse_version(joblib.__version__) < parse_version("0.12"),
    reason="tests not yet supported in joblib <0.12",
)
@skip_if_no_parallel
def test_backend_respected():
    clf = RandomForestClassifier(n_estimators=10, n_jobs=2)

    with joblib.parallel_backend("testing") as (ba, n_jobs):
        clf.fit(X, y)

    assert ba.count > 0

    # predict_proba requires shared memory. Ensure that's honored.
    with joblib.parallel_backend("testing") as (ba, _):
        clf.predict_proba(X)

    assert ba.count == 0


def test_forest_feature_importances_sum():
    X, y = make_classification(
        n_samples=15, n_informative=3, random_state=1, n_classes=3
    )
    clf = RandomForestClassifier(
        min_samples_leaf=5, random_state=42, n_estimators=200
    ).fit(X, y)
    assert math.isclose(1, clf.feature_importances_.sum(), abs_tol=1e-7)


def test_forest_degenerate_feature_importances():
    # build a forest of single node trees. See #13636
    X = np.zeros((10, 10))
    y = np.ones((10,))
    gbr = RandomForestRegressor(n_estimators=10).fit(X, y)
    assert_array_equal(gbr.feature_importances_, np.zeros(10, dtype=np.float64))


@pytest.mark.parametrize("name", FOREST_CLASSIFIERS_REGRESSORS)
def test_max_samples_bootstrap(name):
    # Check invalid `max_samples` values
    est = FOREST_CLASSIFIERS_REGRESSORS[name](bootstrap=False, max_samples=0.5)
    err_msg = (
        r"`max_sample` cannot be set if `bootstrap=False`. "
        r"Either switch to `bootstrap=True` or set "
        r"`max_sample=None`."
    )
    with pytest.raises(ValueError, match=err_msg):
        est.fit(X, y)


@pytest.mark.parametrize("name", FOREST_CLASSIFIERS_REGRESSORS)
@pytest.mark.parametrize(
    "max_samples, exc_type, exc_msg",
    [
        (
            int(1e9),
            ValueError,
            "`max_samples` must be in range 1 to 6 but got value 1000000000",
        ),
        (
            2.0,
            ValueError,
            r"`max_samples` must be in range \(0.0, 1.0\] but got value 2.0",
        ),
        (
            0.0,
            ValueError,
            r"`max_samples` must be in range \(0.0, 1.0\] but got value 0.0",
        ),
        (
            np.nan,
            ValueError,
            r"`max_samples` must be in range \(0.0, 1.0\] but got value nan",
        ),
        (
            np.inf,
            ValueError,
            r"`max_samples` must be in range \(0.0, 1.0\] but got value inf",
        ),
        (
            "str max_samples?!",
            TypeError,
            r"`max_samples` should be int or float, but got " r"type '\<class 'str'\>'",
        ),
        (
            np.ones(2),
            TypeError,
            r"`max_samples` should be int or float, but got type "
            r"'\<class 'numpy.ndarray'\>'",
        ),
    ],
    # Avoid long error messages in test names:
    # https://github.com/scikit-learn/scikit-learn/issues/21362
    ids=lambda x: x[:10].replace("]", "") if isinstance(x, str) else x,
)
def test_max_samples_exceptions(name, max_samples, exc_type, exc_msg):
    # Check invalid `max_samples` values
    est = FOREST_CLASSIFIERS_REGRESSORS[name](bootstrap=True, max_samples=max_samples)
    with pytest.raises(exc_type, match=exc_msg):
        est.fit(X, y)


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
def test_max_samples_boundary_regressors(name):
    X_train, X_test, y_train, y_test = train_test_split(
        X_reg, y_reg, train_size=0.7, test_size=0.3, random_state=0
    )

    ms_1_model = FOREST_REGRESSORS[name](
        bootstrap=True, max_samples=1.0, random_state=0
    )
    ms_1_predict = ms_1_model.fit(X_train, y_train).predict(X_test)

    ms_None_model = FOREST_REGRESSORS[name](
        bootstrap=True, max_samples=None, random_state=0
    )
    ms_None_predict = ms_None_model.fit(X_train, y_train).predict(X_test)

    ms_1_ms = mean_squared_error(ms_1_predict, y_test)
    ms_None_ms = mean_squared_error(ms_None_predict, y_test)

    assert ms_1_ms == pytest.approx(ms_None_ms)


@pytest.mark.parametrize("name", FOREST_CLASSIFIERS)
def test_max_samples_boundary_classifiers(name):
    X_train, X_test, y_train, _ = train_test_split(
        X_large, y_large, random_state=0, stratify=y_large
    )

    ms_1_model = FOREST_CLASSIFIERS[name](
        bootstrap=True, max_samples=1.0, random_state=0
    )
    ms_1_proba = ms_1_model.fit(X_train, y_train).predict_proba(X_test)

    ms_None_model = FOREST_CLASSIFIERS[name](
        bootstrap=True, max_samples=None, random_state=0
    )
    ms_None_proba = ms_None_model.fit(X_train, y_train).predict_proba(X_test)

    np.testing.assert_allclose(ms_1_proba, ms_None_proba)


def test_forest_y_sparse():
    X = [[1, 2, 3]]
    y = csr_matrix([4, 5, 6])
    est = RandomForestClassifier()
    msg = "sparse multilabel-indicator for y is not supported."
    with pytest.raises(ValueError, match=msg):
        est.fit(X, y)


@pytest.mark.parametrize("ForestClass", [RandomForestClassifier, RandomForestRegressor])
def test_little_tree_with_small_max_samples(ForestClass):
    rng = np.random.RandomState(1)

    X = rng.randn(10000, 2)
    y = rng.randn(10000) > 0

    # First fit with no restriction on max samples
    est1 = ForestClass(
        n_estimators=1,
        random_state=rng,
        max_samples=None,
    )

    # Second fit with max samples restricted to just 2
    est2 = ForestClass(
        n_estimators=1,
        random_state=rng,
        max_samples=2,
    )

    est1.fit(X, y)
    est2.fit(X, y)

    tree1 = est1.estimators_[0].tree_
    tree2 = est2.estimators_[0].tree_

    msg = "Tree without `max_samples` restriction should have more nodes"
    assert tree1.node_count > tree2.node_count, msg


# FIXME: remove in 1.2
@pytest.mark.parametrize(
    "Estimator",
    [
        ExtraTreesClassifier,
        ExtraTreesRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
        RandomTreesEmbedding,
    ],
)
def test_n_features_deprecation(Estimator):
    # Check that we raise the proper deprecation warning if accessing
    # `n_features_`.
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 0])
    est = Estimator().fit(X, y)

    with pytest.warns(FutureWarning, match="`n_features_` was deprecated"):
        est.n_features_


# TODO: Remove in v1.2
@pytest.mark.parametrize(
    "old_criterion, new_criterion",
    [
        ("mse", "squared_error"),
        ("mae", "absolute_error"),
    ],
)
def test_criterion_deprecated(old_criterion, new_criterion):
    est1 = RandomForestRegressor(criterion=old_criterion, random_state=0)

    with pytest.warns(
        FutureWarning, match=f"Criterion '{old_criterion}' was deprecated"
    ):
        est1.fit(X, y)

    est2 = RandomForestRegressor(criterion=new_criterion, random_state=0)
    est2.fit(X, y)
    assert_allclose(est1.predict(X), est2.predict(X))


@pytest.mark.parametrize("Forest", FOREST_REGRESSORS)
def test_mse_criterion_object_segfault_smoke_test(Forest):
    # This is a smoke test to ensure that passing a mutable criterion
    # does not cause a segfault when fitting with concurrent threads.
    # Non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/12623
    from sklearn.tree._criterion import MSE

    y = y_reg.reshape(-1, 1)
    n_samples, n_outputs = y.shape
    mse_criterion = MSE(n_outputs, n_samples)
    est = FOREST_REGRESSORS[Forest](n_estimators=2, n_jobs=2, criterion=mse_criterion)

    est.fit(X_reg, y)
