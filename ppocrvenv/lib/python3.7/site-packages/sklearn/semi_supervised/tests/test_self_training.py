from math import ceil

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from sklearn.ensemble import StackingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, make_blobs
from sklearn.metrics import accuracy_score

from sklearn.semi_supervised import SelfTrainingClassifier

# Author: Oliver Rausch <rauscho@ethz.ch>
# License: BSD 3 clause

# load the iris dataset and randomly permute it
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, random_state=0
)

n_labeled_samples = 50

y_train_missing_labels = y_train.copy()
y_train_missing_labels[n_labeled_samples:] = -1
mapping = {0: "A", 1: "B", 2: "C", -1: "-1"}
y_train_missing_strings = np.vectorize(mapping.get)(y_train_missing_labels).astype(
    object
)
y_train_missing_strings[y_train_missing_labels == -1] = -1


def test_missing_predict_proba():
    # Check that an error is thrown if predict_proba is not implemented
    base_estimator = SVC(probability=False, gamma="scale")
    self_training = SelfTrainingClassifier(base_estimator)

    with pytest.raises(ValueError, match=r"base_estimator \(SVC\) should"):
        self_training.fit(X_train, y_train_missing_labels)


def test_none_classifier():
    st = SelfTrainingClassifier(None)
    with pytest.raises(ValueError, match="base_estimator cannot be None"):
        st.fit(X_train, y_train_missing_labels)


@pytest.mark.parametrize("max_iter, threshold", [(-1, 1.0), (-100, -2), (-10, 10)])
def test_invalid_params(max_iter, threshold):
    # Test negative iterations
    base_estimator = SVC(gamma="scale", probability=True)
    st = SelfTrainingClassifier(base_estimator, max_iter=max_iter)
    with pytest.raises(ValueError, match="max_iter must be >= 0 or None"):
        st.fit(X_train, y_train)

    base_estimator = SVC(gamma="scale", probability=True)
    st = SelfTrainingClassifier(base_estimator, threshold=threshold)
    with pytest.raises(ValueError, match="threshold must be in"):
        st.fit(X_train, y_train)


def test_invalid_params_selection_crit():
    st = SelfTrainingClassifier(KNeighborsClassifier(), criterion="foo")

    with pytest.raises(ValueError, match="criterion must be either"):
        st.fit(X_train, y_train)


def test_warns_k_best():
    st = SelfTrainingClassifier(KNeighborsClassifier(), criterion="k_best", k_best=1000)
    with pytest.warns(UserWarning, match="k_best is larger than"):
        st.fit(X_train, y_train_missing_labels)

    assert st.termination_condition_ == "all_labeled"


@pytest.mark.parametrize(
    "base_estimator",
    [KNeighborsClassifier(), SVC(gamma="scale", probability=True, random_state=0)],
)
@pytest.mark.parametrize("selection_crit", ["threshold", "k_best"])
def test_classification(base_estimator, selection_crit):
    # Check classification for various parameter settings.
    # Also assert that predictions for strings and numerical labels are equal.
    # Also test for multioutput classification
    threshold = 0.75
    max_iter = 10
    st = SelfTrainingClassifier(
        base_estimator, max_iter=max_iter, threshold=threshold, criterion=selection_crit
    )
    st.fit(X_train, y_train_missing_labels)
    pred = st.predict(X_test)
    proba = st.predict_proba(X_test)

    st_string = SelfTrainingClassifier(
        base_estimator, max_iter=max_iter, criterion=selection_crit, threshold=threshold
    )
    st_string.fit(X_train, y_train_missing_strings)
    pred_string = st_string.predict(X_test)
    proba_string = st_string.predict_proba(X_test)

    assert_array_equal(np.vectorize(mapping.get)(pred), pred_string)
    assert_array_equal(proba, proba_string)

    assert st.termination_condition_ == st_string.termination_condition_
    # Check consistency between labeled_iter, n_iter and max_iter
    labeled = y_train_missing_labels != -1
    # assert that labeled samples have labeled_iter = 0
    assert_array_equal(st.labeled_iter_ == 0, labeled)
    # assert that labeled samples do not change label during training
    assert_array_equal(y_train_missing_labels[labeled], st.transduction_[labeled])

    # assert that the max of the iterations is less than the total amount of
    # iterations
    assert np.max(st.labeled_iter_) <= st.n_iter_ <= max_iter
    assert np.max(st_string.labeled_iter_) <= st_string.n_iter_ <= max_iter

    # check shapes
    assert st.labeled_iter_.shape == st.transduction_.shape
    assert st_string.labeled_iter_.shape == st_string.transduction_.shape


def test_k_best():
    st = SelfTrainingClassifier(
        KNeighborsClassifier(n_neighbors=1),
        criterion="k_best",
        k_best=10,
        max_iter=None,
    )
    y_train_only_one_label = np.copy(y_train)
    y_train_only_one_label[1:] = -1
    n_samples = y_train.shape[0]

    n_expected_iter = ceil((n_samples - 1) / 10)
    st.fit(X_train, y_train_only_one_label)
    assert st.n_iter_ == n_expected_iter

    # Check labeled_iter_
    assert np.sum(st.labeled_iter_ == 0) == 1
    for i in range(1, n_expected_iter):
        assert np.sum(st.labeled_iter_ == i) == 10
    assert np.sum(st.labeled_iter_ == n_expected_iter) == (n_samples - 1) % 10
    assert st.termination_condition_ == "all_labeled"


def test_sanity_classification():
    base_estimator = SVC(gamma="scale", probability=True)
    base_estimator.fit(X_train[n_labeled_samples:], y_train[n_labeled_samples:])

    st = SelfTrainingClassifier(base_estimator)
    st.fit(X_train, y_train_missing_labels)

    pred1, pred2 = base_estimator.predict(X_test), st.predict(X_test)
    assert not np.array_equal(pred1, pred2)
    score_supervised = accuracy_score(base_estimator.predict(X_test), y_test)
    score_self_training = accuracy_score(st.predict(X_test), y_test)

    assert score_self_training > score_supervised


def test_none_iter():
    # Check that the all samples were labeled after a 'reasonable' number of
    # iterations.
    st = SelfTrainingClassifier(KNeighborsClassifier(), threshold=0.55, max_iter=None)
    st.fit(X_train, y_train_missing_labels)

    assert st.n_iter_ < 10
    assert st.termination_condition_ == "all_labeled"


@pytest.mark.parametrize(
    "base_estimator",
    [KNeighborsClassifier(), SVC(gamma="scale", probability=True, random_state=0)],
)
@pytest.mark.parametrize("y", [y_train_missing_labels, y_train_missing_strings])
def test_zero_iterations(base_estimator, y):
    # Check classification for zero iterations.
    # Fitting a SelfTrainingClassifier with zero iterations should give the
    # same results as fitting a supervised classifier.
    # This also asserts that string arrays work as expected.

    clf1 = SelfTrainingClassifier(base_estimator, max_iter=0)

    clf1.fit(X_train, y)

    clf2 = base_estimator.fit(X_train[:n_labeled_samples], y[:n_labeled_samples])

    assert_array_equal(clf1.predict(X_test), clf2.predict(X_test))
    assert clf1.termination_condition_ == "max_iter"


def test_prefitted_throws_error():
    # Test that passing a pre-fitted classifier and calling predict throws an
    # error
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    st = SelfTrainingClassifier(knn)
    with pytest.raises(
        NotFittedError,
        match="This SelfTrainingClassifier instance is not fitted yet",
    ):
        st.predict(X_train)


@pytest.mark.parametrize("max_iter", range(1, 5))
def test_labeled_iter(max_iter):
    # Check that the amount of datapoints labeled in iteration 0 is equal to
    # the amount of labeled datapoints we passed.
    st = SelfTrainingClassifier(KNeighborsClassifier(), max_iter=max_iter)

    st.fit(X_train, y_train_missing_labels)
    amount_iter_0 = len(st.labeled_iter_[st.labeled_iter_ == 0])
    assert amount_iter_0 == n_labeled_samples
    # Check that the max of the iterations is less than the total amount of
    # iterations
    assert np.max(st.labeled_iter_) <= st.n_iter_ <= max_iter


def test_no_unlabeled():
    # Test that training on a fully labeled dataset produces the same results
    # as training the classifier by itself.
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    st = SelfTrainingClassifier(knn)
    with pytest.warns(UserWarning, match="y contains no unlabeled samples"):
        st.fit(X_train, y_train)
    assert_array_equal(knn.predict(X_test), st.predict(X_test))
    # Assert that all samples were labeled in iteration 0 (since there were no
    # unlabeled samples).
    assert np.all(st.labeled_iter_ == 0)
    assert st.termination_condition_ == "all_labeled"


def test_early_stopping():
    svc = SVC(gamma="scale", probability=True)
    st = SelfTrainingClassifier(svc)
    X_train_easy = [[1], [0], [1], [0.5]]
    y_train_easy = [1, 0, -1, -1]
    # X = [[0.5]] cannot be predicted on with a high confidence, so training
    # stops early
    st.fit(X_train_easy, y_train_easy)
    assert st.n_iter_ == 1
    assert st.termination_condition_ == "no_change"


def test_strings_dtype():
    clf = SelfTrainingClassifier(KNeighborsClassifier())
    X, y = make_blobs(n_samples=30, random_state=0, cluster_std=0.1)
    labels_multiclass = ["one", "two", "three"]

    y_strings = np.take(labels_multiclass, y)

    with pytest.raises(ValueError, match="dtype"):
        clf.fit(X, y_strings)


@pytest.mark.parametrize("verbose", [True, False])
def test_verbose(capsys, verbose):
    clf = SelfTrainingClassifier(KNeighborsClassifier(), verbose=verbose)
    clf.fit(X_train, y_train_missing_labels)

    captured = capsys.readouterr()

    if verbose:
        assert "iteration" in captured.out
    else:
        assert "iteration" not in captured.out


def test_verbose_k_best(capsys):
    st = SelfTrainingClassifier(
        KNeighborsClassifier(n_neighbors=1),
        criterion="k_best",
        k_best=10,
        verbose=True,
        max_iter=None,
    )

    y_train_only_one_label = np.copy(y_train)
    y_train_only_one_label[1:] = -1
    n_samples = y_train.shape[0]

    n_expected_iter = ceil((n_samples - 1) / 10)
    st.fit(X_train, y_train_only_one_label)

    captured = capsys.readouterr()

    msg = "End of iteration {}, added {} new labels."
    for i in range(1, n_expected_iter):
        assert msg.format(i, 10) in captured.out

    assert msg.format(n_expected_iter, (n_samples - 1) % 10) in captured.out


def test_k_best_selects_best():
    # Tests that the labels added by st really are the 10 best labels.
    svc = SVC(gamma="scale", probability=True, random_state=0)
    st = SelfTrainingClassifier(svc, criterion="k_best", max_iter=1, k_best=10)
    has_label = y_train_missing_labels != -1
    st.fit(X_train, y_train_missing_labels)

    got_label = ~has_label & (st.transduction_ != -1)

    svc.fit(X_train[has_label], y_train_missing_labels[has_label])
    pred = svc.predict_proba(X_train[~has_label])
    max_proba = np.max(pred, axis=1)

    most_confident_svc = X_train[~has_label][np.argsort(max_proba)[-10:]]
    added_by_st = X_train[np.where(got_label)].tolist()

    for row in most_confident_svc.tolist():
        assert row in added_by_st


def test_base_estimator_meta_estimator():
    # Check that a meta-estimator relying on an estimator implementing
    # `predict_proba` will work even if it does expose this method before being
    # fitted.
    # Non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/19119

    base_estimator = StackingClassifier(
        estimators=[
            ("svc_1", SVC(probability=True)),
            ("svc_2", SVC(probability=True)),
        ],
        final_estimator=SVC(probability=True),
        cv=2,
    )

    # make sure that the `base_estimator` does not expose `predict_proba`
    # without being fitted
    assert not hasattr(base_estimator, "predict_proba")

    clf = SelfTrainingClassifier(base_estimator=base_estimator)
    clf.fit(X_train, y_train_missing_labels)
    clf.predict_proba(X_test)
