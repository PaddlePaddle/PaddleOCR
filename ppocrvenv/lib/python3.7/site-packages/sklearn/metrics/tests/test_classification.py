from functools import partial
from itertools import product
from itertools import chain
from itertools import permutations
import warnings
import re

import numpy as np
from scipy import linalg
import pytest

from sklearn import datasets
from sklearn import svm

from sklearn.datasets import make_multilabel_classification
from sklearn.preprocessing import label_binarize, LabelBinarizer
from sklearn.utils.validation import check_random_state
from sklearn.utils._testing import assert_almost_equal
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_allclose
from sklearn.utils._testing import assert_no_warnings
from sklearn.utils._testing import ignore_warnings
from sklearn.utils._mocking import MockDataFrame

from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import hinge_loss
from sklearn.metrics import jaccard_score
from sklearn.metrics import log_loss
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import zero_one_loss
from sklearn.metrics import brier_score_loss
from sklearn.metrics import multilabel_confusion_matrix

from sklearn.metrics._classification import _check_targets
from sklearn.exceptions import UndefinedMetricWarning

from scipy.spatial.distance import hamming as sp_hamming

###############################################################################
# Utilities for testing


def make_prediction(dataset=None, binary=False):
    """Make some classification predictions on a toy dataset using a SVC

    If binary is True restrict to a binary classification problem instead of a
    multiclass classification problem
    """

    if dataset is None:
        # import some data to play with
        dataset = datasets.load_iris()

    X = dataset.data
    y = dataset.target

    if binary:
        # restrict to a binary classification task
        X, y = X[y < 2], y[y < 2]

    n_samples, n_features = X.shape
    p = np.arange(n_samples)

    rng = check_random_state(37)
    rng.shuffle(p)
    X, y = X[p], y[p]
    half = int(n_samples / 2)

    # add noisy features to make the problem harder and avoid perfect results
    rng = np.random.RandomState(0)
    X = np.c_[X, rng.randn(n_samples, 200 * n_features)]

    # run classifier, get class probabilities and label predictions
    clf = svm.SVC(kernel="linear", probability=True, random_state=0)
    probas_pred = clf.fit(X[:half], y[:half]).predict_proba(X[half:])

    if binary:
        # only interested in probabilities of the positive case
        # XXX: do we really want a special API for the binary case?
        probas_pred = probas_pred[:, 1]

    y_pred = clf.predict(X[half:])
    y_true = y[half:]
    return y_true, y_pred, probas_pred


###############################################################################
# Tests


def test_classification_report_dictionary_output():

    # Test performance report with dictionary output
    iris = datasets.load_iris()
    y_true, y_pred, _ = make_prediction(dataset=iris, binary=False)

    # print classification report with class names
    expected_report = {
        "setosa": {
            "precision": 0.82608695652173914,
            "recall": 0.79166666666666663,
            "f1-score": 0.8085106382978724,
            "support": 24,
        },
        "versicolor": {
            "precision": 0.33333333333333331,
            "recall": 0.096774193548387094,
            "f1-score": 0.15000000000000002,
            "support": 31,
        },
        "virginica": {
            "precision": 0.41860465116279072,
            "recall": 0.90000000000000002,
            "f1-score": 0.57142857142857151,
            "support": 20,
        },
        "macro avg": {
            "f1-score": 0.5099797365754813,
            "precision": 0.5260083136726211,
            "recall": 0.596146953405018,
            "support": 75,
        },
        "accuracy": 0.5333333333333333,
        "weighted avg": {
            "f1-score": 0.47310435663627154,
            "precision": 0.5137535108414785,
            "recall": 0.5333333333333333,
            "support": 75,
        },
    }

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(iris.target_names)),
        target_names=iris.target_names,
        output_dict=True,
    )

    # assert the 2 dicts are equal.
    assert report.keys() == expected_report.keys()
    for key in expected_report:
        if key == "accuracy":
            assert isinstance(report[key], float)
            assert report[key] == expected_report[key]
        else:
            assert report[key].keys() == expected_report[key].keys()
            for metric in expected_report[key]:
                assert_almost_equal(expected_report[key][metric], report[key][metric])

    assert type(expected_report["setosa"]["precision"]) == float
    assert type(expected_report["macro avg"]["precision"]) == float
    assert type(expected_report["setosa"]["support"]) == int
    assert type(expected_report["macro avg"]["support"]) == int


def test_classification_report_output_dict_empty_input():
    report = classification_report(y_true=[], y_pred=[], output_dict=True)
    expected_report = {
        "accuracy": 0.0,
        "macro avg": {
            "f1-score": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "support": 0,
        },
        "weighted avg": {
            "f1-score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "support": 0,
        },
    }
    assert isinstance(report, dict)
    # assert the 2 dicts are equal.
    assert report.keys() == expected_report.keys()
    for key in expected_report:
        if key == "accuracy":
            assert isinstance(report[key], float)
            assert report[key] == expected_report[key]
        else:
            assert report[key].keys() == expected_report[key].keys()
            for metric in expected_report[key]:
                assert_almost_equal(expected_report[key][metric], report[key][metric])


@pytest.mark.parametrize("zero_division", ["warn", 0, 1])
def test_classification_report_zero_division_warning(zero_division):
    y_true, y_pred = ["a", "b", "c"], ["a", "b", "d"]
    with warnings.catch_warnings(record=True) as record:
        classification_report(
            y_true, y_pred, zero_division=zero_division, output_dict=True
        )
        if zero_division == "warn":
            assert len(record) > 1
            for item in record:
                msg = "Use `zero_division` parameter to control this behavior."
                assert msg in str(item.message)
        else:
            assert not record


def test_multilabel_accuracy_score_subset_accuracy():
    # Dense label indicator matrix format
    y1 = np.array([[0, 1, 1], [1, 0, 1]])
    y2 = np.array([[0, 0, 1], [1, 0, 1]])

    assert accuracy_score(y1, y2) == 0.5
    assert accuracy_score(y1, y1) == 1
    assert accuracy_score(y2, y2) == 1
    assert accuracy_score(y2, np.logical_not(y2)) == 0
    assert accuracy_score(y1, np.logical_not(y1)) == 0
    assert accuracy_score(y1, np.zeros(y1.shape)) == 0
    assert accuracy_score(y2, np.zeros(y1.shape)) == 0


def test_precision_recall_f1_score_binary():
    # Test Precision Recall and F1 Score for binary classification task
    y_true, y_pred, _ = make_prediction(binary=True)

    # detailed measures for each class
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
    assert_array_almost_equal(p, [0.73, 0.85], 2)
    assert_array_almost_equal(r, [0.88, 0.68], 2)
    assert_array_almost_equal(f, [0.80, 0.76], 2)
    assert_array_equal(s, [25, 25])

    # individual scoring function that can be used for grid search: in the
    # binary class case the score is the value of the measure for the positive
    # class (e.g. label == 1). This is deprecated for average != 'binary'.
    for kwargs, my_assert in [
        ({}, assert_no_warnings),
        ({"average": "binary"}, assert_no_warnings),
    ]:
        ps = my_assert(precision_score, y_true, y_pred, **kwargs)
        assert_array_almost_equal(ps, 0.85, 2)

        rs = my_assert(recall_score, y_true, y_pred, **kwargs)
        assert_array_almost_equal(rs, 0.68, 2)

        fs = my_assert(f1_score, y_true, y_pred, **kwargs)
        assert_array_almost_equal(fs, 0.76, 2)

        assert_almost_equal(
            my_assert(fbeta_score, y_true, y_pred, beta=2, **kwargs),
            (1 + 2 ** 2) * ps * rs / (2 ** 2 * ps + rs),
            2,
        )


@ignore_warnings
def test_precision_recall_f_binary_single_class():
    # Test precision, recall and F-scores behave with a single positive or
    # negative class
    # Such a case may occur with non-stratified cross-validation
    assert 1.0 == precision_score([1, 1], [1, 1])
    assert 1.0 == recall_score([1, 1], [1, 1])
    assert 1.0 == f1_score([1, 1], [1, 1])
    assert 1.0 == fbeta_score([1, 1], [1, 1], beta=0)

    assert 0.0 == precision_score([-1, -1], [-1, -1])
    assert 0.0 == recall_score([-1, -1], [-1, -1])
    assert 0.0 == f1_score([-1, -1], [-1, -1])
    assert 0.0 == fbeta_score([-1, -1], [-1, -1], beta=float("inf"))
    assert fbeta_score([-1, -1], [-1, -1], beta=float("inf")) == pytest.approx(
        fbeta_score([-1, -1], [-1, -1], beta=1e5)
    )


@ignore_warnings
def test_precision_recall_f_extra_labels():
    # Test handling of explicit additional (not in input) labels to PRF
    y_true = [1, 3, 3, 2]
    y_pred = [1, 1, 3, 2]
    y_true_bin = label_binarize(y_true, classes=np.arange(5))
    y_pred_bin = label_binarize(y_pred, classes=np.arange(5))
    data = [(y_true, y_pred), (y_true_bin, y_pred_bin)]

    for i, (y_true, y_pred) in enumerate(data):
        # No average: zeros in array
        actual = recall_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average=None)
        assert_array_almost_equal([0.0, 1.0, 1.0, 0.5, 0.0], actual)

        # Macro average is changed
        actual = recall_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average="macro")
        assert_array_almost_equal(np.mean([0.0, 1.0, 1.0, 0.5, 0.0]), actual)

        # No effect otherwise
        for average in ["micro", "weighted", "samples"]:
            if average == "samples" and i == 0:
                continue
            assert_almost_equal(
                recall_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average=average),
                recall_score(y_true, y_pred, labels=None, average=average),
            )

    # Error when introducing invalid label in multilabel case
    # (although it would only affect performance if average='macro'/None)
    for average in [None, "macro", "micro", "samples"]:
        with pytest.raises(ValueError):
            recall_score(y_true_bin, y_pred_bin, labels=np.arange(6), average=average)
        with pytest.raises(ValueError):
            recall_score(
                y_true_bin, y_pred_bin, labels=np.arange(-1, 4), average=average
            )

    # tests non-regression on issue #10307
    y_true = np.array([[0, 1, 1], [1, 0, 0]])
    y_pred = np.array([[1, 1, 1], [1, 0, 1]])
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="samples", labels=[0, 1]
    )
    assert_almost_equal(np.array([p, r, f]), np.array([3 / 4, 1, 5 / 6]))


@ignore_warnings
def test_precision_recall_f_ignored_labels():
    # Test a subset of labels may be requested for PRF
    y_true = [1, 1, 2, 3]
    y_pred = [1, 3, 3, 3]
    y_true_bin = label_binarize(y_true, classes=np.arange(5))
    y_pred_bin = label_binarize(y_pred, classes=np.arange(5))
    data = [(y_true, y_pred), (y_true_bin, y_pred_bin)]

    for i, (y_true, y_pred) in enumerate(data):
        recall_13 = partial(recall_score, y_true, y_pred, labels=[1, 3])
        recall_all = partial(recall_score, y_true, y_pred, labels=None)

        assert_array_almost_equal([0.5, 1.0], recall_13(average=None))
        assert_almost_equal((0.5 + 1.0) / 2, recall_13(average="macro"))
        assert_almost_equal((0.5 * 2 + 1.0 * 1) / 3, recall_13(average="weighted"))
        assert_almost_equal(2.0 / 3, recall_13(average="micro"))

        # ensure the above were meaningful tests:
        for average in ["macro", "weighted", "micro"]:
            assert recall_13(average=average) != recall_all(average=average)


def test_average_precision_score_score_non_binary_class():
    # Test that average_precision_score function returns an error when trying
    # to compute average_precision_score for multiclass task.
    rng = check_random_state(404)
    y_pred = rng.rand(10)

    # y_true contains three different class values
    y_true = rng.randint(0, 3, size=10)
    err_msg = "multiclass format is not supported"
    with pytest.raises(ValueError, match=err_msg):
        average_precision_score(y_true, y_pred)


def test_average_precision_score_duplicate_values():
    # Duplicate values with precision-recall require a different
    # processing than when computing the AUC of a ROC, because the
    # precision-recall curve is a decreasing curve
    # The following situation corresponds to a perfect
    # test statistic, the average_precision_score should be 1
    y_true = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    y_score = [0, 0.1, 0.1, 0.4, 0.5, 0.6, 0.6, 0.9, 0.9, 1, 1]
    assert average_precision_score(y_true, y_score) == 1


def test_average_precision_score_tied_values():
    # Here if we go from left to right in y_true, the 0 values are
    # are separated from the 1 values, so it appears that we've
    # Correctly sorted our classifications. But in fact the first two
    # values have the same score (0.5) and so the first two values
    # could be swapped around, creating an imperfect sorting. This
    # imperfection should come through in the end score, making it less
    # than one.
    y_true = [0, 1, 1]
    y_score = [0.5, 0.5, 0.6]
    assert average_precision_score(y_true, y_score) != 1.0


@ignore_warnings
def test_precision_recall_fscore_support_errors():
    y_true, y_pred, _ = make_prediction(binary=True)

    # Bad beta
    with pytest.raises(ValueError):
        precision_recall_fscore_support(y_true, y_pred, beta=-0.1)

    # Bad pos_label
    with pytest.raises(ValueError):
        precision_recall_fscore_support(y_true, y_pred, pos_label=2, average="binary")

    # Bad average option
    with pytest.raises(ValueError):
        precision_recall_fscore_support([0, 1, 2], [1, 2, 0], average="mega")


def test_precision_recall_f_unused_pos_label():
    # Check warning that pos_label unused when set to non-default value
    # but average != 'binary'; even if data is binary.

    msg = (
        r"Note that pos_label \(set to 2\) is "
        r"ignored when average != 'binary' \(got 'macro'\). You "
        r"may use labels=\[pos_label\] to specify a single "
        "positive class."
    )
    with pytest.warns(UserWarning, match=msg):
        precision_recall_fscore_support(
            [1, 2, 1], [1, 2, 2], pos_label=2, average="macro"
        )


def test_confusion_matrix_binary():
    # Test confusion matrix - binary classification case
    y_true, y_pred, _ = make_prediction(binary=True)

    def test(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        assert_array_equal(cm, [[22, 3], [8, 17]])

        tp, fp, fn, tn = cm.flatten()
        num = tp * tn - fp * fn
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        true_mcc = 0 if den == 0 else num / den
        mcc = matthews_corrcoef(y_true, y_pred)
        assert_array_almost_equal(mcc, true_mcc, decimal=2)
        assert_array_almost_equal(mcc, 0.57, decimal=2)

    test(y_true, y_pred)
    test([str(y) for y in y_true], [str(y) for y in y_pred])


def test_multilabel_confusion_matrix_binary():
    # Test multilabel confusion matrix - binary classification case
    y_true, y_pred, _ = make_prediction(binary=True)

    def test(y_true, y_pred):
        cm = multilabel_confusion_matrix(y_true, y_pred)
        assert_array_equal(cm, [[[17, 8], [3, 22]], [[22, 3], [8, 17]]])

    test(y_true, y_pred)
    test([str(y) for y in y_true], [str(y) for y in y_pred])


def test_multilabel_confusion_matrix_multiclass():
    # Test multilabel confusion matrix - multi-class case
    y_true, y_pred, _ = make_prediction(binary=False)

    def test(y_true, y_pred, string_type=False):
        # compute confusion matrix with default labels introspection
        cm = multilabel_confusion_matrix(y_true, y_pred)
        assert_array_equal(
            cm, [[[47, 4], [5, 19]], [[38, 6], [28, 3]], [[30, 25], [2, 18]]]
        )

        # compute confusion matrix with explicit label ordering
        labels = ["0", "2", "1"] if string_type else [0, 2, 1]
        cm = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
        assert_array_equal(
            cm, [[[47, 4], [5, 19]], [[30, 25], [2, 18]], [[38, 6], [28, 3]]]
        )

        # compute confusion matrix with super set of present labels
        labels = ["0", "2", "1", "3"] if string_type else [0, 2, 1, 3]
        cm = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
        assert_array_equal(
            cm,
            [
                [[47, 4], [5, 19]],
                [[30, 25], [2, 18]],
                [[38, 6], [28, 3]],
                [[75, 0], [0, 0]],
            ],
        )

    test(y_true, y_pred)
    test(list(str(y) for y in y_true), list(str(y) for y in y_pred), string_type=True)


def test_multilabel_confusion_matrix_multilabel():
    # Test multilabel confusion matrix - multilabel-indicator case
    from scipy.sparse import csc_matrix, csr_matrix

    y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    y_pred = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]])
    y_true_csr = csr_matrix(y_true)
    y_pred_csr = csr_matrix(y_pred)
    y_true_csc = csc_matrix(y_true)
    y_pred_csc = csc_matrix(y_pred)

    # cross test different types
    sample_weight = np.array([2, 1, 3])
    real_cm = [[[1, 0], [1, 1]], [[1, 0], [1, 1]], [[0, 2], [1, 0]]]
    trues = [y_true, y_true_csr, y_true_csc]
    preds = [y_pred, y_pred_csr, y_pred_csc]

    for y_true_tmp in trues:
        for y_pred_tmp in preds:
            cm = multilabel_confusion_matrix(y_true_tmp, y_pred_tmp)
            assert_array_equal(cm, real_cm)

    # test support for samplewise
    cm = multilabel_confusion_matrix(y_true, y_pred, samplewise=True)
    assert_array_equal(cm, [[[1, 0], [1, 1]], [[1, 1], [0, 1]], [[0, 1], [2, 0]]])

    # test support for labels
    cm = multilabel_confusion_matrix(y_true, y_pred, labels=[2, 0])
    assert_array_equal(cm, [[[0, 2], [1, 0]], [[1, 0], [1, 1]]])

    # test support for labels with samplewise
    cm = multilabel_confusion_matrix(y_true, y_pred, labels=[2, 0], samplewise=True)
    assert_array_equal(cm, [[[0, 0], [1, 1]], [[1, 1], [0, 0]], [[0, 1], [1, 0]]])

    # test support for sample_weight with sample_wise
    cm = multilabel_confusion_matrix(
        y_true, y_pred, sample_weight=sample_weight, samplewise=True
    )
    assert_array_equal(cm, [[[2, 0], [2, 2]], [[1, 1], [0, 1]], [[0, 3], [6, 0]]])


def test_multilabel_confusion_matrix_errors():
    y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    y_pred = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]])

    # Bad sample_weight
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        multilabel_confusion_matrix(y_true, y_pred, sample_weight=[1, 2])
    with pytest.raises(ValueError, match="should be a 1d array"):
        multilabel_confusion_matrix(
            y_true, y_pred, sample_weight=[[1, 2, 3], [2, 3, 4], [3, 4, 5]]
        )

    # Bad labels
    err_msg = r"All labels must be in \[0, n labels\)"
    with pytest.raises(ValueError, match=err_msg):
        multilabel_confusion_matrix(y_true, y_pred, labels=[-1])
    err_msg = r"All labels must be in \[0, n labels\)"
    with pytest.raises(ValueError, match=err_msg):
        multilabel_confusion_matrix(y_true, y_pred, labels=[3])

    # Using samplewise outside multilabel
    with pytest.raises(ValueError, match="Samplewise metrics"):
        multilabel_confusion_matrix([0, 1, 2], [1, 2, 0], samplewise=True)

    # Bad y_type
    err_msg = "multiclass-multioutput is not supported"
    with pytest.raises(ValueError, match=err_msg):
        multilabel_confusion_matrix([[0, 1, 2], [2, 1, 0]], [[1, 2, 0], [1, 0, 2]])


@pytest.mark.parametrize(
    "normalize, cm_dtype, expected_results",
    [
        ("true", "f", 0.333333333),
        ("pred", "f", 0.333333333),
        ("all", "f", 0.1111111111),
        (None, "i", 2),
    ],
)
def test_confusion_matrix_normalize(normalize, cm_dtype, expected_results):
    y_test = [0, 1, 2] * 6
    y_pred = list(chain(*permutations([0, 1, 2])))
    cm = confusion_matrix(y_test, y_pred, normalize=normalize)
    assert_allclose(cm, expected_results)
    assert cm.dtype.kind == cm_dtype


def test_confusion_matrix_normalize_wrong_option():
    y_test = [0, 0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 0, 0, 0, 0, 0]
    with pytest.raises(ValueError, match="normalize must be one of"):
        confusion_matrix(y_test, y_pred, normalize=True)


def test_confusion_matrix_normalize_single_class():
    y_test = [0, 0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 0, 0, 0, 0, 0]

    cm_true = confusion_matrix(y_test, y_pred, normalize="true")
    assert cm_true.sum() == pytest.approx(2.0)

    # additionally check that no warnings are raised due to a division by zero
    with pytest.warns(None) as rec:
        cm_pred = confusion_matrix(y_test, y_pred, normalize="pred")
    assert not rec
    assert cm_pred.sum() == pytest.approx(1.0)

    with pytest.warns(None) as rec:
        cm_pred = confusion_matrix(y_pred, y_test, normalize="true")
    assert not rec


def test_cohen_kappa():
    # These label vectors reproduce the contingency matrix from Artstein and
    # Poesio (2008), Table 1: np.array([[20, 20], [10, 50]]).
    y1 = np.array([0] * 40 + [1] * 60)
    y2 = np.array([0] * 20 + [1] * 20 + [0] * 10 + [1] * 50)
    kappa = cohen_kappa_score(y1, y2)
    assert_almost_equal(kappa, 0.348, decimal=3)
    assert kappa == cohen_kappa_score(y2, y1)

    # Add spurious labels and ignore them.
    y1 = np.append(y1, [2] * 4)
    y2 = np.append(y2, [2] * 4)
    assert cohen_kappa_score(y1, y2, labels=[0, 1]) == kappa

    assert_almost_equal(cohen_kappa_score(y1, y1), 1.0)

    # Multiclass example: Artstein and Poesio, Table 4.
    y1 = np.array([0] * 46 + [1] * 44 + [2] * 10)
    y2 = np.array([0] * 52 + [1] * 32 + [2] * 16)
    assert_almost_equal(cohen_kappa_score(y1, y2), 0.8013, decimal=4)

    # Weighting example: none, linear, quadratic.
    y1 = np.array([0] * 46 + [1] * 44 + [2] * 10)
    y2 = np.array([0] * 50 + [1] * 40 + [2] * 10)
    assert_almost_equal(cohen_kappa_score(y1, y2), 0.9315, decimal=4)
    assert_almost_equal(cohen_kappa_score(y1, y2, weights="linear"), 0.9412, decimal=4)
    assert_almost_equal(
        cohen_kappa_score(y1, y2, weights="quadratic"), 0.9541, decimal=4
    )


def test_matthews_corrcoef_nan():
    assert matthews_corrcoef([0], [1]) == 0.0
    assert matthews_corrcoef([0, 0], [0, 1]) == 0.0


def test_matthews_corrcoef_against_numpy_corrcoef():
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 2, size=20)
    y_pred = rng.randint(0, 2, size=20)

    assert_almost_equal(
        matthews_corrcoef(y_true, y_pred), np.corrcoef(y_true, y_pred)[0, 1], 10
    )


def test_matthews_corrcoef_against_jurman():
    # Check that the multiclass matthews_corrcoef agrees with the definition
    # presented in Jurman, Riccadonna, Furlanello, (2012). A Comparison of MCC
    # and CEN Error Measures in MultiClass Prediction
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 2, size=20)
    y_pred = rng.randint(0, 2, size=20)
    sample_weight = rng.rand(20)

    C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    N = len(C)
    cov_ytyp = sum(
        [
            C[k, k] * C[m, l] - C[l, k] * C[k, m]
            for k in range(N)
            for m in range(N)
            for l in range(N)
        ]
    )
    cov_ytyt = sum(
        [
            C[:, k].sum()
            * np.sum([C[g, f] for f in range(N) for g in range(N) if f != k])
            for k in range(N)
        ]
    )
    cov_ypyp = np.sum(
        [
            C[k, :].sum()
            * np.sum([C[f, g] for f in range(N) for g in range(N) if f != k])
            for k in range(N)
        ]
    )
    mcc_jurman = cov_ytyp / np.sqrt(cov_ytyt * cov_ypyp)
    mcc_ours = matthews_corrcoef(y_true, y_pred, sample_weight=sample_weight)

    assert_almost_equal(mcc_ours, mcc_jurman, 10)


def test_matthews_corrcoef():
    rng = np.random.RandomState(0)
    y_true = ["a" if i == 0 else "b" for i in rng.randint(0, 2, size=20)]

    # corrcoef of same vectors must be 1
    assert_almost_equal(matthews_corrcoef(y_true, y_true), 1.0)

    # corrcoef, when the two vectors are opposites of each other, should be -1
    y_true_inv = ["b" if i == "a" else "a" for i in y_true]
    assert_almost_equal(matthews_corrcoef(y_true, y_true_inv), -1)

    y_true_inv2 = label_binarize(y_true, classes=["a", "b"])
    y_true_inv2 = np.where(y_true_inv2, "a", "b")
    assert_almost_equal(matthews_corrcoef(y_true, y_true_inv2), -1)

    # For the zero vector case, the corrcoef cannot be calculated and should
    # output 0
    assert_almost_equal(matthews_corrcoef([0, 0, 0, 0], [0, 0, 0, 0]), 0.0)

    # And also for any other vector with 0 variance
    assert_almost_equal(matthews_corrcoef(y_true, ["a"] * len(y_true)), 0.0)

    # These two vectors have 0 correlation and hence mcc should be 0
    y_1 = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]
    y_2 = [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1]
    assert_almost_equal(matthews_corrcoef(y_1, y_2), 0.0)

    # Check that sample weight is able to selectively exclude
    mask = [1] * 10 + [0] * 10
    # Now the first half of the vector elements are alone given a weight of 1
    # and hence the mcc will not be a perfect 0 as in the previous case
    with pytest.raises(AssertionError):
        assert_almost_equal(matthews_corrcoef(y_1, y_2, sample_weight=mask), 0.0)


def test_matthews_corrcoef_multiclass():
    rng = np.random.RandomState(0)
    ord_a = ord("a")
    n_classes = 4
    y_true = [chr(ord_a + i) for i in rng.randint(0, n_classes, size=20)]

    # corrcoef of same vectors must be 1
    assert_almost_equal(matthews_corrcoef(y_true, y_true), 1.0)

    # with multiclass > 2 it is not possible to achieve -1
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred_bad = [2, 2, 0, 0, 1, 1]
    assert_almost_equal(matthews_corrcoef(y_true, y_pred_bad), -0.5)

    # Maximizing false positives and negatives minimizes the MCC
    # The minimum will be different for depending on the input
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred_min = [1, 1, 0, 0, 0, 0]
    assert_almost_equal(matthews_corrcoef(y_true, y_pred_min), -12 / np.sqrt(24 * 16))

    # Zero variance will result in an mcc of zero
    y_true = [0, 1, 2]
    y_pred = [3, 3, 3]
    assert_almost_equal(matthews_corrcoef(y_true, y_pred), 0.0)

    # Also for ground truth with zero variance
    y_true = [3, 3, 3]
    y_pred = [0, 1, 2]
    assert_almost_equal(matthews_corrcoef(y_true, y_pred), 0.0)

    # These two vectors have 0 correlation and hence mcc should be 0
    y_1 = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    y_2 = [1, 1, 1, 2, 2, 2, 0, 0, 0]
    assert_almost_equal(matthews_corrcoef(y_1, y_2), 0.0)

    # We can test that binary assumptions hold using the multiclass computation
    # by masking the weight of samples not in the first two classes

    # Masking the last label should let us get an MCC of -1
    y_true = [0, 0, 1, 1, 2]
    y_pred = [1, 1, 0, 0, 2]
    sample_weight = [1, 1, 1, 1, 0]
    assert_almost_equal(
        matthews_corrcoef(y_true, y_pred, sample_weight=sample_weight), -1
    )

    # For the zero vector case, the corrcoef cannot be calculated and should
    # output 0
    y_true = [0, 0, 1, 2]
    y_pred = [0, 0, 1, 2]
    sample_weight = [1, 1, 0, 0]
    assert_almost_equal(
        matthews_corrcoef(y_true, y_pred, sample_weight=sample_weight), 0.0
    )


@pytest.mark.parametrize("n_points", [100, 10000])
def test_matthews_corrcoef_overflow(n_points):
    # https://github.com/scikit-learn/scikit-learn/issues/9622
    rng = np.random.RandomState(20170906)

    def mcc_safe(y_true, y_pred):
        conf_matrix = confusion_matrix(y_true, y_pred)
        true_pos = conf_matrix[1, 1]
        false_pos = conf_matrix[1, 0]
        false_neg = conf_matrix[0, 1]
        n_points = len(y_true)
        pos_rate = (true_pos + false_neg) / n_points
        activity = (true_pos + false_pos) / n_points
        mcc_numerator = true_pos / n_points - pos_rate * activity
        mcc_denominator = activity * pos_rate * (1 - activity) * (1 - pos_rate)
        return mcc_numerator / np.sqrt(mcc_denominator)

    def random_ys(n_points):  # binary
        x_true = rng.random_sample(n_points)
        x_pred = x_true + 0.2 * (rng.random_sample(n_points) - 0.5)
        y_true = x_true > 0.5
        y_pred = x_pred > 0.5
        return y_true, y_pred

    arr = np.repeat([0.0, 1.0], n_points)  # binary
    assert_almost_equal(matthews_corrcoef(arr, arr), 1.0)
    arr = np.repeat([0.0, 1.0, 2.0], n_points)  # multiclass
    assert_almost_equal(matthews_corrcoef(arr, arr), 1.0)

    y_true, y_pred = random_ys(n_points)
    assert_almost_equal(matthews_corrcoef(y_true, y_true), 1.0)
    assert_almost_equal(matthews_corrcoef(y_true, y_pred), mcc_safe(y_true, y_pred))


def test_precision_recall_f1_score_multiclass():
    # Test Precision Recall and F1 Score for multiclass classification task
    y_true, y_pred, _ = make_prediction(binary=False)

    # compute scores with default labels introspection
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
    assert_array_almost_equal(p, [0.83, 0.33, 0.42], 2)
    assert_array_almost_equal(r, [0.79, 0.09, 0.90], 2)
    assert_array_almost_equal(f, [0.81, 0.15, 0.57], 2)
    assert_array_equal(s, [24, 31, 20])

    # averaging tests
    ps = precision_score(y_true, y_pred, pos_label=1, average="micro")
    assert_array_almost_equal(ps, 0.53, 2)

    rs = recall_score(y_true, y_pred, average="micro")
    assert_array_almost_equal(rs, 0.53, 2)

    fs = f1_score(y_true, y_pred, average="micro")
    assert_array_almost_equal(fs, 0.53, 2)

    ps = precision_score(y_true, y_pred, average="macro")
    assert_array_almost_equal(ps, 0.53, 2)

    rs = recall_score(y_true, y_pred, average="macro")
    assert_array_almost_equal(rs, 0.60, 2)

    fs = f1_score(y_true, y_pred, average="macro")
    assert_array_almost_equal(fs, 0.51, 2)

    ps = precision_score(y_true, y_pred, average="weighted")
    assert_array_almost_equal(ps, 0.51, 2)

    rs = recall_score(y_true, y_pred, average="weighted")
    assert_array_almost_equal(rs, 0.53, 2)

    fs = f1_score(y_true, y_pred, average="weighted")
    assert_array_almost_equal(fs, 0.47, 2)

    with pytest.raises(ValueError):
        precision_score(y_true, y_pred, average="samples")
    with pytest.raises(ValueError):
        recall_score(y_true, y_pred, average="samples")
    with pytest.raises(ValueError):
        f1_score(y_true, y_pred, average="samples")
    with pytest.raises(ValueError):
        fbeta_score(y_true, y_pred, average="samples", beta=0.5)

    # same prediction but with and explicit label ordering
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 2, 1], average=None
    )
    assert_array_almost_equal(p, [0.83, 0.41, 0.33], 2)
    assert_array_almost_equal(r, [0.79, 0.90, 0.10], 2)
    assert_array_almost_equal(f, [0.81, 0.57, 0.15], 2)
    assert_array_equal(s, [24, 20, 31])


@pytest.mark.parametrize("average", ["samples", "micro", "macro", "weighted", None])
def test_precision_refcall_f1_score_multilabel_unordered_labels(average):
    # test that labels need not be sorted in the multilabel case
    y_true = np.array([[1, 1, 0, 0]])
    y_pred = np.array([[0, 0, 1, 1]])
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=[3, 0, 1, 2], warn_for=[], average=average
    )
    assert_array_equal(p, 0)
    assert_array_equal(r, 0)
    assert_array_equal(f, 0)
    if average is None:
        assert_array_equal(s, [0, 1, 1, 0])


def test_precision_recall_f1_score_binary_averaged():
    y_true = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1])

    # compute scores with default labels introspection
    ps, rs, fs, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    assert p == np.mean(ps)
    assert r == np.mean(rs)
    assert f == np.mean(fs)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    support = np.bincount(y_true)
    assert p == np.average(ps, weights=support)
    assert r == np.average(rs, weights=support)
    assert f == np.average(fs, weights=support)


def test_zero_precision_recall():
    # Check that pathological cases do not bring NaNs

    old_error_settings = np.seterr(all="raise")

    try:
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([2, 0, 1, 1, 2, 0])

        assert_almost_equal(precision_score(y_true, y_pred, average="macro"), 0.0, 2)
        assert_almost_equal(recall_score(y_true, y_pred, average="macro"), 0.0, 2)
        assert_almost_equal(f1_score(y_true, y_pred, average="macro"), 0.0, 2)

    finally:
        np.seterr(**old_error_settings)


def test_confusion_matrix_multiclass_subset_labels():
    # Test confusion matrix - multi-class case with subset of labels
    y_true, y_pred, _ = make_prediction(binary=False)

    # compute confusion matrix with only first two labels considered
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    assert_array_equal(cm, [[19, 4], [4, 3]])

    # compute confusion matrix with explicit label ordering for only subset
    # of labels
    cm = confusion_matrix(y_true, y_pred, labels=[2, 1])
    assert_array_equal(cm, [[18, 2], [24, 3]])

    # a label not in y_true should result in zeros for that row/column
    extra_label = np.max(y_true) + 1
    cm = confusion_matrix(y_true, y_pred, labels=[2, extra_label])
    assert_array_equal(cm, [[18, 0], [0, 0]])


@pytest.mark.parametrize(
    "labels, err_msg",
    [
        ([], "'labels' should contains at least one label."),
        ([3, 4], "At least one label specified must be in y_true"),
    ],
    ids=["empty list", "unknown labels"],
)
def test_confusion_matrix_error(labels, err_msg):
    y_true, y_pred, _ = make_prediction(binary=False)
    with pytest.raises(ValueError, match=err_msg):
        confusion_matrix(y_true, y_pred, labels=labels)


@pytest.mark.parametrize(
    "labels", (None, [0, 1], [0, 1, 2]), ids=["None", "binary", "multiclass"]
)
def test_confusion_matrix_on_zero_length_input(labels):
    expected_n_classes = len(labels) if labels else 0
    expected = np.zeros((expected_n_classes, expected_n_classes), dtype=int)
    cm = confusion_matrix([], [], labels=labels)
    assert_array_equal(cm, expected)


def test_confusion_matrix_dtype():
    y = [0, 1, 1]
    weight = np.ones(len(y))
    # confusion_matrix returns int64 by default
    cm = confusion_matrix(y, y)
    assert cm.dtype == np.int64
    # The dtype of confusion_matrix is always 64 bit
    for dtype in [np.bool_, np.int32, np.uint64]:
        cm = confusion_matrix(y, y, sample_weight=weight.astype(dtype, copy=False))
        assert cm.dtype == np.int64
    for dtype in [np.float32, np.float64, None, object]:
        cm = confusion_matrix(y, y, sample_weight=weight.astype(dtype, copy=False))
        assert cm.dtype == np.float64

    # np.iinfo(np.uint32).max should be accumulated correctly
    weight = np.full(len(y), 4294967295, dtype=np.uint32)
    cm = confusion_matrix(y, y, sample_weight=weight)
    assert cm[0, 0] == 4294967295
    assert cm[1, 1] == 8589934590

    # np.iinfo(np.int64).max should cause an overflow
    weight = np.full(len(y), 9223372036854775807, dtype=np.int64)
    cm = confusion_matrix(y, y, sample_weight=weight)
    assert cm[0, 0] == 9223372036854775807
    assert cm[1, 1] == -2


def test_classification_report_multiclass():
    # Test performance report
    iris = datasets.load_iris()
    y_true, y_pred, _ = make_prediction(dataset=iris, binary=False)

    # print classification report with class names
    expected_report = """\
              precision    recall  f1-score   support

      setosa       0.83      0.79      0.81        24
  versicolor       0.33      0.10      0.15        31
   virginica       0.42      0.90      0.57        20

    accuracy                           0.53        75
   macro avg       0.53      0.60      0.51        75
weighted avg       0.51      0.53      0.47        75
"""
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(iris.target_names)),
        target_names=iris.target_names,
    )
    assert report == expected_report


def test_classification_report_multiclass_balanced():
    y_true, y_pred = [0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]

    expected_report = """\
              precision    recall  f1-score   support

           0       0.33      0.33      0.33         3
           1       0.33      0.33      0.33         3
           2       0.33      0.33      0.33         3

    accuracy                           0.33         9
   macro avg       0.33      0.33      0.33         9
weighted avg       0.33      0.33      0.33         9
"""
    report = classification_report(y_true, y_pred)
    assert report == expected_report


def test_classification_report_multiclass_with_label_detection():
    iris = datasets.load_iris()
    y_true, y_pred, _ = make_prediction(dataset=iris, binary=False)

    # print classification report with label detection
    expected_report = """\
              precision    recall  f1-score   support

           0       0.83      0.79      0.81        24
           1       0.33      0.10      0.15        31
           2       0.42      0.90      0.57        20

    accuracy                           0.53        75
   macro avg       0.53      0.60      0.51        75
weighted avg       0.51      0.53      0.47        75
"""
    report = classification_report(y_true, y_pred)
    assert report == expected_report


def test_classification_report_multiclass_with_digits():
    # Test performance report with added digits in floating point values
    iris = datasets.load_iris()
    y_true, y_pred, _ = make_prediction(dataset=iris, binary=False)

    # print classification report with class names
    expected_report = """\
              precision    recall  f1-score   support

      setosa    0.82609   0.79167   0.80851        24
  versicolor    0.33333   0.09677   0.15000        31
   virginica    0.41860   0.90000   0.57143        20

    accuracy                        0.53333        75
   macro avg    0.52601   0.59615   0.50998        75
weighted avg    0.51375   0.53333   0.47310        75
"""
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(iris.target_names)),
        target_names=iris.target_names,
        digits=5,
    )
    assert report == expected_report


def test_classification_report_multiclass_with_string_label():
    y_true, y_pred, _ = make_prediction(binary=False)

    y_true = np.array(["blue", "green", "red"])[y_true]
    y_pred = np.array(["blue", "green", "red"])[y_pred]

    expected_report = """\
              precision    recall  f1-score   support

        blue       0.83      0.79      0.81        24
       green       0.33      0.10      0.15        31
         red       0.42      0.90      0.57        20

    accuracy                           0.53        75
   macro avg       0.53      0.60      0.51        75
weighted avg       0.51      0.53      0.47        75
"""
    report = classification_report(y_true, y_pred)
    assert report == expected_report

    expected_report = """\
              precision    recall  f1-score   support

           a       0.83      0.79      0.81        24
           b       0.33      0.10      0.15        31
           c       0.42      0.90      0.57        20

    accuracy                           0.53        75
   macro avg       0.53      0.60      0.51        75
weighted avg       0.51      0.53      0.47        75
"""
    report = classification_report(y_true, y_pred, target_names=["a", "b", "c"])
    assert report == expected_report


def test_classification_report_multiclass_with_unicode_label():
    y_true, y_pred, _ = make_prediction(binary=False)

    labels = np.array(["blue\xa2", "green\xa2", "red\xa2"])
    y_true = labels[y_true]
    y_pred = labels[y_pred]

    expected_report = """\
              precision    recall  f1-score   support

       blue\xa2       0.83      0.79      0.81        24
      green\xa2       0.33      0.10      0.15        31
        red\xa2       0.42      0.90      0.57        20

    accuracy                           0.53        75
   macro avg       0.53      0.60      0.51        75
weighted avg       0.51      0.53      0.47        75
"""
    report = classification_report(y_true, y_pred)
    assert report == expected_report


def test_classification_report_multiclass_with_long_string_label():
    y_true, y_pred, _ = make_prediction(binary=False)

    labels = np.array(["blue", "green" * 5, "red"])
    y_true = labels[y_true]
    y_pred = labels[y_pred]

    expected_report = """\
                           precision    recall  f1-score   support

                     blue       0.83      0.79      0.81        24
greengreengreengreengreen       0.33      0.10      0.15        31
                      red       0.42      0.90      0.57        20

                 accuracy                           0.53        75
                macro avg       0.53      0.60      0.51        75
             weighted avg       0.51      0.53      0.47        75
"""

    report = classification_report(y_true, y_pred)
    assert report == expected_report


def test_classification_report_labels_target_names_unequal_length():
    y_true = [0, 0, 2, 0, 0]
    y_pred = [0, 2, 2, 0, 0]
    target_names = ["class 0", "class 1", "class 2"]

    msg = "labels size, 2, does not match size of target_names, 3"
    with pytest.warns(UserWarning, match=msg):
        classification_report(y_true, y_pred, labels=[0, 2], target_names=target_names)


def test_classification_report_no_labels_target_names_unequal_length():
    y_true = [0, 0, 2, 0, 0]
    y_pred = [0, 2, 2, 0, 0]
    target_names = ["class 0", "class 1", "class 2"]

    err_msg = (
        "Number of classes, 2, does not "
        "match size of target_names, 3. "
        "Try specifying the labels parameter"
    )
    with pytest.raises(ValueError, match=err_msg):
        classification_report(y_true, y_pred, target_names=target_names)


@ignore_warnings
def test_multilabel_classification_report():
    n_classes = 4
    n_samples = 50

    _, y_true = make_multilabel_classification(
        n_features=1, n_samples=n_samples, n_classes=n_classes, random_state=0
    )

    _, y_pred = make_multilabel_classification(
        n_features=1, n_samples=n_samples, n_classes=n_classes, random_state=1
    )

    expected_report = """\
              precision    recall  f1-score   support

           0       0.50      0.67      0.57        24
           1       0.51      0.74      0.61        27
           2       0.29      0.08      0.12        26
           3       0.52      0.56      0.54        27

   micro avg       0.50      0.51      0.50       104
   macro avg       0.45      0.51      0.46       104
weighted avg       0.45      0.51      0.46       104
 samples avg       0.46      0.42      0.40       104
"""

    report = classification_report(y_true, y_pred)
    assert report == expected_report


def test_multilabel_zero_one_loss_subset():
    # Dense label indicator matrix format
    y1 = np.array([[0, 1, 1], [1, 0, 1]])
    y2 = np.array([[0, 0, 1], [1, 0, 1]])

    assert zero_one_loss(y1, y2) == 0.5
    assert zero_one_loss(y1, y1) == 0
    assert zero_one_loss(y2, y2) == 0
    assert zero_one_loss(y2, np.logical_not(y2)) == 1
    assert zero_one_loss(y1, np.logical_not(y1)) == 1
    assert zero_one_loss(y1, np.zeros(y1.shape)) == 1
    assert zero_one_loss(y2, np.zeros(y1.shape)) == 1


def test_multilabel_hamming_loss():
    # Dense label indicator matrix format
    y1 = np.array([[0, 1, 1], [1, 0, 1]])
    y2 = np.array([[0, 0, 1], [1, 0, 1]])
    w = np.array([1, 3])

    assert hamming_loss(y1, y2) == 1 / 6
    assert hamming_loss(y1, y1) == 0
    assert hamming_loss(y2, y2) == 0
    assert hamming_loss(y2, 1 - y2) == 1
    assert hamming_loss(y1, 1 - y1) == 1
    assert hamming_loss(y1, np.zeros(y1.shape)) == 4 / 6
    assert hamming_loss(y2, np.zeros(y1.shape)) == 0.5
    assert hamming_loss(y1, y2, sample_weight=w) == 1.0 / 12
    assert hamming_loss(y1, 1 - y2, sample_weight=w) == 11.0 / 12
    assert hamming_loss(y1, np.zeros_like(y1), sample_weight=w) == 2.0 / 3
    # sp_hamming only works with 1-D arrays
    assert hamming_loss(y1[0], y2[0]) == sp_hamming(y1[0], y2[0])


def test_jaccard_score_validation():
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 1])
    err_msg = r"pos_label=2 is not a valid label. It should be one of \[0, 1\]"
    with pytest.raises(ValueError, match=err_msg):
        jaccard_score(y_true, y_pred, average="binary", pos_label=2)

    y_true = np.array([[0, 1, 1], [1, 0, 0]])
    y_pred = np.array([[1, 1, 1], [1, 0, 1]])
    msg1 = (
        r"Target is multilabel-indicator but average='binary'. "
        r"Please choose another average setting, one of \[None, "
        r"'micro', 'macro', 'weighted', 'samples'\]."
    )
    with pytest.raises(ValueError, match=msg1):
        jaccard_score(y_true, y_pred, average="binary", pos_label=-1)

    y_true = np.array([0, 1, 1, 0, 2])
    y_pred = np.array([1, 1, 1, 1, 0])
    msg2 = (
        r"Target is multiclass but average='binary'. Please choose "
        r"another average setting, one of \[None, 'micro', 'macro', "
        r"'weighted'\]."
    )
    with pytest.raises(ValueError, match=msg2):
        jaccard_score(y_true, y_pred, average="binary")
    msg3 = "Samplewise metrics are not available outside of multilabel classification."
    with pytest.raises(ValueError, match=msg3):
        jaccard_score(y_true, y_pred, average="samples")

    msg = (
        r"Note that pos_label \(set to 3\) is ignored when "
        r"average != 'binary' \(got 'micro'\). You may use "
        r"labels=\[pos_label\] to specify a single positive "
        "class."
    )
    with pytest.warns(UserWarning, match=msg):
        jaccard_score(y_true, y_pred, average="micro", pos_label=3)


def test_multilabel_jaccard_score(recwarn):
    # Dense label indicator matrix format
    y1 = np.array([[0, 1, 1], [1, 0, 1]])
    y2 = np.array([[0, 0, 1], [1, 0, 1]])

    # size(y1 \inter y2) = [1, 2]
    # size(y1 \union y2) = [2, 2]

    assert jaccard_score(y1, y2, average="samples") == 0.75
    assert jaccard_score(y1, y1, average="samples") == 1
    assert jaccard_score(y2, y2, average="samples") == 1
    assert jaccard_score(y2, np.logical_not(y2), average="samples") == 0
    assert jaccard_score(y1, np.logical_not(y1), average="samples") == 0
    assert jaccard_score(y1, np.zeros(y1.shape), average="samples") == 0
    assert jaccard_score(y2, np.zeros(y1.shape), average="samples") == 0

    y_true = np.array([[0, 1, 1], [1, 0, 0]])
    y_pred = np.array([[1, 1, 1], [1, 0, 1]])
    # average='macro'
    assert_almost_equal(jaccard_score(y_true, y_pred, average="macro"), 2.0 / 3)
    # average='micro'
    assert_almost_equal(jaccard_score(y_true, y_pred, average="micro"), 3.0 / 5)
    # average='samples'
    assert_almost_equal(jaccard_score(y_true, y_pred, average="samples"), 7.0 / 12)
    assert_almost_equal(
        jaccard_score(y_true, y_pred, average="samples", labels=[0, 2]), 1.0 / 2
    )
    assert_almost_equal(
        jaccard_score(y_true, y_pred, average="samples", labels=[1, 2]), 1.0 / 2
    )
    # average=None
    assert_array_equal(
        jaccard_score(y_true, y_pred, average=None), np.array([1.0 / 2, 1.0, 1.0 / 2])
    )

    y_true = np.array([[0, 1, 1], [1, 0, 1]])
    y_pred = np.array([[1, 1, 1], [1, 0, 1]])
    assert_almost_equal(jaccard_score(y_true, y_pred, average="macro"), 5.0 / 6)
    # average='weighted'
    assert_almost_equal(jaccard_score(y_true, y_pred, average="weighted"), 7.0 / 8)

    msg2 = "Got 4 > 2"
    with pytest.raises(ValueError, match=msg2):
        jaccard_score(y_true, y_pred, labels=[4], average="macro")
    msg3 = "Got -1 < 0"
    with pytest.raises(ValueError, match=msg3):
        jaccard_score(y_true, y_pred, labels=[-1], average="macro")

    msg = (
        "Jaccard is ill-defined and being set to 0.0 in labels "
        "with no true or predicted samples."
    )

    with pytest.warns(UndefinedMetricWarning, match=msg):
        assert (
            jaccard_score(np.array([[0, 1]]), np.array([[0, 1]]), average="macro")
            == 0.5
        )

    msg = (
        "Jaccard is ill-defined and being set to 0.0 in samples "
        "with no true or predicted labels."
    )

    with pytest.warns(UndefinedMetricWarning, match=msg):
        assert (
            jaccard_score(
                np.array([[0, 0], [1, 1]]),
                np.array([[0, 0], [1, 1]]),
                average="samples",
            )
            == 0.5
        )

    assert not list(recwarn)


def test_multiclass_jaccard_score(recwarn):
    y_true = ["ant", "ant", "cat", "cat", "ant", "cat", "bird", "bird"]
    y_pred = ["cat", "ant", "cat", "cat", "ant", "bird", "bird", "cat"]
    labels = ["ant", "bird", "cat"]
    lb = LabelBinarizer()
    lb.fit(labels)
    y_true_bin = lb.transform(y_true)
    y_pred_bin = lb.transform(y_pred)
    multi_jaccard_score = partial(jaccard_score, y_true, y_pred)
    bin_jaccard_score = partial(jaccard_score, y_true_bin, y_pred_bin)
    multi_labels_list = [
        ["ant", "bird"],
        ["ant", "cat"],
        ["cat", "bird"],
        ["ant"],
        ["bird"],
        ["cat"],
        None,
    ]
    bin_labels_list = [[0, 1], [0, 2], [2, 1], [0], [1], [2], None]

    # other than average='samples'/'none-samples', test everything else here
    for average in ("macro", "weighted", "micro", None):
        for m_label, b_label in zip(multi_labels_list, bin_labels_list):
            assert_almost_equal(
                multi_jaccard_score(average=average, labels=m_label),
                bin_jaccard_score(average=average, labels=b_label),
            )

    y_true = np.array([[0, 0], [0, 0], [0, 0]])
    y_pred = np.array([[0, 0], [0, 0], [0, 0]])
    with ignore_warnings():
        assert jaccard_score(y_true, y_pred, average="weighted") == 0

    assert not list(recwarn)


def test_average_binary_jaccard_score(recwarn):
    # tp=0, fp=0, fn=1, tn=0
    assert jaccard_score([1], [0], average="binary") == 0.0
    # tp=0, fp=0, fn=0, tn=1
    msg = (
        "Jaccard is ill-defined and being set to 0.0 due to "
        "no true or predicted samples"
    )
    with pytest.warns(UndefinedMetricWarning, match=msg):
        assert jaccard_score([0, 0], [0, 0], average="binary") == 0.0

    # tp=1, fp=0, fn=0, tn=0 (pos_label=0)
    assert jaccard_score([0], [0], pos_label=0, average="binary") == 1.0
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 1, 1])
    assert_almost_equal(jaccard_score(y_true, y_pred, average="binary"), 3.0 / 4)
    assert_almost_equal(
        jaccard_score(y_true, y_pred, average="binary", pos_label=0), 1.0 / 2
    )

    assert not list(recwarn)


def test_jaccard_score_zero_division_warning():
    # check that we raised a warning with default behavior if a zero division
    # happens
    y_true = np.array([[1, 0, 1], [0, 0, 0]])
    y_pred = np.array([[0, 0, 0], [0, 0, 0]])
    msg = (
        "Jaccard is ill-defined and being set to 0.0 in "
        "samples with no true or predicted labels."
        " Use `zero_division` parameter to control this behavior."
    )
    with pytest.warns(UndefinedMetricWarning, match=msg):
        score = jaccard_score(y_true, y_pred, average="samples", zero_division="warn")
        assert score == pytest.approx(0.0)


@pytest.mark.parametrize("zero_division, expected_score", [(0, 0), (1, 0.5)])
def test_jaccard_score_zero_division_set_value(zero_division, expected_score):
    # check that we don't issue warning by passing the zero_division parameter
    y_true = np.array([[1, 0, 1], [0, 0, 0]])
    y_pred = np.array([[0, 0, 0], [0, 0, 0]])
    with pytest.warns(None) as record:
        score = jaccard_score(
            y_true, y_pred, average="samples", zero_division=zero_division
        )
    assert score == pytest.approx(expected_score)
    assert len(record) == 0


@ignore_warnings
def test_precision_recall_f1_score_multilabel_1():
    # Test precision_recall_f1_score on a crafted multilabel example
    # First crafted example

    y_true = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1]])
    y_pred = np.array([[0, 1, 0, 0], [0, 1, 0, 0], [1, 0, 1, 0]])

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)

    # tp = [0, 1, 1, 0]
    # fn = [1, 0, 0, 1]
    # fp = [1, 1, 0, 0]
    # Check per class

    assert_array_almost_equal(p, [0.0, 0.5, 1.0, 0.0], 2)
    assert_array_almost_equal(r, [0.0, 1.0, 1.0, 0.0], 2)
    assert_array_almost_equal(f, [0.0, 1 / 1.5, 1, 0.0], 2)
    assert_array_almost_equal(s, [1, 1, 1, 1], 2)

    f2 = fbeta_score(y_true, y_pred, beta=2, average=None)
    support = s
    assert_array_almost_equal(f2, [0, 0.83, 1, 0], 2)

    # Check macro
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average="macro")
    assert_almost_equal(p, 1.5 / 4)
    assert_almost_equal(r, 0.5)
    assert_almost_equal(f, 2.5 / 1.5 * 0.25)
    assert s is None
    assert_almost_equal(
        fbeta_score(y_true, y_pred, beta=2, average="macro"), np.mean(f2)
    )

    # Check micro
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average="micro")
    assert_almost_equal(p, 0.5)
    assert_almost_equal(r, 0.5)
    assert_almost_equal(f, 0.5)
    assert s is None
    assert_almost_equal(
        fbeta_score(y_true, y_pred, beta=2, average="micro"),
        (1 + 4) * p * r / (4 * p + r),
    )

    # Check weighted
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    assert_almost_equal(p, 1.5 / 4)
    assert_almost_equal(r, 0.5)
    assert_almost_equal(f, 2.5 / 1.5 * 0.25)
    assert s is None
    assert_almost_equal(
        fbeta_score(y_true, y_pred, beta=2, average="weighted"),
        np.average(f2, weights=support),
    )
    # Check samples
    # |h(x_i) inter y_i | = [0, 1, 1]
    # |y_i| = [1, 1, 2]
    # |h(x_i)| = [1, 1, 2]
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average="samples")
    assert_almost_equal(p, 0.5)
    assert_almost_equal(r, 0.5)
    assert_almost_equal(f, 0.5)
    assert s is None
    assert_almost_equal(fbeta_score(y_true, y_pred, beta=2, average="samples"), 0.5)


@ignore_warnings
def test_precision_recall_f1_score_multilabel_2():
    # Test precision_recall_f1_score on a crafted multilabel example 2
    # Second crafted example
    y_true = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0]])
    y_pred = np.array([[0, 0, 0, 1], [0, 0, 0, 1], [1, 1, 0, 0]])

    # tp = [ 0.  1.  0.  0.]
    # fp = [ 1.  0.  0.  2.]
    # fn = [ 1.  1.  1.  0.]

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
    assert_array_almost_equal(p, [0.0, 1.0, 0.0, 0.0], 2)
    assert_array_almost_equal(r, [0.0, 0.5, 0.0, 0.0], 2)
    assert_array_almost_equal(f, [0.0, 0.66, 0.0, 0.0], 2)
    assert_array_almost_equal(s, [1, 2, 1, 0], 2)

    f2 = fbeta_score(y_true, y_pred, beta=2, average=None)
    support = s
    assert_array_almost_equal(f2, [0, 0.55, 0, 0], 2)

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average="micro")
    assert_almost_equal(p, 0.25)
    assert_almost_equal(r, 0.25)
    assert_almost_equal(f, 2 * 0.25 * 0.25 / 0.5)
    assert s is None
    assert_almost_equal(
        fbeta_score(y_true, y_pred, beta=2, average="micro"),
        (1 + 4) * p * r / (4 * p + r),
    )

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average="macro")
    assert_almost_equal(p, 0.25)
    assert_almost_equal(r, 0.125)
    assert_almost_equal(f, 2 / 12)
    assert s is None
    assert_almost_equal(
        fbeta_score(y_true, y_pred, beta=2, average="macro"), np.mean(f2)
    )

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    assert_almost_equal(p, 2 / 4)
    assert_almost_equal(r, 1 / 4)
    assert_almost_equal(f, 2 / 3 * 2 / 4)
    assert s is None
    assert_almost_equal(
        fbeta_score(y_true, y_pred, beta=2, average="weighted"),
        np.average(f2, weights=support),
    )

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average="samples")
    # Check samples
    # |h(x_i) inter y_i | = [0, 0, 1]
    # |y_i| = [1, 1, 2]
    # |h(x_i)| = [1, 1, 2]

    assert_almost_equal(p, 1 / 6)
    assert_almost_equal(r, 1 / 6)
    assert_almost_equal(f, 2 / 4 * 1 / 3)
    assert s is None
    assert_almost_equal(
        fbeta_score(y_true, y_pred, beta=2, average="samples"), 0.1666, 2
    )


@ignore_warnings
@pytest.mark.parametrize("zero_division", ["warn", 0, 1])
def test_precision_recall_f1_score_with_an_empty_prediction(zero_division):
    y_true = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 1, 0]])
    y_pred = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0]])

    # true_pos = [ 0.  1.  1.  0.]
    # false_pos = [ 0.  0.  0.  1.]
    # false_neg = [ 1.  1.  0.  0.]
    zero_division = 1.0 if zero_division == 1.0 else 0.0
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=zero_division
    )
    assert_array_almost_equal(p, [zero_division, 1.0, 1.0, 0.0], 2)
    assert_array_almost_equal(r, [0.0, 0.5, 1.0, zero_division], 2)
    assert_array_almost_equal(f, [0.0, 1 / 1.5, 1, 0.0], 2)
    assert_array_almost_equal(s, [1, 2, 1, 0], 2)

    f2 = fbeta_score(y_true, y_pred, beta=2, average=None, zero_division=zero_division)
    support = s
    assert_array_almost_equal(f2, [0, 0.55, 1, 0], 2)

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=zero_division
    )
    assert_almost_equal(p, (2 + zero_division) / 4)
    assert_almost_equal(r, (1.5 + zero_division) / 4)
    assert_almost_equal(f, 2.5 / (4 * 1.5))
    assert s is None
    assert_almost_equal(
        fbeta_score(y_true, y_pred, beta=2, average="macro"), np.mean(f2)
    )

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=zero_division
    )
    assert_almost_equal(p, 2 / 3)
    assert_almost_equal(r, 0.5)
    assert_almost_equal(f, 2 / 3 / (2 / 3 + 0.5))
    assert s is None
    assert_almost_equal(
        fbeta_score(
            y_true, y_pred, beta=2, average="micro", zero_division=zero_division
        ),
        (1 + 4) * p * r / (4 * p + r),
    )

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=zero_division
    )
    assert_almost_equal(p, 3 / 4 if zero_division == 0 else 1.0)
    assert_almost_equal(r, 0.5)
    assert_almost_equal(f, (2 / 1.5 + 1) / 4)
    assert s is None
    assert_almost_equal(
        fbeta_score(
            y_true, y_pred, beta=2, average="weighted", zero_division=zero_division
        ),
        np.average(f2, weights=support),
    )

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average="samples")
    # |h(x_i) inter y_i | = [0, 0, 2]
    # |y_i| = [1, 1, 2]
    # |h(x_i)| = [0, 1, 2]
    assert_almost_equal(p, 1 / 3)
    assert_almost_equal(r, 1 / 3)
    assert_almost_equal(f, 1 / 3)
    assert s is None
    assert_almost_equal(
        fbeta_score(
            y_true, y_pred, beta=2, average="samples", zero_division=zero_division
        ),
        0.333,
        2,
    )


@pytest.mark.parametrize("beta", [1])
@pytest.mark.parametrize("average", ["macro", "micro", "weighted", "samples"])
@pytest.mark.parametrize("zero_division", [0, 1])
def test_precision_recall_f1_no_labels(beta, average, zero_division):
    y_true = np.zeros((20, 3))
    y_pred = np.zeros_like(y_true)

    p, r, f, s = assert_no_warnings(
        precision_recall_fscore_support,
        y_true,
        y_pred,
        average=average,
        beta=beta,
        zero_division=zero_division,
    )
    fbeta = assert_no_warnings(
        fbeta_score,
        y_true,
        y_pred,
        beta=beta,
        average=average,
        zero_division=zero_division,
    )

    zero_division = float(zero_division)
    assert_almost_equal(p, zero_division)
    assert_almost_equal(r, zero_division)
    assert_almost_equal(f, zero_division)
    assert s is None

    assert_almost_equal(fbeta, float(zero_division))


@pytest.mark.parametrize("average", ["macro", "micro", "weighted", "samples"])
def test_precision_recall_f1_no_labels_check_warnings(average):
    y_true = np.zeros((20, 3))
    y_pred = np.zeros_like(y_true)

    func = precision_recall_fscore_support
    with pytest.warns(UndefinedMetricWarning):
        p, r, f, s = func(y_true, y_pred, average=average, beta=1.0)

    assert_almost_equal(p, 0)
    assert_almost_equal(r, 0)
    assert_almost_equal(f, 0)
    assert s is None

    with pytest.warns(UndefinedMetricWarning):
        fbeta = fbeta_score(y_true, y_pred, average=average, beta=1.0)

    assert_almost_equal(fbeta, 0)


@pytest.mark.parametrize("zero_division", [0, 1])
def test_precision_recall_f1_no_labels_average_none(zero_division):
    y_true = np.zeros((20, 3))
    y_pred = np.zeros_like(y_true)

    # tp = [0, 0, 0]
    # fn = [0, 0, 0]
    # fp = [0, 0, 0]
    # support = [0, 0, 0]
    # |y_hat_i inter y_i | = [0, 0, 0]
    # |y_i| = [0, 0, 0]
    # |y_hat_i| = [0, 0, 0]

    p, r, f, s = assert_no_warnings(
        precision_recall_fscore_support,
        y_true,
        y_pred,
        average=None,
        beta=1.0,
        zero_division=zero_division,
    )
    fbeta = assert_no_warnings(
        fbeta_score, y_true, y_pred, beta=1.0, average=None, zero_division=zero_division
    )

    zero_division = float(zero_division)
    assert_array_almost_equal(p, [zero_division, zero_division, zero_division], 2)
    assert_array_almost_equal(r, [zero_division, zero_division, zero_division], 2)
    assert_array_almost_equal(f, [zero_division, zero_division, zero_division], 2)
    assert_array_almost_equal(s, [0, 0, 0], 2)

    assert_array_almost_equal(fbeta, [zero_division, zero_division, zero_division], 2)


def test_precision_recall_f1_no_labels_average_none_warn():
    y_true = np.zeros((20, 3))
    y_pred = np.zeros_like(y_true)

    # tp = [0, 0, 0]
    # fn = [0, 0, 0]
    # fp = [0, 0, 0]
    # support = [0, 0, 0]
    # |y_hat_i inter y_i | = [0, 0, 0]
    # |y_i| = [0, 0, 0]
    # |y_hat_i| = [0, 0, 0]

    with pytest.warns(UndefinedMetricWarning):
        p, r, f, s = precision_recall_fscore_support(
            y_true, y_pred, average=None, beta=1
        )

    assert_array_almost_equal(p, [0, 0, 0], 2)
    assert_array_almost_equal(r, [0, 0, 0], 2)
    assert_array_almost_equal(f, [0, 0, 0], 2)
    assert_array_almost_equal(s, [0, 0, 0], 2)

    with pytest.warns(UndefinedMetricWarning):
        fbeta = fbeta_score(y_true, y_pred, beta=1, average=None)

    assert_array_almost_equal(fbeta, [0, 0, 0], 2)


def test_prf_warnings():
    # average of per-label scores
    f, w = precision_recall_fscore_support, UndefinedMetricWarning
    for average in [None, "weighted", "macro"]:

        msg = (
            "Precision and F-score are ill-defined and "
            "being set to 0.0 in labels with no predicted samples."
            " Use `zero_division` parameter to control"
            " this behavior."
        )
        with pytest.warns(w, match=msg):
            f([0, 1, 2], [1, 1, 2], average=average)

        msg = (
            "Recall and F-score are ill-defined and "
            "being set to 0.0 in labels with no true samples."
            " Use `zero_division` parameter to control"
            " this behavior."
        )
        with pytest.warns(w, match=msg):
            f([1, 1, 2], [0, 1, 2], average=average)

    # average of per-sample scores
    msg = (
        "Precision and F-score are ill-defined and "
        "being set to 0.0 in samples with no predicted labels."
        " Use `zero_division` parameter to control"
        " this behavior."
    )
    with pytest.warns(w, match=msg):
        f(np.array([[1, 0], [1, 0]]), np.array([[1, 0], [0, 0]]), average="samples")

    msg = (
        "Recall and F-score are ill-defined and "
        "being set to 0.0 in samples with no true labels."
        " Use `zero_division` parameter to control"
        " this behavior."
    )
    with pytest.warns(w, match=msg):
        f(np.array([[1, 0], [0, 0]]), np.array([[1, 0], [1, 0]]), average="samples")

    # single score: micro-average
    msg = (
        "Precision and F-score are ill-defined and "
        "being set to 0.0 due to no predicted samples."
        " Use `zero_division` parameter to control"
        " this behavior."
    )
    with pytest.warns(w, match=msg):
        f(np.array([[1, 1], [1, 1]]), np.array([[0, 0], [0, 0]]), average="micro")

    msg = (
        "Recall and F-score are ill-defined and "
        "being set to 0.0 due to no true samples."
        " Use `zero_division` parameter to control"
        " this behavior."
    )
    with pytest.warns(w, match=msg):
        f(np.array([[0, 0], [0, 0]]), np.array([[1, 1], [1, 1]]), average="micro")

    # single positive label
    msg = (
        "Precision and F-score are ill-defined and "
        "being set to 0.0 due to no predicted samples."
        " Use `zero_division` parameter to control"
        " this behavior."
    )
    with pytest.warns(w, match=msg):
        f([1, 1], [-1, -1], average="binary")

    msg = (
        "Recall and F-score are ill-defined and "
        "being set to 0.0 due to no true samples."
        " Use `zero_division` parameter to control"
        " this behavior."
    )
    with pytest.warns(w, match=msg):
        f([-1, -1], [1, 1], average="binary")

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        precision_recall_fscore_support([0, 0], [0, 0], average="binary")
        msg = (
            "Recall and F-score are ill-defined and "
            "being set to 0.0 due to no true samples."
            " Use `zero_division` parameter to control"
            " this behavior."
        )
        assert str(record.pop().message) == msg
        msg = (
            "Precision and F-score are ill-defined and "
            "being set to 0.0 due to no predicted samples."
            " Use `zero_division` parameter to control"
            " this behavior."
        )
        assert str(record.pop().message) == msg


@pytest.mark.parametrize("zero_division", [0, 1])
def test_prf_no_warnings_if_zero_division_set(zero_division):
    # average of per-label scores
    f = precision_recall_fscore_support
    for average in [None, "weighted", "macro"]:

        assert_no_warnings(
            f, [0, 1, 2], [1, 1, 2], average=average, zero_division=zero_division
        )

        assert_no_warnings(
            f, [1, 1, 2], [0, 1, 2], average=average, zero_division=zero_division
        )

    # average of per-sample scores
    assert_no_warnings(
        f,
        np.array([[1, 0], [1, 0]]),
        np.array([[1, 0], [0, 0]]),
        average="samples",
        zero_division=zero_division,
    )

    assert_no_warnings(
        f,
        np.array([[1, 0], [0, 0]]),
        np.array([[1, 0], [1, 0]]),
        average="samples",
        zero_division=zero_division,
    )

    # single score: micro-average
    assert_no_warnings(
        f,
        np.array([[1, 1], [1, 1]]),
        np.array([[0, 0], [0, 0]]),
        average="micro",
        zero_division=zero_division,
    )

    assert_no_warnings(
        f,
        np.array([[0, 0], [0, 0]]),
        np.array([[1, 1], [1, 1]]),
        average="micro",
        zero_division=zero_division,
    )

    # single positive label
    assert_no_warnings(
        f, [1, 1], [-1, -1], average="binary", zero_division=zero_division
    )

    assert_no_warnings(
        f, [-1, -1], [1, 1], average="binary", zero_division=zero_division
    )

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        precision_recall_fscore_support(
            [0, 0], [0, 0], average="binary", zero_division=zero_division
        )
        assert len(record) == 0


@pytest.mark.parametrize("zero_division", ["warn", 0, 1])
def test_recall_warnings(zero_division):
    assert_no_warnings(
        recall_score,
        np.array([[1, 1], [1, 1]]),
        np.array([[0, 0], [0, 0]]),
        average="micro",
        zero_division=zero_division,
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        recall_score(
            np.array([[0, 0], [0, 0]]),
            np.array([[1, 1], [1, 1]]),
            average="micro",
            zero_division=zero_division,
        )
        if zero_division == "warn":
            assert (
                str(record.pop().message)
                == "Recall is ill-defined and "
                "being set to 0.0 due to no true samples."
                " Use `zero_division` parameter to control"
                " this behavior."
            )
        else:
            assert len(record) == 0

        recall_score([0, 0], [0, 0])
        if zero_division == "warn":
            assert (
                str(record.pop().message)
                == "Recall is ill-defined and "
                "being set to 0.0 due to no true samples."
                " Use `zero_division` parameter to control"
                " this behavior."
            )


@pytest.mark.parametrize("zero_division", ["warn", 0, 1])
def test_precision_warnings(zero_division):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        precision_score(
            np.array([[1, 1], [1, 1]]),
            np.array([[0, 0], [0, 0]]),
            average="micro",
            zero_division=zero_division,
        )
        if zero_division == "warn":
            assert (
                str(record.pop().message)
                == "Precision is ill-defined and "
                "being set to 0.0 due to no predicted samples."
                " Use `zero_division` parameter to control"
                " this behavior."
            )
        else:
            assert len(record) == 0

        precision_score([0, 0], [0, 0])
        if zero_division == "warn":
            assert (
                str(record.pop().message)
                == "Precision is ill-defined and "
                "being set to 0.0 due to no predicted samples."
                " Use `zero_division` parameter to control"
                " this behavior."
            )

    assert_no_warnings(
        precision_score,
        np.array([[0, 0], [0, 0]]),
        np.array([[1, 1], [1, 1]]),
        average="micro",
        zero_division=zero_division,
    )


@pytest.mark.parametrize("zero_division", ["warn", 0, 1])
def test_fscore_warnings(zero_division):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")

        for score in [f1_score, partial(fbeta_score, beta=2)]:
            score(
                np.array([[1, 1], [1, 1]]),
                np.array([[0, 0], [0, 0]]),
                average="micro",
                zero_division=zero_division,
            )
            assert len(record) == 0

            score(
                np.array([[0, 0], [0, 0]]),
                np.array([[1, 1], [1, 1]]),
                average="micro",
                zero_division=zero_division,
            )
            assert len(record) == 0

            score(
                np.array([[0, 0], [0, 0]]),
                np.array([[0, 0], [0, 0]]),
                average="micro",
                zero_division=zero_division,
            )
            if zero_division == "warn":
                assert (
                    str(record.pop().message)
                    == "F-score is ill-defined and "
                    "being set to 0.0 due to no true nor predicted "
                    "samples. Use `zero_division` parameter to "
                    "control this behavior."
                )
            else:
                assert len(record) == 0


def test_prf_average_binary_data_non_binary():
    # Error if user does not explicitly set non-binary average mode
    y_true_mc = [1, 2, 3, 3]
    y_pred_mc = [1, 2, 3, 1]
    msg_mc = (
        r"Target is multiclass but average='binary'. Please "
        r"choose another average setting, one of \["
        r"None, 'micro', 'macro', 'weighted'\]."
    )
    y_true_ind = np.array([[0, 1, 1], [1, 0, 0], [0, 0, 1]])
    y_pred_ind = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    msg_ind = (
        r"Target is multilabel-indicator but average='binary'. Please "
        r"choose another average setting, one of \["
        r"None, 'micro', 'macro', 'weighted', 'samples'\]."
    )

    for y_true, y_pred, msg in [
        (y_true_mc, y_pred_mc, msg_mc),
        (y_true_ind, y_pred_ind, msg_ind),
    ]:
        for metric in [
            precision_score,
            recall_score,
            f1_score,
            partial(fbeta_score, beta=2),
        ]:
            with pytest.raises(ValueError, match=msg):
                metric(y_true, y_pred)


def test__check_targets():
    # Check that _check_targets correctly merges target types, squeezes
    # output and fails if input lengths differ.
    IND = "multilabel-indicator"
    MC = "multiclass"
    BIN = "binary"
    CNT = "continuous"
    MMC = "multiclass-multioutput"
    MCN = "continuous-multioutput"
    # all of length 3
    EXAMPLES = [
        (IND, np.array([[0, 1, 1], [1, 0, 0], [0, 0, 1]])),
        # must not be considered binary
        (IND, np.array([[0, 1], [1, 0], [1, 1]])),
        (MC, [2, 3, 1]),
        (BIN, [0, 1, 1]),
        (CNT, [0.0, 1.5, 1.0]),
        (MC, np.array([[2], [3], [1]])),
        (BIN, np.array([[0], [1], [1]])),
        (CNT, np.array([[0.0], [1.5], [1.0]])),
        (MMC, np.array([[0, 2], [1, 3], [2, 3]])),
        (MCN, np.array([[0.5, 2.0], [1.1, 3.0], [2.0, 3.0]])),
    ]
    # expected type given input types, or None for error
    # (types will be tried in either order)
    EXPECTED = {
        (IND, IND): IND,
        (MC, MC): MC,
        (BIN, BIN): BIN,
        (MC, IND): None,
        (BIN, IND): None,
        (BIN, MC): MC,
        # Disallowed types
        (CNT, CNT): None,
        (MMC, MMC): None,
        (MCN, MCN): None,
        (IND, CNT): None,
        (MC, CNT): None,
        (BIN, CNT): None,
        (MMC, CNT): None,
        (MCN, CNT): None,
        (IND, MMC): None,
        (MC, MMC): None,
        (BIN, MMC): None,
        (MCN, MMC): None,
        (IND, MCN): None,
        (MC, MCN): None,
        (BIN, MCN): None,
    }

    for (type1, y1), (type2, y2) in product(EXAMPLES, repeat=2):
        try:
            expected = EXPECTED[type1, type2]
        except KeyError:
            expected = EXPECTED[type2, type1]
        if expected is None:
            with pytest.raises(ValueError):
                _check_targets(y1, y2)

            if type1 != type2:
                err_msg = (
                    "Classification metrics can't handle a mix "
                    "of {0} and {1} targets".format(type1, type2)
                )
                with pytest.raises(ValueError, match=err_msg):
                    _check_targets(y1, y2)

            else:
                if type1 not in (BIN, MC, IND):
                    err_msg = "{0} is not supported".format(type1)
                    with pytest.raises(ValueError, match=err_msg):
                        _check_targets(y1, y2)

        else:
            merged_type, y1out, y2out = _check_targets(y1, y2)
            assert merged_type == expected
            if merged_type.startswith("multilabel"):
                assert y1out.format == "csr"
                assert y2out.format == "csr"
            else:
                assert_array_equal(y1out, np.squeeze(y1))
                assert_array_equal(y2out, np.squeeze(y2))
            with pytest.raises(ValueError):
                _check_targets(y1[:-1], y2)

    # Make sure seq of seq is not supported
    y1 = [(1, 2), (0, 2, 3)]
    y2 = [(2,), (0, 2)]
    msg = (
        "You appear to be using a legacy multi-label data representation. "
        "Sequence of sequences are no longer supported; use a binary array"
        " or sparse matrix instead - the MultiLabelBinarizer"
        " transformer can convert to this format."
    )
    with pytest.raises(ValueError, match=msg):
        _check_targets(y1, y2)


def test__check_targets_multiclass_with_both_y_true_and_y_pred_binary():
    # https://github.com/scikit-learn/scikit-learn/issues/8098
    y_true = [0, 1]
    y_pred = [0, -1]
    assert _check_targets(y_true, y_pred)[0] == "multiclass"


def test_hinge_loss_binary():
    y_true = np.array([-1, 1, 1, -1])
    pred_decision = np.array([-8.5, 0.5, 1.5, -0.3])
    assert hinge_loss(y_true, pred_decision) == 1.2 / 4

    y_true = np.array([0, 2, 2, 0])
    pred_decision = np.array([-8.5, 0.5, 1.5, -0.3])
    assert hinge_loss(y_true, pred_decision) == 1.2 / 4


def test_hinge_loss_multiclass():
    pred_decision = np.array(
        [
            [+0.36, -0.17, -0.58, -0.99],
            [-0.54, -0.37, -0.48, -0.58],
            [-1.45, -0.58, -0.38, -0.17],
            [-0.54, -0.38, -0.48, -0.58],
            [-2.36, -0.79, -0.27, +0.24],
            [-1.45, -0.58, -0.38, -0.17],
        ]
    )
    y_true = np.array([0, 1, 2, 1, 3, 2])
    dummy_losses = np.array(
        [
            1 - pred_decision[0][0] + pred_decision[0][1],
            1 - pred_decision[1][1] + pred_decision[1][2],
            1 - pred_decision[2][2] + pred_decision[2][3],
            1 - pred_decision[3][1] + pred_decision[3][2],
            1 - pred_decision[4][3] + pred_decision[4][2],
            1 - pred_decision[5][2] + pred_decision[5][3],
        ]
    )
    np.clip(dummy_losses, 0, None, out=dummy_losses)
    dummy_hinge_loss = np.mean(dummy_losses)
    assert hinge_loss(y_true, pred_decision) == dummy_hinge_loss


def test_hinge_loss_multiclass_missing_labels_with_labels_none():
    y_true = np.array([0, 1, 2, 2])
    pred_decision = np.array(
        [
            [+1.27, 0.034, -0.68, -1.40],
            [-1.45, -0.58, -0.38, -0.17],
            [-2.36, -0.79, -0.27, +0.24],
            [-2.36, -0.79, -0.27, +0.24],
        ]
    )
    error_message = (
        "Please include all labels in y_true or pass labels as third argument"
    )
    with pytest.raises(ValueError, match=error_message):
        hinge_loss(y_true, pred_decision)


def test_hinge_loss_multiclass_no_consistent_pred_decision_shape():
    # test for inconsistency between multiclass problem and pred_decision
    # argument
    y_true = np.array([2, 1, 0, 1, 0, 1, 1])
    pred_decision = np.array([0, 1, 2, 1, 0, 2, 1])
    error_message = (
        "The shape of pred_decision cannot be 1d array"
        "with a multiclass target. pred_decision shape "
        "must be (n_samples, n_classes), that is "
        "(7, 3). Got: (7,)"
    )
    with pytest.raises(ValueError, match=re.escape(error_message)):
        hinge_loss(y_true=y_true, pred_decision=pred_decision)

    # test for inconsistency between pred_decision shape and labels number
    pred_decision = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [2, 0], [0, 1], [1, 0]])
    labels = [0, 1, 2]
    error_message = (
        "The shape of pred_decision is not "
        "consistent with the number of classes. "
        "With a multiclass target, pred_decision "
        "shape must be (n_samples, n_classes), that is "
        "(7, 3). Got: (7, 2)"
    )
    with pytest.raises(ValueError, match=re.escape(error_message)):
        hinge_loss(y_true=y_true, pred_decision=pred_decision, labels=labels)


def test_hinge_loss_multiclass_with_missing_labels():
    pred_decision = np.array(
        [
            [+0.36, -0.17, -0.58, -0.99],
            [-0.55, -0.38, -0.48, -0.58],
            [-1.45, -0.58, -0.38, -0.17],
            [-0.55, -0.38, -0.48, -0.58],
            [-1.45, -0.58, -0.38, -0.17],
        ]
    )
    y_true = np.array([0, 1, 2, 1, 2])
    labels = np.array([0, 1, 2, 3])
    dummy_losses = np.array(
        [
            1 - pred_decision[0][0] + pred_decision[0][1],
            1 - pred_decision[1][1] + pred_decision[1][2],
            1 - pred_decision[2][2] + pred_decision[2][3],
            1 - pred_decision[3][1] + pred_decision[3][2],
            1 - pred_decision[4][2] + pred_decision[4][3],
        ]
    )
    np.clip(dummy_losses, 0, None, out=dummy_losses)
    dummy_hinge_loss = np.mean(dummy_losses)
    assert hinge_loss(y_true, pred_decision, labels=labels) == dummy_hinge_loss


def test_hinge_loss_multiclass_missing_labels_only_two_unq_in_y_true():
    # non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/17630
    # check that we can compute the hinge loss when providing an array
    # with labels allowing to not have all labels in y_true
    pred_decision = np.array(
        [
            [+0.36, -0.17, -0.58],
            [-0.15, -0.58, -0.48],
            [-1.45, -0.58, -0.38],
            [-0.55, -0.78, -0.42],
            [-1.45, -0.58, -0.38],
        ]
    )
    y_true = np.array([0, 2, 2, 0, 2])
    labels = np.array([0, 1, 2])
    dummy_losses = np.array(
        [
            1 - pred_decision[0][0] + pred_decision[0][1],
            1 - pred_decision[1][2] + pred_decision[1][0],
            1 - pred_decision[2][2] + pred_decision[2][1],
            1 - pred_decision[3][0] + pred_decision[3][2],
            1 - pred_decision[4][2] + pred_decision[4][1],
        ]
    )
    np.clip(dummy_losses, 0, None, out=dummy_losses)
    dummy_hinge_loss = np.mean(dummy_losses)
    assert_almost_equal(
        hinge_loss(y_true, pred_decision, labels=labels), dummy_hinge_loss
    )


def test_hinge_loss_multiclass_invariance_lists():
    # Currently, invariance of string and integer labels cannot be tested
    # in common invariance tests because invariance tests for multiclass
    # decision functions is not implemented yet.
    y_true = ["blue", "green", "red", "green", "white", "red"]
    pred_decision = [
        [+0.36, -0.17, -0.58, -0.99],
        [-0.55, -0.38, -0.48, -0.58],
        [-1.45, -0.58, -0.38, -0.17],
        [-0.55, -0.38, -0.48, -0.58],
        [-2.36, -0.79, -0.27, +0.24],
        [-1.45, -0.58, -0.38, -0.17],
    ]
    dummy_losses = np.array(
        [
            1 - pred_decision[0][0] + pred_decision[0][1],
            1 - pred_decision[1][1] + pred_decision[1][2],
            1 - pred_decision[2][2] + pred_decision[2][3],
            1 - pred_decision[3][1] + pred_decision[3][2],
            1 - pred_decision[4][3] + pred_decision[4][2],
            1 - pred_decision[5][2] + pred_decision[5][3],
        ]
    )
    np.clip(dummy_losses, 0, None, out=dummy_losses)
    dummy_hinge_loss = np.mean(dummy_losses)
    assert hinge_loss(y_true, pred_decision) == dummy_hinge_loss


def test_log_loss():
    # binary case with symbolic labels ("no" < "yes")
    y_true = ["no", "no", "no", "yes", "yes", "yes"]
    y_pred = np.array(
        [[0.5, 0.5], [0.1, 0.9], [0.01, 0.99], [0.9, 0.1], [0.75, 0.25], [0.001, 0.999]]
    )
    loss = log_loss(y_true, y_pred)
    assert_almost_equal(loss, 1.8817971)

    # multiclass case; adapted from http://bit.ly/RJJHWA
    y_true = [1, 0, 2]
    y_pred = [[0.2, 0.7, 0.1], [0.6, 0.2, 0.2], [0.6, 0.1, 0.3]]
    loss = log_loss(y_true, y_pred, normalize=True)
    assert_almost_equal(loss, 0.6904911)

    # check that we got all the shapes and axes right
    # by doubling the length of y_true and y_pred
    y_true *= 2
    y_pred *= 2
    loss = log_loss(y_true, y_pred, normalize=False)
    assert_almost_equal(loss, 0.6904911 * 6, decimal=6)

    # check eps and handling of absolute zero and one probabilities
    y_pred = np.asarray(y_pred) > 0.5
    loss = log_loss(y_true, y_pred, normalize=True, eps=0.1)
    assert_almost_equal(loss, log_loss(y_true, np.clip(y_pred, 0.1, 0.9)))

    # raise error if number of classes are not equal.
    y_true = [1, 0, 2]
    y_pred = [[0.2, 0.7], [0.6, 0.5], [0.4, 0.1]]
    with pytest.raises(ValueError):
        log_loss(y_true, y_pred)

    # case when y_true is a string array object
    y_true = ["ham", "spam", "spam", "ham"]
    y_pred = [[0.2, 0.7], [0.6, 0.5], [0.4, 0.1], [0.7, 0.2]]
    loss = log_loss(y_true, y_pred)
    assert_almost_equal(loss, 1.0383217, decimal=6)

    # test labels option

    y_true = [2, 2]
    y_pred = [[0.2, 0.7], [0.6, 0.5]]
    y_score = np.array([[0.1, 0.9], [0.1, 0.9]])
    error_str = (
        r"y_true contains only one label \(2\). Please provide "
        r"the true labels explicitly through the labels argument."
    )
    with pytest.raises(ValueError, match=error_str):
        log_loss(y_true, y_pred)

    y_pred = [[0.2, 0.7], [0.6, 0.5], [0.2, 0.3]]
    error_str = "Found input variables with inconsistent numbers of samples: [3, 2]"
    (ValueError, error_str, log_loss, y_true, y_pred)

    # works when the labels argument is used

    true_log_loss = -np.mean(np.log(y_score[:, 1]))
    calculated_log_loss = log_loss(y_true, y_score, labels=[1, 2])
    assert_almost_equal(calculated_log_loss, true_log_loss)

    # ensure labels work when len(np.unique(y_true)) != y_pred.shape[1]
    y_true = [1, 2, 2]
    y_score2 = [[0.2, 0.7, 0.3], [0.6, 0.5, 0.3], [0.3, 0.9, 0.1]]
    loss = log_loss(y_true, y_score2, labels=[1, 2, 3])
    assert_almost_equal(loss, 1.0630345, decimal=6)


def test_log_loss_pandas_input():
    # case when input is a pandas series and dataframe gh-5715
    y_tr = np.array(["ham", "spam", "spam", "ham"])
    y_pr = np.array([[0.2, 0.7], [0.6, 0.5], [0.4, 0.1], [0.7, 0.2]])
    types = [(MockDataFrame, MockDataFrame)]
    try:
        from pandas import Series, DataFrame

        types.append((Series, DataFrame))
    except ImportError:
        pass
    for TrueInputType, PredInputType in types:
        # y_pred dataframe, y_true series
        y_true, y_pred = TrueInputType(y_tr), PredInputType(y_pr)
        loss = log_loss(y_true, y_pred)
        assert_almost_equal(loss, 1.0383217, decimal=6)


def test_brier_score_loss():
    # Check brier_score_loss function
    y_true = np.array([0, 1, 1, 0, 1, 1])
    y_pred = np.array([0.1, 0.8, 0.9, 0.3, 1.0, 0.95])
    true_score = linalg.norm(y_true - y_pred) ** 2 / len(y_true)

    assert_almost_equal(brier_score_loss(y_true, y_true), 0.0)
    assert_almost_equal(brier_score_loss(y_true, y_pred), true_score)
    assert_almost_equal(brier_score_loss(1.0 + y_true, y_pred), true_score)
    assert_almost_equal(brier_score_loss(2 * y_true - 1, y_pred), true_score)
    with pytest.raises(ValueError):
        brier_score_loss(y_true, y_pred[1:])
    with pytest.raises(ValueError):
        brier_score_loss(y_true, y_pred + 1.0)
    with pytest.raises(ValueError):
        brier_score_loss(y_true, y_pred - 1.0)

    # ensure to raise an error for multiclass y_true
    y_true = np.array([0, 1, 2, 0])
    y_pred = np.array([0.8, 0.6, 0.4, 0.2])
    error_message = (
        "Only binary classification is supported. The type of the target is multiclass"
    )

    with pytest.raises(ValueError, match=error_message):
        brier_score_loss(y_true, y_pred)

    # calculate correctly when there's only one class in y_true
    assert_almost_equal(brier_score_loss([-1], [0.4]), 0.16)
    assert_almost_equal(brier_score_loss([0], [0.4]), 0.16)
    assert_almost_equal(brier_score_loss([1], [0.4]), 0.36)
    assert_almost_equal(brier_score_loss(["foo"], [0.4], pos_label="bar"), 0.16)
    assert_almost_equal(brier_score_loss(["foo"], [0.4], pos_label="foo"), 0.36)


def test_balanced_accuracy_score_unseen():
    msg = "y_pred contains classes not in y_true"
    with pytest.warns(UserWarning, match=msg):
        balanced_accuracy_score([0, 0, 0], [0, 0, 1])


@pytest.mark.parametrize(
    "y_true,y_pred",
    [
        (["a", "b", "a", "b"], ["a", "a", "a", "b"]),
        (["a", "b", "c", "b"], ["a", "a", "a", "b"]),
        (["a", "a", "a", "b"], ["a", "b", "c", "b"]),
    ],
)
def test_balanced_accuracy_score(y_true, y_pred):
    macro_recall = recall_score(
        y_true, y_pred, average="macro", labels=np.unique(y_true)
    )
    with ignore_warnings():
        # Warnings are tested in test_balanced_accuracy_score_unseen
        balanced = balanced_accuracy_score(y_true, y_pred)
    assert balanced == pytest.approx(macro_recall)
    adjusted = balanced_accuracy_score(y_true, y_pred, adjusted=True)
    chance = balanced_accuracy_score(y_true, np.full_like(y_true, y_true[0]))
    assert adjusted == (balanced - chance) / (1 - chance)
