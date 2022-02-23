import re
import pytest
import numpy as np
import warnings
from scipy.sparse import csr_matrix

from sklearn import datasets
from sklearn import svm

from sklearn.utils.extmath import softmax
from sklearn.datasets import make_multilabel_classification
from sklearn.random_projection import _sparse_random_matrix
from sklearn.utils.validation import check_array, check_consistent_length
from sklearn.utils.validation import check_random_state

from sklearn.utils._testing import assert_allclose
from sklearn.utils._testing import assert_almost_equal
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_array_almost_equal

from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import coverage_error
from sklearn.metrics import det_curve
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics._ranking import _ndcg_sample_scores, _dcg_sample_scores
from sklearn.metrics import ndcg_score, dcg_score
from sklearn.metrics import top_k_accuracy_score

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


###############################################################################
# Utilities for testing

CURVE_FUNCS = [
    det_curve,
    precision_recall_curve,
    roc_curve,
]


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
    y_score = clf.fit(X[:half], y[:half]).predict_proba(X[half:])

    if binary:
        # only interested in probabilities of the positive case
        # XXX: do we really want a special API for the binary case?
        y_score = y_score[:, 1]

    y_pred = clf.predict(X[half:])
    y_true = y[half:]
    return y_true, y_pred, y_score


###############################################################################
# Tests


def _auc(y_true, y_score):
    """Alternative implementation to check for correctness of
    `roc_auc_score`."""
    pos_label = np.unique(y_true)[1]

    # Count the number of times positive samples are correctly ranked above
    # negative samples.
    pos = y_score[y_true == pos_label]
    neg = y_score[y_true != pos_label]
    diff_matrix = pos.reshape(1, -1) - neg.reshape(-1, 1)
    n_correct = np.sum(diff_matrix > 0)

    return n_correct / float(len(pos) * len(neg))


def _average_precision(y_true, y_score):
    """Alternative implementation to check for correctness of
    `average_precision_score`.

    Note that this implementation fails on some edge cases.
    For example, for constant predictions e.g. [0.5, 0.5, 0.5],
    y_true = [1, 0, 0] returns an average precision of 0.33...
    but y_true = [0, 0, 1] returns 1.0.
    """
    pos_label = np.unique(y_true)[1]
    n_pos = np.sum(y_true == pos_label)
    order = np.argsort(y_score)[::-1]
    y_score = y_score[order]
    y_true = y_true[order]

    score = 0
    for i in range(len(y_score)):
        if y_true[i] == pos_label:
            # Compute precision up to document i
            # i.e, percentage of relevant documents up to document i.
            prec = 0
            for j in range(0, i + 1):
                if y_true[j] == pos_label:
                    prec += 1.0
            prec /= i + 1.0
            score += prec

    return score / n_pos


def _average_precision_slow(y_true, y_score):
    """A second alternative implementation of average precision that closely
    follows the Wikipedia article's definition (see References). This should
    give identical results as `average_precision_score` for all inputs.

    References
    ----------
    .. [1] `Wikipedia entry for the Average precision
       <https://en.wikipedia.org/wiki/Average_precision>`_
    """
    precision, recall, threshold = precision_recall_curve(y_true, y_score)
    precision = list(reversed(precision))
    recall = list(reversed(recall))
    average_precision = 0
    for i in range(1, len(precision)):
        average_precision += precision[i] * (recall[i] - recall[i - 1])
    return average_precision


def _partial_roc_auc_score(y_true, y_predict, max_fpr):
    """Alternative implementation to check for correctness of `roc_auc_score`
    with `max_fpr` set.
    """

    def _partial_roc(y_true, y_predict, max_fpr):
        fpr, tpr, _ = roc_curve(y_true, y_predict)
        new_fpr = fpr[fpr <= max_fpr]
        new_fpr = np.append(new_fpr, max_fpr)
        new_tpr = tpr[fpr <= max_fpr]
        idx_out = np.argmax(fpr > max_fpr)
        idx_in = idx_out - 1
        x_interp = [fpr[idx_in], fpr[idx_out]]
        y_interp = [tpr[idx_in], tpr[idx_out]]
        new_tpr = np.append(new_tpr, np.interp(max_fpr, x_interp, y_interp))
        return (new_fpr, new_tpr)

    new_fpr, new_tpr = _partial_roc(y_true, y_predict, max_fpr)
    partial_auc = auc(new_fpr, new_tpr)

    # Formula (5) from McClish 1989
    fpr1 = 0
    fpr2 = max_fpr
    min_area = 0.5 * (fpr2 - fpr1) * (fpr2 + fpr1)
    max_area = fpr2 - fpr1
    return 0.5 * (1 + (partial_auc - min_area) / (max_area - min_area))


@pytest.mark.parametrize("drop", [True, False])
def test_roc_curve(drop):
    # Test Area under Receiver Operating Characteristic (ROC) curve
    y_true, _, y_score = make_prediction(binary=True)
    expected_auc = _auc(y_true, y_score)

    fpr, tpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=drop)
    roc_auc = auc(fpr, tpr)
    assert_array_almost_equal(roc_auc, expected_auc, decimal=2)
    assert_almost_equal(roc_auc, roc_auc_score(y_true, y_score))
    assert fpr.shape == tpr.shape
    assert fpr.shape == thresholds.shape


def test_roc_curve_end_points():
    # Make sure that roc_curve returns a curve start at 0 and ending and
    # 1 even in corner cases
    rng = np.random.RandomState(0)
    y_true = np.array([0] * 50 + [1] * 50)
    y_pred = rng.randint(3, size=100)
    fpr, tpr, thr = roc_curve(y_true, y_pred, drop_intermediate=True)
    assert fpr[0] == 0
    assert fpr[-1] == 1
    assert fpr.shape == tpr.shape
    assert fpr.shape == thr.shape


def test_roc_returns_consistency():
    # Test whether the returned threshold matches up with tpr
    # make small toy dataset
    y_true, _, y_score = make_prediction(binary=True)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # use the given thresholds to determine the tpr
    tpr_correct = []
    for t in thresholds:
        tp = np.sum((y_score >= t) & y_true)
        p = np.sum(y_true)
        tpr_correct.append(1.0 * tp / p)

    # compare tpr and tpr_correct to see if the thresholds' order was correct
    assert_array_almost_equal(tpr, tpr_correct, decimal=2)
    assert fpr.shape == tpr.shape
    assert fpr.shape == thresholds.shape


def test_roc_curve_multi():
    # roc_curve not applicable for multi-class problems
    y_true, _, y_score = make_prediction(binary=False)

    with pytest.raises(ValueError):
        roc_curve(y_true, y_score)


def test_roc_curve_confidence():
    # roc_curve for confidence scores
    y_true, _, y_score = make_prediction(binary=True)

    fpr, tpr, thresholds = roc_curve(y_true, y_score - 0.5)
    roc_auc = auc(fpr, tpr)
    assert_array_almost_equal(roc_auc, 0.90, decimal=2)
    assert fpr.shape == tpr.shape
    assert fpr.shape == thresholds.shape


def test_roc_curve_hard():
    # roc_curve for hard decisions
    y_true, pred, y_score = make_prediction(binary=True)

    # always predict one
    trivial_pred = np.ones(y_true.shape)
    fpr, tpr, thresholds = roc_curve(y_true, trivial_pred)
    roc_auc = auc(fpr, tpr)
    assert_array_almost_equal(roc_auc, 0.50, decimal=2)
    assert fpr.shape == tpr.shape
    assert fpr.shape == thresholds.shape

    # always predict zero
    trivial_pred = np.zeros(y_true.shape)
    fpr, tpr, thresholds = roc_curve(y_true, trivial_pred)
    roc_auc = auc(fpr, tpr)
    assert_array_almost_equal(roc_auc, 0.50, decimal=2)
    assert fpr.shape == tpr.shape
    assert fpr.shape == thresholds.shape

    # hard decisions
    fpr, tpr, thresholds = roc_curve(y_true, pred)
    roc_auc = auc(fpr, tpr)
    assert_array_almost_equal(roc_auc, 0.78, decimal=2)
    assert fpr.shape == tpr.shape
    assert fpr.shape == thresholds.shape


def test_roc_curve_one_label():
    y_true = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    y_pred = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    # assert there are warnings
    expected_message = (
        "No negative samples in y_true, false positive value should be meaningless"
    )
    with pytest.warns(UndefinedMetricWarning, match=expected_message):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # all true labels, all fpr should be nan
    assert_array_equal(fpr, np.full(len(thresholds), np.nan))
    assert fpr.shape == tpr.shape
    assert fpr.shape == thresholds.shape

    # assert there are warnings
    expected_message = (
        "No positive samples in y_true, true positive value should be meaningless"
    )
    with pytest.warns(UndefinedMetricWarning, match=expected_message):
        fpr, tpr, thresholds = roc_curve([1 - x for x in y_true], y_pred)
    # all negative labels, all tpr should be nan
    assert_array_equal(tpr, np.full(len(thresholds), np.nan))
    assert fpr.shape == tpr.shape
    assert fpr.shape == thresholds.shape


def test_roc_curve_toydata():
    # Binary classification
    y_true = [0, 1]
    y_score = [0, 1]
    tpr, fpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    assert_array_almost_equal(tpr, [0, 0, 1])
    assert_array_almost_equal(fpr, [0, 1, 1])
    assert_almost_equal(roc_auc, 1.0)

    y_true = [0, 1]
    y_score = [1, 0]
    tpr, fpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    assert_array_almost_equal(tpr, [0, 1, 1])
    assert_array_almost_equal(fpr, [0, 0, 1])
    assert_almost_equal(roc_auc, 0.0)

    y_true = [1, 0]
    y_score = [1, 1]
    tpr, fpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    assert_array_almost_equal(tpr, [0, 1])
    assert_array_almost_equal(fpr, [0, 1])
    assert_almost_equal(roc_auc, 0.5)

    y_true = [1, 0]
    y_score = [1, 0]
    tpr, fpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    assert_array_almost_equal(tpr, [0, 0, 1])
    assert_array_almost_equal(fpr, [0, 1, 1])
    assert_almost_equal(roc_auc, 1.0)

    y_true = [1, 0]
    y_score = [0.5, 0.5]
    tpr, fpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    assert_array_almost_equal(tpr, [0, 1])
    assert_array_almost_equal(fpr, [0, 1])
    assert_almost_equal(roc_auc, 0.5)

    y_true = [0, 0]
    y_score = [0.25, 0.75]
    # assert UndefinedMetricWarning because of no positive sample in y_true
    expected_message = (
        "No positive samples in y_true, true positive value should be meaningless"
    )
    with pytest.warns(UndefinedMetricWarning, match=expected_message):
        tpr, fpr, _ = roc_curve(y_true, y_score)

    with pytest.raises(ValueError):
        roc_auc_score(y_true, y_score)
    assert_array_almost_equal(tpr, [0.0, 0.5, 1.0])
    assert_array_almost_equal(fpr, [np.nan, np.nan, np.nan])

    y_true = [1, 1]
    y_score = [0.25, 0.75]
    # assert UndefinedMetricWarning because of no negative sample in y_true
    expected_message = (
        "No negative samples in y_true, false positive value should be meaningless"
    )
    with pytest.warns(UndefinedMetricWarning, match=expected_message):
        tpr, fpr, _ = roc_curve(y_true, y_score)

    with pytest.raises(ValueError):
        roc_auc_score(y_true, y_score)
    assert_array_almost_equal(tpr, [np.nan, np.nan, np.nan])
    assert_array_almost_equal(fpr, [0.0, 0.5, 1.0])

    # Multi-label classification task
    y_true = np.array([[0, 1], [0, 1]])
    y_score = np.array([[0, 1], [0, 1]])
    with pytest.raises(ValueError):
        roc_auc_score(y_true, y_score, average="macro")
    with pytest.raises(ValueError):
        roc_auc_score(y_true, y_score, average="weighted")
    assert_almost_equal(roc_auc_score(y_true, y_score, average="samples"), 1.0)
    assert_almost_equal(roc_auc_score(y_true, y_score, average="micro"), 1.0)

    y_true = np.array([[0, 1], [0, 1]])
    y_score = np.array([[0, 1], [1, 0]])
    with pytest.raises(ValueError):
        roc_auc_score(y_true, y_score, average="macro")
    with pytest.raises(ValueError):
        roc_auc_score(y_true, y_score, average="weighted")
    assert_almost_equal(roc_auc_score(y_true, y_score, average="samples"), 0.5)
    assert_almost_equal(roc_auc_score(y_true, y_score, average="micro"), 0.5)

    y_true = np.array([[1, 0], [0, 1]])
    y_score = np.array([[0, 1], [1, 0]])
    assert_almost_equal(roc_auc_score(y_true, y_score, average="macro"), 0)
    assert_almost_equal(roc_auc_score(y_true, y_score, average="weighted"), 0)
    assert_almost_equal(roc_auc_score(y_true, y_score, average="samples"), 0)
    assert_almost_equal(roc_auc_score(y_true, y_score, average="micro"), 0)

    y_true = np.array([[1, 0], [0, 1]])
    y_score = np.array([[0.5, 0.5], [0.5, 0.5]])
    assert_almost_equal(roc_auc_score(y_true, y_score, average="macro"), 0.5)
    assert_almost_equal(roc_auc_score(y_true, y_score, average="weighted"), 0.5)
    assert_almost_equal(roc_auc_score(y_true, y_score, average="samples"), 0.5)
    assert_almost_equal(roc_auc_score(y_true, y_score, average="micro"), 0.5)


def test_roc_curve_drop_intermediate():
    # Test that drop_intermediate drops the correct thresholds
    y_true = [0, 0, 0, 0, 1, 1]
    y_score = [0.0, 0.2, 0.5, 0.6, 0.7, 1.0]
    tpr, fpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=True)
    assert_array_almost_equal(thresholds, [2.0, 1.0, 0.7, 0.0])

    # Test dropping thresholds with repeating scores
    y_true = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    y_score = [0.0, 0.1, 0.6, 0.6, 0.7, 0.8, 0.9, 0.6, 0.7, 0.8, 0.9, 0.9, 1.0]
    tpr, fpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=True)
    assert_array_almost_equal(thresholds, [2.0, 1.0, 0.9, 0.7, 0.6, 0.0])


def test_roc_curve_fpr_tpr_increasing():
    # Ensure that fpr and tpr returned by roc_curve are increasing.
    # Construct an edge case with float y_score and sample_weight
    # when some adjacent values of fpr and tpr are actually the same.
    y_true = [0, 0, 1, 1, 1]
    y_score = [0.1, 0.7, 0.3, 0.4, 0.5]
    sample_weight = np.repeat(0.2, 5)
    fpr, tpr, _ = roc_curve(y_true, y_score, sample_weight=sample_weight)
    assert (np.diff(fpr) < 0).sum() == 0
    assert (np.diff(tpr) < 0).sum() == 0


def test_auc():
    # Test Area Under Curve (AUC) computation
    x = [0, 1]
    y = [0, 1]
    assert_array_almost_equal(auc(x, y), 0.5)
    x = [1, 0]
    y = [0, 1]
    assert_array_almost_equal(auc(x, y), 0.5)
    x = [1, 0, 0]
    y = [0, 1, 1]
    assert_array_almost_equal(auc(x, y), 0.5)
    x = [0, 1]
    y = [1, 1]
    assert_array_almost_equal(auc(x, y), 1)
    x = [0, 0.5, 1]
    y = [0, 0.5, 1]
    assert_array_almost_equal(auc(x, y), 0.5)


def test_auc_errors():
    # Incompatible shapes
    with pytest.raises(ValueError):
        auc([0.0, 0.5, 1.0], [0.1, 0.2])

    # Too few x values
    with pytest.raises(ValueError):
        auc([0.0], [0.1])

    # x is not in order
    x = [2, 1, 3, 4]
    y = [5, 6, 7, 8]
    error_message = "x is neither increasing nor decreasing : {}".format(np.array(x))
    with pytest.raises(ValueError, match=re.escape(error_message)):
        auc(x, y)


@pytest.mark.parametrize(
    "y_true, labels",
    [
        (np.array([0, 1, 0, 2]), [0, 1, 2]),
        (np.array([0, 1, 0, 2]), None),
        (["a", "b", "a", "c"], ["a", "b", "c"]),
        (["a", "b", "a", "c"], None),
    ],
)
def test_multiclass_ovo_roc_auc_toydata(y_true, labels):
    # Tests the one-vs-one multiclass ROC AUC algorithm
    # on a small example, representative of an expected use case.
    y_scores = np.array(
        [[0.1, 0.8, 0.1], [0.3, 0.4, 0.3], [0.35, 0.5, 0.15], [0, 0.2, 0.8]]
    )

    # Used to compute the expected output.
    # Consider labels 0 and 1:
    # positive label is 0, negative label is 1
    score_01 = roc_auc_score([1, 0, 1], [0.1, 0.3, 0.35])
    # positive label is 1, negative label is 0
    score_10 = roc_auc_score([0, 1, 0], [0.8, 0.4, 0.5])
    average_score_01 = (score_01 + score_10) / 2

    # Consider labels 0 and 2:
    score_02 = roc_auc_score([1, 1, 0], [0.1, 0.35, 0])
    score_20 = roc_auc_score([0, 0, 1], [0.1, 0.15, 0.8])
    average_score_02 = (score_02 + score_20) / 2

    # Consider labels 1 and 2:
    score_12 = roc_auc_score([1, 0], [0.4, 0.2])
    score_21 = roc_auc_score([0, 1], [0.3, 0.8])
    average_score_12 = (score_12 + score_21) / 2

    # Unweighted, one-vs-one multiclass ROC AUC algorithm
    ovo_unweighted_score = (average_score_01 + average_score_02 + average_score_12) / 3
    assert_almost_equal(
        roc_auc_score(y_true, y_scores, labels=labels, multi_class="ovo"),
        ovo_unweighted_score,
    )

    # Weighted, one-vs-one multiclass ROC AUC algorithm
    # Each term is weighted by the prevalence for the positive label.
    pair_scores = [average_score_01, average_score_02, average_score_12]
    prevalence = [0.75, 0.75, 0.50]
    ovo_weighted_score = np.average(pair_scores, weights=prevalence)
    assert_almost_equal(
        roc_auc_score(
            y_true, y_scores, labels=labels, multi_class="ovo", average="weighted"
        ),
        ovo_weighted_score,
    )


@pytest.mark.parametrize(
    "y_true, labels",
    [
        (np.array([0, 2, 0, 2]), [0, 1, 2]),
        (np.array(["a", "d", "a", "d"]), ["a", "b", "d"]),
    ],
)
def test_multiclass_ovo_roc_auc_toydata_binary(y_true, labels):
    # Tests the one-vs-one multiclass ROC AUC algorithm for binary y_true
    #
    # on a small example, representative of an expected use case.
    y_scores = np.array(
        [[0.2, 0.0, 0.8], [0.6, 0.0, 0.4], [0.55, 0.0, 0.45], [0.4, 0.0, 0.6]]
    )

    # Used to compute the expected output.
    # Consider labels 0 and 1:
    # positive label is 0, negative label is 1
    score_01 = roc_auc_score([1, 0, 1, 0], [0.2, 0.6, 0.55, 0.4])
    # positive label is 1, negative label is 0
    score_10 = roc_auc_score([0, 1, 0, 1], [0.8, 0.4, 0.45, 0.6])
    ovo_score = (score_01 + score_10) / 2

    assert_almost_equal(
        roc_auc_score(y_true, y_scores, labels=labels, multi_class="ovo"), ovo_score
    )

    # Weighted, one-vs-one multiclass ROC AUC algorithm
    assert_almost_equal(
        roc_auc_score(
            y_true, y_scores, labels=labels, multi_class="ovo", average="weighted"
        ),
        ovo_score,
    )


@pytest.mark.parametrize(
    "y_true, labels",
    [
        (np.array([0, 1, 2, 2]), None),
        (["a", "b", "c", "c"], None),
        ([0, 1, 2, 2], [0, 1, 2]),
        (["a", "b", "c", "c"], ["a", "b", "c"]),
    ],
)
def test_multiclass_ovr_roc_auc_toydata(y_true, labels):
    # Tests the unweighted, one-vs-rest multiclass ROC AUC algorithm
    # on a small example, representative of an expected use case.
    y_scores = np.array(
        [[1.0, 0.0, 0.0], [0.1, 0.5, 0.4], [0.1, 0.1, 0.8], [0.3, 0.3, 0.4]]
    )
    # Compute the expected result by individually computing the 'one-vs-rest'
    # ROC AUC scores for classes 0, 1, and 2.
    out_0 = roc_auc_score([1, 0, 0, 0], y_scores[:, 0])
    out_1 = roc_auc_score([0, 1, 0, 0], y_scores[:, 1])
    out_2 = roc_auc_score([0, 0, 1, 1], y_scores[:, 2])
    result_unweighted = (out_0 + out_1 + out_2) / 3.0

    assert_almost_equal(
        roc_auc_score(y_true, y_scores, multi_class="ovr", labels=labels),
        result_unweighted,
    )

    # Tests the weighted, one-vs-rest multiclass ROC AUC algorithm
    # on the same input (Provost & Domingos, 2000)
    result_weighted = out_0 * 0.25 + out_1 * 0.25 + out_2 * 0.5
    assert_almost_equal(
        roc_auc_score(
            y_true, y_scores, multi_class="ovr", labels=labels, average="weighted"
        ),
        result_weighted,
    )


@pytest.mark.parametrize(
    "msg, y_true, labels",
    [
        ("Parameter 'labels' must be unique", np.array([0, 1, 2, 2]), [0, 2, 0]),
        (
            "Parameter 'labels' must be unique",
            np.array(["a", "b", "c", "c"]),
            ["a", "a", "b"],
        ),
        (
            "Number of classes in y_true not equal to the number of columns "
            "in 'y_score'",
            np.array([0, 2, 0, 2]),
            None,
        ),
        (
            "Parameter 'labels' must be ordered",
            np.array(["a", "b", "c", "c"]),
            ["a", "c", "b"],
        ),
        (
            "Number of given labels, 2, not equal to the number of columns in "
            "'y_score', 3",
            np.array([0, 1, 2, 2]),
            [0, 1],
        ),
        (
            "Number of given labels, 2, not equal to the number of columns in "
            "'y_score', 3",
            np.array(["a", "b", "c", "c"]),
            ["a", "b"],
        ),
        (
            "Number of given labels, 4, not equal to the number of columns in "
            "'y_score', 3",
            np.array([0, 1, 2, 2]),
            [0, 1, 2, 3],
        ),
        (
            "Number of given labels, 4, not equal to the number of columns in "
            "'y_score', 3",
            np.array(["a", "b", "c", "c"]),
            ["a", "b", "c", "d"],
        ),
        (
            "'y_true' contains labels not in parameter 'labels'",
            np.array(["a", "b", "c", "e"]),
            ["a", "b", "c"],
        ),
        (
            "'y_true' contains labels not in parameter 'labels'",
            np.array(["a", "b", "c", "d"]),
            ["a", "b", "c"],
        ),
        (
            "'y_true' contains labels not in parameter 'labels'",
            np.array([0, 1, 2, 3]),
            [0, 1, 2],
        ),
    ],
)
@pytest.mark.parametrize("multi_class", ["ovo", "ovr"])
def test_roc_auc_score_multiclass_labels_error(msg, y_true, labels, multi_class):
    y_scores = np.array(
        [[0.1, 0.8, 0.1], [0.3, 0.4, 0.3], [0.35, 0.5, 0.15], [0, 0.2, 0.8]]
    )

    with pytest.raises(ValueError, match=msg):
        roc_auc_score(y_true, y_scores, labels=labels, multi_class=multi_class)


@pytest.mark.parametrize(
    "msg, kwargs",
    [
        (
            (
                r"average must be one of \('macro', 'weighted'\) for "
                r"multiclass problems"
            ),
            {"average": "samples", "multi_class": "ovo"},
        ),
        (
            (
                r"average must be one of \('macro', 'weighted'\) for "
                r"multiclass problems"
            ),
            {"average": "micro", "multi_class": "ovr"},
        ),
        (
            (
                r"sample_weight is not supported for multiclass one-vs-one "
                r"ROC AUC, 'sample_weight' must be None in this case"
            ),
            {"multi_class": "ovo", "sample_weight": []},
        ),
        (
            (
                r"Partial AUC computation not available in multiclass setting, "
                r"'max_fpr' must be set to `None`, received `max_fpr=0.5` "
                r"instead"
            ),
            {"multi_class": "ovo", "max_fpr": 0.5},
        ),
        (
            (
                r"multi_class='ovp' is not supported for multiclass ROC AUC, "
                r"multi_class must be in \('ovo', 'ovr'\)"
            ),
            {"multi_class": "ovp"},
        ),
        (r"multi_class must be in \('ovo', 'ovr'\)", {}),
    ],
)
def test_roc_auc_score_multiclass_error(msg, kwargs):
    # Test that roc_auc_score function returns an error when trying
    # to compute multiclass AUC for parameters where an output
    # is not defined.
    rng = check_random_state(404)
    y_score = rng.rand(20, 3)
    y_prob = softmax(y_score)
    y_true = rng.randint(0, 3, size=20)
    with pytest.raises(ValueError, match=msg):
        roc_auc_score(y_true, y_prob, **kwargs)


def test_auc_score_non_binary_class():
    # Test that roc_auc_score function returns an error when trying
    # to compute AUC for non-binary class values.
    rng = check_random_state(404)
    y_pred = rng.rand(10)
    # y_true contains only one class value
    y_true = np.zeros(10, dtype="int")
    err_msg = "ROC AUC score is not defined"
    with pytest.raises(ValueError, match=err_msg):
        roc_auc_score(y_true, y_pred)
    y_true = np.ones(10, dtype="int")
    with pytest.raises(ValueError, match=err_msg):
        roc_auc_score(y_true, y_pred)
    y_true = np.full(10, -1, dtype="int")
    with pytest.raises(ValueError, match=err_msg):
        roc_auc_score(y_true, y_pred)

    with warnings.catch_warnings(record=True):
        rng = check_random_state(404)
        y_pred = rng.rand(10)
        # y_true contains only one class value
        y_true = np.zeros(10, dtype="int")
        with pytest.raises(ValueError, match=err_msg):
            roc_auc_score(y_true, y_pred)
        y_true = np.ones(10, dtype="int")
        with pytest.raises(ValueError, match=err_msg):
            roc_auc_score(y_true, y_pred)
        y_true = np.full(10, -1, dtype="int")
        with pytest.raises(ValueError, match=err_msg):
            roc_auc_score(y_true, y_pred)


@pytest.mark.parametrize("curve_func", CURVE_FUNCS)
def test_binary_clf_curve_multiclass_error(curve_func):
    rng = check_random_state(404)
    y_true = rng.randint(0, 3, size=10)
    y_pred = rng.rand(10)
    msg = "multiclass format is not supported"
    with pytest.raises(ValueError, match=msg):
        curve_func(y_true, y_pred)


@pytest.mark.parametrize("curve_func", CURVE_FUNCS)
def test_binary_clf_curve_implicit_pos_label(curve_func):
    # Check that using string class labels raises an informative
    # error for any supported string dtype:
    msg = (
        "y_true takes value in {'a', 'b'} and pos_label is "
        "not specified: either make y_true take "
        "value in {0, 1} or {-1, 1} or pass pos_label "
        "explicitly."
    )
    with pytest.raises(ValueError, match=msg):
        curve_func(np.array(["a", "b"], dtype="<U1"), [0.0, 1.0])

    with pytest.raises(ValueError, match=msg):
        curve_func(np.array(["a", "b"], dtype=object), [0.0, 1.0])

    # The error message is slightly different for bytes-encoded
    # class labels, but otherwise the behavior is the same:
    msg = (
        "y_true takes value in {b'a', b'b'} and pos_label is "
        "not specified: either make y_true take "
        "value in {0, 1} or {-1, 1} or pass pos_label "
        "explicitly."
    )
    with pytest.raises(ValueError, match=msg):
        curve_func(np.array([b"a", b"b"], dtype="<S1"), [0.0, 1.0])

    # Check that it is possible to use floating point class labels
    # that are interpreted similarly to integer class labels:
    y_pred = [0.0, 1.0, 0.2, 0.42]
    int_curve = curve_func([0, 1, 1, 0], y_pred)
    float_curve = curve_func([0.0, 1.0, 1.0, 0.0], y_pred)
    for int_curve_part, float_curve_part in zip(int_curve, float_curve):
        np.testing.assert_allclose(int_curve_part, float_curve_part)


@pytest.mark.parametrize("curve_func", CURVE_FUNCS)
def test_binary_clf_curve_zero_sample_weight(curve_func):
    y_true = [0, 0, 1, 1, 1]
    y_score = [0.1, 0.2, 0.3, 0.4, 0.5]
    sample_weight = [1, 1, 1, 0.5, 0]

    result_1 = curve_func(y_true, y_score, sample_weight=sample_weight)
    result_2 = curve_func(y_true[:-1], y_score[:-1], sample_weight=sample_weight[:-1])

    for arr_1, arr_2 in zip(result_1, result_2):
        assert_allclose(arr_1, arr_2)


def test_precision_recall_curve():
    y_true, _, y_score = make_prediction(binary=True)
    _test_precision_recall_curve(y_true, y_score)

    # Use {-1, 1} for labels; make sure original labels aren't modified
    y_true[np.where(y_true == 0)] = -1
    y_true_copy = y_true.copy()
    _test_precision_recall_curve(y_true, y_score)
    assert_array_equal(y_true_copy, y_true)

    labels = [1, 0, 0, 1]
    predict_probas = [1, 2, 3, 4]
    p, r, t = precision_recall_curve(labels, predict_probas)
    assert_array_almost_equal(p, np.array([0.5, 0.33333333, 0.5, 1.0, 1.0]))
    assert_array_almost_equal(r, np.array([1.0, 0.5, 0.5, 0.5, 0.0]))
    assert_array_almost_equal(t, np.array([1, 2, 3, 4]))
    assert p.size == r.size
    assert p.size == t.size + 1


def _test_precision_recall_curve(y_true, y_score):
    # Test Precision-Recall and aread under PR curve
    p, r, thresholds = precision_recall_curve(y_true, y_score)
    precision_recall_auc = _average_precision_slow(y_true, y_score)
    assert_array_almost_equal(precision_recall_auc, 0.859, 3)
    assert_array_almost_equal(
        precision_recall_auc, average_precision_score(y_true, y_score)
    )
    # `_average_precision` is not very precise in case of 0.5 ties: be tolerant
    assert_almost_equal(
        _average_precision(y_true, y_score), precision_recall_auc, decimal=2
    )
    assert p.size == r.size
    assert p.size == thresholds.size + 1
    # Smoke test in the case of proba having only one value
    p, r, thresholds = precision_recall_curve(y_true, np.zeros_like(y_score))
    assert p.size == r.size
    assert p.size == thresholds.size + 1


def test_precision_recall_curve_toydata():
    with np.errstate(all="raise"):
        # Binary classification
        y_true = [0, 1]
        y_score = [0, 1]
        p, r, _ = precision_recall_curve(y_true, y_score)
        auc_prc = average_precision_score(y_true, y_score)
        assert_array_almost_equal(p, [1, 1])
        assert_array_almost_equal(r, [1, 0])
        assert_almost_equal(auc_prc, 1.0)

        y_true = [0, 1]
        y_score = [1, 0]
        p, r, _ = precision_recall_curve(y_true, y_score)
        auc_prc = average_precision_score(y_true, y_score)
        assert_array_almost_equal(p, [0.5, 0.0, 1.0])
        assert_array_almost_equal(r, [1.0, 0.0, 0.0])
        # Here we are doing a terrible prediction: we are always getting
        # it wrong, hence the average_precision_score is the accuracy at
        # chance: 50%
        assert_almost_equal(auc_prc, 0.5)

        y_true = [1, 0]
        y_score = [1, 1]
        p, r, _ = precision_recall_curve(y_true, y_score)
        auc_prc = average_precision_score(y_true, y_score)
        assert_array_almost_equal(p, [0.5, 1])
        assert_array_almost_equal(r, [1.0, 0])
        assert_almost_equal(auc_prc, 0.5)

        y_true = [1, 0]
        y_score = [1, 0]
        p, r, _ = precision_recall_curve(y_true, y_score)
        auc_prc = average_precision_score(y_true, y_score)
        assert_array_almost_equal(p, [1, 1])
        assert_array_almost_equal(r, [1, 0])
        assert_almost_equal(auc_prc, 1.0)

        y_true = [1, 0]
        y_score = [0.5, 0.5]
        p, r, _ = precision_recall_curve(y_true, y_score)
        auc_prc = average_precision_score(y_true, y_score)
        assert_array_almost_equal(p, [0.5, 1])
        assert_array_almost_equal(r, [1, 0.0])
        assert_almost_equal(auc_prc, 0.5)

        y_true = [0, 0]
        y_score = [0.25, 0.75]
        with pytest.raises(Exception):
            precision_recall_curve(y_true, y_score)
        with pytest.raises(Exception):
            average_precision_score(y_true, y_score)

        y_true = [1, 1]
        y_score = [0.25, 0.75]
        p, r, _ = precision_recall_curve(y_true, y_score)
        assert_almost_equal(average_precision_score(y_true, y_score), 1.0)
        assert_array_almost_equal(p, [1.0, 1.0, 1.0])
        assert_array_almost_equal(r, [1, 0.5, 0.0])

        # Multi-label classification task
        y_true = np.array([[0, 1], [0, 1]])
        y_score = np.array([[0, 1], [0, 1]])
        with pytest.raises(Exception):
            average_precision_score(y_true, y_score, average="macro")
        with pytest.raises(Exception):
            average_precision_score(y_true, y_score, average="weighted")
        assert_almost_equal(
            average_precision_score(y_true, y_score, average="samples"), 1.0
        )
        assert_almost_equal(
            average_precision_score(y_true, y_score, average="micro"), 1.0
        )

        y_true = np.array([[0, 1], [0, 1]])
        y_score = np.array([[0, 1], [1, 0]])
        with pytest.raises(Exception):
            average_precision_score(y_true, y_score, average="macro")
        with pytest.raises(Exception):
            average_precision_score(y_true, y_score, average="weighted")
        assert_almost_equal(
            average_precision_score(y_true, y_score, average="samples"), 0.75
        )
        assert_almost_equal(
            average_precision_score(y_true, y_score, average="micro"), 0.5
        )

        y_true = np.array([[1, 0], [0, 1]])
        y_score = np.array([[0, 1], [1, 0]])
        assert_almost_equal(
            average_precision_score(y_true, y_score, average="macro"), 0.5
        )
        assert_almost_equal(
            average_precision_score(y_true, y_score, average="weighted"), 0.5
        )
        assert_almost_equal(
            average_precision_score(y_true, y_score, average="samples"), 0.5
        )
        assert_almost_equal(
            average_precision_score(y_true, y_score, average="micro"), 0.5
        )

        y_true = np.array([[1, 0], [0, 1]])
        y_score = np.array([[0.5, 0.5], [0.5, 0.5]])
        assert_almost_equal(
            average_precision_score(y_true, y_score, average="macro"), 0.5
        )
        assert_almost_equal(
            average_precision_score(y_true, y_score, average="weighted"), 0.5
        )
        assert_almost_equal(
            average_precision_score(y_true, y_score, average="samples"), 0.5
        )
        assert_almost_equal(
            average_precision_score(y_true, y_score, average="micro"), 0.5
        )

    with np.errstate(all="ignore"):
        # if one class is never present weighted should not be NaN
        y_true = np.array([[0, 0], [0, 1]])
        y_score = np.array([[0, 0], [0, 1]])
        assert_almost_equal(
            average_precision_score(y_true, y_score, average="weighted"), 1
        )


def test_average_precision_constant_values():
    # Check the average_precision_score of a constant predictor is
    # the TPR

    # Generate a dataset with 25% of positives
    y_true = np.zeros(100, dtype=int)
    y_true[::4] = 1
    # And a constant score
    y_score = np.ones(100)
    # The precision is then the fraction of positive whatever the recall
    # is, as there is only one threshold:
    assert average_precision_score(y_true, y_score) == 0.25


def test_average_precision_score_pos_label_errors():
    # Raise an error when pos_label is not in binary y_true
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    err_msg = r"pos_label=2 is not a valid label. It should be one of \[0, 1\]"
    with pytest.raises(ValueError, match=err_msg):
        average_precision_score(y_true, y_pred, pos_label=2)
    # Raise an error for multilabel-indicator y_true with
    # pos_label other than 1
    y_true = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
    y_pred = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    err_msg = (
        "Parameter pos_label is fixed to 1 for multilabel-indicator y_true. "
        "Do not set pos_label or set pos_label to 1."
    )
    with pytest.raises(ValueError, match=err_msg):
        average_precision_score(y_true, y_pred, pos_label=0)


def test_score_scale_invariance():
    # Test that average_precision_score and roc_auc_score are invariant by
    # the scaling or shifting of probabilities
    # This test was expanded (added scaled_down) in response to github
    # issue #3864 (and others), where overly aggressive rounding was causing
    # problems for users with very small y_score values
    y_true, _, y_score = make_prediction(binary=True)

    roc_auc = roc_auc_score(y_true, y_score)
    roc_auc_scaled_up = roc_auc_score(y_true, 100 * y_score)
    roc_auc_scaled_down = roc_auc_score(y_true, 1e-6 * y_score)
    roc_auc_shifted = roc_auc_score(y_true, y_score - 10)
    assert roc_auc == roc_auc_scaled_up
    assert roc_auc == roc_auc_scaled_down
    assert roc_auc == roc_auc_shifted

    pr_auc = average_precision_score(y_true, y_score)
    pr_auc_scaled_up = average_precision_score(y_true, 100 * y_score)
    pr_auc_scaled_down = average_precision_score(y_true, 1e-6 * y_score)
    pr_auc_shifted = average_precision_score(y_true, y_score - 10)
    assert pr_auc == pr_auc_scaled_up
    assert pr_auc == pr_auc_scaled_down
    assert pr_auc == pr_auc_shifted


@pytest.mark.parametrize(
    "y_true,y_score,expected_fpr,expected_fnr",
    [
        ([0, 0, 1], [0, 0.5, 1], [0], [0]),
        ([0, 0, 1], [0, 0.25, 0.5], [0], [0]),
        ([0, 0, 1], [0.5, 0.75, 1], [0], [0]),
        ([0, 0, 1], [0.25, 0.5, 0.75], [0], [0]),
        ([0, 1, 0], [0, 0.5, 1], [0.5], [0]),
        ([0, 1, 0], [0, 0.25, 0.5], [0.5], [0]),
        ([0, 1, 0], [0.5, 0.75, 1], [0.5], [0]),
        ([0, 1, 0], [0.25, 0.5, 0.75], [0.5], [0]),
        ([0, 1, 1], [0, 0.5, 1], [0.0], [0]),
        ([0, 1, 1], [0, 0.25, 0.5], [0], [0]),
        ([0, 1, 1], [0.5, 0.75, 1], [0], [0]),
        ([0, 1, 1], [0.25, 0.5, 0.75], [0], [0]),
        ([1, 0, 0], [0, 0.5, 1], [1, 1, 0.5], [0, 1, 1]),
        ([1, 0, 0], [0, 0.25, 0.5], [1, 1, 0.5], [0, 1, 1]),
        ([1, 0, 0], [0.5, 0.75, 1], [1, 1, 0.5], [0, 1, 1]),
        ([1, 0, 0], [0.25, 0.5, 0.75], [1, 1, 0.5], [0, 1, 1]),
        ([1, 0, 1], [0, 0.5, 1], [1, 1, 0], [0, 0.5, 0.5]),
        ([1, 0, 1], [0, 0.25, 0.5], [1, 1, 0], [0, 0.5, 0.5]),
        ([1, 0, 1], [0.5, 0.75, 1], [1, 1, 0], [0, 0.5, 0.5]),
        ([1, 0, 1], [0.25, 0.5, 0.75], [1, 1, 0], [0, 0.5, 0.5]),
    ],
)
def test_det_curve_toydata(y_true, y_score, expected_fpr, expected_fnr):
    # Check on a batch of small examples.
    fpr, fnr, _ = det_curve(y_true, y_score)

    assert_allclose(fpr, expected_fpr)
    assert_allclose(fnr, expected_fnr)


@pytest.mark.parametrize(
    "y_true,y_score,expected_fpr,expected_fnr",
    [
        ([1, 0], [0.5, 0.5], [1], [0]),
        ([0, 1], [0.5, 0.5], [1], [0]),
        ([0, 0, 1], [0.25, 0.5, 0.5], [0.5], [0]),
        ([0, 1, 0], [0.25, 0.5, 0.5], [0.5], [0]),
        ([0, 1, 1], [0.25, 0.5, 0.5], [0], [0]),
        ([1, 0, 0], [0.25, 0.5, 0.5], [1], [0]),
        ([1, 0, 1], [0.25, 0.5, 0.5], [1], [0]),
        ([1, 1, 0], [0.25, 0.5, 0.5], [1], [0]),
    ],
)
def test_det_curve_tie_handling(y_true, y_score, expected_fpr, expected_fnr):
    fpr, fnr, _ = det_curve(y_true, y_score)

    assert_allclose(fpr, expected_fpr)
    assert_allclose(fnr, expected_fnr)


def test_det_curve_sanity_check():
    # Exactly duplicated inputs yield the same result.
    assert_allclose(
        det_curve([0, 0, 1], [0, 0.5, 1]),
        det_curve([0, 0, 0, 0, 1, 1], [0, 0, 0.5, 0.5, 1, 1]),
    )


@pytest.mark.parametrize("y_score", [(0), (0.25), (0.5), (0.75), (1)])
def test_det_curve_constant_scores(y_score):
    fpr, fnr, threshold = det_curve(
        y_true=[0, 1, 0, 1, 0, 1], y_score=np.full(6, y_score)
    )

    assert_allclose(fpr, [1])
    assert_allclose(fnr, [0])
    assert_allclose(threshold, [y_score])


@pytest.mark.parametrize(
    "y_true",
    [
        ([0, 0, 0, 0, 0, 1]),
        ([0, 0, 0, 0, 1, 1]),
        ([0, 0, 0, 1, 1, 1]),
        ([0, 0, 1, 1, 1, 1]),
        ([0, 1, 1, 1, 1, 1]),
    ],
)
def test_det_curve_perfect_scores(y_true):
    fpr, fnr, _ = det_curve(y_true=y_true, y_score=y_true)

    assert_allclose(fpr, [0])
    assert_allclose(fnr, [0])


@pytest.mark.parametrize(
    "y_true, y_pred, err_msg",
    [
        ([0, 1], [0, 0.5, 1], "inconsistent numbers of samples"),
        ([0, 1, 1], [0, 0.5], "inconsistent numbers of samples"),
        ([0, 0, 0], [0, 0.5, 1], "Only one class present in y_true"),
        ([1, 1, 1], [0, 0.5, 1], "Only one class present in y_true"),
        (
            ["cancer", "cancer", "not cancer"],
            [0.2, 0.3, 0.8],
            "pos_label is not specified",
        ),
    ],
)
def test_det_curve_bad_input(y_true, y_pred, err_msg):
    # input variables with inconsistent numbers of samples
    with pytest.raises(ValueError, match=err_msg):
        det_curve(y_true, y_pred)


def test_det_curve_pos_label():
    y_true = ["cancer"] * 3 + ["not cancer"] * 7
    y_pred_pos_not_cancer = np.array([0.1, 0.4, 0.6, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9])
    y_pred_pos_cancer = 1 - y_pred_pos_not_cancer

    fpr_pos_cancer, fnr_pos_cancer, th_pos_cancer = det_curve(
        y_true,
        y_pred_pos_cancer,
        pos_label="cancer",
    )
    fpr_pos_not_cancer, fnr_pos_not_cancer, th_pos_not_cancer = det_curve(
        y_true,
        y_pred_pos_not_cancer,
        pos_label="not cancer",
    )

    # check that the first threshold will change depending which label we
    # consider positive
    assert th_pos_cancer[0] == pytest.approx(0.4)
    assert th_pos_not_cancer[0] == pytest.approx(0.2)

    # check for the symmetry of the fpr and fnr
    assert_allclose(fpr_pos_cancer, fnr_pos_not_cancer[::-1])
    assert_allclose(fnr_pos_cancer, fpr_pos_not_cancer[::-1])


def check_lrap_toy(lrap_score):
    # Check on several small example that it works
    assert_almost_equal(lrap_score([[0, 1]], [[0.25, 0.75]]), 1)
    assert_almost_equal(lrap_score([[0, 1]], [[0.75, 0.25]]), 1 / 2)
    assert_almost_equal(lrap_score([[1, 1]], [[0.75, 0.25]]), 1)

    assert_almost_equal(lrap_score([[0, 0, 1]], [[0.25, 0.5, 0.75]]), 1)
    assert_almost_equal(lrap_score([[0, 1, 0]], [[0.25, 0.5, 0.75]]), 1 / 2)
    assert_almost_equal(lrap_score([[0, 1, 1]], [[0.25, 0.5, 0.75]]), 1)
    assert_almost_equal(lrap_score([[1, 0, 0]], [[0.25, 0.5, 0.75]]), 1 / 3)
    assert_almost_equal(
        lrap_score([[1, 0, 1]], [[0.25, 0.5, 0.75]]), (2 / 3 + 1 / 1) / 2
    )
    assert_almost_equal(
        lrap_score([[1, 1, 0]], [[0.25, 0.5, 0.75]]), (2 / 3 + 1 / 2) / 2
    )

    assert_almost_equal(lrap_score([[0, 0, 1]], [[0.75, 0.5, 0.25]]), 1 / 3)
    assert_almost_equal(lrap_score([[0, 1, 0]], [[0.75, 0.5, 0.25]]), 1 / 2)
    assert_almost_equal(
        lrap_score([[0, 1, 1]], [[0.75, 0.5, 0.25]]), (1 / 2 + 2 / 3) / 2
    )
    assert_almost_equal(lrap_score([[1, 0, 0]], [[0.75, 0.5, 0.25]]), 1)
    assert_almost_equal(lrap_score([[1, 0, 1]], [[0.75, 0.5, 0.25]]), (1 + 2 / 3) / 2)
    assert_almost_equal(lrap_score([[1, 1, 0]], [[0.75, 0.5, 0.25]]), 1)
    assert_almost_equal(lrap_score([[1, 1, 1]], [[0.75, 0.5, 0.25]]), 1)

    assert_almost_equal(lrap_score([[0, 0, 1]], [[0.5, 0.75, 0.25]]), 1 / 3)
    assert_almost_equal(lrap_score([[0, 1, 0]], [[0.5, 0.75, 0.25]]), 1)
    assert_almost_equal(lrap_score([[0, 1, 1]], [[0.5, 0.75, 0.25]]), (1 + 2 / 3) / 2)
    assert_almost_equal(lrap_score([[1, 0, 0]], [[0.5, 0.75, 0.25]]), 1 / 2)
    assert_almost_equal(
        lrap_score([[1, 0, 1]], [[0.5, 0.75, 0.25]]), (1 / 2 + 2 / 3) / 2
    )
    assert_almost_equal(lrap_score([[1, 1, 0]], [[0.5, 0.75, 0.25]]), 1)
    assert_almost_equal(lrap_score([[1, 1, 1]], [[0.5, 0.75, 0.25]]), 1)

    # Tie handling
    assert_almost_equal(lrap_score([[1, 0]], [[0.5, 0.5]]), 0.5)
    assert_almost_equal(lrap_score([[0, 1]], [[0.5, 0.5]]), 0.5)
    assert_almost_equal(lrap_score([[1, 1]], [[0.5, 0.5]]), 1)

    assert_almost_equal(lrap_score([[0, 0, 1]], [[0.25, 0.5, 0.5]]), 0.5)
    assert_almost_equal(lrap_score([[0, 1, 0]], [[0.25, 0.5, 0.5]]), 0.5)
    assert_almost_equal(lrap_score([[0, 1, 1]], [[0.25, 0.5, 0.5]]), 1)
    assert_almost_equal(lrap_score([[1, 0, 0]], [[0.25, 0.5, 0.5]]), 1 / 3)
    assert_almost_equal(
        lrap_score([[1, 0, 1]], [[0.25, 0.5, 0.5]]), (2 / 3 + 1 / 2) / 2
    )
    assert_almost_equal(
        lrap_score([[1, 1, 0]], [[0.25, 0.5, 0.5]]), (2 / 3 + 1 / 2) / 2
    )
    assert_almost_equal(lrap_score([[1, 1, 1]], [[0.25, 0.5, 0.5]]), 1)

    assert_almost_equal(lrap_score([[1, 1, 0]], [[0.5, 0.5, 0.5]]), 2 / 3)

    assert_almost_equal(lrap_score([[1, 1, 1, 0]], [[0.5, 0.5, 0.5, 0.5]]), 3 / 4)


def check_zero_or_all_relevant_labels(lrap_score):
    random_state = check_random_state(0)

    for n_labels in range(2, 5):
        y_score = random_state.uniform(size=(1, n_labels))
        y_score_ties = np.zeros_like(y_score)

        # No relevant labels
        y_true = np.zeros((1, n_labels))
        assert lrap_score(y_true, y_score) == 1.0
        assert lrap_score(y_true, y_score_ties) == 1.0

        # Only relevant labels
        y_true = np.ones((1, n_labels))
        assert lrap_score(y_true, y_score) == 1.0
        assert lrap_score(y_true, y_score_ties) == 1.0

    # Degenerate case: only one label
    assert_almost_equal(
        lrap_score([[1], [0], [1], [0]], [[0.5], [0.5], [0.5], [0.5]]), 1.0
    )


def check_lrap_error_raised(lrap_score):
    # Raise value error if not appropriate format
    with pytest.raises(ValueError):
        lrap_score([0, 1, 0], [0.25, 0.3, 0.2])
    with pytest.raises(ValueError):
        lrap_score([0, 1, 2], [[0.25, 0.75, 0.0], [0.7, 0.3, 0.0], [0.8, 0.2, 0.0]])
    with pytest.raises(ValueError):
        lrap_score(
            [(0), (1), (2)], [[0.25, 0.75, 0.0], [0.7, 0.3, 0.0], [0.8, 0.2, 0.0]]
        )

    # Check that y_true.shape != y_score.shape raise the proper exception
    with pytest.raises(ValueError):
        lrap_score([[0, 1], [0, 1]], [0, 1])
    with pytest.raises(ValueError):
        lrap_score([[0, 1], [0, 1]], [[0, 1]])
    with pytest.raises(ValueError):
        lrap_score([[0, 1], [0, 1]], [[0], [1]])
    with pytest.raises(ValueError):
        lrap_score([[0, 1]], [[0, 1], [0, 1]])
    with pytest.raises(ValueError):
        lrap_score([[0], [1]], [[0, 1], [0, 1]])
    with pytest.raises(ValueError):
        lrap_score([[0, 1], [0, 1]], [[0], [1]])


def check_lrap_only_ties(lrap_score):
    # Check tie handling in score
    # Basic check with only ties and increasing label space
    for n_labels in range(2, 10):
        y_score = np.ones((1, n_labels))

        # Check for growing number of consecutive relevant
        for n_relevant in range(1, n_labels):
            # Check for a bunch of positions
            for pos in range(n_labels - n_relevant):
                y_true = np.zeros((1, n_labels))
                y_true[0, pos : pos + n_relevant] = 1
                assert_almost_equal(lrap_score(y_true, y_score), n_relevant / n_labels)


def check_lrap_without_tie_and_increasing_score(lrap_score):
    # Check that Label ranking average precision works for various
    # Basic check with increasing label space size and decreasing score
    for n_labels in range(2, 10):
        y_score = n_labels - (np.arange(n_labels).reshape((1, n_labels)) + 1)

        # First and last
        y_true = np.zeros((1, n_labels))
        y_true[0, 0] = 1
        y_true[0, -1] = 1
        assert_almost_equal(lrap_score(y_true, y_score), (2 / n_labels + 1) / 2)

        # Check for growing number of consecutive relevant label
        for n_relevant in range(1, n_labels):
            # Check for a bunch of position
            for pos in range(n_labels - n_relevant):
                y_true = np.zeros((1, n_labels))
                y_true[0, pos : pos + n_relevant] = 1
                assert_almost_equal(
                    lrap_score(y_true, y_score),
                    sum(
                        (r + 1) / ((pos + r + 1) * n_relevant)
                        for r in range(n_relevant)
                    ),
                )


def _my_lrap(y_true, y_score):
    """Simple implementation of label ranking average precision"""
    check_consistent_length(y_true, y_score)
    y_true = check_array(y_true)
    y_score = check_array(y_score)
    n_samples, n_labels = y_true.shape
    score = np.empty((n_samples,))
    for i in range(n_samples):
        # The best rank correspond to 1. Rank higher than 1 are worse.
        # The best inverse ranking correspond to n_labels.
        unique_rank, inv_rank = np.unique(y_score[i], return_inverse=True)
        n_ranks = unique_rank.size
        rank = n_ranks - inv_rank

        # Rank need to be corrected to take into account ties
        # ex: rank 1 ex aequo means that both label are rank 2.
        corr_rank = np.bincount(rank, minlength=n_ranks + 1).cumsum()
        rank = corr_rank[rank]

        relevant = y_true[i].nonzero()[0]
        if relevant.size == 0 or relevant.size == n_labels:
            score[i] = 1
            continue

        score[i] = 0.0
        for label in relevant:
            # Let's count the number of relevant label with better rank
            # (smaller rank).
            n_ranked_above = sum(rank[r] <= rank[label] for r in relevant)

            # Weight by the rank of the actual label
            score[i] += n_ranked_above / rank[label]

        score[i] /= relevant.size

    return score.mean()


def check_alternative_lrap_implementation(
    lrap_score, n_classes=5, n_samples=20, random_state=0
):
    _, y_true = make_multilabel_classification(
        n_features=1,
        allow_unlabeled=False,
        random_state=random_state,
        n_classes=n_classes,
        n_samples=n_samples,
    )

    # Score with ties
    y_score = _sparse_random_matrix(
        n_components=y_true.shape[0],
        n_features=y_true.shape[1],
        random_state=random_state,
    )

    if hasattr(y_score, "toarray"):
        y_score = y_score.toarray()
    score_lrap = label_ranking_average_precision_score(y_true, y_score)
    score_my_lrap = _my_lrap(y_true, y_score)
    assert_almost_equal(score_lrap, score_my_lrap)

    # Uniform score
    random_state = check_random_state(random_state)
    y_score = random_state.uniform(size=(n_samples, n_classes))
    score_lrap = label_ranking_average_precision_score(y_true, y_score)
    score_my_lrap = _my_lrap(y_true, y_score)
    assert_almost_equal(score_lrap, score_my_lrap)


@pytest.mark.parametrize(
    "check",
    (
        check_lrap_toy,
        check_lrap_without_tie_and_increasing_score,
        check_lrap_only_ties,
        check_zero_or_all_relevant_labels,
    ),
)
@pytest.mark.parametrize("func", (label_ranking_average_precision_score, _my_lrap))
def test_label_ranking_avp(check, func):
    check(func)


def test_lrap_error_raised():
    check_lrap_error_raised(label_ranking_average_precision_score)


@pytest.mark.parametrize("n_samples", (1, 2, 8, 20))
@pytest.mark.parametrize("n_classes", (2, 5, 10))
@pytest.mark.parametrize("random_state", range(1))
def test_alternative_lrap_implementation(n_samples, n_classes, random_state):

    check_alternative_lrap_implementation(
        label_ranking_average_precision_score, n_classes, n_samples, random_state
    )


def test_lrap_sample_weighting_zero_labels():
    # Degenerate sample labeling (e.g., zero labels for a sample) is a valid
    # special case for lrap (the sample is considered to achieve perfect
    # precision), but this case is not tested in test_common.
    # For these test samples, the APs are 0.5, 0.75, and 1.0 (default for zero
    # labels).
    y_true = np.array([[1, 0, 0, 0], [1, 0, 0, 1], [0, 0, 0, 0]], dtype=bool)
    y_score = np.array(
        [[0.3, 0.4, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]
    )
    samplewise_lraps = np.array([0.5, 0.75, 1.0])
    sample_weight = np.array([1.0, 1.0, 0.0])

    assert_almost_equal(
        label_ranking_average_precision_score(
            y_true, y_score, sample_weight=sample_weight
        ),
        np.sum(sample_weight * samplewise_lraps) / np.sum(sample_weight),
    )


def test_coverage_error():
    # Toy case
    assert_almost_equal(coverage_error([[0, 1]], [[0.25, 0.75]]), 1)
    assert_almost_equal(coverage_error([[0, 1]], [[0.75, 0.25]]), 2)
    assert_almost_equal(coverage_error([[1, 1]], [[0.75, 0.25]]), 2)
    assert_almost_equal(coverage_error([[0, 0]], [[0.75, 0.25]]), 0)

    assert_almost_equal(coverage_error([[0, 0, 0]], [[0.25, 0.5, 0.75]]), 0)
    assert_almost_equal(coverage_error([[0, 0, 1]], [[0.25, 0.5, 0.75]]), 1)
    assert_almost_equal(coverage_error([[0, 1, 0]], [[0.25, 0.5, 0.75]]), 2)
    assert_almost_equal(coverage_error([[0, 1, 1]], [[0.25, 0.5, 0.75]]), 2)
    assert_almost_equal(coverage_error([[1, 0, 0]], [[0.25, 0.5, 0.75]]), 3)
    assert_almost_equal(coverage_error([[1, 0, 1]], [[0.25, 0.5, 0.75]]), 3)
    assert_almost_equal(coverage_error([[1, 1, 0]], [[0.25, 0.5, 0.75]]), 3)
    assert_almost_equal(coverage_error([[1, 1, 1]], [[0.25, 0.5, 0.75]]), 3)

    assert_almost_equal(coverage_error([[0, 0, 0]], [[0.75, 0.5, 0.25]]), 0)
    assert_almost_equal(coverage_error([[0, 0, 1]], [[0.75, 0.5, 0.25]]), 3)
    assert_almost_equal(coverage_error([[0, 1, 0]], [[0.75, 0.5, 0.25]]), 2)
    assert_almost_equal(coverage_error([[0, 1, 1]], [[0.75, 0.5, 0.25]]), 3)
    assert_almost_equal(coverage_error([[1, 0, 0]], [[0.75, 0.5, 0.25]]), 1)
    assert_almost_equal(coverage_error([[1, 0, 1]], [[0.75, 0.5, 0.25]]), 3)
    assert_almost_equal(coverage_error([[1, 1, 0]], [[0.75, 0.5, 0.25]]), 2)
    assert_almost_equal(coverage_error([[1, 1, 1]], [[0.75, 0.5, 0.25]]), 3)

    assert_almost_equal(coverage_error([[0, 0, 0]], [[0.5, 0.75, 0.25]]), 0)
    assert_almost_equal(coverage_error([[0, 0, 1]], [[0.5, 0.75, 0.25]]), 3)
    assert_almost_equal(coverage_error([[0, 1, 0]], [[0.5, 0.75, 0.25]]), 1)
    assert_almost_equal(coverage_error([[0, 1, 1]], [[0.5, 0.75, 0.25]]), 3)
    assert_almost_equal(coverage_error([[1, 0, 0]], [[0.5, 0.75, 0.25]]), 2)
    assert_almost_equal(coverage_error([[1, 0, 1]], [[0.5, 0.75, 0.25]]), 3)
    assert_almost_equal(coverage_error([[1, 1, 0]], [[0.5, 0.75, 0.25]]), 2)
    assert_almost_equal(coverage_error([[1, 1, 1]], [[0.5, 0.75, 0.25]]), 3)

    # Non trivial case
    assert_almost_equal(
        coverage_error([[0, 1, 0], [1, 1, 0]], [[0.1, 10.0, -3], [0, 1, 3]]),
        (1 + 3) / 2.0,
    )

    assert_almost_equal(
        coverage_error(
            [[0, 1, 0], [1, 1, 0], [0, 1, 1]], [[0.1, 10, -3], [0, 1, 3], [0, 2, 0]]
        ),
        (1 + 3 + 3) / 3.0,
    )

    assert_almost_equal(
        coverage_error(
            [[0, 1, 0], [1, 1, 0], [0, 1, 1]], [[0.1, 10, -3], [3, 1, 3], [0, 2, 0]]
        ),
        (1 + 3 + 3) / 3.0,
    )


def test_coverage_tie_handling():
    assert_almost_equal(coverage_error([[0, 0]], [[0.5, 0.5]]), 0)
    assert_almost_equal(coverage_error([[1, 0]], [[0.5, 0.5]]), 2)
    assert_almost_equal(coverage_error([[0, 1]], [[0.5, 0.5]]), 2)
    assert_almost_equal(coverage_error([[1, 1]], [[0.5, 0.5]]), 2)

    assert_almost_equal(coverage_error([[0, 0, 0]], [[0.25, 0.5, 0.5]]), 0)
    assert_almost_equal(coverage_error([[0, 0, 1]], [[0.25, 0.5, 0.5]]), 2)
    assert_almost_equal(coverage_error([[0, 1, 0]], [[0.25, 0.5, 0.5]]), 2)
    assert_almost_equal(coverage_error([[0, 1, 1]], [[0.25, 0.5, 0.5]]), 2)
    assert_almost_equal(coverage_error([[1, 0, 0]], [[0.25, 0.5, 0.5]]), 3)
    assert_almost_equal(coverage_error([[1, 0, 1]], [[0.25, 0.5, 0.5]]), 3)
    assert_almost_equal(coverage_error([[1, 1, 0]], [[0.25, 0.5, 0.5]]), 3)
    assert_almost_equal(coverage_error([[1, 1, 1]], [[0.25, 0.5, 0.5]]), 3)


def test_label_ranking_loss():
    assert_almost_equal(label_ranking_loss([[0, 1]], [[0.25, 0.75]]), 0)
    assert_almost_equal(label_ranking_loss([[0, 1]], [[0.75, 0.25]]), 1)

    assert_almost_equal(label_ranking_loss([[0, 0, 1]], [[0.25, 0.5, 0.75]]), 0)
    assert_almost_equal(label_ranking_loss([[0, 1, 0]], [[0.25, 0.5, 0.75]]), 1 / 2)
    assert_almost_equal(label_ranking_loss([[0, 1, 1]], [[0.25, 0.5, 0.75]]), 0)
    assert_almost_equal(label_ranking_loss([[1, 0, 0]], [[0.25, 0.5, 0.75]]), 2 / 2)
    assert_almost_equal(label_ranking_loss([[1, 0, 1]], [[0.25, 0.5, 0.75]]), 1 / 2)
    assert_almost_equal(label_ranking_loss([[1, 1, 0]], [[0.25, 0.5, 0.75]]), 2 / 2)

    # Undefined metrics -  the ranking doesn't matter
    assert_almost_equal(label_ranking_loss([[0, 0]], [[0.75, 0.25]]), 0)
    assert_almost_equal(label_ranking_loss([[1, 1]], [[0.75, 0.25]]), 0)
    assert_almost_equal(label_ranking_loss([[0, 0]], [[0.5, 0.5]]), 0)
    assert_almost_equal(label_ranking_loss([[1, 1]], [[0.5, 0.5]]), 0)

    assert_almost_equal(label_ranking_loss([[0, 0, 0]], [[0.5, 0.75, 0.25]]), 0)
    assert_almost_equal(label_ranking_loss([[1, 1, 1]], [[0.5, 0.75, 0.25]]), 0)
    assert_almost_equal(label_ranking_loss([[0, 0, 0]], [[0.25, 0.5, 0.5]]), 0)
    assert_almost_equal(label_ranking_loss([[1, 1, 1]], [[0.25, 0.5, 0.5]]), 0)

    # Non trivial case
    assert_almost_equal(
        label_ranking_loss([[0, 1, 0], [1, 1, 0]], [[0.1, 10.0, -3], [0, 1, 3]]),
        (0 + 2 / 2) / 2.0,
    )

    assert_almost_equal(
        label_ranking_loss(
            [[0, 1, 0], [1, 1, 0], [0, 1, 1]], [[0.1, 10, -3], [0, 1, 3], [0, 2, 0]]
        ),
        (0 + 2 / 2 + 1 / 2) / 3.0,
    )

    assert_almost_equal(
        label_ranking_loss(
            [[0, 1, 0], [1, 1, 0], [0, 1, 1]], [[0.1, 10, -3], [3, 1, 3], [0, 2, 0]]
        ),
        (0 + 2 / 2 + 1 / 2) / 3.0,
    )

    # Sparse csr matrices
    assert_almost_equal(
        label_ranking_loss(
            csr_matrix(np.array([[0, 1, 0], [1, 1, 0]])), [[0.1, 10, -3], [3, 1, 3]]
        ),
        (0 + 2 / 2) / 2.0,
    )


def test_ranking_appropriate_input_shape():
    # Check that y_true.shape != y_score.shape raise the proper exception
    with pytest.raises(ValueError):
        label_ranking_loss([[0, 1], [0, 1]], [0, 1])
    with pytest.raises(ValueError):
        label_ranking_loss([[0, 1], [0, 1]], [[0, 1]])
    with pytest.raises(ValueError):
        label_ranking_loss([[0, 1], [0, 1]], [[0], [1]])
    with pytest.raises(ValueError):
        label_ranking_loss([[0, 1]], [[0, 1], [0, 1]])
    with pytest.raises(ValueError):
        label_ranking_loss([[0], [1]], [[0, 1], [0, 1]])
    with pytest.raises(ValueError):
        label_ranking_loss([[0, 1], [0, 1]], [[0], [1]])


def test_ranking_loss_ties_handling():
    # Tie handling
    assert_almost_equal(label_ranking_loss([[1, 0]], [[0.5, 0.5]]), 1)
    assert_almost_equal(label_ranking_loss([[0, 1]], [[0.5, 0.5]]), 1)
    assert_almost_equal(label_ranking_loss([[0, 0, 1]], [[0.25, 0.5, 0.5]]), 1 / 2)
    assert_almost_equal(label_ranking_loss([[0, 1, 0]], [[0.25, 0.5, 0.5]]), 1 / 2)
    assert_almost_equal(label_ranking_loss([[0, 1, 1]], [[0.25, 0.5, 0.5]]), 0)
    assert_almost_equal(label_ranking_loss([[1, 0, 0]], [[0.25, 0.5, 0.5]]), 1)
    assert_almost_equal(label_ranking_loss([[1, 0, 1]], [[0.25, 0.5, 0.5]]), 1)
    assert_almost_equal(label_ranking_loss([[1, 1, 0]], [[0.25, 0.5, 0.5]]), 1)


def test_dcg_score():
    _, y_true = make_multilabel_classification(random_state=0, n_classes=10)
    y_score = -y_true + 1
    _test_dcg_score_for(y_true, y_score)
    y_true, y_score = np.random.RandomState(0).random_sample((2, 100, 10))
    _test_dcg_score_for(y_true, y_score)


def _test_dcg_score_for(y_true, y_score):
    discount = np.log2(np.arange(y_true.shape[1]) + 2)
    ideal = _dcg_sample_scores(y_true, y_true)
    score = _dcg_sample_scores(y_true, y_score)
    assert (score <= ideal).all()
    assert (_dcg_sample_scores(y_true, y_true, k=5) <= ideal).all()
    assert ideal.shape == (y_true.shape[0],)
    assert score.shape == (y_true.shape[0],)
    assert ideal == pytest.approx((np.sort(y_true)[:, ::-1] / discount).sum(axis=1))


def test_dcg_ties():
    y_true = np.asarray([np.arange(5)])
    y_score = np.zeros(y_true.shape)
    dcg = _dcg_sample_scores(y_true, y_score)
    dcg_ignore_ties = _dcg_sample_scores(y_true, y_score, ignore_ties=True)
    discounts = 1 / np.log2(np.arange(2, 7))
    assert dcg == pytest.approx([discounts.sum() * y_true.mean()])
    assert dcg_ignore_ties == pytest.approx([(discounts * y_true[:, ::-1]).sum()])
    y_score[0, 3:] = 1
    dcg = _dcg_sample_scores(y_true, y_score)
    dcg_ignore_ties = _dcg_sample_scores(y_true, y_score, ignore_ties=True)
    assert dcg_ignore_ties == pytest.approx([(discounts * y_true[:, ::-1]).sum()])
    assert dcg == pytest.approx(
        [
            discounts[:2].sum() * y_true[0, 3:].mean()
            + discounts[2:].sum() * y_true[0, :3].mean()
        ]
    )


def test_ndcg_ignore_ties_with_k():
    a = np.arange(12).reshape((2, 6))
    assert ndcg_score(a, a, k=3, ignore_ties=True) == pytest.approx(
        ndcg_score(a, a, k=3, ignore_ties=True)
    )


def test_ndcg_invariant():
    y_true = np.arange(70).reshape(7, 10)
    y_score = y_true + np.random.RandomState(0).uniform(-0.2, 0.2, size=y_true.shape)
    ndcg = ndcg_score(y_true, y_score)
    ndcg_no_ties = ndcg_score(y_true, y_score, ignore_ties=True)
    assert ndcg == pytest.approx(ndcg_no_ties)
    assert ndcg == pytest.approx(1.0)
    y_score += 1000
    assert ndcg_score(y_true, y_score) == pytest.approx(1.0)


@pytest.mark.parametrize("ignore_ties", [True, False])
def test_ndcg_toy_examples(ignore_ties):
    y_true = 3 * np.eye(7)[:5]
    y_score = np.tile(np.arange(6, -1, -1), (5, 1))
    y_score_noisy = y_score + np.random.RandomState(0).uniform(
        -0.2, 0.2, size=y_score.shape
    )
    assert _dcg_sample_scores(
        y_true, y_score, ignore_ties=ignore_ties
    ) == pytest.approx(3 / np.log2(np.arange(2, 7)))
    assert _dcg_sample_scores(
        y_true, y_score_noisy, ignore_ties=ignore_ties
    ) == pytest.approx(3 / np.log2(np.arange(2, 7)))
    assert _ndcg_sample_scores(
        y_true, y_score, ignore_ties=ignore_ties
    ) == pytest.approx(1 / np.log2(np.arange(2, 7)))
    assert _dcg_sample_scores(
        y_true, y_score, log_base=10, ignore_ties=ignore_ties
    ) == pytest.approx(3 / np.log10(np.arange(2, 7)))
    assert ndcg_score(y_true, y_score, ignore_ties=ignore_ties) == pytest.approx(
        (1 / np.log2(np.arange(2, 7))).mean()
    )
    assert dcg_score(y_true, y_score, ignore_ties=ignore_ties) == pytest.approx(
        (3 / np.log2(np.arange(2, 7))).mean()
    )
    y_true = 3 * np.ones((5, 7))
    expected_dcg_score = (3 / np.log2(np.arange(2, 9))).sum()
    assert _dcg_sample_scores(
        y_true, y_score, ignore_ties=ignore_ties
    ) == pytest.approx(expected_dcg_score * np.ones(5))
    assert _ndcg_sample_scores(
        y_true, y_score, ignore_ties=ignore_ties
    ) == pytest.approx(np.ones(5))
    assert dcg_score(y_true, y_score, ignore_ties=ignore_ties) == pytest.approx(
        expected_dcg_score
    )
    assert ndcg_score(y_true, y_score, ignore_ties=ignore_ties) == pytest.approx(1.0)


def test_ndcg_score():
    _, y_true = make_multilabel_classification(random_state=0, n_classes=10)
    y_score = -y_true + 1
    _test_ndcg_score_for(y_true, y_score)
    y_true, y_score = np.random.RandomState(0).random_sample((2, 100, 10))
    _test_ndcg_score_for(y_true, y_score)


def _test_ndcg_score_for(y_true, y_score):
    ideal = _ndcg_sample_scores(y_true, y_true)
    score = _ndcg_sample_scores(y_true, y_score)
    assert (score <= ideal).all()
    all_zero = (y_true == 0).all(axis=1)
    assert ideal[~all_zero] == pytest.approx(np.ones((~all_zero).sum()))
    assert ideal[all_zero] == pytest.approx(np.zeros(all_zero.sum()))
    assert score[~all_zero] == pytest.approx(
        _dcg_sample_scores(y_true, y_score)[~all_zero]
        / _dcg_sample_scores(y_true, y_true)[~all_zero]
    )
    assert score[all_zero] == pytest.approx(np.zeros(all_zero.sum()))
    assert ideal.shape == (y_true.shape[0],)
    assert score.shape == (y_true.shape[0],)


def test_partial_roc_auc_score():
    # Check `roc_auc_score` for max_fpr != `None`
    y_true = np.array([0, 0, 1, 1])
    assert roc_auc_score(y_true, y_true, max_fpr=1) == 1
    assert roc_auc_score(y_true, y_true, max_fpr=0.001) == 1
    with pytest.raises(ValueError):
        assert roc_auc_score(y_true, y_true, max_fpr=-0.1)
    with pytest.raises(ValueError):
        assert roc_auc_score(y_true, y_true, max_fpr=1.1)
    with pytest.raises(ValueError):
        assert roc_auc_score(y_true, y_true, max_fpr=0)

    y_scores = np.array([0.1, 0, 0.1, 0.01])
    roc_auc_with_max_fpr_one = roc_auc_score(y_true, y_scores, max_fpr=1)
    unconstrained_roc_auc = roc_auc_score(y_true, y_scores)
    assert roc_auc_with_max_fpr_one == unconstrained_roc_auc
    assert roc_auc_score(y_true, y_scores, max_fpr=0.3) == 0.5

    y_true, y_pred, _ = make_prediction(binary=True)
    for max_fpr in np.linspace(1e-4, 1, 5):
        assert_almost_equal(
            roc_auc_score(y_true, y_pred, max_fpr=max_fpr),
            _partial_roc_auc_score(y_true, y_pred, max_fpr),
        )


@pytest.mark.parametrize(
    "y_true, k, true_score",
    [
        ([0, 1, 2, 3], 1, 0.25),
        ([0, 1, 2, 3], 2, 0.5),
        ([0, 1, 2, 3], 3, 0.75),
    ],
)
def test_top_k_accuracy_score(y_true, k, true_score):
    y_score = np.array(
        [
            [0.4, 0.3, 0.2, 0.1],
            [0.1, 0.3, 0.4, 0.2],
            [0.4, 0.1, 0.2, 0.3],
            [0.3, 0.2, 0.4, 0.1],
        ]
    )
    score = top_k_accuracy_score(y_true, y_score, k=k)
    assert score == pytest.approx(true_score)


@pytest.mark.parametrize(
    "y_score, k, true_score",
    [
        (np.array([-1, -1, 1, 1]), 1, 1),
        (np.array([-1, 1, -1, 1]), 1, 0.5),
        (np.array([-1, 1, -1, 1]), 2, 1),
        (np.array([0.2, 0.2, 0.7, 0.7]), 1, 1),
        (np.array([0.2, 0.7, 0.2, 0.7]), 1, 0.5),
        (np.array([0.2, 0.7, 0.2, 0.7]), 2, 1),
    ],
)
def test_top_k_accuracy_score_binary(y_score, k, true_score):
    y_true = [0, 0, 1, 1]

    threshold = 0.5 if y_score.min() >= 0 and y_score.max() <= 1 else 0
    y_pred = (y_score > threshold).astype(np.int64) if k == 1 else y_true

    score = top_k_accuracy_score(y_true, y_score, k=k)
    score_acc = accuracy_score(y_true, y_pred)

    assert score == score_acc == pytest.approx(true_score)


@pytest.mark.parametrize(
    "y_true, true_score, labels",
    [
        (np.array([0, 1, 1, 2]), 0.75, [0, 1, 2, 3]),
        (np.array([0, 1, 1, 1]), 0.5, [0, 1, 2, 3]),
        (np.array([1, 1, 1, 1]), 0.5, [0, 1, 2, 3]),
        (np.array(["a", "e", "e", "a"]), 0.75, ["a", "b", "d", "e"]),
    ],
)
@pytest.mark.parametrize("labels_as_ndarray", [True, False])
def test_top_k_accuracy_score_multiclass_with_labels(
    y_true, true_score, labels, labels_as_ndarray
):
    """Test when labels and y_score are multiclass."""
    if labels_as_ndarray:
        labels = np.asarray(labels)
    y_score = np.array(
        [
            [0.4, 0.3, 0.2, 0.1],
            [0.1, 0.3, 0.4, 0.2],
            [0.4, 0.1, 0.2, 0.3],
            [0.3, 0.2, 0.4, 0.1],
        ]
    )

    score = top_k_accuracy_score(y_true, y_score, k=2, labels=labels)
    assert score == pytest.approx(true_score)


def test_top_k_accuracy_score_increasing():
    # Make sure increasing k leads to a higher score
    X, y = datasets.make_classification(
        n_classes=10, n_samples=1000, n_informative=10, random_state=0
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    clf = LogisticRegression(random_state=0)
    clf.fit(X_train, y_train)

    for X, y in zip((X_train, X_test), (y_train, y_test)):
        scores = [
            top_k_accuracy_score(y, clf.predict_proba(X), k=k) for k in range(2, 10)
        ]

        assert np.all(np.diff(scores) > 0)


@pytest.mark.parametrize(
    "y_true, k, true_score",
    [
        ([0, 1, 2, 3], 1, 0.25),
        ([0, 1, 2, 3], 2, 0.5),
        ([0, 1, 2, 3], 3, 1),
    ],
)
def test_top_k_accuracy_score_ties(y_true, k, true_score):
    # Make sure highest indices labels are chosen first in case of ties
    y_score = np.array(
        [
            [5, 5, 7, 0],
            [1, 5, 5, 5],
            [0, 0, 3, 3],
            [1, 1, 1, 1],
        ]
    )
    assert top_k_accuracy_score(y_true, y_score, k=k) == pytest.approx(true_score)


@pytest.mark.parametrize(
    "y_true, k",
    [
        ([0, 1, 2, 3], 4),
        ([0, 1, 2, 3], 5),
    ],
)
def test_top_k_accuracy_score_warning(y_true, k):
    y_score = np.array(
        [
            [0.4, 0.3, 0.2, 0.1],
            [0.1, 0.4, 0.3, 0.2],
            [0.2, 0.1, 0.4, 0.3],
            [0.3, 0.2, 0.1, 0.4],
        ]
    )
    expected_message = (
        r"'k' \(\d+\) greater than or equal to 'n_classes' \(\d+\) will result in a "
        "perfect score and is therefore meaningless."
    )
    with pytest.warns(UndefinedMetricWarning, match=expected_message):
        score = top_k_accuracy_score(y_true, y_score, k=k)
    assert score == 1


@pytest.mark.parametrize(
    "y_true, labels, msg",
    [
        (
            [0, 0.57, 1, 2],
            None,
            "y type must be 'binary' or 'multiclass', got 'continuous'",
        ),
        (
            [0, 1, 2, 3],
            None,
            r"Number of classes in 'y_true' \(4\) not equal to the number of "
            r"classes in 'y_score' \(3\).",
        ),
        (
            ["c", "c", "a", "b"],
            ["a", "b", "c", "c"],
            "Parameter 'labels' must be unique.",
        ),
        (["c", "c", "a", "b"], ["a", "c", "b"], "Parameter 'labels' must be ordered."),
        (
            [0, 0, 1, 2],
            [0, 1, 2, 3],
            r"Number of given labels \(4\) not equal to the number of classes in "
            r"'y_score' \(3\).",
        ),
        (
            [0, 0, 1, 2],
            [0, 1, 3],
            "'y_true' contains labels not in parameter 'labels'.",
        ),
    ],
)
def test_top_k_accuracy_score_error(y_true, labels, msg):
    y_score = np.array(
        [
            [0.2, 0.1, 0.7],
            [0.4, 0.3, 0.3],
            [0.3, 0.4, 0.3],
            [0.4, 0.5, 0.1],
        ]
    )
    with pytest.raises(ValueError, match=msg):
        top_k_accuracy_score(y_true, y_score, k=2, labels=labels)
