from functools import partial
from inspect import signature
from itertools import product
from itertools import chain
from itertools import permutations

import numpy as np
import scipy.sparse as sp

import pytest

from sklearn.datasets import make_multilabel_classification
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _num_samples
from sklearn.utils.validation import check_random_state
from sklearn.utils import shuffle

from sklearn.utils._testing import assert_allclose
from sklearn.utils._testing import assert_almost_equal
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_array_less
from sklearn.utils._testing import ignore_warnings

from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import coverage_error
from sklearn.metrics import d2_tweedie_score
from sklearn.metrics import det_curve
from sklearn.metrics import explained_variance_score
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import hinge_loss
from sklearn.metrics import jaccard_score
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import log_loss
from sklearn.metrics import max_error
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_tweedie_deviance
from sklearn.metrics import mean_poisson_deviance
from sklearn.metrics import mean_gamma_deviance
from sklearn.metrics import median_absolute_error
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import mean_pinball_loss
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import r2_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import zero_one_loss
from sklearn.metrics import ndcg_score
from sklearn.metrics import dcg_score
from sklearn.metrics import top_k_accuracy_score

from sklearn.metrics._base import _average_binary_score


# Note toward developers about metric testing
# -------------------------------------------
# It is often possible to write one general test for several metrics:
#
#   - invariance properties, e.g. invariance to sample order
#   - common behavior for an argument, e.g. the "normalize" with value True
#     will return the mean of the metrics and with value False will return
#     the sum of the metrics.
#
# In order to improve the overall metric testing, it is a good idea to write
# first a specific test for the given metric and then add a general test for
# all metrics that have the same behavior.
#
# Two types of datastructures are used in order to implement this system:
# dictionaries of metrics and lists of metrics with common properties.
#
# Dictionaries of metrics
# ------------------------
# The goal of having those dictionaries is to have an easy way to call a
# particular metric and associate a name to each function:
#
#   - REGRESSION_METRICS: all regression metrics.
#   - CLASSIFICATION_METRICS: all classification metrics
#     which compare a ground truth and the estimated targets as returned by a
#     classifier.
#   - THRESHOLDED_METRICS: all classification metrics which
#     compare a ground truth and a score, e.g. estimated probabilities or
#     decision function (format might vary)
#
# Those dictionaries will be used to test systematically some invariance
# properties, e.g. invariance toward several input layout.
#

REGRESSION_METRICS = {
    "max_error": max_error,
    "mean_absolute_error": mean_absolute_error,
    "mean_squared_error": mean_squared_error,
    "mean_pinball_loss": mean_pinball_loss,
    "median_absolute_error": median_absolute_error,
    "mean_absolute_percentage_error": mean_absolute_percentage_error,
    "explained_variance_score": explained_variance_score,
    "r2_score": partial(r2_score, multioutput="variance_weighted"),
    "mean_normal_deviance": partial(mean_tweedie_deviance, power=0),
    "mean_poisson_deviance": mean_poisson_deviance,
    "mean_gamma_deviance": mean_gamma_deviance,
    "mean_compound_poisson_deviance": partial(mean_tweedie_deviance, power=1.4),
    "d2_tweedie_score": partial(d2_tweedie_score, power=1.4),
}

CLASSIFICATION_METRICS = {
    "accuracy_score": accuracy_score,
    "balanced_accuracy_score": balanced_accuracy_score,
    "adjusted_balanced_accuracy_score": partial(balanced_accuracy_score, adjusted=True),
    "unnormalized_accuracy_score": partial(accuracy_score, normalize=False),
    # `confusion_matrix` returns absolute values and hence behaves unnormalized
    # . Naming it with an unnormalized_ prefix is necessary for this module to
    # skip sample_weight scaling checks which will fail for unnormalized
    # metrics.
    "unnormalized_confusion_matrix": confusion_matrix,
    "normalized_confusion_matrix": lambda *args, **kwargs: (
        confusion_matrix(*args, **kwargs).astype("float")
        / confusion_matrix(*args, **kwargs).sum(axis=1)[:, np.newaxis]
    ),
    "unnormalized_multilabel_confusion_matrix": multilabel_confusion_matrix,
    "unnormalized_multilabel_confusion_matrix_sample": partial(
        multilabel_confusion_matrix, samplewise=True
    ),
    "hamming_loss": hamming_loss,
    "zero_one_loss": zero_one_loss,
    "unnormalized_zero_one_loss": partial(zero_one_loss, normalize=False),
    # These are needed to test averaging
    "jaccard_score": jaccard_score,
    "precision_score": precision_score,
    "recall_score": recall_score,
    "f1_score": f1_score,
    "f2_score": partial(fbeta_score, beta=2),
    "f0.5_score": partial(fbeta_score, beta=0.5),
    "matthews_corrcoef_score": matthews_corrcoef,
    "weighted_f0.5_score": partial(fbeta_score, average="weighted", beta=0.5),
    "weighted_f1_score": partial(f1_score, average="weighted"),
    "weighted_f2_score": partial(fbeta_score, average="weighted", beta=2),
    "weighted_precision_score": partial(precision_score, average="weighted"),
    "weighted_recall_score": partial(recall_score, average="weighted"),
    "weighted_jaccard_score": partial(jaccard_score, average="weighted"),
    "micro_f0.5_score": partial(fbeta_score, average="micro", beta=0.5),
    "micro_f1_score": partial(f1_score, average="micro"),
    "micro_f2_score": partial(fbeta_score, average="micro", beta=2),
    "micro_precision_score": partial(precision_score, average="micro"),
    "micro_recall_score": partial(recall_score, average="micro"),
    "micro_jaccard_score": partial(jaccard_score, average="micro"),
    "macro_f0.5_score": partial(fbeta_score, average="macro", beta=0.5),
    "macro_f1_score": partial(f1_score, average="macro"),
    "macro_f2_score": partial(fbeta_score, average="macro", beta=2),
    "macro_precision_score": partial(precision_score, average="macro"),
    "macro_recall_score": partial(recall_score, average="macro"),
    "macro_jaccard_score": partial(jaccard_score, average="macro"),
    "samples_f0.5_score": partial(fbeta_score, average="samples", beta=0.5),
    "samples_f1_score": partial(f1_score, average="samples"),
    "samples_f2_score": partial(fbeta_score, average="samples", beta=2),
    "samples_precision_score": partial(precision_score, average="samples"),
    "samples_recall_score": partial(recall_score, average="samples"),
    "samples_jaccard_score": partial(jaccard_score, average="samples"),
    "cohen_kappa_score": cohen_kappa_score,
}


def precision_recall_curve_padded_thresholds(*args, **kwargs):
    """
    The dimensions of precision-recall pairs and the threshold array as
    returned by the precision_recall_curve do not match. See
    func:`sklearn.metrics.precision_recall_curve`

    This prevents implicit conversion of return value triple to an higher
    dimensional np.array of dtype('float64') (it will be of dtype('object)
    instead). This again is needed for assert_array_equal to work correctly.

    As a workaround we pad the threshold array with NaN values to match
    the dimension of precision and recall arrays respectively.
    """
    precision, recall, thresholds = precision_recall_curve(*args, **kwargs)

    pad_threshholds = len(precision) - len(thresholds)

    return np.array(
        [
            precision,
            recall,
            np.pad(
                thresholds.astype(np.float64),
                pad_width=(0, pad_threshholds),
                mode="constant",
                constant_values=[np.nan],
            ),
        ]
    )


CURVE_METRICS = {
    "roc_curve": roc_curve,
    "precision_recall_curve": precision_recall_curve_padded_thresholds,
    "det_curve": det_curve,
}

THRESHOLDED_METRICS = {
    "coverage_error": coverage_error,
    "label_ranking_loss": label_ranking_loss,
    "log_loss": log_loss,
    "unnormalized_log_loss": partial(log_loss, normalize=False),
    "hinge_loss": hinge_loss,
    "brier_score_loss": brier_score_loss,
    "roc_auc_score": roc_auc_score,  # default: average="macro"
    "weighted_roc_auc": partial(roc_auc_score, average="weighted"),
    "samples_roc_auc": partial(roc_auc_score, average="samples"),
    "micro_roc_auc": partial(roc_auc_score, average="micro"),
    "ovr_roc_auc": partial(roc_auc_score, average="macro", multi_class="ovr"),
    "weighted_ovr_roc_auc": partial(
        roc_auc_score, average="weighted", multi_class="ovr"
    ),
    "ovo_roc_auc": partial(roc_auc_score, average="macro", multi_class="ovo"),
    "weighted_ovo_roc_auc": partial(
        roc_auc_score, average="weighted", multi_class="ovo"
    ),
    "partial_roc_auc": partial(roc_auc_score, max_fpr=0.5),
    "average_precision_score": average_precision_score,  # default: average="macro"
    "weighted_average_precision_score": partial(
        average_precision_score, average="weighted"
    ),
    "samples_average_precision_score": partial(
        average_precision_score, average="samples"
    ),
    "micro_average_precision_score": partial(average_precision_score, average="micro"),
    "label_ranking_average_precision_score": label_ranking_average_precision_score,
    "ndcg_score": ndcg_score,
    "dcg_score": dcg_score,
    "top_k_accuracy_score": top_k_accuracy_score,
}

ALL_METRICS = dict()
ALL_METRICS.update(THRESHOLDED_METRICS)
ALL_METRICS.update(CLASSIFICATION_METRICS)
ALL_METRICS.update(REGRESSION_METRICS)
ALL_METRICS.update(CURVE_METRICS)

# Lists of metrics with common properties
# ---------------------------------------
# Lists of metrics with common properties are used to test systematically some
# functionalities and invariance, e.g. SYMMETRIC_METRICS lists all metrics that
# are symmetric with respect to their input argument y_true and y_pred.
#
# When you add a new metric or functionality, check if a general test
# is already written.

# Those metrics don't support binary inputs
METRIC_UNDEFINED_BINARY = {
    "samples_f0.5_score",
    "samples_f1_score",
    "samples_f2_score",
    "samples_precision_score",
    "samples_recall_score",
    "samples_jaccard_score",
    "coverage_error",
    "unnormalized_multilabel_confusion_matrix_sample",
    "label_ranking_loss",
    "label_ranking_average_precision_score",
    "dcg_score",
    "ndcg_score",
}

# Those metrics don't support multiclass inputs
METRIC_UNDEFINED_MULTICLASS = {
    "brier_score_loss",
    "micro_roc_auc",
    "samples_roc_auc",
    "partial_roc_auc",
    "roc_auc_score",
    "weighted_roc_auc",
    "average_precision_score",
    "weighted_average_precision_score",
    "micro_average_precision_score",
    "samples_average_precision_score",
    "jaccard_score",
    # with default average='binary', multiclass is prohibited
    "precision_score",
    "recall_score",
    "f1_score",
    "f2_score",
    "f0.5_score",
    # curves
    "roc_curve",
    "precision_recall_curve",
    "det_curve",
}

# Metric undefined with "binary" or "multiclass" input
METRIC_UNDEFINED_BINARY_MULTICLASS = METRIC_UNDEFINED_BINARY.union(
    METRIC_UNDEFINED_MULTICLASS
)

# Metrics with an "average" argument
METRICS_WITH_AVERAGING = {
    "precision_score",
    "recall_score",
    "f1_score",
    "f2_score",
    "f0.5_score",
    "jaccard_score",
}

# Threshold-based metrics with an "average" argument
THRESHOLDED_METRICS_WITH_AVERAGING = {
    "roc_auc_score",
    "average_precision_score",
    "partial_roc_auc",
}

# Metrics with a "pos_label" argument
METRICS_WITH_POS_LABEL = {
    "roc_curve",
    "precision_recall_curve",
    "det_curve",
    "brier_score_loss",
    "precision_score",
    "recall_score",
    "f1_score",
    "f2_score",
    "f0.5_score",
    "jaccard_score",
    "average_precision_score",
    "weighted_average_precision_score",
    "micro_average_precision_score",
    "samples_average_precision_score",
}

# Metrics with a "labels" argument
# TODO: Handle multi_class metrics that has a labels argument as well as a
# decision function argument. e.g hinge_loss
METRICS_WITH_LABELS = {
    "unnormalized_confusion_matrix",
    "normalized_confusion_matrix",
    "roc_curve",
    "precision_recall_curve",
    "det_curve",
    "precision_score",
    "recall_score",
    "f1_score",
    "f2_score",
    "f0.5_score",
    "jaccard_score",
    "weighted_f0.5_score",
    "weighted_f1_score",
    "weighted_f2_score",
    "weighted_precision_score",
    "weighted_recall_score",
    "weighted_jaccard_score",
    "micro_f0.5_score",
    "micro_f1_score",
    "micro_f2_score",
    "micro_precision_score",
    "micro_recall_score",
    "micro_jaccard_score",
    "macro_f0.5_score",
    "macro_f1_score",
    "macro_f2_score",
    "macro_precision_score",
    "macro_recall_score",
    "macro_jaccard_score",
    "unnormalized_multilabel_confusion_matrix",
    "unnormalized_multilabel_confusion_matrix_sample",
    "cohen_kappa_score",
}

# Metrics with a "normalize" option
METRICS_WITH_NORMALIZE_OPTION = {
    "accuracy_score",
    "top_k_accuracy_score",
    "zero_one_loss",
}

# Threshold-based metrics with "multilabel-indicator" format support
THRESHOLDED_MULTILABEL_METRICS = {
    "log_loss",
    "unnormalized_log_loss",
    "roc_auc_score",
    "weighted_roc_auc",
    "samples_roc_auc",
    "micro_roc_auc",
    "partial_roc_auc",
    "average_precision_score",
    "weighted_average_precision_score",
    "samples_average_precision_score",
    "micro_average_precision_score",
    "coverage_error",
    "label_ranking_loss",
    "ndcg_score",
    "dcg_score",
    "label_ranking_average_precision_score",
}

# Classification metrics with  "multilabel-indicator" format
MULTILABELS_METRICS = {
    "accuracy_score",
    "unnormalized_accuracy_score",
    "hamming_loss",
    "zero_one_loss",
    "unnormalized_zero_one_loss",
    "weighted_f0.5_score",
    "weighted_f1_score",
    "weighted_f2_score",
    "weighted_precision_score",
    "weighted_recall_score",
    "weighted_jaccard_score",
    "macro_f0.5_score",
    "macro_f1_score",
    "macro_f2_score",
    "macro_precision_score",
    "macro_recall_score",
    "macro_jaccard_score",
    "micro_f0.5_score",
    "micro_f1_score",
    "micro_f2_score",
    "micro_precision_score",
    "micro_recall_score",
    "micro_jaccard_score",
    "unnormalized_multilabel_confusion_matrix",
    "samples_f0.5_score",
    "samples_f1_score",
    "samples_f2_score",
    "samples_precision_score",
    "samples_recall_score",
    "samples_jaccard_score",
}

# Regression metrics with "multioutput-continuous" format support
MULTIOUTPUT_METRICS = {
    "mean_absolute_error",
    "median_absolute_error",
    "mean_squared_error",
    "r2_score",
    "explained_variance_score",
    "mean_absolute_percentage_error",
    "mean_pinball_loss",
}

# Symmetric with respect to their input arguments y_true and y_pred
# metric(y_true, y_pred) == metric(y_pred, y_true).
SYMMETRIC_METRICS = {
    "accuracy_score",
    "unnormalized_accuracy_score",
    "hamming_loss",
    "zero_one_loss",
    "unnormalized_zero_one_loss",
    "micro_jaccard_score",
    "macro_jaccard_score",
    "jaccard_score",
    "samples_jaccard_score",
    "f1_score",
    "micro_f1_score",
    "macro_f1_score",
    "weighted_recall_score",
    # P = R = F = accuracy in multiclass case
    "micro_f0.5_score",
    "micro_f1_score",
    "micro_f2_score",
    "micro_precision_score",
    "micro_recall_score",
    "matthews_corrcoef_score",
    "mean_absolute_error",
    "mean_squared_error",
    "median_absolute_error",
    "max_error",
    # Pinball loss is only symmetric for alpha=0.5 which is the default.
    "mean_pinball_loss",
    "cohen_kappa_score",
    "mean_normal_deviance",
}

# Asymmetric with respect to their input arguments y_true and y_pred
# metric(y_true, y_pred) != metric(y_pred, y_true).
NOT_SYMMETRIC_METRICS = {
    "balanced_accuracy_score",
    "adjusted_balanced_accuracy_score",
    "explained_variance_score",
    "r2_score",
    "unnormalized_confusion_matrix",
    "normalized_confusion_matrix",
    "roc_curve",
    "precision_recall_curve",
    "det_curve",
    "precision_score",
    "recall_score",
    "f2_score",
    "f0.5_score",
    "weighted_f0.5_score",
    "weighted_f1_score",
    "weighted_f2_score",
    "weighted_precision_score",
    "weighted_jaccard_score",
    "unnormalized_multilabel_confusion_matrix",
    "macro_f0.5_score",
    "macro_f2_score",
    "macro_precision_score",
    "macro_recall_score",
    "log_loss",
    "hinge_loss",
    "mean_gamma_deviance",
    "mean_poisson_deviance",
    "mean_compound_poisson_deviance",
    "d2_tweedie_score",
    "mean_absolute_percentage_error",
}


# No Sample weight support
METRICS_WITHOUT_SAMPLE_WEIGHT = {
    "median_absolute_error",
    "max_error",
    "ovo_roc_auc",
    "weighted_ovo_roc_auc",
}

METRICS_REQUIRE_POSITIVE_Y = {
    "mean_poisson_deviance",
    "mean_gamma_deviance",
    "mean_compound_poisson_deviance",
    "d2_tweedie_score",
}


def _require_positive_targets(y1, y2):
    """Make targets strictly positive"""
    offset = abs(min(y1.min(), y2.min())) + 1
    y1 += offset
    y2 += offset
    return y1, y2


def test_symmetry_consistency():

    # We shouldn't forget any metrics
    assert (
        SYMMETRIC_METRICS
        | NOT_SYMMETRIC_METRICS
        | set(THRESHOLDED_METRICS)
        | METRIC_UNDEFINED_BINARY_MULTICLASS
    ) == set(ALL_METRICS)

    assert (SYMMETRIC_METRICS & NOT_SYMMETRIC_METRICS) == set()


@pytest.mark.parametrize("name", sorted(SYMMETRIC_METRICS))
def test_symmetric_metric(name):
    # Test the symmetry of score and loss functions
    random_state = check_random_state(0)
    y_true = random_state.randint(0, 2, size=(20,))
    y_pred = random_state.randint(0, 2, size=(20,))

    if name in METRICS_REQUIRE_POSITIVE_Y:
        y_true, y_pred = _require_positive_targets(y_true, y_pred)

    y_true_bin = random_state.randint(0, 2, size=(20, 25))
    y_pred_bin = random_state.randint(0, 2, size=(20, 25))

    metric = ALL_METRICS[name]
    if name in METRIC_UNDEFINED_BINARY:
        if name in MULTILABELS_METRICS:
            assert_allclose(
                metric(y_true_bin, y_pred_bin),
                metric(y_pred_bin, y_true_bin),
                err_msg="%s is not symmetric" % name,
            )
        else:
            assert False, "This case is currently unhandled"
    else:
        assert_allclose(
            metric(y_true, y_pred),
            metric(y_pred, y_true),
            err_msg="%s is not symmetric" % name,
        )


@pytest.mark.parametrize("name", sorted(NOT_SYMMETRIC_METRICS))
def test_not_symmetric_metric(name):
    # Test the symmetry of score and loss functions
    random_state = check_random_state(0)
    y_true = random_state.randint(0, 2, size=(20,))
    y_pred = random_state.randint(0, 2, size=(20,))

    if name in METRICS_REQUIRE_POSITIVE_Y:
        y_true, y_pred = _require_positive_targets(y_true, y_pred)

    metric = ALL_METRICS[name]

    # use context manager to supply custom error message
    with pytest.raises(AssertionError):
        assert_array_equal(metric(y_true, y_pred), metric(y_pred, y_true))
        raise ValueError("%s seems to be symmetric" % name)


@pytest.mark.parametrize(
    "name", sorted(set(ALL_METRICS) - METRIC_UNDEFINED_BINARY_MULTICLASS)
)
def test_sample_order_invariance(name):
    random_state = check_random_state(0)
    y_true = random_state.randint(0, 2, size=(20,))
    y_pred = random_state.randint(0, 2, size=(20,))

    if name in METRICS_REQUIRE_POSITIVE_Y:
        y_true, y_pred = _require_positive_targets(y_true, y_pred)

    y_true_shuffle, y_pred_shuffle = shuffle(y_true, y_pred, random_state=0)

    with ignore_warnings():
        metric = ALL_METRICS[name]
        assert_allclose(
            metric(y_true, y_pred),
            metric(y_true_shuffle, y_pred_shuffle),
            err_msg="%s is not sample order invariant" % name,
        )


@ignore_warnings
def test_sample_order_invariance_multilabel_and_multioutput():
    random_state = check_random_state(0)

    # Generate some data
    y_true = random_state.randint(0, 2, size=(20, 25))
    y_pred = random_state.randint(0, 2, size=(20, 25))
    y_score = random_state.normal(size=y_true.shape)

    y_true_shuffle, y_pred_shuffle, y_score_shuffle = shuffle(
        y_true, y_pred, y_score, random_state=0
    )

    for name in MULTILABELS_METRICS:
        metric = ALL_METRICS[name]
        assert_allclose(
            metric(y_true, y_pred),
            metric(y_true_shuffle, y_pred_shuffle),
            err_msg="%s is not sample order invariant" % name,
        )

    for name in THRESHOLDED_MULTILABEL_METRICS:
        metric = ALL_METRICS[name]
        assert_allclose(
            metric(y_true, y_score),
            metric(y_true_shuffle, y_score_shuffle),
            err_msg="%s is not sample order invariant" % name,
        )

    for name in MULTIOUTPUT_METRICS:
        metric = ALL_METRICS[name]
        assert_allclose(
            metric(y_true, y_score),
            metric(y_true_shuffle, y_score_shuffle),
            err_msg="%s is not sample order invariant" % name,
        )
        assert_allclose(
            metric(y_true, y_pred),
            metric(y_true_shuffle, y_pred_shuffle),
            err_msg="%s is not sample order invariant" % name,
        )


@pytest.mark.parametrize(
    "name", sorted(set(ALL_METRICS) - METRIC_UNDEFINED_BINARY_MULTICLASS)
)
def test_format_invariance_with_1d_vectors(name):
    random_state = check_random_state(0)
    y1 = random_state.randint(0, 2, size=(20,))
    y2 = random_state.randint(0, 2, size=(20,))

    if name in METRICS_REQUIRE_POSITIVE_Y:
        y1, y2 = _require_positive_targets(y1, y2)

    y1_list = list(y1)
    y2_list = list(y2)

    y1_1d, y2_1d = np.array(y1), np.array(y2)
    assert_array_equal(y1_1d.ndim, 1)
    assert_array_equal(y2_1d.ndim, 1)
    y1_column = np.reshape(y1_1d, (-1, 1))
    y2_column = np.reshape(y2_1d, (-1, 1))
    y1_row = np.reshape(y1_1d, (1, -1))
    y2_row = np.reshape(y2_1d, (1, -1))

    with ignore_warnings():
        metric = ALL_METRICS[name]

        measure = metric(y1, y2)

        assert_allclose(
            metric(y1_list, y2_list),
            measure,
            err_msg="%s is not representation invariant with list" % name,
        )

        assert_allclose(
            metric(y1_1d, y2_1d),
            measure,
            err_msg="%s is not representation invariant with np-array-1d" % name,
        )

        assert_allclose(
            metric(y1_column, y2_column),
            measure,
            err_msg="%s is not representation invariant with np-array-column" % name,
        )

        # Mix format support
        assert_allclose(
            metric(y1_1d, y2_list),
            measure,
            err_msg="%s is not representation invariant with mix np-array-1d and list"
            % name,
        )

        assert_allclose(
            metric(y1_list, y2_1d),
            measure,
            err_msg="%s is not representation invariant with mix np-array-1d and list"
            % name,
        )

        assert_allclose(
            metric(y1_1d, y2_column),
            measure,
            err_msg=(
                "%s is not representation invariant with mix "
                "np-array-1d and np-array-column"
            )
            % name,
        )

        assert_allclose(
            metric(y1_column, y2_1d),
            measure,
            err_msg=(
                "%s is not representation invariant with mix "
                "np-array-1d and np-array-column"
            )
            % name,
        )

        assert_allclose(
            metric(y1_list, y2_column),
            measure,
            err_msg=(
                "%s is not representation invariant with mix list and np-array-column"
            )
            % name,
        )

        assert_allclose(
            metric(y1_column, y2_list),
            measure,
            err_msg=(
                "%s is not representation invariant with mix list and np-array-column"
            )
            % name,
        )

        # These mix representations aren't allowed
        with pytest.raises(ValueError):
            metric(y1_1d, y2_row)
        with pytest.raises(ValueError):
            metric(y1_row, y2_1d)
        with pytest.raises(ValueError):
            metric(y1_list, y2_row)
        with pytest.raises(ValueError):
            metric(y1_row, y2_list)
        with pytest.raises(ValueError):
            metric(y1_column, y2_row)
        with pytest.raises(ValueError):
            metric(y1_row, y2_column)

        # NB: We do not test for y1_row, y2_row as these may be
        # interpreted as multilabel or multioutput data.
        if name not in (
            MULTIOUTPUT_METRICS | THRESHOLDED_MULTILABEL_METRICS | MULTILABELS_METRICS
        ):
            with pytest.raises(ValueError):
                metric(y1_row, y2_row)


@pytest.mark.parametrize(
    "name", sorted(set(CLASSIFICATION_METRICS) - METRIC_UNDEFINED_BINARY_MULTICLASS)
)
def test_classification_invariance_string_vs_numbers_labels(name):
    # Ensure that classification metrics with string labels are invariant
    random_state = check_random_state(0)
    y1 = random_state.randint(0, 2, size=(20,))
    y2 = random_state.randint(0, 2, size=(20,))

    y1_str = np.array(["eggs", "spam"])[y1]
    y2_str = np.array(["eggs", "spam"])[y2]

    pos_label_str = "spam"
    labels_str = ["eggs", "spam"]

    with ignore_warnings():
        metric = CLASSIFICATION_METRICS[name]
        measure_with_number = metric(y1, y2)

        # Ugly, but handle case with a pos_label and label
        metric_str = metric
        if name in METRICS_WITH_POS_LABEL:
            metric_str = partial(metric_str, pos_label=pos_label_str)

        measure_with_str = metric_str(y1_str, y2_str)

        assert_array_equal(
            measure_with_number,
            measure_with_str,
            err_msg="{0} failed string vs number invariance test".format(name),
        )

        measure_with_strobj = metric_str(y1_str.astype("O"), y2_str.astype("O"))
        assert_array_equal(
            measure_with_number,
            measure_with_strobj,
            err_msg="{0} failed string object vs number invariance test".format(name),
        )

        if name in METRICS_WITH_LABELS:
            metric_str = partial(metric_str, labels=labels_str)
            measure_with_str = metric_str(y1_str, y2_str)
            assert_array_equal(
                measure_with_number,
                measure_with_str,
                err_msg="{0} failed string vs number  invariance test".format(name),
            )

            measure_with_strobj = metric_str(y1_str.astype("O"), y2_str.astype("O"))
            assert_array_equal(
                measure_with_number,
                measure_with_strobj,
                err_msg="{0} failed string vs number  invariance test".format(name),
            )


@pytest.mark.parametrize("name", THRESHOLDED_METRICS)
def test_thresholded_invariance_string_vs_numbers_labels(name):
    # Ensure that thresholded metrics with string labels are invariant
    random_state = check_random_state(0)
    y1 = random_state.randint(0, 2, size=(20,))
    y2 = random_state.randint(0, 2, size=(20,))

    y1_str = np.array(["eggs", "spam"])[y1]

    pos_label_str = "spam"

    with ignore_warnings():
        metric = THRESHOLDED_METRICS[name]
        if name not in METRIC_UNDEFINED_BINARY:
            # Ugly, but handle case with a pos_label and label
            metric_str = metric
            if name in METRICS_WITH_POS_LABEL:
                metric_str = partial(metric_str, pos_label=pos_label_str)

            measure_with_number = metric(y1, y2)
            measure_with_str = metric_str(y1_str, y2)
            assert_array_equal(
                measure_with_number,
                measure_with_str,
                err_msg="{0} failed string vs number invariance test".format(name),
            )

            measure_with_strobj = metric_str(y1_str.astype("O"), y2)
            assert_array_equal(
                measure_with_number,
                measure_with_strobj,
                err_msg="{0} failed string object vs number invariance test".format(
                    name
                ),
            )
        else:
            # TODO those metrics doesn't support string label yet
            with pytest.raises(ValueError):
                metric(y1_str, y2)
            with pytest.raises(ValueError):
                metric(y1_str.astype("O"), y2)


invalids_nan_inf = [
    ([0, 1], [np.inf, np.inf]),
    ([0, 1], [np.nan, np.nan]),
    ([0, 1], [np.nan, np.inf]),
    ([0, 1], [np.inf, 1]),
    ([0, 1], [np.nan, 1]),
]


@pytest.mark.parametrize(
    "metric", chain(THRESHOLDED_METRICS.values(), REGRESSION_METRICS.values())
)
@pytest.mark.parametrize("y_true, y_score", invalids_nan_inf)
def test_regression_thresholded_inf_nan_input(metric, y_true, y_score):
    with pytest.raises(ValueError, match="contains NaN, infinity"):
        metric(y_true, y_score)


@pytest.mark.parametrize("metric", CLASSIFICATION_METRICS.values())
@pytest.mark.parametrize(
    "y_true, y_score",
    invalids_nan_inf +
    # Add an additional case for classification only
    # non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/6809
    [([np.nan, 1, 2], [1, 2, 3])],  # type: ignore
)
def test_classification_inf_nan_input(metric, y_true, y_score):
    """check that classification metrics raise a message mentioning the
    occurrence of non-finite values in the target vectors."""
    err_msg = "Input contains NaN, infinity or a value too large"
    with pytest.raises(ValueError, match=err_msg):
        metric(y_true, y_score)


@pytest.mark.parametrize("metric", CLASSIFICATION_METRICS.values())
def test_classification_binary_continuous_input(metric):
    """check that classification metrics raise a message of mixed type data
    with continuous/binary target vectors."""
    y_true, y_score = ["a", "b", "a"], [0.1, 0.2, 0.3]
    err_msg = (
        "Classification metrics can't handle a mix of binary and continuous targets"
    )
    with pytest.raises(ValueError, match=err_msg):
        metric(y_true, y_score)


@ignore_warnings
def check_single_sample(name):
    # Non-regression test: scores should work with a single sample.
    # This is important for leave-one-out cross validation.
    # Score functions tested are those that formerly called np.squeeze,
    # which turns an array of size 1 into a 0-d array (!).
    metric = ALL_METRICS[name]

    # assert that no exception is thrown
    if name in METRICS_REQUIRE_POSITIVE_Y:
        values = [1, 2]
    else:
        values = [0, 1]
    for i, j in product(values, repeat=2):
        metric([i], [j])


@ignore_warnings
def check_single_sample_multioutput(name):
    metric = ALL_METRICS[name]
    for i, j, k, l in product([0, 1], repeat=4):
        metric(np.array([[i, j]]), np.array([[k, l]]))


@pytest.mark.parametrize(
    "name",
    sorted(
        set(ALL_METRICS)
        # Those metrics are not always defined with one sample
        # or in multiclass classification
        - METRIC_UNDEFINED_BINARY_MULTICLASS
        - set(THRESHOLDED_METRICS)
    ),
)
def test_single_sample(name):
    check_single_sample(name)


@pytest.mark.parametrize("name", sorted(MULTIOUTPUT_METRICS | MULTILABELS_METRICS))
def test_single_sample_multioutput(name):
    check_single_sample_multioutput(name)


@pytest.mark.parametrize("name", sorted(MULTIOUTPUT_METRICS))
def test_multioutput_number_of_output_differ(name):
    y_true = np.array([[1, 0, 0, 1], [0, 1, 1, 1], [1, 1, 0, 1]])
    y_pred = np.array([[0, 0], [1, 0], [0, 0]])

    metric = ALL_METRICS[name]
    with pytest.raises(ValueError):
        metric(y_true, y_pred)


@pytest.mark.parametrize("name", sorted(MULTIOUTPUT_METRICS))
def test_multioutput_regression_invariance_to_dimension_shuffling(name):
    # test invariance to dimension shuffling
    random_state = check_random_state(0)
    y_true = random_state.uniform(0, 2, size=(20, 5))
    y_pred = random_state.uniform(0, 2, size=(20, 5))

    metric = ALL_METRICS[name]
    error = metric(y_true, y_pred)

    for _ in range(3):
        perm = random_state.permutation(y_true.shape[1])
        assert_allclose(
            metric(y_true[:, perm], y_pred[:, perm]),
            error,
            err_msg="%s is not dimension shuffling invariant" % (name),
        )


@ignore_warnings
def test_multilabel_representation_invariance():
    # Generate some data
    n_classes = 4
    n_samples = 50

    _, y1 = make_multilabel_classification(
        n_features=1,
        n_classes=n_classes,
        random_state=0,
        n_samples=n_samples,
        allow_unlabeled=True,
    )
    _, y2 = make_multilabel_classification(
        n_features=1,
        n_classes=n_classes,
        random_state=1,
        n_samples=n_samples,
        allow_unlabeled=True,
    )

    # To make sure at least one empty label is present
    y1 = np.vstack([y1, [[0] * n_classes]])
    y2 = np.vstack([y2, [[0] * n_classes]])

    y1_sparse_indicator = sp.coo_matrix(y1)
    y2_sparse_indicator = sp.coo_matrix(y2)

    y1_list_array_indicator = list(y1)
    y2_list_array_indicator = list(y2)

    y1_list_list_indicator = [list(a) for a in y1_list_array_indicator]
    y2_list_list_indicator = [list(a) for a in y2_list_array_indicator]

    for name in MULTILABELS_METRICS:
        metric = ALL_METRICS[name]

        # XXX cruel hack to work with partial functions
        if isinstance(metric, partial):
            metric.__module__ = "tmp"
            metric.__name__ = name

        measure = metric(y1, y2)

        # Check representation invariance
        assert_allclose(
            metric(y1_sparse_indicator, y2_sparse_indicator),
            measure,
            err_msg=(
                "%s failed representation invariance between "
                "dense and sparse indicator formats."
            )
            % name,
        )
        assert_almost_equal(
            metric(y1_list_list_indicator, y2_list_list_indicator),
            measure,
            err_msg=(
                "%s failed representation invariance  "
                "between dense array and list of list "
                "indicator formats."
            )
            % name,
        )
        assert_almost_equal(
            metric(y1_list_array_indicator, y2_list_array_indicator),
            measure,
            err_msg=(
                "%s failed representation invariance  "
                "between dense and list of array "
                "indicator formats."
            )
            % name,
        )


@pytest.mark.parametrize("name", sorted(MULTILABELS_METRICS))
def test_raise_value_error_multilabel_sequences(name):
    # make sure the multilabel-sequence format raises ValueError
    multilabel_sequences = [
        [[1], [2], [0, 1]],
        [(), (2), (0, 1)],
        [[]],
        [()],
        np.array([[], [1, 2]], dtype="object"),
    ]

    metric = ALL_METRICS[name]
    for seq in multilabel_sequences:
        with pytest.raises(ValueError):
            metric(seq, seq)


@pytest.mark.parametrize("name", sorted(METRICS_WITH_NORMALIZE_OPTION))
def test_normalize_option_binary_classification(name):
    # Test in the binary case
    n_classes = 2
    n_samples = 20
    random_state = check_random_state(0)

    y_true = random_state.randint(0, n_classes, size=(n_samples,))
    y_pred = random_state.randint(0, n_classes, size=(n_samples,))
    y_score = random_state.normal(size=y_true.shape)

    metrics = ALL_METRICS[name]
    pred = y_score if name in THRESHOLDED_METRICS else y_pred
    measure_normalized = metrics(y_true, pred, normalize=True)
    measure_not_normalized = metrics(y_true, pred, normalize=False)

    assert_array_less(
        -1.0 * measure_normalized,
        0,
        err_msg="We failed to test correctly the normalize option",
    )

    assert_allclose(
        measure_normalized,
        measure_not_normalized / n_samples,
        err_msg=f"Failed with {name}",
    )


@pytest.mark.parametrize("name", sorted(METRICS_WITH_NORMALIZE_OPTION))
def test_normalize_option_multiclass_classification(name):
    # Test in the multiclass case
    n_classes = 4
    n_samples = 20
    random_state = check_random_state(0)

    y_true = random_state.randint(0, n_classes, size=(n_samples,))
    y_pred = random_state.randint(0, n_classes, size=(n_samples,))
    y_score = random_state.uniform(size=(n_samples, n_classes))

    metrics = ALL_METRICS[name]
    pred = y_score if name in THRESHOLDED_METRICS else y_pred
    measure_normalized = metrics(y_true, pred, normalize=True)
    measure_not_normalized = metrics(y_true, pred, normalize=False)

    assert_array_less(
        -1.0 * measure_normalized,
        0,
        err_msg="We failed to test correctly the normalize option",
    )

    assert_allclose(
        measure_normalized,
        measure_not_normalized / n_samples,
        err_msg=f"Failed with {name}",
    )


@pytest.mark.parametrize(
    "name", sorted(METRICS_WITH_NORMALIZE_OPTION.intersection(MULTILABELS_METRICS))
)
def test_normalize_option_multilabel_classification(name):
    # Test in the multilabel case
    n_classes = 4
    n_samples = 100
    random_state = check_random_state(0)

    # for both random_state 0 and 1, y_true and y_pred has at least one
    # unlabelled entry
    _, y_true = make_multilabel_classification(
        n_features=1,
        n_classes=n_classes,
        random_state=0,
        allow_unlabeled=True,
        n_samples=n_samples,
    )
    _, y_pred = make_multilabel_classification(
        n_features=1,
        n_classes=n_classes,
        random_state=1,
        allow_unlabeled=True,
        n_samples=n_samples,
    )

    y_score = random_state.uniform(size=y_true.shape)

    # To make sure at least one empty label is present
    y_true += [0] * n_classes
    y_pred += [0] * n_classes

    metrics = ALL_METRICS[name]
    pred = y_score if name in THRESHOLDED_METRICS else y_pred
    measure_normalized = metrics(y_true, pred, normalize=True)
    measure_not_normalized = metrics(y_true, pred, normalize=False)

    assert_array_less(
        -1.0 * measure_normalized,
        0,
        err_msg="We failed to test correctly the normalize option",
    )

    assert_allclose(
        measure_normalized,
        measure_not_normalized / n_samples,
        err_msg=f"Failed with {name}",
    )


@ignore_warnings
def _check_averaging(
    metric, y_true, y_pred, y_true_binarize, y_pred_binarize, is_multilabel
):
    n_samples, n_classes = y_true_binarize.shape

    # No averaging
    label_measure = metric(y_true, y_pred, average=None)
    assert_allclose(
        label_measure,
        [
            metric(y_true_binarize[:, i], y_pred_binarize[:, i])
            for i in range(n_classes)
        ],
    )

    # Micro measure
    micro_measure = metric(y_true, y_pred, average="micro")
    assert_allclose(
        micro_measure, metric(y_true_binarize.ravel(), y_pred_binarize.ravel())
    )

    # Macro measure
    macro_measure = metric(y_true, y_pred, average="macro")
    assert_allclose(macro_measure, np.mean(label_measure))

    # Weighted measure
    weights = np.sum(y_true_binarize, axis=0, dtype=int)

    if np.sum(weights) != 0:
        weighted_measure = metric(y_true, y_pred, average="weighted")
        assert_allclose(weighted_measure, np.average(label_measure, weights=weights))
    else:
        weighted_measure = metric(y_true, y_pred, average="weighted")
        assert_allclose(weighted_measure, 0)

    # Sample measure
    if is_multilabel:
        sample_measure = metric(y_true, y_pred, average="samples")
        assert_allclose(
            sample_measure,
            np.mean(
                [
                    metric(y_true_binarize[i], y_pred_binarize[i])
                    for i in range(n_samples)
                ]
            ),
        )

    with pytest.raises(ValueError):
        metric(y_true, y_pred, average="unknown")
    with pytest.raises(ValueError):
        metric(y_true, y_pred, average="garbage")


def check_averaging(name, y_true, y_true_binarize, y_pred, y_pred_binarize, y_score):
    is_multilabel = type_of_target(y_true).startswith("multilabel")

    metric = ALL_METRICS[name]

    if name in METRICS_WITH_AVERAGING:
        _check_averaging(
            metric, y_true, y_pred, y_true_binarize, y_pred_binarize, is_multilabel
        )
    elif name in THRESHOLDED_METRICS_WITH_AVERAGING:
        _check_averaging(
            metric, y_true, y_score, y_true_binarize, y_score, is_multilabel
        )
    else:
        raise ValueError("Metric is not recorded as having an average option")


@pytest.mark.parametrize("name", sorted(METRICS_WITH_AVERAGING))
def test_averaging_multiclass(name):
    n_samples, n_classes = 50, 3
    random_state = check_random_state(0)
    y_true = random_state.randint(0, n_classes, size=(n_samples,))
    y_pred = random_state.randint(0, n_classes, size=(n_samples,))
    y_score = random_state.uniform(size=(n_samples, n_classes))

    lb = LabelBinarizer().fit(y_true)
    y_true_binarize = lb.transform(y_true)
    y_pred_binarize = lb.transform(y_pred)

    check_averaging(name, y_true, y_true_binarize, y_pred, y_pred_binarize, y_score)


@pytest.mark.parametrize(
    "name", sorted(METRICS_WITH_AVERAGING | THRESHOLDED_METRICS_WITH_AVERAGING)
)
def test_averaging_multilabel(name):
    n_samples, n_classes = 40, 5
    _, y = make_multilabel_classification(
        n_features=1,
        n_classes=n_classes,
        random_state=5,
        n_samples=n_samples,
        allow_unlabeled=False,
    )
    y_true = y[:20]
    y_pred = y[20:]
    y_score = check_random_state(0).normal(size=(20, n_classes))
    y_true_binarize = y_true
    y_pred_binarize = y_pred

    check_averaging(name, y_true, y_true_binarize, y_pred, y_pred_binarize, y_score)


@pytest.mark.parametrize("name", sorted(METRICS_WITH_AVERAGING))
def test_averaging_multilabel_all_zeroes(name):
    y_true = np.zeros((20, 3))
    y_pred = np.zeros((20, 3))
    y_score = np.zeros((20, 3))
    y_true_binarize = y_true
    y_pred_binarize = y_pred

    check_averaging(name, y_true, y_true_binarize, y_pred, y_pred_binarize, y_score)


def test_averaging_binary_multilabel_all_zeroes():
    y_true = np.zeros((20, 3))
    y_pred = np.zeros((20, 3))
    y_true_binarize = y_true
    y_pred_binarize = y_pred
    # Test _average_binary_score for weight.sum() == 0
    binary_metric = lambda y_true, y_score, average="macro": _average_binary_score(
        precision_score, y_true, y_score, average
    )
    _check_averaging(
        binary_metric,
        y_true,
        y_pred,
        y_true_binarize,
        y_pred_binarize,
        is_multilabel=True,
    )


@pytest.mark.parametrize("name", sorted(METRICS_WITH_AVERAGING))
def test_averaging_multilabel_all_ones(name):
    y_true = np.ones((20, 3))
    y_pred = np.ones((20, 3))
    y_score = np.ones((20, 3))
    y_true_binarize = y_true
    y_pred_binarize = y_pred

    check_averaging(name, y_true, y_true_binarize, y_pred, y_pred_binarize, y_score)


@ignore_warnings
def check_sample_weight_invariance(name, metric, y1, y2):
    rng = np.random.RandomState(0)
    sample_weight = rng.randint(1, 10, size=len(y1))

    # top_k_accuracy_score always lead to a perfect score for k > 1 in the
    # binary case
    metric = partial(metric, k=1) if name == "top_k_accuracy_score" else metric

    # check that unit weights gives the same score as no weight
    unweighted_score = metric(y1, y2, sample_weight=None)

    assert_allclose(
        unweighted_score,
        metric(y1, y2, sample_weight=np.ones(shape=len(y1))),
        err_msg="For %s sample_weight=None is not equivalent to sample_weight=ones"
        % name,
    )

    # check that the weighted and unweighted scores are unequal
    weighted_score = metric(y1, y2, sample_weight=sample_weight)

    # use context manager to supply custom error message
    with pytest.raises(AssertionError):
        assert_allclose(unweighted_score, weighted_score)
        raise ValueError(
            "Unweighted and weighted scores are unexpectedly "
            "almost equal (%s) and (%s) "
            "for %s" % (unweighted_score, weighted_score, name)
        )

    # check that sample_weight can be a list
    weighted_score_list = metric(y1, y2, sample_weight=sample_weight.tolist())
    assert_allclose(
        weighted_score,
        weighted_score_list,
        err_msg=(
            "Weighted scores for array and list "
            "sample_weight input are not equal (%s != %s) for %s"
        )
        % (weighted_score, weighted_score_list, name),
    )

    # check that integer weights is the same as repeated samples
    repeat_weighted_score = metric(
        np.repeat(y1, sample_weight, axis=0),
        np.repeat(y2, sample_weight, axis=0),
        sample_weight=None,
    )
    assert_allclose(
        weighted_score,
        repeat_weighted_score,
        err_msg="Weighting %s is not equal to repeating samples" % name,
    )

    # check that ignoring a fraction of the samples is equivalent to setting
    # the corresponding weights to zero
    sample_weight_subset = sample_weight[1::2]
    sample_weight_zeroed = np.copy(sample_weight)
    sample_weight_zeroed[::2] = 0
    y1_subset = y1[1::2]
    y2_subset = y2[1::2]
    weighted_score_subset = metric(
        y1_subset, y2_subset, sample_weight=sample_weight_subset
    )
    weighted_score_zeroed = metric(y1, y2, sample_weight=sample_weight_zeroed)
    assert_allclose(
        weighted_score_subset,
        weighted_score_zeroed,
        err_msg=(
            "Zeroing weights does not give the same result as "
            "removing the corresponding samples (%s != %s) for %s"
        )
        % (weighted_score_zeroed, weighted_score_subset, name),
    )

    if not name.startswith("unnormalized"):
        # check that the score is invariant under scaling of the weights by a
        # common factor
        for scaling in [2, 0.3]:
            assert_allclose(
                weighted_score,
                metric(y1, y2, sample_weight=sample_weight * scaling),
                err_msg="%s sample_weight is not invariant under scaling" % name,
            )

    # Check that if number of samples in y_true and sample_weight are not
    # equal, meaningful error is raised.
    error_message = (
        r"Found input variables with inconsistent numbers of "
        r"samples: \[{}, {}, {}\]".format(
            _num_samples(y1), _num_samples(y2), _num_samples(sample_weight) * 2
        )
    )
    with pytest.raises(ValueError, match=error_message):
        metric(y1, y2, sample_weight=np.hstack([sample_weight, sample_weight]))


@pytest.mark.parametrize(
    "name",
    sorted(
        set(ALL_METRICS).intersection(set(REGRESSION_METRICS))
        - METRICS_WITHOUT_SAMPLE_WEIGHT
    ),
)
def test_regression_sample_weight_invariance(name):
    n_samples = 50
    random_state = check_random_state(0)
    # regression
    y_true = random_state.random_sample(size=(n_samples,))
    y_pred = random_state.random_sample(size=(n_samples,))
    metric = ALL_METRICS[name]
    check_sample_weight_invariance(name, metric, y_true, y_pred)


@pytest.mark.parametrize(
    "name",
    sorted(
        set(ALL_METRICS)
        - set(REGRESSION_METRICS)
        - METRICS_WITHOUT_SAMPLE_WEIGHT
        - METRIC_UNDEFINED_BINARY
    ),
)
def test_binary_sample_weight_invariance(name):
    # binary
    n_samples = 50
    random_state = check_random_state(0)
    y_true = random_state.randint(0, 2, size=(n_samples,))
    y_pred = random_state.randint(0, 2, size=(n_samples,))
    y_score = random_state.random_sample(size=(n_samples,))
    metric = ALL_METRICS[name]
    if name in THRESHOLDED_METRICS:
        check_sample_weight_invariance(name, metric, y_true, y_score)
    else:
        check_sample_weight_invariance(name, metric, y_true, y_pred)


@pytest.mark.parametrize(
    "name",
    sorted(
        set(ALL_METRICS)
        - set(REGRESSION_METRICS)
        - METRICS_WITHOUT_SAMPLE_WEIGHT
        - METRIC_UNDEFINED_BINARY_MULTICLASS
    ),
)
def test_multiclass_sample_weight_invariance(name):
    # multiclass
    n_samples = 50
    random_state = check_random_state(0)
    y_true = random_state.randint(0, 5, size=(n_samples,))
    y_pred = random_state.randint(0, 5, size=(n_samples,))
    y_score = random_state.random_sample(size=(n_samples, 5))
    metric = ALL_METRICS[name]
    if name in THRESHOLDED_METRICS:
        # softmax
        temp = np.exp(-y_score)
        y_score_norm = temp / temp.sum(axis=-1).reshape(-1, 1)
        check_sample_weight_invariance(name, metric, y_true, y_score_norm)
    else:
        check_sample_weight_invariance(name, metric, y_true, y_pred)


@pytest.mark.parametrize(
    "name",
    sorted(
        (MULTILABELS_METRICS | THRESHOLDED_MULTILABEL_METRICS | MULTIOUTPUT_METRICS)
        - METRICS_WITHOUT_SAMPLE_WEIGHT
    ),
)
def test_multilabel_sample_weight_invariance(name):
    # multilabel indicator
    random_state = check_random_state(0)
    _, ya = make_multilabel_classification(
        n_features=1, n_classes=10, random_state=0, n_samples=50, allow_unlabeled=False
    )
    _, yb = make_multilabel_classification(
        n_features=1, n_classes=10, random_state=1, n_samples=50, allow_unlabeled=False
    )
    y_true = np.vstack([ya, yb])
    y_pred = np.vstack([ya, ya])
    y_score = random_state.randint(1, 4, size=y_true.shape)

    metric = ALL_METRICS[name]
    if name in THRESHOLDED_METRICS:
        check_sample_weight_invariance(name, metric, y_true, y_score)
    else:
        check_sample_weight_invariance(name, metric, y_true, y_pred)


@ignore_warnings
def test_no_averaging_labels():
    # test labels argument when not using averaging
    # in multi-class and multi-label cases
    y_true_multilabel = np.array([[1, 1, 0, 0], [1, 1, 0, 0]])
    y_pred_multilabel = np.array([[0, 0, 1, 1], [0, 1, 1, 0]])
    y_true_multiclass = np.array([0, 1, 2])
    y_pred_multiclass = np.array([0, 2, 3])
    labels = np.array([3, 0, 1, 2])
    _, inverse_labels = np.unique(labels, return_inverse=True)

    for name in METRICS_WITH_AVERAGING:
        for y_true, y_pred in [
            [y_true_multiclass, y_pred_multiclass],
            [y_true_multilabel, y_pred_multilabel],
        ]:
            if name not in MULTILABELS_METRICS and y_pred.ndim > 1:
                continue

            metric = ALL_METRICS[name]

            score_labels = metric(y_true, y_pred, labels=labels, average=None)
            score = metric(y_true, y_pred, average=None)
            assert_array_equal(score_labels, score[inverse_labels])


@pytest.mark.parametrize(
    "name", sorted(MULTILABELS_METRICS - {"unnormalized_multilabel_confusion_matrix"})
)
def test_multilabel_label_permutations_invariance(name):
    random_state = check_random_state(0)
    n_samples, n_classes = 20, 4

    y_true = random_state.randint(0, 2, size=(n_samples, n_classes))
    y_score = random_state.randint(0, 2, size=(n_samples, n_classes))

    metric = ALL_METRICS[name]
    score = metric(y_true, y_score)

    for perm in permutations(range(n_classes), n_classes):
        y_score_perm = y_score[:, perm]
        y_true_perm = y_true[:, perm]

        current_score = metric(y_true_perm, y_score_perm)
        assert_almost_equal(score, current_score)


@pytest.mark.parametrize(
    "name", sorted(THRESHOLDED_MULTILABEL_METRICS | MULTIOUTPUT_METRICS)
)
def test_thresholded_multilabel_multioutput_permutations_invariance(name):
    random_state = check_random_state(0)
    n_samples, n_classes = 20, 4
    y_true = random_state.randint(0, 2, size=(n_samples, n_classes))
    y_score = random_state.normal(size=y_true.shape)

    # Makes sure all samples have at least one label. This works around errors
    # when running metrics where average="sample"
    y_true[y_true.sum(1) == 4, 0] = 0
    y_true[y_true.sum(1) == 0, 0] = 1

    metric = ALL_METRICS[name]
    score = metric(y_true, y_score)

    for perm in permutations(range(n_classes), n_classes):
        y_score_perm = y_score[:, perm]
        y_true_perm = y_true[:, perm]

        current_score = metric(y_true_perm, y_score_perm)
        if metric == mean_absolute_percentage_error:
            assert np.isfinite(current_score)
            assert current_score > 1e6
            # Here we are not comparing the values in case of MAPE because
            # whenever y_true value is exactly zero, the MAPE value doesn't
            # signify anything. Thus, in this case we are just expecting
            # very large finite value.
        else:
            assert_almost_equal(score, current_score)


@pytest.mark.parametrize(
    "name", sorted(set(THRESHOLDED_METRICS) - METRIC_UNDEFINED_BINARY_MULTICLASS)
)
def test_thresholded_metric_permutation_invariance(name):
    n_samples, n_classes = 100, 3
    random_state = check_random_state(0)

    y_score = random_state.rand(n_samples, n_classes)
    temp = np.exp(-y_score)
    y_score = temp / temp.sum(axis=-1).reshape(-1, 1)
    y_true = random_state.randint(0, n_classes, size=n_samples)

    metric = ALL_METRICS[name]
    score = metric(y_true, y_score)
    for perm in permutations(range(n_classes), n_classes):
        inverse_perm = np.zeros(n_classes, dtype=int)
        inverse_perm[list(perm)] = np.arange(n_classes)
        y_score_perm = y_score[:, inverse_perm]
        y_true_perm = np.take(perm, y_true)

        current_score = metric(y_true_perm, y_score_perm)
        assert_almost_equal(score, current_score)


@pytest.mark.parametrize("metric_name", CLASSIFICATION_METRICS)
def test_metrics_consistent_type_error(metric_name):
    # check that an understable message is raised when the type between y_true
    # and y_pred mismatch
    rng = np.random.RandomState(42)
    y1 = np.array(["spam"] * 3 + ["eggs"] * 2, dtype=object)
    y2 = rng.randint(0, 2, size=y1.size)

    err_msg = "Labels in y_true and y_pred should be of the same type."
    with pytest.raises(TypeError, match=err_msg):
        CLASSIFICATION_METRICS[metric_name](y1, y2)


@pytest.mark.parametrize(
    "metric, y_pred_threshold",
    [
        (average_precision_score, True),
        (brier_score_loss, True),
        (f1_score, False),
        (partial(fbeta_score, beta=1), False),
        (jaccard_score, False),
        (precision_recall_curve, True),
        (precision_score, False),
        (recall_score, False),
        (roc_curve, True),
    ],
)
@pytest.mark.parametrize("dtype_y_str", [str, object])
def test_metrics_pos_label_error_str(metric, y_pred_threshold, dtype_y_str):
    # check that the error message if `pos_label` is not specified and the
    # targets is made of strings.
    rng = np.random.RandomState(42)
    y1 = np.array(["spam"] * 3 + ["eggs"] * 2, dtype=dtype_y_str)
    y2 = rng.randint(0, 2, size=y1.size)

    if not y_pred_threshold:
        y2 = np.array(["spam", "eggs"], dtype=dtype_y_str)[y2]

    err_msg_pos_label_None = (
        "y_true takes value in {'eggs', 'spam'} and pos_label is not "
        "specified: either make y_true take value in {0, 1} or {-1, 1} or "
        "pass pos_label explicit"
    )
    err_msg_pos_label_1 = (
        r"pos_label=1 is not a valid label. It should be one of " r"\['eggs', 'spam'\]"
    )

    pos_label_default = signature(metric).parameters["pos_label"].default

    err_msg = err_msg_pos_label_1 if pos_label_default == 1 else err_msg_pos_label_None
    with pytest.raises(ValueError, match=err_msg):
        metric(y1, y2)
