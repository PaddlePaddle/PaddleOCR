from functools import partial
from itertools import chain

import pytest
import numpy as np

from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import rand_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import mutual_info_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.cluster import silhouette_score
from sklearn.metrics.cluster import calinski_harabasz_score
from sklearn.metrics.cluster import davies_bouldin_score

from sklearn.utils._testing import assert_allclose


# Dictionaries of metrics
# ------------------------
# The goal of having those dictionaries is to have an easy way to call a
# particular metric and associate a name to each function:
#   - SUPERVISED_METRICS: all supervised cluster metrics - (when given a
# ground truth value)
#   - UNSUPERVISED_METRICS: all unsupervised cluster metrics
#
# Those dictionaries will be used to test systematically some invariance
# properties, e.g. invariance toward several input layout.
#

SUPERVISED_METRICS = {
    "adjusted_mutual_info_score": adjusted_mutual_info_score,
    "adjusted_rand_score": adjusted_rand_score,
    "rand_score": rand_score,
    "completeness_score": completeness_score,
    "homogeneity_score": homogeneity_score,
    "mutual_info_score": mutual_info_score,
    "normalized_mutual_info_score": normalized_mutual_info_score,
    "v_measure_score": v_measure_score,
    "fowlkes_mallows_score": fowlkes_mallows_score,
}

UNSUPERVISED_METRICS = {
    "silhouette_score": silhouette_score,
    "silhouette_manhattan": partial(silhouette_score, metric="manhattan"),
    "calinski_harabasz_score": calinski_harabasz_score,
    "davies_bouldin_score": davies_bouldin_score,
}

# Lists of metrics with common properties
# ---------------------------------------
# Lists of metrics with common properties are used to test systematically some
# functionalities and invariance, e.g. SYMMETRIC_METRICS lists all metrics
# that are symmetric with respect to their input argument y_true and y_pred.
#
# --------------------------------------------------------------------
# Symmetric with respect to their input arguments y_true and y_pred.
# Symmetric metrics only apply to supervised clusters.
SYMMETRIC_METRICS = [
    "adjusted_rand_score",
    "rand_score",
    "v_measure_score",
    "mutual_info_score",
    "adjusted_mutual_info_score",
    "normalized_mutual_info_score",
    "fowlkes_mallows_score",
]

NON_SYMMETRIC_METRICS = ["homogeneity_score", "completeness_score"]

# Metrics whose upper bound is 1
NORMALIZED_METRICS = [
    "adjusted_rand_score",
    "rand_score",
    "homogeneity_score",
    "completeness_score",
    "v_measure_score",
    "adjusted_mutual_info_score",
    "fowlkes_mallows_score",
    "normalized_mutual_info_score",
]


rng = np.random.RandomState(0)
y1 = rng.randint(3, size=30)
y2 = rng.randint(3, size=30)


def test_symmetric_non_symmetric_union():
    assert sorted(SYMMETRIC_METRICS + NON_SYMMETRIC_METRICS) == sorted(
        SUPERVISED_METRICS
    )


# 0.22 AMI and NMI changes
@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize(
    "metric_name, y1, y2", [(name, y1, y2) for name in SYMMETRIC_METRICS]
)
def test_symmetry(metric_name, y1, y2):
    metric = SUPERVISED_METRICS[metric_name]
    assert metric(y1, y2) == pytest.approx(metric(y2, y1))


@pytest.mark.parametrize(
    "metric_name, y1, y2", [(name, y1, y2) for name in NON_SYMMETRIC_METRICS]
)
def test_non_symmetry(metric_name, y1, y2):
    metric = SUPERVISED_METRICS[metric_name]
    assert metric(y1, y2) != pytest.approx(metric(y2, y1))


# 0.22 AMI and NMI changes
@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize("metric_name", NORMALIZED_METRICS)
def test_normalized_output(metric_name):
    upper_bound_1 = [0, 0, 0, 1, 1, 1]
    upper_bound_2 = [0, 0, 0, 1, 1, 1]
    metric = SUPERVISED_METRICS[metric_name]
    assert metric([0, 0, 0, 1, 1], [0, 0, 0, 1, 2]) > 0.0
    assert metric([0, 0, 1, 1, 2], [0, 0, 1, 1, 1]) > 0.0
    assert metric([0, 0, 0, 1, 2], [0, 1, 1, 1, 1]) < 1.0
    assert metric([0, 0, 0, 1, 2], [0, 1, 1, 1, 1]) < 1.0
    assert metric(upper_bound_1, upper_bound_2) == pytest.approx(1.0)

    lower_bound_1 = [0, 0, 0, 0, 0, 0]
    lower_bound_2 = [0, 1, 2, 3, 4, 5]
    score = np.array(
        [metric(lower_bound_1, lower_bound_2), metric(lower_bound_2, lower_bound_1)]
    )
    assert not (score < 0).any()


# 0.22 AMI and NMI changes
@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize("metric_name", chain(SUPERVISED_METRICS, UNSUPERVISED_METRICS))
def test_permute_labels(metric_name):
    # All clustering metrics do not change score due to permutations of labels
    # that is when 0 and 1 exchanged.
    y_label = np.array([0, 0, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 1, 1, 0])
    if metric_name in SUPERVISED_METRICS:
        metric = SUPERVISED_METRICS[metric_name]
        score_1 = metric(y_pred, y_label)
        assert_allclose(score_1, metric(1 - y_pred, y_label))
        assert_allclose(score_1, metric(1 - y_pred, 1 - y_label))
        assert_allclose(score_1, metric(y_pred, 1 - y_label))
    else:
        metric = UNSUPERVISED_METRICS[metric_name]
        X = np.random.randint(10, size=(7, 10))
        score_1 = metric(X, y_pred)
        assert_allclose(score_1, metric(X, 1 - y_pred))


# 0.22 AMI and NMI changes
@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize("metric_name", chain(SUPERVISED_METRICS, UNSUPERVISED_METRICS))
# For all clustering metrics Input parameters can be both
# in the form of arrays lists, positive, negative or string
def test_format_invariance(metric_name):
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 1, 2, 3, 4, 5, 6, 7]

    def generate_formats(y):
        y = np.array(y)
        yield y, "array of ints"
        yield y.tolist(), "list of ints"
        yield [str(x) + "-a" for x in y.tolist()], "list of strs"
        yield (
            np.array([str(x) + "-a" for x in y.tolist()], dtype=object),
            "array of strs",
        )
        yield y - 1, "including negative ints"
        yield y + 1, "strictly positive ints"

    if metric_name in SUPERVISED_METRICS:
        metric = SUPERVISED_METRICS[metric_name]
        score_1 = metric(y_true, y_pred)
        y_true_gen = generate_formats(y_true)
        y_pred_gen = generate_formats(y_pred)
        for (y_true_fmt, fmt_name), (y_pred_fmt, _) in zip(y_true_gen, y_pred_gen):
            assert score_1 == metric(y_true_fmt, y_pred_fmt)
    else:
        metric = UNSUPERVISED_METRICS[metric_name]
        X = np.random.randint(10, size=(8, 10))
        score_1 = metric(X, y_true)
        assert score_1 == metric(X.astype(float), y_true)
        y_true_gen = generate_formats(y_true)
        for (y_true_fmt, fmt_name) in y_true_gen:
            assert score_1 == metric(X, y_true_fmt)


@pytest.mark.parametrize("metric", SUPERVISED_METRICS.values())
def test_single_sample(metric):
    # only the supervised metrics support single sample
    for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        metric([i], [j])


@pytest.mark.parametrize(
    "metric_name, metric_func", dict(SUPERVISED_METRICS, **UNSUPERVISED_METRICS).items()
)
def test_inf_nan_input(metric_name, metric_func):
    if metric_name in SUPERVISED_METRICS:
        invalids = [
            ([0, 1], [np.inf, np.inf]),
            ([0, 1], [np.nan, np.nan]),
            ([0, 1], [np.nan, np.inf]),
        ]
    else:
        X = np.random.randint(10, size=(2, 10))
        invalids = [(X, [np.inf, np.inf]), (X, [np.nan, np.nan]), (X, [np.nan, np.inf])]
    with pytest.raises(ValueError, match="contains NaN, infinity"):
        for args in invalids:
            metric_func(*args)
