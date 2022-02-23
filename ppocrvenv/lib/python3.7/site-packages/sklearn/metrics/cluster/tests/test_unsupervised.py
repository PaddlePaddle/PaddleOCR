import numpy as np
import scipy.sparse as sp
import pytest
from scipy.sparse import csr_matrix

from sklearn import datasets
from sklearn.utils._testing import assert_array_equal
from sklearn.metrics.cluster import silhouette_score
from sklearn.metrics.cluster import silhouette_samples
from sklearn.metrics import pairwise_distances
from sklearn.metrics.cluster import calinski_harabasz_score
from sklearn.metrics.cluster import davies_bouldin_score


def test_silhouette():
    # Tests the Silhouette Coefficient.
    dataset = datasets.load_iris()
    X_dense = dataset.data
    X_csr = csr_matrix(X_dense)
    X_dok = sp.dok_matrix(X_dense)
    X_lil = sp.lil_matrix(X_dense)
    y = dataset.target

    for X in [X_dense, X_csr, X_dok, X_lil]:
        D = pairwise_distances(X, metric="euclidean")
        # Given that the actual labels are used, we can assume that S would be
        # positive.
        score_precomputed = silhouette_score(D, y, metric="precomputed")
        assert score_precomputed > 0
        # Test without calculating D
        score_euclidean = silhouette_score(X, y, metric="euclidean")
        pytest.approx(score_precomputed, score_euclidean)

        if X is X_dense:
            score_dense_without_sampling = score_precomputed
        else:
            pytest.approx(score_euclidean, score_dense_without_sampling)

        # Test with sampling
        score_precomputed = silhouette_score(
            D, y, metric="precomputed", sample_size=int(X.shape[0] / 2), random_state=0
        )
        score_euclidean = silhouette_score(
            X, y, metric="euclidean", sample_size=int(X.shape[0] / 2), random_state=0
        )
        assert score_precomputed > 0
        assert score_euclidean > 0
        pytest.approx(score_euclidean, score_precomputed)

        if X is X_dense:
            score_dense_with_sampling = score_precomputed
        else:
            pytest.approx(score_euclidean, score_dense_with_sampling)


def test_cluster_size_1():
    # Assert Silhouette Coefficient == 0 when there is 1 sample in a cluster
    # (cluster 0). We also test the case where there are identical samples
    # as the only members of a cluster (cluster 2). To our knowledge, this case
    # is not discussed in reference material, and we choose for it a sample
    # score of 1.
    X = [[0.0], [1.0], [1.0], [2.0], [3.0], [3.0]]
    labels = np.array([0, 1, 1, 1, 2, 2])

    # Cluster 0: 1 sample -> score of 0 by Rousseeuw's convention
    # Cluster 1: intra-cluster = [.5, .5, 1]
    #            inter-cluster = [1, 1, 1]
    #            silhouette    = [.5, .5, 0]
    # Cluster 2: intra-cluster = [0, 0]
    #            inter-cluster = [arbitrary, arbitrary]
    #            silhouette    = [1., 1.]

    silhouette = silhouette_score(X, labels)
    assert not np.isnan(silhouette)
    ss = silhouette_samples(X, labels)
    assert_array_equal(ss, [0, 0.5, 0.5, 0, 1, 1])


def test_silhouette_paper_example():
    # Explicitly check per-sample results against Rousseeuw (1987)
    # Data from Table 1
    lower = [
        5.58,
        7.00,
        6.50,
        7.08,
        7.00,
        3.83,
        4.83,
        5.08,
        8.17,
        5.83,
        2.17,
        5.75,
        6.67,
        6.92,
        4.92,
        6.42,
        5.00,
        5.58,
        6.00,
        4.67,
        6.42,
        3.42,
        5.50,
        6.42,
        6.42,
        5.00,
        3.92,
        6.17,
        2.50,
        4.92,
        6.25,
        7.33,
        4.50,
        2.25,
        6.33,
        2.75,
        6.08,
        6.67,
        4.25,
        2.67,
        6.00,
        6.17,
        6.17,
        6.92,
        6.17,
        5.25,
        6.83,
        4.50,
        3.75,
        5.75,
        5.42,
        6.08,
        5.83,
        6.67,
        3.67,
        4.75,
        3.00,
        6.08,
        6.67,
        5.00,
        5.58,
        4.83,
        6.17,
        5.67,
        6.50,
        6.92,
    ]
    D = np.zeros((12, 12))
    D[np.tril_indices(12, -1)] = lower
    D += D.T

    names = [
        "BEL",
        "BRA",
        "CHI",
        "CUB",
        "EGY",
        "FRA",
        "IND",
        "ISR",
        "USA",
        "USS",
        "YUG",
        "ZAI",
    ]

    # Data from Figure 2
    labels1 = [1, 1, 2, 2, 1, 1, 2, 1, 1, 2, 2, 1]
    expected1 = {
        "USA": 0.43,
        "BEL": 0.39,
        "FRA": 0.35,
        "ISR": 0.30,
        "BRA": 0.22,
        "EGY": 0.20,
        "ZAI": 0.19,
        "CUB": 0.40,
        "USS": 0.34,
        "CHI": 0.33,
        "YUG": 0.26,
        "IND": -0.04,
    }
    score1 = 0.28

    # Data from Figure 3
    labels2 = [1, 2, 3, 3, 1, 1, 2, 1, 1, 3, 3, 2]
    expected2 = {
        "USA": 0.47,
        "FRA": 0.44,
        "BEL": 0.42,
        "ISR": 0.37,
        "EGY": 0.02,
        "ZAI": 0.28,
        "BRA": 0.25,
        "IND": 0.17,
        "CUB": 0.48,
        "USS": 0.44,
        "YUG": 0.31,
        "CHI": 0.31,
    }
    score2 = 0.33

    for labels, expected, score in [
        (labels1, expected1, score1),
        (labels2, expected2, score2),
    ]:
        expected = [expected[name] for name in names]
        # we check to 2dp because that's what's in the paper
        pytest.approx(
            expected,
            silhouette_samples(D, np.array(labels), metric="precomputed"),
            abs=1e-2,
        )
        pytest.approx(
            score, silhouette_score(D, np.array(labels), metric="precomputed"), abs=1e-2
        )


def test_correct_labelsize():
    # Assert 1 < n_labels < n_samples
    dataset = datasets.load_iris()
    X = dataset.data

    # n_labels = n_samples
    y = np.arange(X.shape[0])
    err_msg = (
        r"Number of labels is %d\. Valid values are 2 "
        r"to n_samples - 1 \(inclusive\)" % len(np.unique(y))
    )
    with pytest.raises(ValueError, match=err_msg):
        silhouette_score(X, y)

    # n_labels = 1
    y = np.zeros(X.shape[0])
    err_msg = (
        r"Number of labels is %d\. Valid values are 2 "
        r"to n_samples - 1 \(inclusive\)" % len(np.unique(y))
    )
    with pytest.raises(ValueError, match=err_msg):
        silhouette_score(X, y)


def test_non_encoded_labels():
    dataset = datasets.load_iris()
    X = dataset.data
    labels = dataset.target
    assert silhouette_score(X, labels * 2 + 10) == silhouette_score(X, labels)
    assert_array_equal(
        silhouette_samples(X, labels * 2 + 10), silhouette_samples(X, labels)
    )


def test_non_numpy_labels():
    dataset = datasets.load_iris()
    X = dataset.data
    y = dataset.target
    assert silhouette_score(list(X), list(y)) == silhouette_score(X, y)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_silhouette_nonzero_diag(dtype):
    # Make sure silhouette_samples requires diagonal to be zero.
    # Non-regression test for #12178

    # Construct a zero-diagonal matrix
    dists = pairwise_distances(
        np.array([[0.2, 0.1, 0.12, 1.34, 1.11, 1.6]], dtype=dtype).T
    )
    labels = [0, 0, 0, 1, 1, 1]

    # small values on the diagonal are OK
    dists[2][2] = np.finfo(dists.dtype).eps * 10
    silhouette_samples(dists, labels, metric="precomputed")

    # values bigger than eps * 100 are not
    dists[2][2] = np.finfo(dists.dtype).eps * 1000
    with pytest.raises(ValueError, match="contains non-zero"):
        silhouette_samples(dists, labels, metric="precomputed")


def assert_raises_on_only_one_label(func):
    """Assert message when there is only one label"""
    rng = np.random.RandomState(seed=0)
    with pytest.raises(ValueError, match="Number of labels is"):
        func(rng.rand(10, 2), np.zeros(10))


def assert_raises_on_all_points_same_cluster(func):
    """Assert message when all point are in different clusters"""
    rng = np.random.RandomState(seed=0)
    with pytest.raises(ValueError, match="Number of labels is"):
        func(rng.rand(10, 2), np.arange(10))


def test_calinski_harabasz_score():
    assert_raises_on_only_one_label(calinski_harabasz_score)

    assert_raises_on_all_points_same_cluster(calinski_harabasz_score)

    # Assert the value is 1. when all samples are equals
    assert 1.0 == calinski_harabasz_score(np.ones((10, 2)), [0] * 5 + [1] * 5)

    # Assert the value is 0. when all the mean cluster are equal
    assert 0.0 == calinski_harabasz_score([[-1, -1], [1, 1]] * 10, [0] * 10 + [1] * 10)

    # General case (with non numpy arrays)
    X = (
        [[0, 0], [1, 1]] * 5
        + [[3, 3], [4, 4]] * 5
        + [[0, 4], [1, 3]] * 5
        + [[3, 1], [4, 0]] * 5
    )
    labels = [0] * 10 + [1] * 10 + [2] * 10 + [3] * 10
    pytest.approx(calinski_harabasz_score(X, labels), 45 * (40 - 4) / (5 * (4 - 1)))


def test_davies_bouldin_score():
    assert_raises_on_only_one_label(davies_bouldin_score)
    assert_raises_on_all_points_same_cluster(davies_bouldin_score)

    # Assert the value is 0. when all samples are equals
    assert davies_bouldin_score(np.ones((10, 2)), [0] * 5 + [1] * 5) == pytest.approx(
        0.0
    )

    # Assert the value is 0. when all the mean cluster are equal
    assert davies_bouldin_score(
        [[-1, -1], [1, 1]] * 10, [0] * 10 + [1] * 10
    ) == pytest.approx(0.0)

    # General case (with non numpy arrays)
    X = (
        [[0, 0], [1, 1]] * 5
        + [[3, 3], [4, 4]] * 5
        + [[0, 4], [1, 3]] * 5
        + [[3, 1], [4, 0]] * 5
    )
    labels = [0] * 10 + [1] * 10 + [2] * 10 + [3] * 10
    pytest.approx(davies_bouldin_score(X, labels), 2 * np.sqrt(0.5) / 3)

    # Ensure divide by zero warning is not raised in general case
    with pytest.warns(None) as record:
        davies_bouldin_score(X, labels)
    div_zero_warnings = [
        warning
        for warning in record
        if "divide by zero encountered" in warning.message.args[0]
    ]
    assert len(div_zero_warnings) == 0

    # General case - cluster have one sample
    X = [[0, 0], [2, 2], [3, 3], [5, 5]]
    labels = [0, 0, 1, 2]
    pytest.approx(davies_bouldin_score(X, labels), (5.0 / 4) / 3)
