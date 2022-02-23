"""
Testing for mean shift clustering methods

"""

import numpy as np
import warnings
import pytest

from scipy import sparse

from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_allclose

from sklearn.cluster import MeanShift
from sklearn.cluster import mean_shift
from sklearn.cluster import estimate_bandwidth
from sklearn.cluster import get_bin_seeds
from sklearn.datasets import make_blobs
from sklearn.metrics import v_measure_score


n_clusters = 3
centers = np.array([[1, 1], [-1, -1], [1, -1]]) + 10
X, _ = make_blobs(
    n_samples=300,
    n_features=2,
    centers=centers,
    cluster_std=0.4,
    shuffle=True,
    random_state=11,
)


def test_estimate_bandwidth():
    # Test estimate_bandwidth
    bandwidth = estimate_bandwidth(X, n_samples=200)
    assert 0.9 <= bandwidth <= 1.5


def test_estimate_bandwidth_1sample():
    # Test estimate_bandwidth when n_samples=1 and quantile<1, so that
    # n_neighbors is set to 1.
    bandwidth = estimate_bandwidth(X, n_samples=1, quantile=0.3)
    assert bandwidth == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize(
    "bandwidth, cluster_all, expected, first_cluster_label",
    [(1.2, True, 3, 0), (1.2, False, 4, -1)],
)
def test_mean_shift(bandwidth, cluster_all, expected, first_cluster_label):
    # Test MeanShift algorithm
    ms = MeanShift(bandwidth=bandwidth, cluster_all=cluster_all)
    labels = ms.fit(X).labels_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    assert n_clusters_ == expected
    assert labels_unique[0] == first_cluster_label

    cluster_centers, labels_mean_shift = mean_shift(X, cluster_all=cluster_all)
    labels_mean_shift_unique = np.unique(labels_mean_shift)
    n_clusters_mean_shift = len(labels_mean_shift_unique)
    assert n_clusters_mean_shift == expected
    assert labels_mean_shift_unique[0] == first_cluster_label


def test_mean_shift_negative_bandwidth():
    bandwidth = -1
    ms = MeanShift(bandwidth=bandwidth)
    msg = r"bandwidth needs to be greater than zero or None," r" got -1\.000000"
    with pytest.raises(ValueError, match=msg):
        ms.fit(X)


def test_estimate_bandwidth_with_sparse_matrix():
    # Test estimate_bandwidth with sparse matrix
    X = sparse.lil_matrix((1000, 1000))
    msg = "A sparse matrix was passed, but dense data is required."
    with pytest.raises(TypeError, match=msg):
        estimate_bandwidth(X)


def test_parallel():
    centers = np.array([[1, 1], [-1, -1], [1, -1]]) + 10
    X, _ = make_blobs(
        n_samples=50,
        n_features=2,
        centers=centers,
        cluster_std=0.4,
        shuffle=True,
        random_state=11,
    )

    ms1 = MeanShift(n_jobs=2)
    ms1.fit(X)

    ms2 = MeanShift()
    ms2.fit(X)

    assert_array_almost_equal(ms1.cluster_centers_, ms2.cluster_centers_)
    assert_array_equal(ms1.labels_, ms2.labels_)


def test_meanshift_predict():
    # Test MeanShift.predict
    ms = MeanShift(bandwidth=1.2)
    labels = ms.fit_predict(X)
    labels2 = ms.predict(X)
    assert_array_equal(labels, labels2)


def test_meanshift_all_orphans():
    # init away from the data, crash with a sensible warning
    ms = MeanShift(bandwidth=0.1, seeds=[[-9, -9], [-10, -10]])
    msg = "No point was within bandwidth=0.1"
    with pytest.raises(ValueError, match=msg):
        ms.fit(
            X,
        )


def test_unfitted():
    # Non-regression: before fit, there should be not fitted attributes.
    ms = MeanShift()
    assert not hasattr(ms, "cluster_centers_")
    assert not hasattr(ms, "labels_")


def test_cluster_intensity_tie():
    X = np.array([[1, 1], [2, 1], [1, 0], [4, 7], [3, 5], [3, 6]])
    c1 = MeanShift(bandwidth=2).fit(X)

    X = np.array([[4, 7], [3, 5], [3, 6], [1, 1], [2, 1], [1, 0]])
    c2 = MeanShift(bandwidth=2).fit(X)
    assert_array_equal(c1.labels_, [1, 1, 1, 0, 0, 0])
    assert_array_equal(c2.labels_, [0, 0, 0, 1, 1, 1])


def test_bin_seeds():
    # Test the bin seeding technique which can be used in the mean shift
    # algorithm
    # Data is just 6 points in the plane
    X = np.array(
        [[1.0, 1.0], [1.4, 1.4], [1.8, 1.2], [2.0, 1.0], [2.1, 1.1], [0.0, 0.0]]
    )

    # With a bin coarseness of 1.0 and min_bin_freq of 1, 3 bins should be
    # found
    ground_truth = {(1.0, 1.0), (2.0, 1.0), (0.0, 0.0)}
    test_bins = get_bin_seeds(X, 1, 1)
    test_result = set(tuple(p) for p in test_bins)
    assert len(ground_truth.symmetric_difference(test_result)) == 0

    # With a bin coarseness of 1.0 and min_bin_freq of 2, 2 bins should be
    # found
    ground_truth = {(1.0, 1.0), (2.0, 1.0)}
    test_bins = get_bin_seeds(X, 1, 2)
    test_result = set(tuple(p) for p in test_bins)
    assert len(ground_truth.symmetric_difference(test_result)) == 0

    # With a bin size of 0.01 and min_bin_freq of 1, 6 bins should be found
    # we bail and use the whole data here.
    with warnings.catch_warnings(record=True):
        test_bins = get_bin_seeds(X, 0.01, 1)
    assert_array_almost_equal(test_bins, X)

    # tight clusters around [0, 0] and [1, 1], only get two bins
    X, _ = make_blobs(
        n_samples=100,
        n_features=2,
        centers=[[0, 0], [1, 1]],
        cluster_std=0.1,
        random_state=0,
    )
    test_bins = get_bin_seeds(X, 1)
    assert_array_equal(test_bins, [[0, 0], [1, 1]])


@pytest.mark.parametrize("max_iter", [1, 100])
def test_max_iter(max_iter):
    clusters1, _ = mean_shift(X, max_iter=max_iter)
    ms = MeanShift(max_iter=max_iter).fit(X)
    clusters2 = ms.cluster_centers_

    assert ms.n_iter_ <= ms.max_iter
    assert len(clusters1) == len(clusters2)

    for c1, c2 in zip(clusters1, clusters2):
        assert np.allclose(c1, c2)


def test_mean_shift_zero_bandwidth():
    # Check that mean shift works when the estimated bandwidth is 0.
    X = np.array([1, 1, 1, 2, 2, 2, 3, 3]).reshape(-1, 1)

    # estimate_bandwidth with default args returns 0 on this dataset
    bandwidth = estimate_bandwidth(X)
    assert bandwidth == 0

    # get_bin_seeds with a 0 bin_size should return the dataset itself
    assert get_bin_seeds(X, bin_size=bandwidth) is X

    # MeanShift with binning and a 0 estimated bandwidth should be equivalent
    # to no binning.
    ms_binning = MeanShift(bin_seeding=True, bandwidth=None).fit(X)
    ms_nobinning = MeanShift(bin_seeding=False).fit(X)
    expected_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2])

    assert v_measure_score(ms_binning.labels_, expected_labels) == 1
    assert v_measure_score(ms_nobinning.labels_, expected_labels) == 1
    assert_allclose(ms_binning.cluster_centers_, ms_nobinning.cluster_centers_)
