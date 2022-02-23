"""
Tests for the birch clustering algorithm.
"""

from scipy import sparse
import numpy as np
import pytest

from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.cluster import Birch
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.metrics import pairwise_distances_argmin, v_measure_score

from sklearn.utils._testing import assert_almost_equal
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_array_almost_equal


def test_n_samples_leaves_roots():
    # Sanity check for the number of samples in leaves and roots
    X, y = make_blobs(n_samples=10)
    brc = Birch()
    brc.fit(X)
    n_samples_root = sum([sc.n_samples_ for sc in brc.root_.subclusters_])
    n_samples_leaves = sum(
        [sc.n_samples_ for leaf in brc._get_leaves() for sc in leaf.subclusters_]
    )
    assert n_samples_leaves == X.shape[0]
    assert n_samples_root == X.shape[0]


def test_partial_fit():
    # Test that fit is equivalent to calling partial_fit multiple times
    X, y = make_blobs(n_samples=100)
    brc = Birch(n_clusters=3)
    brc.fit(X)
    brc_partial = Birch(n_clusters=None)
    brc_partial.partial_fit(X[:50])
    brc_partial.partial_fit(X[50:])
    assert_array_almost_equal(brc_partial.subcluster_centers_, brc.subcluster_centers_)

    # Test that same global labels are obtained after calling partial_fit
    # with None
    brc_partial.set_params(n_clusters=3)
    brc_partial.partial_fit(None)
    assert_array_equal(brc_partial.subcluster_labels_, brc.subcluster_labels_)


def test_birch_predict():
    # Test the predict method predicts the nearest centroid.
    rng = np.random.RandomState(0)
    X = generate_clustered_data(n_clusters=3, n_features=3, n_samples_per_cluster=10)

    # n_samples * n_samples_per_cluster
    shuffle_indices = np.arange(30)
    rng.shuffle(shuffle_indices)
    X_shuffle = X[shuffle_indices, :]
    brc = Birch(n_clusters=4, threshold=1.0)
    brc.fit(X_shuffle)
    centroids = brc.subcluster_centers_
    assert_array_equal(brc.labels_, brc.predict(X_shuffle))
    nearest_centroid = pairwise_distances_argmin(X_shuffle, centroids)
    assert_almost_equal(v_measure_score(nearest_centroid, brc.labels_), 1.0)


def test_n_clusters():
    # Test that n_clusters param works properly
    X, y = make_blobs(n_samples=100, centers=10)
    brc1 = Birch(n_clusters=10)
    brc1.fit(X)
    assert len(brc1.subcluster_centers_) > 10
    assert len(np.unique(brc1.labels_)) == 10

    # Test that n_clusters = Agglomerative Clustering gives
    # the same results.
    gc = AgglomerativeClustering(n_clusters=10)
    brc2 = Birch(n_clusters=gc)
    brc2.fit(X)
    assert_array_equal(brc1.subcluster_labels_, brc2.subcluster_labels_)
    assert_array_equal(brc1.labels_, brc2.labels_)

    # Test that the wrong global clustering step raises an Error.
    clf = ElasticNet()
    brc3 = Birch(n_clusters=clf)
    with pytest.raises(ValueError):
        brc3.fit(X)

    # Test that a small number of clusters raises a warning.
    brc4 = Birch(threshold=10000.0)
    with pytest.warns(ConvergenceWarning):
        brc4.fit(X)


def test_sparse_X():
    # Test that sparse and dense data give same results
    X, y = make_blobs(n_samples=100, centers=10)
    brc = Birch(n_clusters=10)
    brc.fit(X)

    csr = sparse.csr_matrix(X)
    brc_sparse = Birch(n_clusters=10)
    brc_sparse.fit(csr)

    assert_array_equal(brc.labels_, brc_sparse.labels_)
    assert_array_almost_equal(brc.subcluster_centers_, brc_sparse.subcluster_centers_)


def test_partial_fit_second_call_error_checks():
    # second partial fit calls will error when n_features is not consistent
    # with the first call
    X, y = make_blobs(n_samples=100)
    brc = Birch(n_clusters=3)
    brc.partial_fit(X, y)

    msg = "X has 1 features, but Birch is expecting 2 features"
    with pytest.raises(ValueError, match=msg):
        brc.partial_fit(X[:, [0]], y)


def check_branching_factor(node, branching_factor):
    subclusters = node.subclusters_
    assert branching_factor >= len(subclusters)
    for cluster in subclusters:
        if cluster.child_:
            check_branching_factor(cluster.child_, branching_factor)


def test_branching_factor():
    # Test that nodes have at max branching_factor number of subclusters
    X, y = make_blobs()
    branching_factor = 9

    # Purposefully set a low threshold to maximize the subclusters.
    brc = Birch(n_clusters=None, branching_factor=branching_factor, threshold=0.01)
    brc.fit(X)
    check_branching_factor(brc.root_, branching_factor)
    brc = Birch(n_clusters=3, branching_factor=branching_factor, threshold=0.01)
    brc.fit(X)
    check_branching_factor(brc.root_, branching_factor)

    # Raises error when branching_factor is set to one.
    brc = Birch(n_clusters=None, branching_factor=1, threshold=0.01)
    with pytest.raises(ValueError):
        brc.fit(X)


def check_threshold(birch_instance, threshold):
    """Use the leaf linked list for traversal"""
    current_leaf = birch_instance.dummy_leaf_.next_leaf_
    while current_leaf:
        subclusters = current_leaf.subclusters_
        for sc in subclusters:
            assert threshold >= sc.radius
        current_leaf = current_leaf.next_leaf_


def test_threshold():
    # Test that the leaf subclusters have a threshold lesser than radius
    X, y = make_blobs(n_samples=80, centers=4)
    brc = Birch(threshold=0.5, n_clusters=None)
    brc.fit(X)
    check_threshold(brc, 0.5)

    brc = Birch(threshold=5.0, n_clusters=None)
    brc.fit(X)
    check_threshold(brc, 5.0)


def test_birch_n_clusters_long_int():
    # Check that birch supports n_clusters with np.int64 dtype, for instance
    # coming from np.arange. #16484
    X, _ = make_blobs(random_state=0)
    n_clusters = np.int64(5)
    Birch(n_clusters=n_clusters).fit(X)


# TODO: Remove in 1.2
@pytest.mark.parametrize("attribute", ["fit_", "partial_fit_"])
def test_birch_fit_attributes_deprecated(attribute):
    """Test that fit_ and partial_fit_ attributes are deprecated."""
    msg = f"`{attribute}` is deprecated in 1.0 and will be removed in 1.2"
    X, y = make_blobs(n_samples=10)
    brc = Birch().fit(X, y)

    with pytest.warns(FutureWarning, match=msg):
        getattr(brc, attribute)
