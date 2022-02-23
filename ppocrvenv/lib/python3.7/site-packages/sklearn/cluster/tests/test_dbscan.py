"""
Tests for DBSCAN clustering algorithm
"""

import pickle

import numpy as np

import warnings

from scipy.spatial import distance
from scipy import sparse

import pytest

from sklearn.utils._testing import assert_array_equal
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.cluster import dbscan
from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.metrics.pairwise import pairwise_distances


n_clusters = 3
X = generate_clustered_data(n_clusters=n_clusters)


def test_dbscan_similarity():
    # Tests the DBSCAN algorithm with a similarity array.
    # Parameters chosen specifically for this task.
    eps = 0.15
    min_samples = 10
    # Compute similarities
    D = distance.squareform(distance.pdist(X))
    D /= np.max(D)
    # Compute DBSCAN
    core_samples, labels = dbscan(
        D, metric="precomputed", eps=eps, min_samples=min_samples
    )
    # number of clusters, ignoring noise if present
    n_clusters_1 = len(set(labels)) - (1 if -1 in labels else 0)

    assert n_clusters_1 == n_clusters

    db = DBSCAN(metric="precomputed", eps=eps, min_samples=min_samples)
    labels = db.fit(D).labels_

    n_clusters_2 = len(set(labels)) - int(-1 in labels)
    assert n_clusters_2 == n_clusters


def test_dbscan_feature():
    # Tests the DBSCAN algorithm with a feature vector array.
    # Parameters chosen specifically for this task.
    # Different eps to other test, because distance is not normalised.
    eps = 0.8
    min_samples = 10
    metric = "euclidean"
    # Compute DBSCAN
    # parameters chosen for task
    core_samples, labels = dbscan(X, metric=metric, eps=eps, min_samples=min_samples)

    # number of clusters, ignoring noise if present
    n_clusters_1 = len(set(labels)) - int(-1 in labels)
    assert n_clusters_1 == n_clusters

    db = DBSCAN(metric=metric, eps=eps, min_samples=min_samples)
    labels = db.fit(X).labels_

    n_clusters_2 = len(set(labels)) - int(-1 in labels)
    assert n_clusters_2 == n_clusters


def test_dbscan_sparse():
    core_sparse, labels_sparse = dbscan(sparse.lil_matrix(X), eps=0.8, min_samples=10)
    core_dense, labels_dense = dbscan(X, eps=0.8, min_samples=10)
    assert_array_equal(core_dense, core_sparse)
    assert_array_equal(labels_dense, labels_sparse)


@pytest.mark.parametrize("include_self", [False, True])
def test_dbscan_sparse_precomputed(include_self):
    D = pairwise_distances(X)
    nn = NearestNeighbors(radius=0.9).fit(X)
    X_ = X if include_self else None
    D_sparse = nn.radius_neighbors_graph(X=X_, mode="distance")
    # Ensure it is sparse not merely on diagonals:
    assert D_sparse.nnz < D.shape[0] * (D.shape[0] - 1)
    core_sparse, labels_sparse = dbscan(
        D_sparse, eps=0.8, min_samples=10, metric="precomputed"
    )
    core_dense, labels_dense = dbscan(D, eps=0.8, min_samples=10, metric="precomputed")
    assert_array_equal(core_dense, core_sparse)
    assert_array_equal(labels_dense, labels_sparse)


def test_dbscan_sparse_precomputed_different_eps():
    # test that precomputed neighbors graph is filtered if computed with
    # a radius larger than DBSCAN's eps.
    lower_eps = 0.2
    nn = NearestNeighbors(radius=lower_eps).fit(X)
    D_sparse = nn.radius_neighbors_graph(X, mode="distance")
    dbscan_lower = dbscan(D_sparse, eps=lower_eps, metric="precomputed")

    higher_eps = lower_eps + 0.7
    nn = NearestNeighbors(radius=higher_eps).fit(X)
    D_sparse = nn.radius_neighbors_graph(X, mode="distance")
    dbscan_higher = dbscan(D_sparse, eps=lower_eps, metric="precomputed")

    assert_array_equal(dbscan_lower[0], dbscan_higher[0])
    assert_array_equal(dbscan_lower[1], dbscan_higher[1])


@pytest.mark.parametrize("use_sparse", [True, False])
@pytest.mark.parametrize("metric", ["precomputed", "minkowski"])
def test_dbscan_input_not_modified(use_sparse, metric):
    # test that the input is not modified by dbscan
    X = np.random.RandomState(0).rand(10, 10)
    X = sparse.csr_matrix(X) if use_sparse else X
    X_copy = X.copy()
    dbscan(X, metric=metric)

    if use_sparse:
        assert_array_equal(X.toarray(), X_copy.toarray())
    else:
        assert_array_equal(X, X_copy)


def test_dbscan_no_core_samples():
    rng = np.random.RandomState(0)
    X = rng.rand(40, 10)
    X[X < 0.8] = 0

    for X_ in [X, sparse.csr_matrix(X)]:
        db = DBSCAN(min_samples=6).fit(X_)
        assert_array_equal(db.components_, np.empty((0, X_.shape[1])))
        assert_array_equal(db.labels_, -1)
        assert db.core_sample_indices_.shape == (0,)


def test_dbscan_callable():
    # Tests the DBSCAN algorithm with a callable metric.
    # Parameters chosen specifically for this task.
    # Different eps to other test, because distance is not normalised.
    eps = 0.8
    min_samples = 10
    # metric is the function reference, not the string key.
    metric = distance.euclidean
    # Compute DBSCAN
    # parameters chosen for task
    core_samples, labels = dbscan(
        X, metric=metric, eps=eps, min_samples=min_samples, algorithm="ball_tree"
    )

    # number of clusters, ignoring noise if present
    n_clusters_1 = len(set(labels)) - int(-1 in labels)
    assert n_clusters_1 == n_clusters

    db = DBSCAN(metric=metric, eps=eps, min_samples=min_samples, algorithm="ball_tree")
    labels = db.fit(X).labels_

    n_clusters_2 = len(set(labels)) - int(-1 in labels)
    assert n_clusters_2 == n_clusters


def test_dbscan_metric_params():
    # Tests that DBSCAN works with the metrics_params argument.
    eps = 0.8
    min_samples = 10
    p = 1

    # Compute DBSCAN with metric_params arg

    with warnings.catch_warnings(record=True) as warns:
        db = DBSCAN(
            metric="minkowski",
            metric_params={"p": p},
            eps=eps,
            p=None,
            min_samples=min_samples,
            algorithm="ball_tree",
        ).fit(X)
    assert not warns, warns[0].message
    core_sample_1, labels_1 = db.core_sample_indices_, db.labels_

    # Test that sample labels are the same as passing Minkowski 'p' directly
    db = DBSCAN(
        metric="minkowski", eps=eps, min_samples=min_samples, algorithm="ball_tree", p=p
    ).fit(X)
    core_sample_2, labels_2 = db.core_sample_indices_, db.labels_

    assert_array_equal(core_sample_1, core_sample_2)
    assert_array_equal(labels_1, labels_2)

    # Minkowski with p=1 should be equivalent to Manhattan distance
    db = DBSCAN(
        metric="manhattan", eps=eps, min_samples=min_samples, algorithm="ball_tree"
    ).fit(X)
    core_sample_3, labels_3 = db.core_sample_indices_, db.labels_

    assert_array_equal(core_sample_1, core_sample_3)
    assert_array_equal(labels_1, labels_3)

    with pytest.warns(
        SyntaxWarning,
        match=(
            "Parameter p is found in metric_params. "
            "The corresponding parameter from __init__ "
            "is ignored."
        ),
    ):
        # Test that checks p is ignored in favor of metric_params={'p': <val>}
        db = DBSCAN(
            metric="minkowski",
            metric_params={"p": p},
            eps=eps,
            p=p + 1,
            min_samples=min_samples,
            algorithm="ball_tree",
        ).fit(X)
        core_sample_4, labels_4 = db.core_sample_indices_, db.labels_

    assert_array_equal(core_sample_1, core_sample_4)
    assert_array_equal(labels_1, labels_4)


def test_dbscan_balltree():
    # Tests the DBSCAN algorithm with balltree for neighbor calculation.
    eps = 0.8
    min_samples = 10

    D = pairwise_distances(X)
    core_samples, labels = dbscan(
        D, metric="precomputed", eps=eps, min_samples=min_samples
    )

    # number of clusters, ignoring noise if present
    n_clusters_1 = len(set(labels)) - int(-1 in labels)
    assert n_clusters_1 == n_clusters

    db = DBSCAN(p=2.0, eps=eps, min_samples=min_samples, algorithm="ball_tree")
    labels = db.fit(X).labels_

    n_clusters_2 = len(set(labels)) - int(-1 in labels)
    assert n_clusters_2 == n_clusters

    db = DBSCAN(p=2.0, eps=eps, min_samples=min_samples, algorithm="kd_tree")
    labels = db.fit(X).labels_

    n_clusters_3 = len(set(labels)) - int(-1 in labels)
    assert n_clusters_3 == n_clusters

    db = DBSCAN(p=1.0, eps=eps, min_samples=min_samples, algorithm="ball_tree")
    labels = db.fit(X).labels_

    n_clusters_4 = len(set(labels)) - int(-1 in labels)
    assert n_clusters_4 == n_clusters

    db = DBSCAN(leaf_size=20, eps=eps, min_samples=min_samples, algorithm="ball_tree")
    labels = db.fit(X).labels_

    n_clusters_5 = len(set(labels)) - int(-1 in labels)
    assert n_clusters_5 == n_clusters


def test_input_validation():
    # DBSCAN.fit should accept a list of lists.
    X = [[1.0, 2.0], [3.0, 4.0]]
    DBSCAN().fit(X)  # must not raise exception


@pytest.mark.parametrize(
    "args",
    [
        {"eps": -1.0},
        {"algorithm": "blah"},
        {"metric": "blah"},
        {"leaf_size": -1},
        {"p": -1},
    ],
)
def test_dbscan_badargs(args):
    # Test bad argument values: these should all raise ValueErrors
    with pytest.raises(ValueError):
        dbscan(X, **args)


def test_pickle():
    obj = DBSCAN()
    s = pickle.dumps(obj)
    assert type(pickle.loads(s)) == obj.__class__


def test_boundaries():
    # ensure min_samples is inclusive of core point
    core, _ = dbscan([[0], [1]], eps=2, min_samples=2)
    assert 0 in core
    # ensure eps is inclusive of circumference
    core, _ = dbscan([[0], [1], [1]], eps=1, min_samples=2)
    assert 0 in core
    core, _ = dbscan([[0], [1], [1]], eps=0.99, min_samples=2)
    assert 0 not in core


def test_weighted_dbscan():
    # ensure sample_weight is validated
    with pytest.raises(ValueError):
        dbscan([[0], [1]], sample_weight=[2])
    with pytest.raises(ValueError):
        dbscan([[0], [1]], sample_weight=[2, 3, 4])

    # ensure sample_weight has an effect
    assert_array_equal([], dbscan([[0], [1]], sample_weight=None, min_samples=6)[0])
    assert_array_equal([], dbscan([[0], [1]], sample_weight=[5, 5], min_samples=6)[0])
    assert_array_equal([0], dbscan([[0], [1]], sample_weight=[6, 5], min_samples=6)[0])
    assert_array_equal(
        [0, 1], dbscan([[0], [1]], sample_weight=[6, 6], min_samples=6)[0]
    )

    # points within eps of each other:
    assert_array_equal(
        [0, 1], dbscan([[0], [1]], eps=1.5, sample_weight=[5, 1], min_samples=6)[0]
    )
    # and effect of non-positive and non-integer sample_weight:
    assert_array_equal(
        [], dbscan([[0], [1]], sample_weight=[5, 0], eps=1.5, min_samples=6)[0]
    )
    assert_array_equal(
        [0, 1], dbscan([[0], [1]], sample_weight=[5.9, 0.1], eps=1.5, min_samples=6)[0]
    )
    assert_array_equal(
        [0, 1], dbscan([[0], [1]], sample_weight=[6, 0], eps=1.5, min_samples=6)[0]
    )
    assert_array_equal(
        [], dbscan([[0], [1]], sample_weight=[6, -1], eps=1.5, min_samples=6)[0]
    )

    # for non-negative sample_weight, cores should be identical to repetition
    rng = np.random.RandomState(42)
    sample_weight = rng.randint(0, 5, X.shape[0])
    core1, label1 = dbscan(X, sample_weight=sample_weight)
    assert len(label1) == len(X)

    X_repeated = np.repeat(X, sample_weight, axis=0)
    core_repeated, label_repeated = dbscan(X_repeated)
    core_repeated_mask = np.zeros(X_repeated.shape[0], dtype=bool)
    core_repeated_mask[core_repeated] = True
    core_mask = np.zeros(X.shape[0], dtype=bool)
    core_mask[core1] = True
    assert_array_equal(np.repeat(core_mask, sample_weight), core_repeated_mask)

    # sample_weight should work with precomputed distance matrix
    D = pairwise_distances(X)
    core3, label3 = dbscan(D, sample_weight=sample_weight, metric="precomputed")
    assert_array_equal(core1, core3)
    assert_array_equal(label1, label3)

    # sample_weight should work with estimator
    est = DBSCAN().fit(X, sample_weight=sample_weight)
    core4 = est.core_sample_indices_
    label4 = est.labels_
    assert_array_equal(core1, core4)
    assert_array_equal(label1, label4)

    est = DBSCAN()
    label5 = est.fit_predict(X, sample_weight=sample_weight)
    core5 = est.core_sample_indices_
    assert_array_equal(core1, core5)
    assert_array_equal(label1, label5)
    assert_array_equal(label1, est.labels_)


@pytest.mark.parametrize("algorithm", ["brute", "kd_tree", "ball_tree"])
def test_dbscan_core_samples_toy(algorithm):
    X = [[0], [2], [3], [4], [6], [8], [10]]
    n_samples = len(X)

    # Degenerate case: every sample is a core sample, either with its own
    # cluster or including other close core samples.
    core_samples, labels = dbscan(X, algorithm=algorithm, eps=1, min_samples=1)
    assert_array_equal(core_samples, np.arange(n_samples))
    assert_array_equal(labels, [0, 1, 1, 1, 2, 3, 4])

    # With eps=1 and min_samples=2 only the 3 samples from the denser area
    # are core samples. All other points are isolated and considered noise.
    core_samples, labels = dbscan(X, algorithm=algorithm, eps=1, min_samples=2)
    assert_array_equal(core_samples, [1, 2, 3])
    assert_array_equal(labels, [-1, 0, 0, 0, -1, -1, -1])

    # Only the sample in the middle of the dense area is core. Its two
    # neighbors are edge samples. Remaining samples are noise.
    core_samples, labels = dbscan(X, algorithm=algorithm, eps=1, min_samples=3)
    assert_array_equal(core_samples, [2])
    assert_array_equal(labels, [-1, 0, 0, 0, -1, -1, -1])

    # It's no longer possible to extract core samples with eps=1:
    # everything is noise.
    core_samples, labels = dbscan(X, algorithm=algorithm, eps=1, min_samples=4)
    assert_array_equal(core_samples, [])
    assert_array_equal(labels, np.full(n_samples, -1.0))


def test_dbscan_precomputed_metric_with_degenerate_input_arrays():
    # see https://github.com/scikit-learn/scikit-learn/issues/4641 for
    # more details
    X = np.eye(10)
    labels = DBSCAN(eps=0.5, metric="precomputed").fit(X).labels_
    assert len(set(labels)) == 1

    X = np.zeros((10, 10))
    labels = DBSCAN(eps=0.5, metric="precomputed").fit(X).labels_
    assert len(set(labels)) == 1


def test_dbscan_precomputed_metric_with_initial_rows_zero():
    # sample matrix with initial two row all zero
    ar = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0],
            [0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.3],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
            [0.0, 0.0, 0.0, 0.0, 0.3, 0.1, 0.0],
        ]
    )
    matrix = sparse.csr_matrix(ar)
    labels = DBSCAN(eps=0.2, metric="precomputed", min_samples=2).fit(matrix).labels_
    assert_array_equal(labels, [-1, -1, 0, 0, 0, 1, 1])
