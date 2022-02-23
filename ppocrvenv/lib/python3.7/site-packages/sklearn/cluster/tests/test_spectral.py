"""Testing for Spectral Clustering methods"""
import re

import numpy as np
from scipy import sparse
from scipy.linalg import LinAlgError

import pytest

import pickle

from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_array_equal

from sklearn.cluster import SpectralClustering, spectral_clustering
from sklearn.cluster._spectral import discretize
from sklearn.feature_extraction import img_to_graph
from sklearn.metrics import pairwise_distances
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import kernel_metrics, rbf_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_blobs

try:
    from pyamg import smoothed_aggregation_solver  # noqa

    amg_loaded = True
except ImportError:
    amg_loaded = False


@pytest.mark.parametrize("eigen_solver", ("arpack", "lobpcg"))
@pytest.mark.parametrize("assign_labels", ("kmeans", "discretize"))
def test_spectral_clustering(eigen_solver, assign_labels):
    S = np.array(
        [
            [1.0, 1.0, 1.0, 0.2, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.2, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.2, 0.0, 0.0, 0.0],
            [0.2, 0.2, 0.2, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )

    for mat in (S, sparse.csr_matrix(S)):
        model = SpectralClustering(
            random_state=0,
            n_clusters=2,
            affinity="precomputed",
            eigen_solver=eigen_solver,
            assign_labels=assign_labels,
        ).fit(mat)
        labels = model.labels_
        if labels[0] == 0:
            labels = 1 - labels

        assert adjusted_rand_score(labels, [1, 1, 1, 0, 0, 0, 0]) == 1

        model_copy = pickle.loads(pickle.dumps(model))
        assert model_copy.n_clusters == model.n_clusters
        assert model_copy.eigen_solver == model.eigen_solver
        assert_array_equal(model_copy.labels_, model.labels_)


def test_spectral_unknown_mode():
    # Test that SpectralClustering fails with an unknown mode set.
    centers = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 10.0, 10.0],
            [20.0, 20.0, 20.0],
        ]
    )
    X, true_labels = make_blobs(
        n_samples=100, centers=centers, cluster_std=1.0, random_state=42
    )
    D = pairwise_distances(X)  # Distance matrix
    S = np.max(D) - D  # Similarity matrix
    S = sparse.coo_matrix(S)
    with pytest.raises(ValueError):
        spectral_clustering(S, n_clusters=2, random_state=0, eigen_solver="<unknown>")


def test_spectral_unknown_assign_labels():
    # Test that SpectralClustering fails with an unknown assign_labels set.
    centers = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 10.0, 10.0],
            [20.0, 20.0, 20.0],
        ]
    )
    X, true_labels = make_blobs(
        n_samples=100, centers=centers, cluster_std=1.0, random_state=42
    )
    D = pairwise_distances(X)  # Distance matrix
    S = np.max(D) - D  # Similarity matrix
    S = sparse.coo_matrix(S)
    with pytest.raises(ValueError):
        spectral_clustering(S, n_clusters=2, random_state=0, assign_labels="<unknown>")


def test_spectral_clustering_sparse():
    X, y = make_blobs(
        n_samples=20, random_state=0, centers=[[1, 1], [-1, -1]], cluster_std=0.01
    )

    S = rbf_kernel(X, gamma=1)
    S = np.maximum(S - 1e-4, 0)
    S = sparse.coo_matrix(S)

    labels = (
        SpectralClustering(random_state=0, n_clusters=2, affinity="precomputed")
        .fit(S)
        .labels_
    )
    assert adjusted_rand_score(y, labels) == 1


def test_precomputed_nearest_neighbors_filtering():
    # Test precomputed graph filtering when containing too many neighbors
    X, y = make_blobs(
        n_samples=200, random_state=0, centers=[[1, 1], [-1, -1]], cluster_std=0.01
    )

    n_neighbors = 2
    results = []
    for additional_neighbors in [0, 10]:
        nn = NearestNeighbors(n_neighbors=n_neighbors + additional_neighbors).fit(X)
        graph = nn.kneighbors_graph(X, mode="connectivity")
        labels = (
            SpectralClustering(
                random_state=0,
                n_clusters=2,
                affinity="precomputed_nearest_neighbors",
                n_neighbors=n_neighbors,
            )
            .fit(graph)
            .labels_
        )
        results.append(labels)

    assert_array_equal(results[0], results[1])


def test_affinities():
    # Note: in the following, random_state has been selected to have
    # a dataset that yields a stable eigen decomposition both when built
    # on OSX and Linux
    X, y = make_blobs(
        n_samples=20, random_state=0, centers=[[1, 1], [-1, -1]], cluster_std=0.01
    )
    # nearest neighbors affinity
    sp = SpectralClustering(n_clusters=2, affinity="nearest_neighbors", random_state=0)
    with pytest.warns(UserWarning, match="not fully connected"):
        sp.fit(X)
    assert adjusted_rand_score(y, sp.labels_) == 1

    sp = SpectralClustering(n_clusters=2, gamma=2, random_state=0)
    labels = sp.fit(X).labels_
    assert adjusted_rand_score(y, labels) == 1

    X = check_random_state(10).rand(10, 5) * 10

    kernels_available = kernel_metrics()
    for kern in kernels_available:
        # Additive chi^2 gives a negative similarity matrix which
        # doesn't make sense for spectral clustering
        if kern != "additive_chi2":
            sp = SpectralClustering(n_clusters=2, affinity=kern, random_state=0)
            labels = sp.fit(X).labels_
            assert (X.shape[0],) == labels.shape

    sp = SpectralClustering(n_clusters=2, affinity=lambda x, y: 1, random_state=0)
    labels = sp.fit(X).labels_
    assert (X.shape[0],) == labels.shape

    def histogram(x, y, **kwargs):
        # Histogram kernel implemented as a callable.
        assert kwargs == {}  # no kernel_params that we didn't ask for
        return np.minimum(x, y).sum()

    sp = SpectralClustering(n_clusters=2, affinity=histogram, random_state=0)
    labels = sp.fit(X).labels_
    assert (X.shape[0],) == labels.shape

    # raise error on unknown affinity
    sp = SpectralClustering(n_clusters=2, affinity="<unknown>")
    with pytest.raises(ValueError):
        sp.fit(X)


@pytest.mark.parametrize("n_samples", [50, 100, 150, 500])
def test_discretize(n_samples):
    # Test the discretize using a noise assignment matrix
    random_state = np.random.RandomState(seed=8)
    for n_class in range(2, 10):
        # random class labels
        y_true = random_state.randint(0, n_class + 1, n_samples)
        y_true = np.array(y_true, float)
        # noise class assignment matrix
        y_indicator = sparse.coo_matrix(
            (np.ones(n_samples), (np.arange(n_samples), y_true)),
            shape=(n_samples, n_class + 1),
        )
        y_true_noisy = y_indicator.toarray() + 0.1 * random_state.randn(
            n_samples, n_class + 1
        )
        y_pred = discretize(y_true_noisy, random_state=random_state)
        assert adjusted_rand_score(y_true, y_pred) > 0.8


# TODO: Remove when pyamg does replaces sp.rand call with np.random.rand
# https://github.com/scikit-learn/scikit-learn/issues/15913
@pytest.mark.filterwarnings(
    "ignore:scipy.rand is deprecated:DeprecationWarning:pyamg.*"
)
# TODO: Remove when pyamg removes the use of np.float
@pytest.mark.filterwarnings(
    "ignore:`np.float` is a deprecated alias:DeprecationWarning:pyamg.*"
)
# TODO: Remove when pyamg removes the use of pinv2
@pytest.mark.filterwarnings(
    "ignore:scipy.linalg.pinv2 is deprecated:DeprecationWarning:pyamg.*"
)
def test_spectral_clustering_with_arpack_amg_solvers():
    # Test that spectral_clustering is the same for arpack and amg solver
    # Based on toy example from plot_segmentation_toy.py

    # a small two coin image
    x, y = np.indices((40, 40))

    center1, center2 = (14, 12), (20, 25)
    radius1, radius2 = 8, 7

    circle1 = (x - center1[0]) ** 2 + (y - center1[1]) ** 2 < radius1 ** 2
    circle2 = (x - center2[0]) ** 2 + (y - center2[1]) ** 2 < radius2 ** 2

    circles = circle1 | circle2
    mask = circles.copy()
    img = circles.astype(float)

    graph = img_to_graph(img, mask=mask)
    graph.data = np.exp(-graph.data / graph.data.std())

    labels_arpack = spectral_clustering(
        graph, n_clusters=2, eigen_solver="arpack", random_state=0
    )

    assert len(np.unique(labels_arpack)) == 2

    if amg_loaded:
        labels_amg = spectral_clustering(
            graph, n_clusters=2, eigen_solver="amg", random_state=0
        )
        assert adjusted_rand_score(labels_arpack, labels_amg) == 1
    else:
        with pytest.raises(ValueError):
            spectral_clustering(graph, n_clusters=2, eigen_solver="amg", random_state=0)


def test_n_components():
    # Test that after adding n_components, result is different and
    # n_components = n_clusters by default
    X, y = make_blobs(
        n_samples=20, random_state=0, centers=[[1, 1], [-1, -1]], cluster_std=0.01
    )
    sp = SpectralClustering(n_clusters=2, random_state=0)
    labels = sp.fit(X).labels_
    # set n_components = n_cluster and test if result is the same
    labels_same_ncomp = (
        SpectralClustering(n_clusters=2, n_components=2, random_state=0).fit(X).labels_
    )
    # test that n_components=n_clusters by default
    assert_array_equal(labels, labels_same_ncomp)

    # test that n_components affect result
    # n_clusters=8 by default, and set n_components=2
    labels_diff_ncomp = (
        SpectralClustering(n_components=2, random_state=0).fit(X).labels_
    )
    assert not np.array_equal(labels, labels_diff_ncomp)


@pytest.mark.parametrize("assign_labels", ("kmeans", "discretize"))
def test_verbose(assign_labels, capsys):
    # Check verbose mode of KMeans for better coverage.
    X, y = make_blobs(
        n_samples=20, random_state=0, centers=[[1, 1], [-1, -1]], cluster_std=0.01
    )

    SpectralClustering(n_clusters=2, random_state=42, verbose=1).fit(X)

    captured = capsys.readouterr()

    assert re.search(r"Computing label assignment using", captured.out)

    if assign_labels == "kmeans":
        assert re.search(r"Initialization complete", captured.out)
        assert re.search(r"Iteration [0-9]+, inertia", captured.out)


# TODO: Remove in 1.1
@pytest.mark.parametrize("affinity", ["precomputed", "precomputed_nearest_neighbors"])
def test_pairwise_is_deprecated(affinity):
    sp = SpectralClustering(affinity=affinity)
    msg = r"Attribute `_pairwise` was deprecated in version 0\.24"
    with pytest.warns(FutureWarning, match=msg):
        sp._pairwise


def test_spectral_clustering_np_matrix_raises():
    """Check that spectral_clustering raises an informative error when passed
    a np.matrix. See #10993"""
    X = np.matrix([[0.0, 2.0], [2.0, 0.0]])

    msg = r"spectral_clustering does not support passing in affinity as an np\.matrix"
    with pytest.raises(TypeError, match=msg):
        spectral_clustering(X)


def test_spectral_clustering_not_infinite_loop(capsys, monkeypatch):
    """Check that discretize raises LinAlgError when svd never converges.

    Non-regression test for #21380
    """

    def new_svd(*args, **kwargs):
        raise LinAlgError()

    monkeypatch.setattr(np.linalg, "svd", new_svd)
    vectors = np.ones((10, 4))

    with pytest.raises(LinAlgError, match="SVD did not converge"):
        discretize(vectors)
