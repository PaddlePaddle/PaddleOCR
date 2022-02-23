from itertools import product
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
import pytest

from sklearn import datasets
from sklearn import manifold
from sklearn import neighbors
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.metrics.pairwise import pairwise_distances

from scipy.sparse import rand as sparse_rand

eigen_solvers = ["auto", "dense", "arpack"]
path_methods = ["auto", "FW", "D"]


def test_isomap_simple_grid():
    # Isomap should preserve distances when all neighbors are used
    N_per_side = 5
    Npts = N_per_side ** 2
    n_neighbors = Npts - 1

    # grid of equidistant points in 2D, n_components = n_dim
    X = np.array(list(product(range(N_per_side), repeat=2)))

    # distances from each point to all others
    G = neighbors.kneighbors_graph(X, n_neighbors, mode="distance").toarray()

    for eigen_solver in eigen_solvers:
        for path_method in path_methods:
            clf = manifold.Isomap(
                n_neighbors=n_neighbors,
                n_components=2,
                eigen_solver=eigen_solver,
                path_method=path_method,
            )
            clf.fit(X)

            G_iso = neighbors.kneighbors_graph(
                clf.embedding_, n_neighbors, mode="distance"
            ).toarray()
            assert_array_almost_equal(G, G_iso)


def test_isomap_reconstruction_error():
    # Same setup as in test_isomap_simple_grid, with an added dimension
    N_per_side = 5
    Npts = N_per_side ** 2
    n_neighbors = Npts - 1

    # grid of equidistant points in 2D, n_components = n_dim
    X = np.array(list(product(range(N_per_side), repeat=2)))

    # add noise in a third dimension
    rng = np.random.RandomState(0)
    noise = 0.1 * rng.randn(Npts, 1)
    X = np.concatenate((X, noise), 1)

    # compute input kernel
    G = neighbors.kneighbors_graph(X, n_neighbors, mode="distance").toarray()

    centerer = preprocessing.KernelCenterer()
    K = centerer.fit_transform(-0.5 * G ** 2)

    for eigen_solver in eigen_solvers:
        for path_method in path_methods:
            clf = manifold.Isomap(
                n_neighbors=n_neighbors,
                n_components=2,
                eigen_solver=eigen_solver,
                path_method=path_method,
            )
            clf.fit(X)

            # compute output kernel
            G_iso = neighbors.kneighbors_graph(
                clf.embedding_, n_neighbors, mode="distance"
            ).toarray()

            K_iso = centerer.fit_transform(-0.5 * G_iso ** 2)

            # make sure error agrees
            reconstruction_error = np.linalg.norm(K - K_iso) / Npts
            assert_almost_equal(reconstruction_error, clf.reconstruction_error())


def test_transform():
    n_samples = 200
    n_components = 10
    noise_scale = 0.01

    # Create S-curve dataset
    X, y = datasets.make_s_curve(n_samples, random_state=0)

    # Compute isomap embedding
    iso = manifold.Isomap(n_components=n_components)
    X_iso = iso.fit_transform(X)

    # Re-embed a noisy version of the points
    rng = np.random.RandomState(0)
    noise = noise_scale * rng.randn(*X.shape)
    X_iso2 = iso.transform(X + noise)

    # Make sure the rms error on re-embedding is comparable to noise_scale
    assert np.sqrt(np.mean((X_iso - X_iso2) ** 2)) < 2 * noise_scale


def test_pipeline():
    # check that Isomap works fine as a transformer in a Pipeline
    # only checks that no error is raised.
    # TODO check that it actually does something useful
    X, y = datasets.make_blobs(random_state=0)
    clf = pipeline.Pipeline(
        [("isomap", manifold.Isomap()), ("clf", neighbors.KNeighborsClassifier())]
    )
    clf.fit(X, y)
    assert 0.9 < clf.score(X, y)


def test_pipeline_with_nearest_neighbors_transformer():
    # Test chaining NearestNeighborsTransformer and Isomap with
    # neighbors_algorithm='precomputed'
    algorithm = "auto"
    n_neighbors = 10

    X, _ = datasets.make_blobs(random_state=0)
    X2, _ = datasets.make_blobs(random_state=1)

    # compare the chained version and the compact version
    est_chain = pipeline.make_pipeline(
        neighbors.KNeighborsTransformer(
            n_neighbors=n_neighbors, algorithm=algorithm, mode="distance"
        ),
        manifold.Isomap(n_neighbors=n_neighbors, metric="precomputed"),
    )
    est_compact = manifold.Isomap(
        n_neighbors=n_neighbors, neighbors_algorithm=algorithm
    )

    Xt_chain = est_chain.fit_transform(X)
    Xt_compact = est_compact.fit_transform(X)
    assert_array_almost_equal(Xt_chain, Xt_compact)

    Xt_chain = est_chain.transform(X2)
    Xt_compact = est_compact.transform(X2)
    assert_array_almost_equal(Xt_chain, Xt_compact)


def test_different_metric():
    # Test that the metric parameters work correctly, and default to euclidean
    def custom_metric(x1, x2):
        return np.sqrt(np.sum(x1 ** 2 + x2 ** 2))

    # metric, p, is_euclidean
    metrics = [
        ("euclidean", 2, True),
        ("manhattan", 1, False),
        ("minkowski", 1, False),
        ("minkowski", 2, True),
        (custom_metric, 2, False),
    ]

    X, _ = datasets.make_blobs(random_state=0)
    reference = manifold.Isomap().fit_transform(X)

    for metric, p, is_euclidean in metrics:
        embedding = manifold.Isomap(metric=metric, p=p).fit_transform(X)

        if is_euclidean:
            assert_array_almost_equal(embedding, reference)
        else:
            with pytest.raises(AssertionError, match="not almost equal"):
                assert_array_almost_equal(embedding, reference)


def test_isomap_clone_bug():
    # regression test for bug reported in #6062
    model = manifold.Isomap()
    for n_neighbors in [10, 15, 20]:
        model.set_params(n_neighbors=n_neighbors)
        model.fit(np.random.rand(50, 2))
        assert model.nbrs_.n_neighbors == n_neighbors


def test_sparse_input():
    X = sparse_rand(100, 3, density=0.1, format="csr")

    # Should not error
    for eigen_solver in eigen_solvers:
        for path_method in path_methods:
            clf = manifold.Isomap(
                n_components=2,
                eigen_solver=eigen_solver,
                path_method=path_method,
                n_neighbors=8,
            )
            clf.fit(X)


def test_multiple_connected_components():
    # Test that a warning is raised when the graph has multiple components
    X = np.array([0, 1, 2, 5, 6, 7])[:, None]
    with pytest.warns(UserWarning, match="number of connected components"):
        manifold.Isomap(n_neighbors=2).fit(X)


def test_multiple_connected_components_metric_precomputed():
    # Test that an error is raised when the graph has multiple components
    # and when X is a precomputed neighbors graph.
    X = np.array([0, 1, 2, 5, 6, 7])[:, None]

    # works with a precomputed distance matrix (dense)
    X_distances = pairwise_distances(X)
    with pytest.warns(UserWarning, match="number of connected components"):
        manifold.Isomap(n_neighbors=1, metric="precomputed").fit(X_distances)

    # does not work with a precomputed neighbors graph (sparse)
    X_graph = neighbors.kneighbors_graph(X, n_neighbors=2, mode="distance")
    with pytest.raises(RuntimeError, match="number of connected components"):
        manifold.Isomap(n_neighbors=1, metric="precomputed").fit(X_graph)
