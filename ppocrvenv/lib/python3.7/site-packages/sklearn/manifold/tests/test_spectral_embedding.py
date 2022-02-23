import pytest

import numpy as np

from scipy import sparse
from scipy.sparse import csgraph
from scipy.linalg import eigh

from sklearn.manifold import SpectralEmbedding
from sklearn.manifold._spectral_embedding import _graph_is_connected
from sklearn.manifold._spectral_embedding import _graph_connected_component
from sklearn.manifold import spectral_embedding
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.utils.extmath import _deterministic_vector_sign_flip
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_array_equal


# non centered, sparse centers to check the
centers = np.array(
    [
        [0.0, 5.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 4.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 5.0, 1.0],
    ]
)
n_samples = 1000
n_clusters, n_features = centers.shape
S, true_labels = make_blobs(
    n_samples=n_samples, centers=centers, cluster_std=1.0, random_state=42
)


def _assert_equal_with_sign_flipping(A, B, tol=0.0):
    """Check array A and B are equal with possible sign flipping on
    each columns"""
    tol_squared = tol ** 2
    for A_col, B_col in zip(A.T, B.T):
        assert (
            np.max((A_col - B_col) ** 2) <= tol_squared
            or np.max((A_col + B_col) ** 2) <= tol_squared
        )


def test_sparse_graph_connected_component():
    rng = np.random.RandomState(42)
    n_samples = 300
    boundaries = [0, 42, 121, 200, n_samples]
    p = rng.permutation(n_samples)
    connections = []

    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        group = p[start:stop]
        # Connect all elements within the group at least once via an
        # arbitrary path that spans the group.
        for i in range(len(group) - 1):
            connections.append((group[i], group[i + 1]))

        # Add some more random connections within the group
        min_idx, max_idx = 0, len(group) - 1
        n_random_connections = 1000
        source = rng.randint(min_idx, max_idx, size=n_random_connections)
        target = rng.randint(min_idx, max_idx, size=n_random_connections)
        connections.extend(zip(group[source], group[target]))

    # Build a symmetric affinity matrix
    row_idx, column_idx = tuple(np.array(connections).T)
    data = rng.uniform(0.1, 42, size=len(connections))
    affinity = sparse.coo_matrix((data, (row_idx, column_idx)))
    affinity = 0.5 * (affinity + affinity.T)

    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        component_1 = _graph_connected_component(affinity, p[start])
        component_size = stop - start
        assert component_1.sum() == component_size

        # We should retrieve the same component mask by starting by both ends
        # of the group
        component_2 = _graph_connected_component(affinity, p[stop - 1])
        assert component_2.sum() == component_size
        assert_array_equal(component_1, component_2)


def test_spectral_embedding_two_components(seed=36):
    # Test spectral embedding with two components
    random_state = np.random.RandomState(seed)
    n_sample = 100
    affinity = np.zeros(shape=[n_sample * 2, n_sample * 2])
    # first component
    affinity[0:n_sample, 0:n_sample] = (
        np.abs(random_state.randn(n_sample, n_sample)) + 2
    )
    # second component
    affinity[n_sample::, n_sample::] = (
        np.abs(random_state.randn(n_sample, n_sample)) + 2
    )

    # Test of internal _graph_connected_component before connection
    component = _graph_connected_component(affinity, 0)
    assert component[:n_sample].all()
    assert not component[n_sample:].any()
    component = _graph_connected_component(affinity, -1)
    assert not component[:n_sample].any()
    assert component[n_sample:].all()

    # connection
    affinity[0, n_sample + 1] = 1
    affinity[n_sample + 1, 0] = 1
    affinity.flat[:: 2 * n_sample + 1] = 0
    affinity = 0.5 * (affinity + affinity.T)

    true_label = np.zeros(shape=2 * n_sample)
    true_label[0:n_sample] = 1

    se_precomp = SpectralEmbedding(
        n_components=1, affinity="precomputed", random_state=np.random.RandomState(seed)
    )
    embedded_coordinate = se_precomp.fit_transform(affinity)
    # Some numpy versions are touchy with types
    embedded_coordinate = se_precomp.fit_transform(affinity.astype(np.float32))
    # thresholding on the first components using 0.
    label_ = np.array(embedded_coordinate.ravel() < 0, dtype="float")
    assert normalized_mutual_info_score(true_label, label_) == pytest.approx(1.0)


@pytest.mark.parametrize("X", [S, sparse.csr_matrix(S)], ids=["dense", "sparse"])
def test_spectral_embedding_precomputed_affinity(X, seed=36):
    # Test spectral embedding with precomputed kernel
    gamma = 1.0
    se_precomp = SpectralEmbedding(
        n_components=2, affinity="precomputed", random_state=np.random.RandomState(seed)
    )
    se_rbf = SpectralEmbedding(
        n_components=2,
        affinity="rbf",
        gamma=gamma,
        random_state=np.random.RandomState(seed),
    )
    embed_precomp = se_precomp.fit_transform(rbf_kernel(X, gamma=gamma))
    embed_rbf = se_rbf.fit_transform(X)
    assert_array_almost_equal(se_precomp.affinity_matrix_, se_rbf.affinity_matrix_)
    _assert_equal_with_sign_flipping(embed_precomp, embed_rbf, 0.05)


def test_precomputed_nearest_neighbors_filtering():
    # Test precomputed graph filtering when containing too many neighbors
    n_neighbors = 2
    results = []
    for additional_neighbors in [0, 10]:
        nn = NearestNeighbors(n_neighbors=n_neighbors + additional_neighbors).fit(S)
        graph = nn.kneighbors_graph(S, mode="connectivity")
        embedding = (
            SpectralEmbedding(
                random_state=0,
                n_components=2,
                affinity="precomputed_nearest_neighbors",
                n_neighbors=n_neighbors,
            )
            .fit(graph)
            .embedding_
        )
        results.append(embedding)

    assert_array_equal(results[0], results[1])


@pytest.mark.parametrize("X", [S, sparse.csr_matrix(S)], ids=["dense", "sparse"])
def test_spectral_embedding_callable_affinity(X, seed=36):
    # Test spectral embedding with callable affinity
    gamma = 0.9
    kern = rbf_kernel(S, gamma=gamma)
    se_callable = SpectralEmbedding(
        n_components=2,
        affinity=(lambda x: rbf_kernel(x, gamma=gamma)),
        gamma=gamma,
        random_state=np.random.RandomState(seed),
    )
    se_rbf = SpectralEmbedding(
        n_components=2,
        affinity="rbf",
        gamma=gamma,
        random_state=np.random.RandomState(seed),
    )
    embed_rbf = se_rbf.fit_transform(X)
    embed_callable = se_callable.fit_transform(X)
    assert_array_almost_equal(se_callable.affinity_matrix_, se_rbf.affinity_matrix_)
    assert_array_almost_equal(kern, se_rbf.affinity_matrix_)
    _assert_equal_with_sign_flipping(embed_rbf, embed_callable, 0.05)


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
def test_spectral_embedding_amg_solver(seed=36):
    # Test spectral embedding with amg solver
    pytest.importorskip("pyamg")

    se_amg = SpectralEmbedding(
        n_components=2,
        affinity="nearest_neighbors",
        eigen_solver="amg",
        n_neighbors=5,
        random_state=np.random.RandomState(seed),
    )
    se_arpack = SpectralEmbedding(
        n_components=2,
        affinity="nearest_neighbors",
        eigen_solver="arpack",
        n_neighbors=5,
        random_state=np.random.RandomState(seed),
    )
    embed_amg = se_amg.fit_transform(S)
    embed_arpack = se_arpack.fit_transform(S)
    _assert_equal_with_sign_flipping(embed_amg, embed_arpack, 1e-5)

    # same with special case in which amg is not actually used
    # regression test for #10715
    # affinity between nodes
    row = [0, 0, 1, 2, 3, 3, 4]
    col = [1, 2, 2, 3, 4, 5, 5]
    val = [100, 100, 100, 1, 100, 100, 100]

    affinity = sparse.coo_matrix(
        (val + val, (row + col, col + row)), shape=(6, 6)
    ).toarray()
    se_amg.affinity = "precomputed"
    se_arpack.affinity = "precomputed"
    embed_amg = se_amg.fit_transform(affinity)
    embed_arpack = se_arpack.fit_transform(affinity)
    _assert_equal_with_sign_flipping(embed_amg, embed_arpack, 1e-5)


# TODO: Remove filterwarnings when pyamg does replaces sp.rand call with
# np.random.rand:
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
def test_spectral_embedding_amg_solver_failure():
    # Non-regression test for amg solver failure (issue #13393 on github)
    pytest.importorskip("pyamg")
    seed = 36
    num_nodes = 100
    X = sparse.rand(num_nodes, num_nodes, density=0.1, random_state=seed)
    upper = sparse.triu(X) - sparse.diags(X.diagonal())
    sym_matrix = upper + upper.T
    embedding = spectral_embedding(
        sym_matrix, n_components=10, eigen_solver="amg", random_state=0
    )

    # Check that the learned embedding is stable w.r.t. random solver init:
    for i in range(3):
        new_embedding = spectral_embedding(
            sym_matrix, n_components=10, eigen_solver="amg", random_state=i + 1
        )
        _assert_equal_with_sign_flipping(embedding, new_embedding, tol=0.05)


@pytest.mark.filterwarnings("ignore:the behavior of nmi will change in version 0.22")
def test_pipeline_spectral_clustering(seed=36):
    # Test using pipeline to do spectral clustering
    random_state = np.random.RandomState(seed)
    se_rbf = SpectralEmbedding(
        n_components=n_clusters, affinity="rbf", random_state=random_state
    )
    se_knn = SpectralEmbedding(
        n_components=n_clusters,
        affinity="nearest_neighbors",
        n_neighbors=5,
        random_state=random_state,
    )
    for se in [se_rbf, se_knn]:
        km = KMeans(n_clusters=n_clusters, random_state=random_state)
        km.fit(se.fit_transform(S))
        assert_array_almost_equal(
            normalized_mutual_info_score(km.labels_, true_labels), 1.0, 2
        )


def test_spectral_embedding_unknown_eigensolver(seed=36):
    # Test that SpectralClustering fails with an unknown eigensolver
    se = SpectralEmbedding(
        n_components=1,
        affinity="precomputed",
        random_state=np.random.RandomState(seed),
        eigen_solver="<unknown>",
    )
    with pytest.raises(ValueError):
        se.fit(S)


def test_spectral_embedding_unknown_affinity(seed=36):
    # Test that SpectralClustering fails with an unknown affinity type
    se = SpectralEmbedding(
        n_components=1, affinity="<unknown>", random_state=np.random.RandomState(seed)
    )
    with pytest.raises(ValueError):
        se.fit(S)


def test_connectivity(seed=36):
    # Test that graph connectivity test works as expected
    graph = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1],
        ]
    )
    assert not _graph_is_connected(graph)
    assert not _graph_is_connected(sparse.csr_matrix(graph))
    assert not _graph_is_connected(sparse.csc_matrix(graph))
    graph = np.array(
        [
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1],
        ]
    )
    assert _graph_is_connected(graph)
    assert _graph_is_connected(sparse.csr_matrix(graph))
    assert _graph_is_connected(sparse.csc_matrix(graph))


def test_spectral_embedding_deterministic():
    # Test that Spectral Embedding is deterministic
    random_state = np.random.RandomState(36)
    data = random_state.randn(10, 30)
    sims = rbf_kernel(data)
    embedding_1 = spectral_embedding(sims)
    embedding_2 = spectral_embedding(sims)
    assert_array_almost_equal(embedding_1, embedding_2)


def test_spectral_embedding_unnormalized():
    # Test that spectral_embedding is also processing unnormalized laplacian
    # correctly
    random_state = np.random.RandomState(36)
    data = random_state.randn(10, 30)
    sims = rbf_kernel(data)
    n_components = 8
    embedding_1 = spectral_embedding(
        sims, norm_laplacian=False, n_components=n_components, drop_first=False
    )

    # Verify using manual computation with dense eigh
    laplacian, dd = csgraph.laplacian(sims, normed=False, return_diag=True)
    _, diffusion_map = eigh(laplacian)
    embedding_2 = diffusion_map.T[:n_components]
    embedding_2 = _deterministic_vector_sign_flip(embedding_2).T

    assert_array_almost_equal(embedding_1, embedding_2)


def test_spectral_embedding_first_eigen_vector():
    # Test that the first eigenvector of spectral_embedding
    # is constant and that the second is not (for a connected graph)
    random_state = np.random.RandomState(36)
    data = random_state.randn(10, 30)
    sims = rbf_kernel(data)
    n_components = 2

    for seed in range(10):
        embedding = spectral_embedding(
            sims,
            norm_laplacian=False,
            n_components=n_components,
            drop_first=False,
            random_state=seed,
        )

        assert np.std(embedding[:, 0]) == pytest.approx(0)
        assert np.std(embedding[:, 1]) > 1e-3


# TODO: Remove in 1.1
@pytest.mark.parametrize("affinity", ["precomputed", "precomputed_nearest_neighbors"])
def test_spectral_embedding_pairwise_deprecated(affinity):
    se = SpectralEmbedding(affinity=affinity)
    msg = r"Attribute `_pairwise` was deprecated in version 0\.24"
    with pytest.warns(FutureWarning, match=msg):
        se._pairwise
