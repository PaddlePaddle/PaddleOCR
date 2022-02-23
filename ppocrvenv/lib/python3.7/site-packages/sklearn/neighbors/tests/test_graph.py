import numpy as np

from sklearn.metrics import euclidean_distances
from sklearn.neighbors import KNeighborsTransformer, RadiusNeighborsTransformer
from sklearn.neighbors._base import _is_sorted_by_data


def test_transformer_result():
    # Test the number of neighbors returned
    n_neighbors = 5
    n_samples_fit = 20
    n_queries = 18
    n_features = 10

    rng = np.random.RandomState(42)
    X = rng.randn(n_samples_fit, n_features)
    X2 = rng.randn(n_queries, n_features)
    radius = np.percentile(euclidean_distances(X), 10)

    # with n_neighbors
    for mode in ["distance", "connectivity"]:
        add_one = mode == "distance"
        nnt = KNeighborsTransformer(n_neighbors=n_neighbors, mode=mode)
        Xt = nnt.fit_transform(X)
        assert Xt.shape == (n_samples_fit, n_samples_fit)
        assert Xt.data.shape == (n_samples_fit * (n_neighbors + add_one),)
        assert Xt.format == "csr"
        assert _is_sorted_by_data(Xt)

        X2t = nnt.transform(X2)
        assert X2t.shape == (n_queries, n_samples_fit)
        assert X2t.data.shape == (n_queries * (n_neighbors + add_one),)
        assert X2t.format == "csr"
        assert _is_sorted_by_data(X2t)

    # with radius
    for mode in ["distance", "connectivity"]:
        add_one = mode == "distance"
        nnt = RadiusNeighborsTransformer(radius=radius, mode=mode)
        Xt = nnt.fit_transform(X)
        assert Xt.shape == (n_samples_fit, n_samples_fit)
        assert not Xt.data.shape == (n_samples_fit * (n_neighbors + add_one),)
        assert Xt.format == "csr"
        assert _is_sorted_by_data(Xt)

        X2t = nnt.transform(X2)
        assert X2t.shape == (n_queries, n_samples_fit)
        assert not X2t.data.shape == (n_queries * (n_neighbors + add_one),)
        assert X2t.format == "csr"
        assert _is_sorted_by_data(X2t)


def _has_explicit_diagonal(X):
    """Return True if the diagonal is explicitly stored"""
    X = X.tocoo()
    explicit = X.row[X.row == X.col]
    return len(explicit) == X.shape[0]


def test_explicit_diagonal():
    # Test that the diagonal is explicitly stored in the sparse graph
    n_neighbors = 5
    n_samples_fit, n_samples_transform, n_features = 20, 18, 10
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples_fit, n_features)
    X2 = rng.randn(n_samples_transform, n_features)

    nnt = KNeighborsTransformer(n_neighbors=n_neighbors)
    Xt = nnt.fit_transform(X)
    assert _has_explicit_diagonal(Xt)
    assert np.all(Xt.data.reshape(n_samples_fit, n_neighbors + 1)[:, 0] == 0)

    Xt = nnt.transform(X)
    assert _has_explicit_diagonal(Xt)
    assert np.all(Xt.data.reshape(n_samples_fit, n_neighbors + 1)[:, 0] == 0)

    # Using transform on new data should not always have zero diagonal
    X2t = nnt.transform(X2)
    assert not _has_explicit_diagonal(X2t)
