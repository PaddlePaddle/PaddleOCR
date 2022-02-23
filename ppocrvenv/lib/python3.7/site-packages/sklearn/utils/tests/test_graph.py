import pytest
import numpy as np
from scipy.sparse.csgraph import connected_components

from sklearn.neighbors import kneighbors_graph
from sklearn.utils.graph import _fix_connected_components
from sklearn.metrics.pairwise import pairwise_distances


def test_fix_connected_components():
    # Test that _fix_connected_components reduces the number of component to 1.
    X = np.array([0, 1, 2, 5, 6, 7])[:, None]
    graph = kneighbors_graph(X, n_neighbors=2, mode="distance")

    n_connected_components, labels = connected_components(graph)
    assert n_connected_components > 1

    graph = _fix_connected_components(X, graph, n_connected_components, labels)

    n_connected_components, labels = connected_components(graph)
    assert n_connected_components == 1


def test_fix_connected_components_precomputed():
    # Test that _fix_connected_components accepts precomputed distance matrix.
    X = np.array([0, 1, 2, 5, 6, 7])[:, None]
    graph = kneighbors_graph(X, n_neighbors=2, mode="distance")

    n_connected_components, labels = connected_components(graph)
    assert n_connected_components > 1

    distances = pairwise_distances(X)
    graph = _fix_connected_components(
        distances, graph, n_connected_components, labels, metric="precomputed"
    )

    n_connected_components, labels = connected_components(graph)
    assert n_connected_components == 1

    # but it does not work with precomputed neighbors graph
    with pytest.raises(RuntimeError, match="does not work with a sparse"):
        _fix_connected_components(
            graph, graph, n_connected_components, labels, metric="precomputed"
        )


def test_fix_connected_components_wrong_mode():
    # Test that the an error is raised if the mode string is incorrect.
    X = np.array([0, 1, 2, 5, 6, 7])[:, None]
    graph = kneighbors_graph(X, n_neighbors=2, mode="distance")
    n_connected_components, labels = connected_components(graph)

    with pytest.raises(ValueError, match="Unknown mode"):
        graph = _fix_connected_components(
            X, graph, n_connected_components, labels, mode="foo"
        )


def test_fix_connected_components_connectivity_mode():
    # Test that the connectivity mode fill new connections with ones.
    X = np.array([0, 1, 6, 7])[:, None]
    graph = kneighbors_graph(X, n_neighbors=1, mode="connectivity")
    n_connected_components, labels = connected_components(graph)
    graph = _fix_connected_components(
        X, graph, n_connected_components, labels, mode="connectivity"
    )
    assert np.all(graph.data == 1)


def test_fix_connected_components_distance_mode():
    # Test that the distance mode does not fill new connections with ones.
    X = np.array([0, 1, 6, 7])[:, None]
    graph = kneighbors_graph(X, n_neighbors=1, mode="distance")
    assert np.all(graph.data == 1)

    n_connected_components, labels = connected_components(graph)
    graph = _fix_connected_components(
        X, graph, n_connected_components, labels, mode="distance"
    )
    assert not np.all(graph.data == 1)
