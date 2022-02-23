import networkx as nx

import pytest

np = pytest.importorskip("numpy")


def test_non_randomness():
    G = nx.karate_club_graph()
    np.testing.assert_almost_equal(nx.non_randomness(G, 2)[0], 11.7, decimal=2)
    np.testing.assert_almost_equal(
        nx.non_randomness(G)[0], 7.21, decimal=2
    )  # infers 3 communities


def test_non_connected():
    G = nx.Graph()
    G.add_edge(1, 2)
    G.add_node(3)
    with pytest.raises(nx.NetworkXException):
        nx.non_randomness(G)


def test_self_loops():
    G = nx.Graph()
    G.add_edge(1, 2)
    G.add_edge(1, 1)
    with pytest.raises(nx.NetworkXError):
        nx.non_randomness(G)
