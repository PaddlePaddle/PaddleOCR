"""Unit tests for the :mod:`networkx.algorithms.community.utils` module.

"""

import networkx as nx
from networkx.algorithms.community import is_partition


def test_is_partition():
    G = nx.empty_graph(3)
    assert is_partition(G, [{0, 1}, {2}])
    assert is_partition(G, ({0, 1}, {2}))
    assert is_partition(G, ([0, 1], [2]))
    assert is_partition(G, [[0, 1], [2]])


def test_not_covering():
    G = nx.empty_graph(3)
    assert not is_partition(G, [{0}, {1}])


def test_not_disjoint():
    G = nx.empty_graph(3)
    assert not is_partition(G, [{0, 1}, {1, 2}])


def test_not_node():
    G = nx.empty_graph(3)
    assert not is_partition(G, [{0, 1}, {3}])
