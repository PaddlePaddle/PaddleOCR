"""Unit tests for the :mod:`networkx.algorithms.bipartite.redundancy` module.

"""

import pytest

from networkx import cycle_graph
from networkx import NetworkXError
from networkx.algorithms.bipartite import complete_bipartite_graph
from networkx.algorithms.bipartite import node_redundancy


def test_no_redundant_nodes():
    G = complete_bipartite_graph(2, 2)
    rc = node_redundancy(G)
    assert all(redundancy == 1 for redundancy in rc.values())


def test_redundant_nodes():
    G = cycle_graph(6)
    edge = {0, 3}
    G.add_edge(*edge)
    redundancy = node_redundancy(G)
    for v in edge:
        assert redundancy[v] == 2 / 3
    for v in set(G) - edge:
        assert redundancy[v] == 1


def test_not_enough_neighbors():
    with pytest.raises(NetworkXError):
        G = complete_bipartite_graph(1, 2)
        node_redundancy(G)
