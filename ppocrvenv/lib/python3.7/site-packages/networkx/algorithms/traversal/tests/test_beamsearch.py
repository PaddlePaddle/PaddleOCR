"""Unit tests for the beam search functions."""

import networkx as nx


def identity(x):
    return x


class TestBeamSearch:
    """Unit tests for the beam search function."""

    def test_narrow(self):
        """Tests that a narrow beam width may cause an incomplete search."""
        # In this search, we enqueue only the neighbor 3 at the first
        # step, then only the neighbor 2 at the second step. Once at
        # node 2, the search chooses node 3, since it has a higher value
        # that node 1, but node 3 has already been visited, so the
        # search terminates.
        G = nx.cycle_graph(4)
        edges = nx.bfs_beam_edges(G, 0, identity, width=1)
        assert list(edges) == [(0, 3), (3, 2)]

    def test_wide(self):
        G = nx.cycle_graph(4)
        edges = nx.bfs_beam_edges(G, 0, identity, width=2)
        assert list(edges) == [(0, 3), (0, 1), (3, 2)]
