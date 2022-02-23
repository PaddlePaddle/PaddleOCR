"""Unit tests for bridge-finding algorithms."""

import networkx as nx


class TestBridges:
    """Unit tests for the bridge-finding function."""

    def test_single_bridge(self):
        edges = [
            # DFS tree edges.
            (1, 2),
            (2, 3),
            (3, 4),
            (3, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            (5, 9),
            (9, 10),
            # Nontree edges.
            (1, 3),
            (1, 4),
            (2, 5),
            (5, 10),
            (6, 8),
        ]
        G = nx.Graph(edges)
        source = 1
        bridges = list(nx.bridges(G, source))
        assert bridges == [(5, 6)]

    def test_barbell_graph(self):
        # The (3, 0) barbell graph has two triangles joined by a single edge.
        G = nx.barbell_graph(3, 0)
        source = 0
        bridges = list(nx.bridges(G, source))
        assert bridges == [(2, 3)]


class TestLocalBridges:
    """Unit tests for the local_bridge function."""

    @classmethod
    def setup_class(cls):
        cls.BB = nx.barbell_graph(4, 0)
        cls.square = nx.cycle_graph(4)
        cls.tri = nx.cycle_graph(3)

    def test_nospan(self):
        expected = {(3, 4), (4, 3)}
        assert next(nx.local_bridges(self.BB, with_span=False)) in expected
        assert set(nx.local_bridges(self.square, with_span=False)) == self.square.edges
        assert list(nx.local_bridges(self.tri, with_span=False)) == []

    def test_no_weight(self):
        inf = float("inf")
        expected = {(3, 4, inf), (4, 3, inf)}
        assert next(nx.local_bridges(self.BB)) in expected
        expected = {(u, v, 3) for u, v, in self.square.edges}
        assert set(nx.local_bridges(self.square)) == expected
        assert list(nx.local_bridges(self.tri)) == []

    def test_weight(self):
        inf = float("inf")
        G = self.square.copy()

        G.edges[1, 2]["weight"] = 2
        expected = {(u, v, 5 - wt) for u, v, wt in G.edges(data="weight", default=1)}
        assert set(nx.local_bridges(G, weight="weight")) == expected

        expected = {(u, v, 6) for u, v in G.edges}
        lb = nx.local_bridges(G, weight=lambda u, v, d: 2)
        assert set(lb) == expected
