import networkx as nx


class TestMinEdgeCover:
    """Tests for :func:`networkx.algorithms.min_edge_cover`"""

    def test_empty_graph(self):
        G = nx.Graph()
        assert nx.min_edge_cover(G) == set()

    def test_graph_with_loop(self):
        G = nx.Graph()
        G.add_edge(0, 0)
        assert nx.min_edge_cover(G) == {(0, 0)}

    def test_graph_single_edge(self):
        G = nx.Graph()
        G.add_edge(0, 1)
        assert nx.min_edge_cover(G) in ({(0, 1)}, {(1, 0)})

    def test_bipartite_explicit(self):
        G = nx.Graph()
        G.add_nodes_from([1, 2, 3, 4], bipartite=0)
        G.add_nodes_from(["a", "b", "c"], bipartite=1)
        G.add_edges_from([(1, "a"), (1, "b"), (2, "b"), (2, "c"), (3, "c"), (4, "a")])
        min_cover = nx.min_edge_cover(
            G, nx.algorithms.bipartite.matching.eppstein_matching
        )
        min_cover2 = nx.min_edge_cover(G)
        assert nx.is_edge_cover(G, min_cover)
        assert len(min_cover) == 8

    def test_complete_graph(self):
        G = nx.complete_graph(10)
        min_cover = nx.min_edge_cover(G)
        assert nx.is_edge_cover(G, min_cover)
        assert len(min_cover) == 5


class TestIsEdgeCover:
    """Tests for :func:`networkx.algorithms.is_edge_cover`"""

    def test_empty_graph(self):
        G = nx.Graph()
        assert nx.is_edge_cover(G, set())

    def test_graph_with_loop(self):
        G = nx.Graph()
        G.add_edge(1, 1)
        assert nx.is_edge_cover(G, {(1, 1)})

    def test_graph_single_edge(self):
        G = nx.Graph()
        G.add_edge(0, 1)
        assert nx.is_edge_cover(G, {(0, 0), (1, 1)})
        assert nx.is_edge_cover(G, {(0, 1), (1, 0)})
        assert nx.is_edge_cover(G, {(0, 1)})
        assert not nx.is_edge_cover(G, {(0, 0)})
