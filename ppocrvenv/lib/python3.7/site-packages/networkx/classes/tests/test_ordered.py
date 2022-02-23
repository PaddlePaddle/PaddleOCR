import networkx as nx


class TestOrdered:
    # Just test instantiation.
    def test_graph(self):
        G = nx.OrderedGraph()

    def test_digraph(self):
        G = nx.OrderedDiGraph()

    def test_multigraph(self):
        G = nx.OrderedMultiGraph()

    def test_multidigraph(self):
        G = nx.OrderedMultiDiGraph()


class TestOrderedFeatures:
    @classmethod
    def setup_class(cls):
        cls.G = nx.OrderedDiGraph()
        cls.G.add_nodes_from([1, 2, 3])
        cls.G.add_edges_from([(2, 3), (1, 3)])

    def test_subgraph_order(self):
        G = self.G
        G_sub = G.subgraph([1, 2, 3])
        assert list(G.nodes) == list(G_sub.nodes)
        assert list(G.edges) == list(G_sub.edges)
        assert list(G.pred[3]) == list(G_sub.pred[3])
        assert [2, 1] == list(G_sub.pred[3])
        assert [] == list(G_sub.succ[3])

        G_sub = nx.induced_subgraph(G, [1, 2, 3])
        assert list(G.nodes) == list(G_sub.nodes)
        assert list(G.edges) == list(G_sub.edges)
        assert list(G.pred[3]) == list(G_sub.pred[3])
        assert [2, 1] == list(G_sub.pred[3])
        assert [] == list(G_sub.succ[3])
