import pytest
import networkx as nx


class TestSubsetBetweennessCentrality:
    def test_K5(self):
        """Betweenness Centrality Subset: K5"""
        G = nx.complete_graph(5)
        b = nx.betweenness_centrality_subset(
            G, sources=[0], targets=[1, 3], weight=None
        )
        b_answer = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-7)

    def test_P5_directed(self):
        """Betweenness Centrality Subset: P5 directed"""
        G = nx.DiGraph()
        nx.add_path(G, range(5))
        b_answer = {0: 0, 1: 1, 2: 1, 3: 0, 4: 0, 5: 0}
        b = nx.betweenness_centrality_subset(G, sources=[0], targets=[3], weight=None)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-7)

    def test_P5(self):
        """Betweenness Centrality Subset: P5"""
        G = nx.Graph()
        nx.add_path(G, range(5))
        b_answer = {0: 0, 1: 0.5, 2: 0.5, 3: 0, 4: 0, 5: 0}
        b = nx.betweenness_centrality_subset(G, sources=[0], targets=[3], weight=None)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-7)

    def test_P5_multiple_target(self):
        """Betweenness Centrality Subset: P5 multiple target"""
        G = nx.Graph()
        nx.add_path(G, range(5))
        b_answer = {0: 0, 1: 1, 2: 1, 3: 0.5, 4: 0, 5: 0}
        b = nx.betweenness_centrality_subset(
            G, sources=[0], targets=[3, 4], weight=None
        )
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-7)

    def test_box(self):
        """Betweenness Centrality Subset: box"""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
        b_answer = {0: 0, 1: 0.25, 2: 0.25, 3: 0}
        b = nx.betweenness_centrality_subset(G, sources=[0], targets=[3], weight=None)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-7)

    def test_box_and_path(self):
        """Betweenness Centrality Subset: box and path"""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5)])
        b_answer = {0: 0, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0, 5: 0}
        b = nx.betweenness_centrality_subset(
            G, sources=[0], targets=[3, 4], weight=None
        )
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-7)

    def test_box_and_path2(self):
        """Betweenness Centrality Subset: box and path multiple target"""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (1, 20), (20, 3), (3, 4)])
        b_answer = {0: 0, 1: 1.0, 2: 0.5, 20: 0.5, 3: 0.5, 4: 0}
        b = nx.betweenness_centrality_subset(
            G, sources=[0], targets=[3, 4], weight=None
        )
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-7)

    def test_diamond_multi_path(self):
        """Betweenness Centrality Subset: Diamond Multi Path"""
        G = nx.Graph()
        G.add_edges_from(
            [
                (1, 2),
                (1, 3),
                (1, 4),
                (1, 5),
                (1, 10),
                (10, 11),
                (11, 12),
                (12, 9),
                (2, 6),
                (3, 6),
                (4, 6),
                (5, 7),
                (7, 8),
                (6, 8),
                (8, 9),
            ]
        )
        b = nx.betweenness_centrality_subset(G, sources=[1], targets=[9], weight=None)

        expected_b = {
            1: 0,
            2: 1.0 / 10,
            3: 1.0 / 10,
            4: 1.0 / 10,
            5: 1.0 / 10,
            6: 3.0 / 10,
            7: 1.0 / 10,
            8: 4.0 / 10,
            9: 0,
            10: 1.0 / 10,
            11: 1.0 / 10,
            12: 1.0 / 10,
        }

        for n in sorted(G):
            assert b[n] == pytest.approx(expected_b[n], abs=1e-7)


class TestBetweennessCentralitySources:
    def test_K5(self):
        """Betweenness Centrality Sources: K5"""
        G = nx.complete_graph(5)
        b = nx.betweenness_centrality_source(G, weight=None, normalized=False)
        b_answer = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-7)

    def test_P3(self):
        """Betweenness Centrality Sources: P3"""
        G = nx.path_graph(3)
        b_answer = {0: 0.0, 1: 1.0, 2: 0.0}
        b = nx.betweenness_centrality_source(G, weight=None, normalized=True)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-7)


class TestEdgeSubsetBetweennessCentrality:
    def test_K5(self):
        """Edge betweenness subset centrality: K5"""
        G = nx.complete_graph(5)
        b = nx.edge_betweenness_centrality_subset(
            G, sources=[0], targets=[1, 3], weight=None
        )
        b_answer = dict.fromkeys(G.edges(), 0)
        b_answer[(0, 3)] = b_answer[(0, 1)] = 0.5
        for n in sorted(G.edges()):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-7)

    def test_P5_directed(self):
        """Edge betweenness subset centrality: P5 directed"""
        G = nx.DiGraph()
        nx.add_path(G, range(5))
        b_answer = dict.fromkeys(G.edges(), 0)
        b_answer[(0, 1)] = b_answer[(1, 2)] = b_answer[(2, 3)] = 1
        b = nx.edge_betweenness_centrality_subset(
            G, sources=[0], targets=[3], weight=None
        )
        for n in sorted(G.edges()):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-7)

    def test_P5(self):
        """Edge betweenness subset centrality: P5"""
        G = nx.Graph()
        nx.add_path(G, range(5))
        b_answer = dict.fromkeys(G.edges(), 0)
        b_answer[(0, 1)] = b_answer[(1, 2)] = b_answer[(2, 3)] = 0.5
        b = nx.edge_betweenness_centrality_subset(
            G, sources=[0], targets=[3], weight=None
        )
        for n in sorted(G.edges()):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-7)

    def test_P5_multiple_target(self):
        """Edge betweenness subset centrality: P5 multiple target"""
        G = nx.Graph()
        nx.add_path(G, range(5))
        b_answer = dict.fromkeys(G.edges(), 0)
        b_answer[(0, 1)] = b_answer[(1, 2)] = b_answer[(2, 3)] = 1
        b_answer[(3, 4)] = 0.5
        b = nx.edge_betweenness_centrality_subset(
            G, sources=[0], targets=[3, 4], weight=None
        )
        for n in sorted(G.edges()):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-7)

    def test_box(self):
        """Edge betweenness subset centrality: box"""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
        b_answer = dict.fromkeys(G.edges(), 0)
        b_answer[(0, 1)] = b_answer[(0, 2)] = 0.25
        b_answer[(1, 3)] = b_answer[(2, 3)] = 0.25
        b = nx.edge_betweenness_centrality_subset(
            G, sources=[0], targets=[3], weight=None
        )
        for n in sorted(G.edges()):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-7)

    def test_box_and_path(self):
        """Edge betweenness subset centrality: box and path"""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5)])
        b_answer = dict.fromkeys(G.edges(), 0)
        b_answer[(0, 1)] = b_answer[(0, 2)] = 0.5
        b_answer[(1, 3)] = b_answer[(2, 3)] = 0.5
        b_answer[(3, 4)] = 0.5
        b = nx.edge_betweenness_centrality_subset(
            G, sources=[0], targets=[3, 4], weight=None
        )
        for n in sorted(G.edges()):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-7)

    def test_box_and_path2(self):
        """Edge betweenness subset centrality: box and path multiple target"""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (1, 20), (20, 3), (3, 4)])
        b_answer = dict.fromkeys(G.edges(), 0)
        b_answer[(0, 1)] = 1.0
        b_answer[(1, 20)] = b_answer[(3, 20)] = 0.5
        b_answer[(1, 2)] = b_answer[(2, 3)] = 0.5
        b_answer[(3, 4)] = 0.5
        b = nx.edge_betweenness_centrality_subset(
            G, sources=[0], targets=[3, 4], weight=None
        )
        for n in sorted(G.edges()):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-7)
