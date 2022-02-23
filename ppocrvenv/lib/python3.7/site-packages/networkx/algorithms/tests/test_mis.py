"""
Tests for maximal (not maximum) independent sets.

"""

import pytest
import networkx as nx
import random


class TestMaximalIndependantSet:
    def setup(self):
        self.florentine = nx.Graph()
        self.florentine.add_edge("Acciaiuoli", "Medici")
        self.florentine.add_edge("Castellani", "Peruzzi")
        self.florentine.add_edge("Castellani", "Strozzi")
        self.florentine.add_edge("Castellani", "Barbadori")
        self.florentine.add_edge("Medici", "Barbadori")
        self.florentine.add_edge("Medici", "Ridolfi")
        self.florentine.add_edge("Medici", "Tornabuoni")
        self.florentine.add_edge("Medici", "Albizzi")
        self.florentine.add_edge("Medici", "Salviati")
        self.florentine.add_edge("Salviati", "Pazzi")
        self.florentine.add_edge("Peruzzi", "Strozzi")
        self.florentine.add_edge("Peruzzi", "Bischeri")
        self.florentine.add_edge("Strozzi", "Ridolfi")
        self.florentine.add_edge("Strozzi", "Bischeri")
        self.florentine.add_edge("Ridolfi", "Tornabuoni")
        self.florentine.add_edge("Tornabuoni", "Guadagni")
        self.florentine.add_edge("Albizzi", "Ginori")
        self.florentine.add_edge("Albizzi", "Guadagni")
        self.florentine.add_edge("Bischeri", "Guadagni")
        self.florentine.add_edge("Guadagni", "Lamberteschi")

    def test_random_seed(self):
        G = nx.complete_graph(5)
        for node in G:
            assert nx.maximal_independent_set(G, [node], seed=1) == [node]

    def test_K5(self):
        """Maximal independent set: K5"""
        G = nx.complete_graph(5)
        for node in G:
            assert nx.maximal_independent_set(G, [node]) == [node]

    def test_K55(self):
        """Maximal independent set: K55"""
        G = nx.complete_graph(55)
        for node in G:
            assert nx.maximal_independent_set(G, [node]) == [node]

    def test_exception(self):
        """Bad input should raise exception."""
        G = self.florentine
        pytest.raises(nx.NetworkXUnfeasible, nx.maximal_independent_set, G, ["Smith"])
        pytest.raises(
            nx.NetworkXUnfeasible, nx.maximal_independent_set, G, ["Salviati", "Pazzi"]
        )

    def test_digraph_exception(self):
        G = nx.DiGraph([(1, 2), (3, 4)])
        pytest.raises(nx.NetworkXNotImplemented, nx.maximal_independent_set, G)

    def test_florentine_family(self):
        G = self.florentine
        indep = nx.maximal_independent_set(G, ["Medici", "Bischeri"])
        assert sorted(indep) == sorted(
            ["Medici", "Bischeri", "Castellani", "Pazzi", "Ginori", "Lamberteschi"]
        )

    def test_bipartite(self):
        G = nx.complete_bipartite_graph(12, 34)
        indep = nx.maximal_independent_set(G, [4, 5, 9, 10])
        assert sorted(indep) == list(range(12))

    def test_random_graphs(self):
        """Generate 50 random graphs of different types and sizes and
        make sure that all sets are independent and maximal."""
        for i in range(0, 50, 10):
            G = nx.random_graphs.erdos_renyi_graph(i * 10 + 1, random.random())
            IS = nx.maximal_independent_set(G)
            assert not list(G.subgraph(IS).edges())
            neighbors_of_MIS = set.union(*(set(G.neighbors(v)) for v in IS))
            for v in set(G.nodes()).difference(IS):
                assert v in neighbors_of_MIS
