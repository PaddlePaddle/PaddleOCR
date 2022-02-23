"""Unit tests for the :mod:`networkx.algorithms.tree.mst` module."""

import pytest

import networkx as nx
from networkx.utils import nodes_equal, edges_equal


def test_unknown_algorithm():
    with pytest.raises(ValueError):
        nx.minimum_spanning_tree(nx.Graph(), algorithm="random")


class MinimumSpanningTreeTestBase:
    """Base class for test classes for minimum spanning tree algorithms.

    This class contains some common tests that will be inherited by
    subclasses. Each subclass must have a class attribute
    :data:`algorithm` that is a string representing the algorithm to
    run, as described under the ``algorithm`` keyword argument for the
    :func:`networkx.minimum_spanning_edges` function.  Subclasses can
    then implement any algorithm-specific tests.

    """

    def setup_method(self, method):
        """Creates an example graph and stores the expected minimum and
        maximum spanning tree edges.

        """
        # This stores the class attribute `algorithm` in an instance attribute.
        self.algo = self.algorithm
        # This example graph comes from Wikipedia:
        # https://en.wikipedia.org/wiki/Kruskal's_algorithm
        edges = [
            (0, 1, 7),
            (0, 3, 5),
            (1, 2, 8),
            (1, 3, 9),
            (1, 4, 7),
            (2, 4, 5),
            (3, 4, 15),
            (3, 5, 6),
            (4, 5, 8),
            (4, 6, 9),
            (5, 6, 11),
        ]
        self.G = nx.Graph()
        self.G.add_weighted_edges_from(edges)
        self.minimum_spanning_edgelist = [
            (0, 1, {"weight": 7}),
            (0, 3, {"weight": 5}),
            (1, 4, {"weight": 7}),
            (2, 4, {"weight": 5}),
            (3, 5, {"weight": 6}),
            (4, 6, {"weight": 9}),
        ]
        self.maximum_spanning_edgelist = [
            (0, 1, {"weight": 7}),
            (1, 2, {"weight": 8}),
            (1, 3, {"weight": 9}),
            (3, 4, {"weight": 15}),
            (4, 6, {"weight": 9}),
            (5, 6, {"weight": 11}),
        ]

    def test_minimum_edges(self):
        edges = nx.minimum_spanning_edges(self.G, algorithm=self.algo)
        # Edges from the spanning edges functions don't come in sorted
        # orientation, so we need to sort each edge individually.
        actual = sorted((min(u, v), max(u, v), d) for u, v, d in edges)
        assert edges_equal(actual, self.minimum_spanning_edgelist)

    def test_maximum_edges(self):
        edges = nx.maximum_spanning_edges(self.G, algorithm=self.algo)
        # Edges from the spanning edges functions don't come in sorted
        # orientation, so we need to sort each edge individually.
        actual = sorted((min(u, v), max(u, v), d) for u, v, d in edges)
        assert edges_equal(actual, self.maximum_spanning_edgelist)

    def test_without_data(self):
        edges = nx.minimum_spanning_edges(self.G, algorithm=self.algo, data=False)
        # Edges from the spanning edges functions don't come in sorted
        # orientation, so we need to sort each edge individually.
        actual = sorted((min(u, v), max(u, v)) for u, v in edges)
        expected = [(u, v) for u, v, d in self.minimum_spanning_edgelist]
        assert edges_equal(actual, expected)

    def test_nan_weights(self):
        # Edge weights NaN never appear in the spanning tree. see #2164
        G = self.G
        G.add_edge(0, 12, weight=float("nan"))
        edges = nx.minimum_spanning_edges(
            G, algorithm=self.algo, data=False, ignore_nan=True
        )
        actual = sorted((min(u, v), max(u, v)) for u, v in edges)
        expected = [(u, v) for u, v, d in self.minimum_spanning_edgelist]
        assert edges_equal(actual, expected)
        # Now test for raising exception
        edges = nx.minimum_spanning_edges(
            G, algorithm=self.algo, data=False, ignore_nan=False
        )
        with pytest.raises(ValueError):
            list(edges)
        # test default for ignore_nan as False
        edges = nx.minimum_spanning_edges(G, algorithm=self.algo, data=False)
        with pytest.raises(ValueError):
            list(edges)

    def test_nan_weights_order(self):
        # now try again with a nan edge at the beginning of G.nodes
        edges = [
            (0, 1, 7),
            (0, 3, 5),
            (1, 2, 8),
            (1, 3, 9),
            (1, 4, 7),
            (2, 4, 5),
            (3, 4, 15),
            (3, 5, 6),
            (4, 5, 8),
            (4, 6, 9),
            (5, 6, 11),
        ]
        G = nx.Graph()
        G.add_weighted_edges_from([(u + 1, v + 1, wt) for u, v, wt in edges])
        G.add_edge(0, 7, weight=float("nan"))
        edges = nx.minimum_spanning_edges(
            G, algorithm=self.algo, data=False, ignore_nan=True
        )
        actual = sorted((min(u, v), max(u, v)) for u, v in edges)
        shift = [(u + 1, v + 1) for u, v, d in self.minimum_spanning_edgelist]
        assert edges_equal(actual, shift)

    def test_isolated_node(self):
        # now try again with an isolated node
        edges = [
            (0, 1, 7),
            (0, 3, 5),
            (1, 2, 8),
            (1, 3, 9),
            (1, 4, 7),
            (2, 4, 5),
            (3, 4, 15),
            (3, 5, 6),
            (4, 5, 8),
            (4, 6, 9),
            (5, 6, 11),
        ]
        G = nx.Graph()
        G.add_weighted_edges_from([(u + 1, v + 1, wt) for u, v, wt in edges])
        G.add_node(0)
        edges = nx.minimum_spanning_edges(
            G, algorithm=self.algo, data=False, ignore_nan=True
        )
        actual = sorted((min(u, v), max(u, v)) for u, v in edges)
        shift = [(u + 1, v + 1) for u, v, d in self.minimum_spanning_edgelist]
        assert edges_equal(actual, shift)

    def test_minimum_tree(self):
        T = nx.minimum_spanning_tree(self.G, algorithm=self.algo)
        actual = sorted(T.edges(data=True))
        assert edges_equal(actual, self.minimum_spanning_edgelist)

    def test_maximum_tree(self):
        T = nx.maximum_spanning_tree(self.G, algorithm=self.algo)
        actual = sorted(T.edges(data=True))
        assert edges_equal(actual, self.maximum_spanning_edgelist)

    def test_disconnected(self):
        G = nx.Graph([(0, 1, dict(weight=1)), (2, 3, dict(weight=2))])
        T = nx.minimum_spanning_tree(G, algorithm=self.algo)
        assert nodes_equal(list(T), list(range(4)))
        assert edges_equal(list(T.edges()), [(0, 1), (2, 3)])

    def test_empty_graph(self):
        G = nx.empty_graph(3)
        T = nx.minimum_spanning_tree(G, algorithm=self.algo)
        assert nodes_equal(sorted(T), list(range(3)))
        assert T.number_of_edges() == 0

    def test_attributes(self):
        G = nx.Graph()
        G.add_edge(1, 2, weight=1, color="red", distance=7)
        G.add_edge(2, 3, weight=1, color="green", distance=2)
        G.add_edge(1, 3, weight=10, color="blue", distance=1)
        G.graph["foo"] = "bar"
        T = nx.minimum_spanning_tree(G, algorithm=self.algo)
        assert T.graph == G.graph
        assert nodes_equal(T, G)
        for u, v in T.edges():
            assert T.adj[u][v] == G.adj[u][v]

    def test_weight_attribute(self):
        G = nx.Graph()
        G.add_edge(0, 1, weight=1, distance=7)
        G.add_edge(0, 2, weight=30, distance=1)
        G.add_edge(1, 2, weight=1, distance=1)
        G.add_node(3)
        T = nx.minimum_spanning_tree(G, algorithm=self.algo, weight="distance")
        assert nodes_equal(sorted(T), list(range(4)))
        assert edges_equal(sorted(T.edges()), [(0, 2), (1, 2)])
        T = nx.maximum_spanning_tree(G, algorithm=self.algo, weight="distance")
        assert nodes_equal(sorted(T), list(range(4)))
        assert edges_equal(sorted(T.edges()), [(0, 1), (0, 2)])


class TestBoruvka(MinimumSpanningTreeTestBase):
    """Unit tests for computing a minimum (or maximum) spanning tree
    using Borůvka's algorithm.

    """

    algorithm = "boruvka"

    def test_unicode_name(self):
        """Tests that using a Unicode string can correctly indicate
        Borůvka's algorithm.

        """
        edges = nx.minimum_spanning_edges(self.G, algorithm="borůvka")
        # Edges from the spanning edges functions don't come in sorted
        # orientation, so we need to sort each edge individually.
        actual = sorted((min(u, v), max(u, v), d) for u, v, d in edges)
        assert edges_equal(actual, self.minimum_spanning_edgelist)


class MultigraphMSTTestBase(MinimumSpanningTreeTestBase):
    # Abstract class

    def test_multigraph_keys_min(self):
        """Tests that the minimum spanning edges of a multigraph
        preserves edge keys.

        """
        G = nx.MultiGraph()
        G.add_edge(0, 1, key="a", weight=2)
        G.add_edge(0, 1, key="b", weight=1)
        min_edges = nx.minimum_spanning_edges
        mst_edges = min_edges(G, algorithm=self.algo, data=False)
        assert edges_equal([(0, 1, "b")], list(mst_edges))

    def test_multigraph_keys_max(self):
        """Tests that the maximum spanning edges of a multigraph
        preserves edge keys.

        """
        G = nx.MultiGraph()
        G.add_edge(0, 1, key="a", weight=2)
        G.add_edge(0, 1, key="b", weight=1)
        max_edges = nx.maximum_spanning_edges
        mst_edges = max_edges(G, algorithm=self.algo, data=False)
        assert edges_equal([(0, 1, "a")], list(mst_edges))


class TestKruskal(MultigraphMSTTestBase):
    """Unit tests for computing a minimum (or maximum) spanning tree
    using Kruskal's algorithm.

    """

    algorithm = "kruskal"


class TestPrim(MultigraphMSTTestBase):
    """Unit tests for computing a minimum (or maximum) spanning tree
    using Prim's algorithm.

    """

    algorithm = "prim"

    def test_multigraph_keys_tree(self):
        G = nx.MultiGraph()
        G.add_edge(0, 1, key="a", weight=2)
        G.add_edge(0, 1, key="b", weight=1)
        T = nx.minimum_spanning_tree(G)
        assert edges_equal([(0, 1, 1)], list(T.edges(data="weight")))

    def test_multigraph_keys_tree_max(self):
        G = nx.MultiGraph()
        G.add_edge(0, 1, key="a", weight=2)
        G.add_edge(0, 1, key="b", weight=1)
        T = nx.maximum_spanning_tree(G)
        assert edges_equal([(0, 1, 2)], list(T.edges(data="weight")))
