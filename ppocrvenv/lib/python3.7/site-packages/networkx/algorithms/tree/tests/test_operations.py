"""Unit tests for the :mod:`networkx.algorithms.tree.operations` module.

"""

import networkx as nx
from networkx.utils import nodes_equal, edges_equal


class TestJoin:
    """Unit tests for the :func:`networkx.tree.join` function."""

    def test_empty_sequence(self):
        """Tests that joining the empty sequence results in the tree
        with one node.

        """
        T = nx.join([])
        assert len(T) == 1
        assert T.number_of_edges() == 0

    def test_single(self):
        """Tests that joining just one tree yields a tree with one more
        node.

        """
        T = nx.empty_graph(1)
        actual = nx.join([(T, 0)])
        expected = nx.path_graph(2)
        assert nodes_equal(list(expected), list(actual))
        assert edges_equal(list(expected.edges()), list(actual.edges()))

    def test_basic(self):
        """Tests for joining multiple subtrees at a root node."""
        trees = [(nx.full_rary_tree(2, 2 ** 2 - 1), 0) for i in range(2)]
        actual = nx.join(trees)
        expected = nx.full_rary_tree(2, 2 ** 3 - 1)
        assert nx.is_isomorphic(actual, expected)
