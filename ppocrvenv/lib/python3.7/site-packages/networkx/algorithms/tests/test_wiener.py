"""Unit tests for the :mod:`networkx.algorithms.wiener` module."""


from networkx import complete_graph
from networkx import DiGraph
from networkx import empty_graph
from networkx import path_graph
from networkx import wiener_index


class TestWienerIndex:
    """Unit tests for computing the Wiener index of a graph."""

    def test_disconnected_graph(self):
        """Tests that the Wiener index of a disconnected graph is
        positive infinity.

        """
        assert wiener_index(empty_graph(2)) == float("inf")

    def test_directed(self):
        """Tests that each pair of nodes in the directed graph is
        counted once when computing the Wiener index.

        """
        G = complete_graph(3)
        H = DiGraph(G)
        assert (2 * wiener_index(G)) == wiener_index(H)

    def test_complete_graph(self):
        """Tests that the Wiener index of the complete graph is simply
        the number of edges.

        """
        n = 10
        G = complete_graph(n)
        assert wiener_index(G) == (n * (n - 1) / 2)

    def test_path_graph(self):
        """Tests that the Wiener index of the path graph is correctly
        computed.

        """
        # In P_n, there are n - 1 pairs of vertices at distance one, n -
        # 2 pairs at distance two, n - 3 at distance three, ..., 1 at
        # distance n - 1, so the Wiener index should be
        #
        #     1 * (n - 1) + 2 * (n - 2) + ... + (n - 2) * 2 + (n - 1) * 1
        #
        # For example, in P_5,
        #
        #     1 * 4 + 2 * 3 + 3 * 2 + 4 * 1 = 2 (1 * 4 + 2 * 3)
        #
        # and in P_6,
        #
        #     1 * 5 + 2 * 4 + 3 * 3 + 4 * 2 + 5 * 1 = 2 (1 * 5 + 2 * 4) + 3 * 3
        #
        # assuming n is *odd*, this gives the formula
        #
        #     2 \sum_{i = 1}^{(n - 1) / 2} [i * (n - i)]
        #
        # assuming n is *even*, this gives the formula
        #
        #     2 \sum_{i = 1}^{n / 2} [i * (n - i)] - (n / 2) ** 2
        #
        n = 9
        G = path_graph(n)
        expected = 2 * sum(i * (n - i) for i in range(1, (n // 2) + 1))
        actual = wiener_index(G)
        assert expected == actual
