from itertools import chain
from itertools import combinations

import pytest

import networkx as nx
from networkx.algorithms.community import label_propagation_communities
from networkx.algorithms.community import asyn_lpa_communities


def test_directed_not_supported():
    with pytest.raises(nx.NetworkXNotImplemented):
        # not supported for directed graphs
        test = nx.DiGraph()
        test.add_edge("a", "b")
        test.add_edge("a", "c")
        test.add_edge("b", "d")
        result = label_propagation_communities(test)


def test_iterator_vs_iterable():
    G = nx.empty_graph("a")
    assert list(label_propagation_communities(G)) == [{"a"}]
    for community in label_propagation_communities(G):
        assert community == {"a"}
    pytest.raises(TypeError, next, label_propagation_communities(G))


def test_one_node():
    test = nx.Graph()
    test.add_node("a")

    # The expected communities are:
    ground_truth = {frozenset(["a"])}

    communities = label_propagation_communities(test)
    result = {frozenset(c) for c in communities}
    assert result == ground_truth


def test_unconnected_communities():
    test = nx.Graph()
    # community 1
    test.add_edge("a", "c")
    test.add_edge("a", "d")
    test.add_edge("d", "c")
    # community 2
    test.add_edge("b", "e")
    test.add_edge("e", "f")
    test.add_edge("f", "b")

    # The expected communities are:
    ground_truth = {frozenset(["a", "c", "d"]), frozenset(["b", "e", "f"])}

    communities = label_propagation_communities(test)
    result = {frozenset(c) for c in communities}
    assert result == ground_truth


def test_connected_communities():
    test = nx.Graph()
    # community 1
    test.add_edge("a", "b")
    test.add_edge("c", "a")
    test.add_edge("c", "b")
    test.add_edge("d", "a")
    test.add_edge("d", "b")
    test.add_edge("d", "c")
    test.add_edge("e", "a")
    test.add_edge("e", "b")
    test.add_edge("e", "c")
    test.add_edge("e", "d")
    # community 2
    test.add_edge("1", "2")
    test.add_edge("3", "1")
    test.add_edge("3", "2")
    test.add_edge("4", "1")
    test.add_edge("4", "2")
    test.add_edge("4", "3")
    test.add_edge("5", "1")
    test.add_edge("5", "2")
    test.add_edge("5", "3")
    test.add_edge("5", "4")
    # edge between community 1 and 2
    test.add_edge("a", "1")
    # community 3
    test.add_edge("x", "y")
    # community 4 with only a single node
    test.add_node("z")

    # The expected communities are:
    ground_truth1 = {
        frozenset(["a", "b", "c", "d", "e"]),
        frozenset(["1", "2", "3", "4", "5"]),
        frozenset(["x", "y"]),
        frozenset(["z"]),
    }
    ground_truth2 = {
        frozenset(["a", "b", "c", "d", "e", "1", "2", "3", "4", "5"]),
        frozenset(["x", "y"]),
        frozenset(["z"]),
    }
    ground_truth = (ground_truth1, ground_truth2)

    communities = label_propagation_communities(test)
    result = {frozenset(c) for c in communities}
    assert result in ground_truth


def test_termination():
    # ensure termination of asyn_lpa_communities in two cases
    # that led to an endless loop in a previous version
    test1 = nx.karate_club_graph()
    test2 = nx.caveman_graph(2, 10)
    test2.add_edges_from([(0, 20), (20, 10)])
    asyn_lpa_communities(test1)
    asyn_lpa_communities(test2)


class TestAsynLpaCommunities:
    def _check_communities(self, G, expected):
        """Checks that the communities computed from the given graph ``G``
        using the :func:`~networkx.asyn_lpa_communities` function match
        the set of nodes given in ``expected``.

        ``expected`` must be a :class:`set` of :class:`frozenset`
        instances, each element of which is a node in the graph.

        """
        communities = asyn_lpa_communities(G)
        result = {frozenset(c) for c in communities}
        assert result == expected

    def test_null_graph(self):
        G = nx.null_graph()
        ground_truth = set()
        self._check_communities(G, ground_truth)

    def test_single_node(self):
        G = nx.empty_graph(1)
        ground_truth = {frozenset([0])}
        self._check_communities(G, ground_truth)

    def test_simple_communities(self):
        # This graph is the disjoint union of two triangles.
        G = nx.Graph(["ab", "ac", "bc", "de", "df", "fe"])
        ground_truth = {frozenset("abc"), frozenset("def")}
        self._check_communities(G, ground_truth)

    def test_seed_argument(self):
        G = nx.Graph(["ab", "ac", "bc", "de", "df", "fe"])
        ground_truth = {frozenset("abc"), frozenset("def")}
        communities = asyn_lpa_communities(G, seed=1)
        result = {frozenset(c) for c in communities}
        assert result == ground_truth

    def test_several_communities(self):
        # This graph is the disjoint union of five triangles.
        ground_truth = {frozenset(range(3 * i, 3 * (i + 1))) for i in range(5)}
        edges = chain.from_iterable(combinations(c, 2) for c in ground_truth)
        G = nx.Graph(edges)
        self._check_communities(G, ground_truth)
