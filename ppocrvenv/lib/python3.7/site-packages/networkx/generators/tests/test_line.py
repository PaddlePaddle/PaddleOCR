import networkx as nx
import pytest

import networkx.generators.line as line
from networkx.utils import edges_equal


def test_node_func():
    # graph
    G = nx.Graph()
    G.add_edge(1, 2)
    nf = line._node_func(G)
    assert nf(1, 2) == (1, 2)
    assert nf(2, 1) == (1, 2)

    # multigraph
    G = nx.MultiGraph()
    G.add_edge(1, 2)
    G.add_edge(1, 2)
    nf = line._node_func(G)
    assert nf(1, 2, 0) == (1, 2, 0)
    assert nf(2, 1, 0) == (1, 2, 0)


def test_edge_func():
    # graph
    G = nx.Graph()
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    ef = line._edge_func(G)
    expected = [(1, 2), (2, 3)]
    assert edges_equal(ef(), expected)

    # digraph
    G = nx.MultiDiGraph()
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    G.add_edge(2, 3)
    ef = line._edge_func(G)
    expected = [(1, 2, 0), (2, 3, 0), (2, 3, 1)]
    result = sorted(ef())
    assert expected == result


def test_sorted_edge():
    assert (1, 2) == line._sorted_edge(1, 2)
    assert (1, 2) == line._sorted_edge(2, 1)


class TestGeneratorLine:
    def test_star(self):
        G = nx.star_graph(5)
        L = nx.line_graph(G)
        assert nx.is_isomorphic(L, nx.complete_graph(5))

    def test_path(self):
        G = nx.path_graph(5)
        L = nx.line_graph(G)
        assert nx.is_isomorphic(L, nx.path_graph(4))

    def test_cycle(self):
        G = nx.cycle_graph(5)
        L = nx.line_graph(G)
        assert nx.is_isomorphic(L, G)

    def test_digraph1(self):
        G = nx.DiGraph()
        G.add_edges_from([(0, 1), (0, 2), (0, 3)])
        L = nx.line_graph(G)
        # no edge graph, but with nodes
        assert L.adj == {(0, 1): {}, (0, 2): {}, (0, 3): {}}

    def test_digraph2(self):
        G = nx.DiGraph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3)])
        L = nx.line_graph(G)
        assert edges_equal(L.edges(), [((0, 1), (1, 2)), ((1, 2), (2, 3))])

    def test_create1(self):
        G = nx.DiGraph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3)])
        L = nx.line_graph(G, create_using=nx.Graph())
        assert edges_equal(L.edges(), [((0, 1), (1, 2)), ((1, 2), (2, 3))])

    def test_create2(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3)])
        L = nx.line_graph(G, create_using=nx.DiGraph())
        assert edges_equal(L.edges(), [((0, 1), (1, 2)), ((1, 2), (2, 3))])


class TestGeneratorInverseLine:
    def test_example(self):
        G = nx.Graph()
        G_edges = [
            [1, 2],
            [1, 3],
            [1, 4],
            [1, 5],
            [2, 3],
            [2, 5],
            [2, 6],
            [2, 7],
            [3, 4],
            [3, 5],
            [6, 7],
            [6, 8],
            [7, 8],
        ]
        G.add_edges_from(G_edges)
        H = nx.inverse_line_graph(G)
        solution = nx.Graph()
        solution_edges = [
            ("a", "b"),
            ("a", "c"),
            ("a", "d"),
            ("a", "e"),
            ("c", "d"),
            ("e", "f"),
            ("e", "g"),
            ("f", "g"),
        ]
        solution.add_edges_from(solution_edges)
        assert nx.is_isomorphic(H, solution)

    def test_example_2(self):
        G = nx.Graph()
        G_edges = [[1, 2], [1, 3], [2, 3], [3, 4], [3, 5], [4, 5]]
        G.add_edges_from(G_edges)
        H = nx.inverse_line_graph(G)
        solution = nx.Graph()
        solution_edges = [("a", "c"), ("b", "c"), ("c", "d"), ("d", "e"), ("d", "f")]
        solution.add_edges_from(solution_edges)
        assert nx.is_isomorphic(H, solution)

    def test_pair(self):
        G = nx.path_graph(2)
        H = nx.inverse_line_graph(G)
        solution = nx.path_graph(3)
        assert nx.is_isomorphic(H, solution)

    def test_line(self):
        G = nx.path_graph(5)
        solution = nx.path_graph(6)
        H = nx.inverse_line_graph(G)
        assert nx.is_isomorphic(H, solution)

    def test_triangle_graph(self):
        G = nx.complete_graph(3)
        H = nx.inverse_line_graph(G)
        alternative_solution = nx.Graph()
        alternative_solution.add_edges_from([[0, 1], [0, 2], [0, 3]])
        # there are two alternative inverse line graphs for this case
        # so long as we get one of them the test should pass
        assert nx.is_isomorphic(H, G) or nx.is_isomorphic(H, alternative_solution)

    def test_cycle(self):
        G = nx.cycle_graph(5)
        H = nx.inverse_line_graph(G)
        assert nx.is_isomorphic(H, G)

    def test_empty(self):
        G = nx.Graph()
        H = nx.inverse_line_graph(G)
        assert nx.is_isomorphic(H, nx.complete_graph(1))

    def test_K1(self):
        G = nx.complete_graph(1)
        H = nx.inverse_line_graph(G)
        solution = nx.path_graph(2)
        assert nx.is_isomorphic(H, solution)

    def test_claw(self):
        # This is the simplest non-line graph
        G = nx.Graph()
        G_edges = [[0, 1], [0, 2], [0, 3]]
        G.add_edges_from(G_edges)
        pytest.raises(nx.NetworkXError, nx.inverse_line_graph, G)

    def test_non_line_graph(self):
        # These are other non-line graphs

        # wheel graph with 6 nodes
        G = nx.Graph()
        G_edges = [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [0, 5],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 1],
        ]
        G.add_edges_from(G_edges)
        pytest.raises(nx.NetworkXError, nx.inverse_line_graph, G)

        #   3---4---5
        #  / \ / \ /
        # 0---1---2
        G = nx.Graph()
        G_edges = [
            [0, 1],
            [1, 2],
            [3, 4],
            [4, 5],
            [0, 3],
            [1, 3],
            [1, 4],
            [2, 4],
            [2, 5],
        ]
        G.add_edges_from(G_edges)
        pytest.raises(nx.NetworkXError, nx.inverse_line_graph, G)

        # K_5 minus an edge
        K5me = nx.complete_graph(5)
        K5me.remove_edge(0, 1)
        pytest.raises(nx.NetworkXError, nx.inverse_line_graph, K5me)

    def test_wrong_graph_type(self):
        G = nx.DiGraph()
        G_edges = [[0, 1], [0, 2], [0, 3]]
        G.add_edges_from(G_edges)
        pytest.raises(nx.NetworkXNotImplemented, nx.inverse_line_graph, G)

        G = nx.MultiGraph()
        G_edges = [[0, 1], [0, 2], [0, 3]]
        G.add_edges_from(G_edges)
        pytest.raises(nx.NetworkXNotImplemented, nx.inverse_line_graph, G)

    def test_line_inverse_line_complete(self):
        G = nx.complete_graph(10)
        H = nx.line_graph(G)
        J = nx.inverse_line_graph(H)
        assert nx.is_isomorphic(G, J)

    def test_line_inverse_line_path(self):
        G = nx.path_graph(10)
        H = nx.line_graph(G)
        J = nx.inverse_line_graph(H)
        assert nx.is_isomorphic(G, J)

    def test_line_inverse_line_hypercube(self):
        G = nx.hypercube_graph(5)
        H = nx.line_graph(G)
        J = nx.inverse_line_graph(H)
        assert nx.is_isomorphic(G, J)

    def test_line_inverse_line_cycle(self):
        G = nx.cycle_graph(10)
        H = nx.line_graph(G)
        J = nx.inverse_line_graph(H)
        assert nx.is_isomorphic(G, J)

    def test_line_inverse_line_star(self):
        G = nx.star_graph(20)
        H = nx.line_graph(G)
        J = nx.inverse_line_graph(H)
        assert nx.is_isomorphic(G, J)

    def test_line_inverse_line_multipartite(self):
        G = nx.complete_multipartite_graph(3, 4, 5)
        H = nx.line_graph(G)
        J = nx.inverse_line_graph(H)
        assert nx.is_isomorphic(G, J)

    def test_line_inverse_line_dgm(self):
        G = nx.dorogovtsev_goltsev_mendes_graph(4)
        H = nx.line_graph(G)
        J = nx.inverse_line_graph(H)
        assert nx.is_isomorphic(G, J)
