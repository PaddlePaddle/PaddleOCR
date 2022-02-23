import pytest
import networkx as nx
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic

is_isomorphic = graph_could_be_isomorphic

"""Generators - Small
=====================

Some small graphs
"""

null = nx.null_graph()


class TestGeneratorsSmall:
    def test_make_small_graph(self):
        d = ["adjacencylist", "Bull Graph", 5, [[2, 3], [1, 3, 4], [1, 2, 5], [2], [3]]]
        G = nx.make_small_graph(d)
        assert is_isomorphic(G, nx.bull_graph())

        # Test small graph creation error with wrong ltype
        d[0] = "erroneouslist"
        pytest.raises(nx.NetworkXError, nx.make_small_graph, graph_description=d)

    def test__LCF_graph(self):
        # If n<=0, then return the null_graph
        G = nx.LCF_graph(-10, [1, 2], 100)
        assert is_isomorphic(G, null)
        G = nx.LCF_graph(0, [1, 2], 3)
        assert is_isomorphic(G, null)
        G = nx.LCF_graph(0, [1, 2], 10)
        assert is_isomorphic(G, null)

        # Test that LCF(n,[],0) == cycle_graph(n)
        for a, b, c in [(5, [], 0), (10, [], 0), (5, [], 1), (10, [], 10)]:
            G = nx.LCF_graph(a, b, c)
            assert is_isomorphic(G, nx.cycle_graph(a))

        # Generate the utility graph K_{3,3}
        G = nx.LCF_graph(6, [3, -3], 3)
        utility_graph = nx.complete_bipartite_graph(3, 3)
        assert is_isomorphic(G, utility_graph)

    def test_properties_named_small_graphs(self):
        G = nx.bull_graph()
        assert G.number_of_nodes() == 5
        assert G.number_of_edges() == 5
        assert sorted(d for n, d in G.degree()) == [1, 1, 2, 3, 3]
        assert nx.diameter(G) == 3
        assert nx.radius(G) == 2

        G = nx.chvatal_graph()
        assert G.number_of_nodes() == 12
        assert G.number_of_edges() == 24
        assert list(d for n, d in G.degree()) == 12 * [4]
        assert nx.diameter(G) == 2
        assert nx.radius(G) == 2

        G = nx.cubical_graph()
        assert G.number_of_nodes() == 8
        assert G.number_of_edges() == 12
        assert list(d for n, d in G.degree()) == 8 * [3]
        assert nx.diameter(G) == 3
        assert nx.radius(G) == 3

        G = nx.desargues_graph()
        assert G.number_of_nodes() == 20
        assert G.number_of_edges() == 30
        assert list(d for n, d in G.degree()) == 20 * [3]

        G = nx.diamond_graph()
        assert G.number_of_nodes() == 4
        assert sorted(d for n, d in G.degree()) == [2, 2, 3, 3]
        assert nx.diameter(G) == 2
        assert nx.radius(G) == 1

        G = nx.dodecahedral_graph()
        assert G.number_of_nodes() == 20
        assert G.number_of_edges() == 30
        assert list(d for n, d in G.degree()) == 20 * [3]
        assert nx.diameter(G) == 5
        assert nx.radius(G) == 5

        G = nx.frucht_graph()
        assert G.number_of_nodes() == 12
        assert G.number_of_edges() == 18
        assert list(d for n, d in G.degree()) == 12 * [3]
        assert nx.diameter(G) == 4
        assert nx.radius(G) == 3

        G = nx.heawood_graph()
        assert G.number_of_nodes() == 14
        assert G.number_of_edges() == 21
        assert list(d for n, d in G.degree()) == 14 * [3]
        assert nx.diameter(G) == 3
        assert nx.radius(G) == 3

        G = nx.hoffman_singleton_graph()
        assert G.number_of_nodes() == 50
        assert G.number_of_edges() == 175
        assert list(d for n, d in G.degree()) == 50 * [7]
        assert nx.diameter(G) == 2
        assert nx.radius(G) == 2

        G = nx.house_graph()
        assert G.number_of_nodes() == 5
        assert G.number_of_edges() == 6
        assert sorted(d for n, d in G.degree()) == [2, 2, 2, 3, 3]
        assert nx.diameter(G) == 2
        assert nx.radius(G) == 2

        G = nx.house_x_graph()
        assert G.number_of_nodes() == 5
        assert G.number_of_edges() == 8
        assert sorted(d for n, d in G.degree()) == [2, 3, 3, 4, 4]
        assert nx.diameter(G) == 2
        assert nx.radius(G) == 1

        G = nx.icosahedral_graph()
        assert G.number_of_nodes() == 12
        assert G.number_of_edges() == 30
        assert list(d for n, d in G.degree()) == [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        assert nx.diameter(G) == 3
        assert nx.radius(G) == 3

        G = nx.krackhardt_kite_graph()
        assert G.number_of_nodes() == 10
        assert G.number_of_edges() == 18
        assert sorted(d for n, d in G.degree()) == [1, 2, 3, 3, 3, 4, 4, 5, 5, 6]

        G = nx.moebius_kantor_graph()
        assert G.number_of_nodes() == 16
        assert G.number_of_edges() == 24
        assert list(d for n, d in G.degree()) == 16 * [3]
        assert nx.diameter(G) == 4

        G = nx.octahedral_graph()
        assert G.number_of_nodes() == 6
        assert G.number_of_edges() == 12
        assert list(d for n, d in G.degree()) == 6 * [4]
        assert nx.diameter(G) == 2
        assert nx.radius(G) == 2

        G = nx.pappus_graph()
        assert G.number_of_nodes() == 18
        assert G.number_of_edges() == 27
        assert list(d for n, d in G.degree()) == 18 * [3]
        assert nx.diameter(G) == 4

        G = nx.petersen_graph()
        assert G.number_of_nodes() == 10
        assert G.number_of_edges() == 15
        assert list(d for n, d in G.degree()) == 10 * [3]
        assert nx.diameter(G) == 2
        assert nx.radius(G) == 2

        G = nx.sedgewick_maze_graph()
        assert G.number_of_nodes() == 8
        assert G.number_of_edges() == 10
        assert sorted(d for n, d in G.degree()) == [1, 2, 2, 2, 3, 3, 3, 4]

        G = nx.tetrahedral_graph()
        assert G.number_of_nodes() == 4
        assert G.number_of_edges() == 6
        assert list(d for n, d in G.degree()) == [3, 3, 3, 3]
        assert nx.diameter(G) == 1
        assert nx.radius(G) == 1

        G = nx.truncated_cube_graph()
        assert G.number_of_nodes() == 24
        assert G.number_of_edges() == 36
        assert list(d for n, d in G.degree()) == 24 * [3]

        G = nx.truncated_tetrahedron_graph()
        assert G.number_of_nodes() == 12
        assert G.number_of_edges() == 18
        assert list(d for n, d in G.degree()) == 12 * [3]

        G = nx.tutte_graph()
        assert G.number_of_nodes() == 46
        assert G.number_of_edges() == 69
        assert list(d for n, d in G.degree()) == 46 * [3]

        # Test create_using with directed or multigraphs on small graphs
        pytest.raises(nx.NetworkXError, nx.tutte_graph, create_using=nx.DiGraph)
        MG = nx.tutte_graph(create_using=nx.MultiGraph)
        assert sorted(MG.edges()) == sorted(G.edges())
