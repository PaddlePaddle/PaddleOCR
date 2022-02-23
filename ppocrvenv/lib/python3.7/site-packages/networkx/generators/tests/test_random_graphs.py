"""Unit tests for the :mod:`networkx.generators.random_graphs` module.

"""
import networkx as nx
import pytest


class TestGeneratorsRandom:
    def test_random_graph(self):
        seed = 42
        G = nx.gnp_random_graph(100, 0.25, seed)
        G = nx.gnp_random_graph(100, 0.25, seed, directed=True)
        G = nx.binomial_graph(100, 0.25, seed)
        G = nx.erdos_renyi_graph(100, 0.25, seed)
        G = nx.fast_gnp_random_graph(100, 0.25, seed)
        G = nx.fast_gnp_random_graph(100, 0.25, seed, directed=True)
        G = nx.gnm_random_graph(100, 20, seed)
        G = nx.gnm_random_graph(100, 20, seed, directed=True)
        G = nx.dense_gnm_random_graph(100, 20, seed)

        G = nx.watts_strogatz_graph(10, 2, 0.25, seed)
        assert len(G) == 10
        assert G.number_of_edges() == 10

        G = nx.connected_watts_strogatz_graph(10, 2, 0.1, tries=10, seed=seed)
        assert len(G) == 10
        assert G.number_of_edges() == 10
        pytest.raises(
            nx.NetworkXError, nx.connected_watts_strogatz_graph, 10, 2, 0.1, tries=0
        )

        G = nx.watts_strogatz_graph(10, 4, 0.25, seed)
        assert len(G) == 10
        assert G.number_of_edges() == 20

        G = nx.newman_watts_strogatz_graph(10, 2, 0.0, seed)
        assert len(G) == 10
        assert G.number_of_edges() == 10

        G = nx.newman_watts_strogatz_graph(10, 4, 0.25, seed)
        assert len(G) == 10
        assert G.number_of_edges() >= 20

        G = nx.barabasi_albert_graph(100, 1, seed)
        G = nx.barabasi_albert_graph(100, 3, seed)
        assert G.number_of_edges() == (97 * 3)

        G = nx.barabasi_albert_graph(100, 3, seed, nx.complete_graph(5))
        assert G.number_of_edges() == (10 + 95 * 3)

        G = nx.extended_barabasi_albert_graph(100, 1, 0, 0, seed)
        assert G.number_of_edges() == 99
        G = nx.extended_barabasi_albert_graph(100, 3, 0, 0, seed)
        assert G.number_of_edges() == 97 * 3
        G = nx.extended_barabasi_albert_graph(100, 1, 0, 0.5, seed)
        assert G.number_of_edges() == 99
        G = nx.extended_barabasi_albert_graph(100, 2, 0.5, 0, seed)
        assert G.number_of_edges() > 100 * 3
        assert G.number_of_edges() < 100 * 4

        G = nx.extended_barabasi_albert_graph(100, 2, 0.3, 0.3, seed)
        assert G.number_of_edges() > 100 * 2
        assert G.number_of_edges() < 100 * 4

        G = nx.powerlaw_cluster_graph(100, 1, 1.0, seed)
        G = nx.powerlaw_cluster_graph(100, 3, 0.0, seed)
        assert G.number_of_edges() == (97 * 3)

        G = nx.random_regular_graph(10, 20, seed)

        pytest.raises(nx.NetworkXError, nx.random_regular_graph, 3, 21)
        pytest.raises(nx.NetworkXError, nx.random_regular_graph, 33, 21)

        constructor = [(10, 20, 0.8), (20, 40, 0.8)]
        G = nx.random_shell_graph(constructor, seed)

        def is_caterpillar(g):
            """
            A tree is a caterpillar iff all nodes of degree >=3 are surrounded
            by at most two nodes of degree two or greater.
            ref: http://mathworld.wolfram.com/CaterpillarGraph.html
            """
            deg_over_3 = [n for n in g if g.degree(n) >= 3]
            for n in deg_over_3:
                nbh_deg_over_2 = [nbh for nbh in g.neighbors(n) if g.degree(nbh) >= 2]
                if not len(nbh_deg_over_2) <= 2:
                    return False
            return True

        def is_lobster(g):
            """
            A tree is a lobster if it has the property that the removal of leaf
            nodes leaves a caterpillar graph (Gallian 2007)
            ref: http://mathworld.wolfram.com/LobsterGraph.html
            """
            non_leafs = [n for n in g if g.degree(n) > 1]
            return is_caterpillar(g.subgraph(non_leafs))

        G = nx.random_lobster(10, 0.1, 0.5, seed)
        assert max([G.degree(n) for n in G.nodes()]) > 3
        assert is_lobster(G)
        pytest.raises(nx.NetworkXError, nx.random_lobster, 10, 0.1, 1, seed)
        pytest.raises(nx.NetworkXError, nx.random_lobster, 10, 1, 1, seed)
        pytest.raises(nx.NetworkXError, nx.random_lobster, 10, 1, 0.5, seed)

        # docstring says this should be a caterpillar
        G = nx.random_lobster(10, 0.1, 0.0, seed)
        assert is_caterpillar(G)

        # difficult to find seed that requires few tries
        seq = nx.random_powerlaw_tree_sequence(10, 3, seed=14, tries=1)
        G = nx.random_powerlaw_tree(10, 3, seed=14, tries=1)

    def test_dual_barabasi_albert(self, m1=1, m2=4, p=0.5):
        """
        Tests that the dual BA random graph generated behaves consistently.

        Tests the exceptions are raised as expected.

        The graphs generation are repeated several times to prevent lucky shots

        """
        seeds = [42, 314, 2718]
        initial_graph = nx.complete_graph(10)

        for seed in seeds:

            # This should be BA with m = m1
            BA1 = nx.barabasi_albert_graph(100, m1, seed)
            DBA1 = nx.dual_barabasi_albert_graph(100, m1, m2, 1, seed)
            assert BA1.edges() == DBA1.edges()

            # This should be BA with m = m2
            BA2 = nx.barabasi_albert_graph(100, m2, seed)
            DBA2 = nx.dual_barabasi_albert_graph(100, m1, m2, 0, seed)
            assert BA2.edges() == DBA2.edges()

            BA3 = nx.barabasi_albert_graph(100, m1, seed)
            DBA3 = nx.dual_barabasi_albert_graph(100, m1, m1, p, seed)
            # We can't compare edges here since randomness is "consumed" when drawing
            # between m1 and m2
            assert BA3.size() == DBA3.size()

            DBA = nx.dual_barabasi_albert_graph(100, m1, m2, p, seed, initial_graph)
            BA1 = nx.barabasi_albert_graph(100, m1, seed, initial_graph)
            BA2 = nx.barabasi_albert_graph(100, m2, seed, initial_graph)
            assert (
                min(BA1.size(), BA2.size()) <= DBA.size() <= max(BA1.size(), BA2.size())
            )

        # Testing exceptions
        dbag = nx.dual_barabasi_albert_graph
        pytest.raises(nx.NetworkXError, dbag, m1, m1, m2, 0)
        pytest.raises(nx.NetworkXError, dbag, m2, m1, m2, 0)
        pytest.raises(nx.NetworkXError, dbag, 100, m1, m2, -0.5)
        pytest.raises(nx.NetworkXError, dbag, 100, m1, m2, 1.5)
        initial = nx.complete_graph(max(m1, m2) - 1)
        pytest.raises(nx.NetworkXError, dbag, 100, m1, m2, p, initial_graph=initial)

    def test_extended_barabasi_albert(self, m=2):
        """
        Tests that the extended BA random graph generated behaves consistently.

        Tests the exceptions are raised as expected.

        The graphs generation are repeated several times to prevent lucky-shots

        """
        seeds = [42, 314, 2718]

        for seed in seeds:
            BA_model = nx.barabasi_albert_graph(100, m, seed)
            BA_model_edges = BA_model.number_of_edges()

            # This behaves just like BA, the number of edges must be the same
            G1 = nx.extended_barabasi_albert_graph(100, m, 0, 0, seed)
            assert G1.size() == BA_model_edges

            # More than twice more edges should have been added
            G1 = nx.extended_barabasi_albert_graph(100, m, 0.8, 0, seed)
            assert G1.size() > BA_model_edges * 2

            # Only edge rewiring, so the number of edges less than original
            G2 = nx.extended_barabasi_albert_graph(100, m, 0, 0.8, seed)
            assert G2.size() == BA_model_edges

            # Mixed scenario: less edges than G1 and more edges than G2
            G3 = nx.extended_barabasi_albert_graph(100, m, 0.3, 0.3, seed)
            assert G3.size() > G2.size()
            assert G3.size() < G1.size()

        # Testing exceptions
        ebag = nx.extended_barabasi_albert_graph
        pytest.raises(nx.NetworkXError, ebag, m, m, 0, 0)
        pytest.raises(nx.NetworkXError, ebag, 1, 0.5, 0, 0)
        pytest.raises(nx.NetworkXError, ebag, 100, 2, 0.5, 0.5)

    def test_random_zero_regular_graph(self):
        """Tests that a 0-regular graph has the correct number of nodes and
        edges.

        """
        seed = 42
        G = nx.random_regular_graph(0, 10, seed)
        assert len(G) == 10
        assert sum(1 for _ in G.edges()) == 0

    def test_gnp(self):
        for generator in [
            nx.gnp_random_graph,
            nx.binomial_graph,
            nx.erdos_renyi_graph,
            nx.fast_gnp_random_graph,
        ]:
            G = generator(10, -1.1)
            assert len(G) == 10
            assert sum(1 for _ in G.edges()) == 0

            G = generator(10, 0.1)
            assert len(G) == 10

            G = generator(10, 0.1, seed=42)
            assert len(G) == 10

            G = generator(10, 1.1)
            assert len(G) == 10
            assert sum(1 for _ in G.edges()) == 45

            G = generator(10, -1.1, directed=True)
            assert G.is_directed()
            assert len(G) == 10
            assert sum(1 for _ in G.edges()) == 0

            G = generator(10, 0.1, directed=True)
            assert G.is_directed()
            assert len(G) == 10

            G = generator(10, 1.1, directed=True)
            assert G.is_directed()
            assert len(G) == 10
            assert sum(1 for _ in G.edges()) == 90

            # assert that random graphs generate all edges for p close to 1
            edges = 0
            runs = 100
            for i in range(runs):
                edges += sum(1 for _ in generator(10, 0.99999, directed=True).edges())
            assert abs(edges / float(runs) - 90) <= runs * 2.0 / 100

    def test_gnm(self):
        G = nx.gnm_random_graph(10, 3)
        assert len(G) == 10
        assert sum(1 for _ in G.edges()) == 3

        G = nx.gnm_random_graph(10, 3, seed=42)
        assert len(G) == 10
        assert sum(1 for _ in G.edges()) == 3

        G = nx.gnm_random_graph(10, 100)
        assert len(G) == 10
        assert sum(1 for _ in G.edges()) == 45

        G = nx.gnm_random_graph(10, 100, directed=True)
        assert len(G) == 10
        assert sum(1 for _ in G.edges()) == 90

        G = nx.gnm_random_graph(10, -1.1)
        assert len(G) == 10
        assert sum(1 for _ in G.edges()) == 0

    def test_watts_strogatz_big_k(self):
        # Test to make sure than n <= k
        pytest.raises(nx.NetworkXError, nx.watts_strogatz_graph, 10, 11, 0.25)
        pytest.raises(nx.NetworkXError, nx.newman_watts_strogatz_graph, 10, 11, 0.25)

        # could create an infinite loop, now doesn't
        # infinite loop used to occur when a node has degree n-1 and needs to rewire
        nx.watts_strogatz_graph(10, 9, 0.25, seed=0)
        nx.newman_watts_strogatz_graph(10, 9, 0.5, seed=0)

        # Test k==n scenario
        nx.watts_strogatz_graph(10, 10, 0.25, seed=0)
        nx.newman_watts_strogatz_graph(10, 10, 0.25, seed=0)

    def test_random_kernel_graph(self):
        def integral(u, w, z):
            return c * (z - w)

        def root(u, w, r):
            return r / c + w

        c = 1
        graph = nx.random_kernel_graph(1000, integral, root)
        graph = nx.random_kernel_graph(1000, integral, root, seed=42)
        assert len(graph) == 1000
