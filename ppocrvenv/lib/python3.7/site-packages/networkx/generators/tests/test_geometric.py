from itertools import combinations
from math import sqrt
import random


import networkx as nx
from networkx.generators.geometric import euclidean


def l1dist(x, y):
    return sum(abs(a - b) for a, b in zip(x, y))


class TestRandomGeometricGraph:
    """Unit tests for the :func:`~networkx.random_geometric_graph`
    function.

    """

    def test_number_of_nodes(self):
        G = nx.random_geometric_graph(50, 0.25, seed=42)
        assert len(G) == 50
        G = nx.random_geometric_graph(range(50), 0.25, seed=42)
        assert len(G) == 50

    def test_distances(self):
        """Tests that pairs of vertices adjacent if and only if they are
        within the prescribed radius.

        """
        # Use the Euclidean metric, the default according to the
        # documentation.
        dist = euclidean
        G = nx.random_geometric_graph(50, 0.25)
        for u, v in combinations(G, 2):
            # Adjacent vertices must be within the given distance.
            if v in G[u]:
                assert dist(G.nodes[u]["pos"], G.nodes[v]["pos"]) <= 0.25
            # Nonadjacent vertices must be at greater distance.
            else:
                assert not dist(G.nodes[u]["pos"], G.nodes[v]["pos"]) <= 0.25

    def test_p(self):
        """Tests for providing an alternate distance metric to the
        generator.

        """
        # Use the L1 metric.
        dist = l1dist
        G = nx.random_geometric_graph(50, 0.25, p=1)
        for u, v in combinations(G, 2):
            # Adjacent vertices must be within the given distance.
            if v in G[u]:
                assert dist(G.nodes[u]["pos"], G.nodes[v]["pos"]) <= 0.25
            # Nonadjacent vertices must be at greater distance.
            else:
                assert not dist(G.nodes[u]["pos"], G.nodes[v]["pos"]) <= 0.25

    def test_node_names(self):
        """Tests using values other than sequential numbers as node IDs."""
        import string

        nodes = list(string.ascii_lowercase)
        G = nx.random_geometric_graph(nodes, 0.25)
        assert len(G) == len(nodes)

        dist = euclidean
        for u, v in combinations(G, 2):
            # Adjacent vertices must be within the given distance.
            if v in G[u]:
                assert dist(G.nodes[u]["pos"], G.nodes[v]["pos"]) <= 0.25
            # Nonadjacent vertices must be at greater distance.
            else:
                assert not dist(G.nodes[u]["pos"], G.nodes[v]["pos"]) <= 0.25


class TestSoftRandomGeometricGraph:
    """Unit tests for the :func:`~networkx.soft_random_geometric_graph`
    function.

    """

    def test_number_of_nodes(self):
        G = nx.soft_random_geometric_graph(50, 0.25, seed=42)
        assert len(G) == 50
        G = nx.soft_random_geometric_graph(range(50), 0.25, seed=42)
        assert len(G) == 50

    def test_distances(self):
        """Tests that pairs of vertices adjacent if and only if they are
        within the prescribed radius.

        """
        # Use the Euclidean metric, the default according to the
        # documentation.
        def dist(x, y):
            return sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))

        G = nx.soft_random_geometric_graph(50, 0.25)
        for u, v in combinations(G, 2):
            # Adjacent vertices must be within the given distance.
            if v in G[u]:
                assert dist(G.nodes[u]["pos"], G.nodes[v]["pos"]) <= 0.25

    def test_p(self):
        """Tests for providing an alternate distance metric to the
        generator.

        """
        # Use the L1 metric.
        def dist(x, y):
            return sum(abs(a - b) for a, b in zip(x, y))

        G = nx.soft_random_geometric_graph(50, 0.25, p=1)
        for u, v in combinations(G, 2):
            # Adjacent vertices must be within the given distance.
            if v in G[u]:
                assert dist(G.nodes[u]["pos"], G.nodes[v]["pos"]) <= 0.25

    def test_node_names(self):
        """Tests using values other than sequential numbers as node IDs."""
        import string

        nodes = list(string.ascii_lowercase)
        G = nx.soft_random_geometric_graph(nodes, 0.25)
        assert len(G) == len(nodes)

        def dist(x, y):
            return sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))

        for u, v in combinations(G, 2):
            # Adjacent vertices must be within the given distance.
            if v in G[u]:
                assert dist(G.nodes[u]["pos"], G.nodes[v]["pos"]) <= 0.25

    def test_p_dist_default(self):
        """Tests default p_dict = 0.5 returns graph with edge count <= RGG with
        same n, radius, dim and positions

        """
        nodes = 50
        dim = 2
        pos = {v: [random.random() for i in range(dim)] for v in range(nodes)}
        RGG = nx.random_geometric_graph(50, 0.25, pos=pos)
        SRGG = nx.soft_random_geometric_graph(50, 0.25, pos=pos)
        assert len(SRGG.edges()) <= len(RGG.edges())

    def test_p_dist_zero(self):
        """Tests if p_dict = 0 returns disconencted graph with 0 edges"""

        def p_dist(dist):
            return 0

        G = nx.soft_random_geometric_graph(50, 0.25, p_dist=p_dist)
        assert len(G.edges) == 0


def join(G, u, v, theta, alpha, metric):
    """Returns ``True`` if and only if the nodes whose attributes are
    ``du`` and ``dv`` should be joined, according to the threshold
    condition for geographical threshold graphs.

    ``G`` is an undirected NetworkX graph, and ``u`` and ``v`` are nodes
    in that graph. The nodes must have node attributes ``'pos'`` and
    ``'weight'``.

    ``metric`` is a distance metric.

    """
    du, dv = G.nodes[u], G.nodes[v]
    u_pos, v_pos = du["pos"], dv["pos"]
    u_weight, v_weight = du["weight"], dv["weight"]
    return (u_weight + v_weight) * metric(u_pos, v_pos) ** alpha >= theta


class TestGeographicalThresholdGraph:
    """Unit tests for the :func:`~networkx.geographical_threshold_graph`
    function.

    """

    def test_number_of_nodes(self):
        G = nx.geographical_threshold_graph(50, 100, seed=42)
        assert len(G) == 50
        G = nx.geographical_threshold_graph(range(50), 100, seed=42)
        assert len(G) == 50

    def test_distances(self):
        """Tests that pairs of vertices adjacent if and only if their
        distances meet the given threshold.

        """
        # Use the Euclidean metric and alpha = -2
        # the default according to the documentation.
        dist = euclidean
        G = nx.geographical_threshold_graph(50, 10)
        for u, v in combinations(G, 2):
            # Adjacent vertices must exceed the threshold.
            if v in G[u]:
                assert join(G, u, v, 10, -2, dist)
            # Nonadjacent vertices must not exceed the threshold.
            else:
                assert not join(G, u, v, 10, -2, dist)

    def test_metric(self):
        """Tests for providing an alternate distance metric to the
        generator.

        """
        # Use the L1 metric.
        dist = l1dist
        G = nx.geographical_threshold_graph(50, 10, metric=dist)
        for u, v in combinations(G, 2):
            # Adjacent vertices must exceed the threshold.
            if v in G[u]:
                assert join(G, u, v, 10, -2, dist)
            # Nonadjacent vertices must not exceed the threshold.
            else:
                assert not join(G, u, v, 10, -2, dist)

    def test_p_dist_zero(self):
        """Tests if p_dict = 0 returns disconencted graph with 0 edges"""

        def p_dist(dist):
            return 0

        G = nx.geographical_threshold_graph(50, 1, p_dist=p_dist)
        assert len(G.edges) == 0


class TestWaxmanGraph:
    """Unit tests for the :func:`~networkx.waxman_graph` function."""

    def test_number_of_nodes_1(self):
        G = nx.waxman_graph(50, 0.5, 0.1, seed=42)
        assert len(G) == 50
        G = nx.waxman_graph(range(50), 0.5, 0.1, seed=42)
        assert len(G) == 50

    def test_number_of_nodes_2(self):
        G = nx.waxman_graph(50, 0.5, 0.1, L=1)
        assert len(G) == 50
        G = nx.waxman_graph(range(50), 0.5, 0.1, L=1)
        assert len(G) == 50

    def test_metric(self):
        """Tests for providing an alternate distance metric to the
        generator.

        """
        # Use the L1 metric.
        dist = l1dist
        G = nx.waxman_graph(50, 0.5, 0.1, metric=dist)
        assert len(G) == 50


class TestNavigableSmallWorldGraph:
    def test_navigable_small_world(self):
        G = nx.navigable_small_world_graph(5, p=1, q=0, seed=42)
        gg = nx.grid_2d_graph(5, 5).to_directed()
        assert nx.is_isomorphic(G, gg)

        G = nx.navigable_small_world_graph(5, p=1, q=0, dim=3)
        gg = nx.grid_graph([5, 5, 5]).to_directed()
        assert nx.is_isomorphic(G, gg)

        G = nx.navigable_small_world_graph(5, p=1, q=0, dim=1)
        gg = nx.grid_graph([5]).to_directed()
        assert nx.is_isomorphic(G, gg)


class TestThresholdedRandomGeometricGraph:
    """Unit tests for the :func:`~networkx.thresholded_random_geometric_graph`
    function.

    """

    def test_number_of_nodes(self):
        G = nx.thresholded_random_geometric_graph(50, 0.2, 0.1, seed=42)
        assert len(G) == 50
        G = nx.thresholded_random_geometric_graph(range(50), 0.2, 0.1)
        assert len(G) == 50

    def test_distances(self):
        """Tests that pairs of vertices adjacent if and only if they are
        within the prescribed radius.

        """
        # Use the Euclidean metric, the default according to the
        # documentation.
        def dist(x, y):
            return sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))

        G = nx.thresholded_random_geometric_graph(50, 0.25, 0.1)
        for u, v in combinations(G, 2):
            # Adjacent vertices must be within the given distance.
            if v in G[u]:
                assert dist(G.nodes[u]["pos"], G.nodes[v]["pos"]) <= 0.25

    def test_p(self):
        """Tests for providing an alternate distance metric to the
        generator.

        """
        # Use the L1 metric.
        def dist(x, y):
            return sum(abs(a - b) for a, b in zip(x, y))

        G = nx.thresholded_random_geometric_graph(50, 0.25, 0.1, p=1)
        for u, v in combinations(G, 2):
            # Adjacent vertices must be within the given distance.
            if v in G[u]:
                assert dist(G.nodes[u]["pos"], G.nodes[v]["pos"]) <= 0.25

    def test_node_names(self):
        """Tests using values other than sequential numbers as node IDs."""
        import string

        nodes = list(string.ascii_lowercase)
        G = nx.thresholded_random_geometric_graph(nodes, 0.25, 0.1)
        assert len(G) == len(nodes)

        def dist(x, y):
            return sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))

        for u, v in combinations(G, 2):
            # Adjacent vertices must be within the given distance.
            if v in G[u]:
                assert dist(G.nodes[u]["pos"], G.nodes[v]["pos"]) <= 0.25

    def test_theta(self):
        """Tests that pairs of vertices adjacent if and only if their sum
        weights exceeds the threshold parameter theta.
        """
        G = nx.thresholded_random_geometric_graph(50, 0.25, 0.1)

        for u, v in combinations(G, 2):
            # Adjacent vertices must be within the given distance.
            if v in G[u]:
                assert (G.nodes[u]["weight"] + G.nodes[v]["weight"]) >= 0.1
