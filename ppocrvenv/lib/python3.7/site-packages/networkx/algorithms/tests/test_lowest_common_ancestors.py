import pytest
from itertools import chain, combinations, product

import networkx as nx

tree_all_pairs_lca = nx.tree_all_pairs_lowest_common_ancestor
all_pairs_lca = nx.all_pairs_lowest_common_ancestor


def get_pair(dictionary, n1, n2):
    if (n1, n2) in dictionary:
        return dictionary[n1, n2]
    else:
        return dictionary[n2, n1]


class TestTreeLCA:
    @classmethod
    def setup_class(cls):
        cls.DG = nx.DiGraph()
        edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
        cls.DG.add_edges_from(edges)
        cls.ans = dict(tree_all_pairs_lca(cls.DG, 0))
        gold = {(n, n): n for n in cls.DG}
        gold.update({(0, i): 0 for i in range(1, 7)})
        gold.update(
            {
                (1, 2): 0,
                (1, 3): 1,
                (1, 4): 1,
                (1, 5): 0,
                (1, 6): 0,
                (2, 3): 0,
                (2, 4): 0,
                (2, 5): 2,
                (2, 6): 2,
                (3, 4): 1,
                (3, 5): 0,
                (3, 6): 0,
                (4, 5): 0,
                (4, 6): 0,
                (5, 6): 2,
            }
        )

        cls.gold = gold

    @staticmethod
    def assert_has_same_pairs(d1, d2):
        for (a, b) in ((min(pair), max(pair)) for pair in chain(d1, d2)):
            assert get_pair(d1, a, b) == get_pair(d2, a, b)

    def test_tree_all_pairs_lowest_common_ancestor1(self):
        """Specifying the root is optional."""
        assert dict(tree_all_pairs_lca(self.DG)) == self.ans

    def test_tree_all_pairs_lowest_common_ancestor2(self):
        """Specifying only some pairs gives only those pairs."""
        test_pairs = [(0, 1), (0, 1), (1, 0)]
        ans = dict(tree_all_pairs_lca(self.DG, 0, test_pairs))
        assert (0, 1) in ans and (1, 0) in ans
        assert len(ans) == 2

    def test_tree_all_pairs_lowest_common_ancestor3(self):
        """Specifying no pairs same as specifying all."""
        all_pairs = chain(combinations(self.DG, 2), ((node, node) for node in self.DG))

        ans = dict(tree_all_pairs_lca(self.DG, 0, all_pairs))
        self.assert_has_same_pairs(ans, self.ans)

    def test_tree_all_pairs_lowest_common_ancestor4(self):
        """Gives the right answer."""
        ans = dict(tree_all_pairs_lca(self.DG))
        self.assert_has_same_pairs(self.gold, ans)

    def test_tree_all_pairs_lowest_common_ancestor5(self):
        """Handles invalid input correctly."""
        empty_digraph = tree_all_pairs_lca(nx.DiGraph())
        pytest.raises(nx.NetworkXPointlessConcept, list, empty_digraph)

        bad_pairs_digraph = tree_all_pairs_lca(self.DG, pairs=[(-1, -2)])
        pytest.raises(nx.NodeNotFound, list, bad_pairs_digraph)

    def test_tree_all_pairs_lowest_common_ancestor6(self):
        """Works on subtrees."""
        ans = dict(tree_all_pairs_lca(self.DG, 1))
        gold = {
            pair: lca
            for (pair, lca) in self.gold.items()
            if all(n in (1, 3, 4) for n in pair)
        }
        self.assert_has_same_pairs(gold, ans)

    def test_tree_all_pairs_lowest_common_ancestor7(self):
        """Works on disconnected nodes."""
        G = nx.DiGraph()
        G.add_node(1)
        assert {(1, 1): 1} == dict(tree_all_pairs_lca(G))

        G.add_node(0)
        assert {(1, 1): 1} == dict(tree_all_pairs_lca(G, 1))
        assert {(0, 0): 0} == dict(tree_all_pairs_lca(G, 0))

        pytest.raises(nx.NetworkXError, list, tree_all_pairs_lca(G))

    def test_tree_all_pairs_lowest_common_ancestor8(self):
        """Raises right errors if not a tree."""
        # Cycle
        G = nx.DiGraph([(1, 2), (2, 1)])
        pytest.raises(nx.NetworkXError, list, tree_all_pairs_lca(G))
        # DAG
        G = nx.DiGraph([(0, 2), (1, 2)])
        pytest.raises(nx.NetworkXError, list, tree_all_pairs_lca(G))

    def test_tree_all_pairs_lowest_common_ancestor9(self):
        """Test that pairs works correctly as a generator."""
        pairs = iter([(0, 1), (0, 1), (1, 0)])
        some_pairs = dict(tree_all_pairs_lca(self.DG, 0, pairs))
        assert (0, 1) in some_pairs and (1, 0) in some_pairs
        assert len(some_pairs) == 2

    def test_tree_all_pairs_lowest_common_ancestor10(self):
        """Test that pairs not in the graph raises error."""
        lca = tree_all_pairs_lca(self.DG, 0, [(-1, -1)])
        pytest.raises(nx.NodeNotFound, list, lca)
        # check if node is None
        lca = tree_all_pairs_lca(self.DG, None, [(-1, -1)])
        pytest.raises(nx.NodeNotFound, list, lca)

    def test_tree_all_pairs_lowest_common_ancestor12(self):
        """Test that tree routine bails on DAGs."""
        G = nx.DiGraph([(3, 4), (5, 4)])
        pytest.raises(nx.NetworkXError, list, tree_all_pairs_lca(G))

    def test_not_implemented_for(self):
        NNI = nx.NetworkXNotImplemented
        G = nx.Graph([(0, 1)])
        with pytest.raises(NNI):
            next(tree_all_pairs_lca(G))
        with pytest.raises(NNI):
            next(all_pairs_lca(G))
        pytest.raises(NNI, nx.lowest_common_ancestor, G, 0, 1)
        G = nx.MultiGraph([(0, 1)])
        with pytest.raises(NNI):
            next(tree_all_pairs_lca(G))
        with pytest.raises(NNI):
            next(all_pairs_lca(G))
        pytest.raises(NNI, nx.lowest_common_ancestor, G, 0, 1)
        G = nx.MultiDiGraph([(0, 1)])
        with pytest.raises(NNI):
            next(tree_all_pairs_lca(G))
        with pytest.raises(NNI):
            next(all_pairs_lca(G))
        pytest.raises(NNI, nx.lowest_common_ancestor, G, 0, 1)

    def test_tree_all_pairs_lowest_common_ancestor13(self):
        """Test that it works on non-empty trees with no LCAs."""
        G = nx.DiGraph()
        G.add_node(3)
        ans = list(tree_all_pairs_lca(G))
        assert ans == [((3, 3), 3)]


class TestDAGLCA:
    @classmethod
    def setup_class(cls):
        cls.DG = nx.DiGraph()
        nx.add_path(cls.DG, (0, 1, 2, 3))
        nx.add_path(cls.DG, (0, 4, 3))
        nx.add_path(cls.DG, (0, 5, 6, 8, 3))
        nx.add_path(cls.DG, (5, 7, 8))
        cls.DG.add_edge(6, 2)
        cls.DG.add_edge(7, 2)

        cls.root_distance = nx.shortest_path_length(cls.DG, source=0)

        cls.gold = {
            (1, 1): 1,
            (1, 2): 1,
            (1, 3): 1,
            (1, 4): 0,
            (1, 5): 0,
            (1, 6): 0,
            (1, 7): 0,
            (1, 8): 0,
            (2, 2): 2,
            (2, 3): 2,
            (2, 4): 0,
            (2, 5): 5,
            (2, 6): 6,
            (2, 7): 7,
            (2, 8): 7,
            (3, 3): 8,
            (3, 4): 4,
            (3, 5): 5,
            (3, 6): 6,
            (3, 7): 7,
            (3, 8): 8,
            (4, 4): 4,
            (4, 5): 0,
            (4, 6): 0,
            (4, 7): 0,
            (4, 8): 0,
            (5, 5): 5,
            (5, 6): 5,
            (5, 7): 5,
            (5, 8): 5,
            (6, 6): 6,
            (6, 7): 5,
            (6, 8): 6,
            (7, 7): 7,
            (7, 8): 7,
            (8, 8): 8,
        }
        cls.gold.update(((0, n), 0) for n in cls.DG)

    def assert_lca_dicts_same(self, d1, d2, G=None):
        """Checks if d1 and d2 contain the same pairs and
        have a node at the same distance from root for each.
        If G is None use self.DG."""
        if G is None:
            G = self.DG
            root_distance = self.root_distance
        else:
            roots = [n for n, deg in G.in_degree if deg == 0]
            assert len(roots) == 1
            root_distance = nx.shortest_path_length(G, source=roots[0])

        for a, b in ((min(pair), max(pair)) for pair in chain(d1, d2)):
            assert (
                root_distance[get_pair(d1, a, b)] == root_distance[get_pair(d2, a, b)]
            )

    def test_all_pairs_lowest_common_ancestor1(self):
        """Produces the correct results."""
        self.assert_lca_dicts_same(dict(all_pairs_lca(self.DG)), self.gold)

    def test_all_pairs_lowest_common_ancestor2(self):
        """Produces the correct results when all pairs given."""
        all_pairs = list(product(self.DG.nodes(), self.DG.nodes()))
        ans = all_pairs_lca(self.DG, pairs=all_pairs)
        self.assert_lca_dicts_same(dict(ans), self.gold)

    def test_all_pairs_lowest_common_ancestor3(self):
        """Produces the correct results when all pairs given as a generator."""
        all_pairs = product(self.DG.nodes(), self.DG.nodes())
        ans = all_pairs_lca(self.DG, pairs=all_pairs)
        self.assert_lca_dicts_same(dict(ans), self.gold)

    def test_all_pairs_lowest_common_ancestor4(self):
        """Graph with two roots."""
        G = self.DG.copy()
        G.add_edge(9, 10)
        G.add_edge(9, 4)
        gold = self.gold.copy()
        gold[9, 9] = 9
        gold[9, 10] = 9
        gold[9, 4] = 9
        gold[9, 3] = 9
        gold[10, 4] = 9
        gold[10, 3] = 9
        gold[10, 10] = 10

        testing = dict(all_pairs_lca(G))

        G.add_edge(-1, 9)
        G.add_edge(-1, 0)
        self.assert_lca_dicts_same(testing, gold, G)

    def test_all_pairs_lowest_common_ancestor5(self):
        """Test that pairs not in the graph raises error."""
        pytest.raises(nx.NodeNotFound, all_pairs_lca, self.DG, [(-1, -1)])

    def test_all_pairs_lowest_common_ancestor6(self):
        """Test that pairs with no LCA specified emits nothing."""
        G = self.DG.copy()
        G.add_node(-1)
        gen = all_pairs_lca(G, [(-1, -1), (-1, 0)])
        assert dict(gen) == {(-1, -1): -1}

    def test_all_pairs_lowest_common_ancestor7(self):
        """Test that LCA on null graph bails."""
        pytest.raises(nx.NetworkXPointlessConcept, all_pairs_lca, nx.DiGraph())

    def test_all_pairs_lowest_common_ancestor8(self):
        """Test that LCA on non-dags bails."""
        pytest.raises(nx.NetworkXError, all_pairs_lca, nx.DiGraph([(3, 4), (4, 3)]))

    def test_all_pairs_lowest_common_ancestor9(self):
        """Test that it works on non-empty graphs with no LCAs."""
        G = nx.DiGraph()
        G.add_node(3)
        ans = list(all_pairs_lca(G))
        assert ans == [((3, 3), 3)]

    def test_lowest_common_ancestor1(self):
        """Test that the one-pair function works on default."""
        G = nx.DiGraph([(0, 1), (2, 1)])
        sentinel = object()
        assert nx.lowest_common_ancestor(G, 0, 2, default=sentinel) is sentinel

    def test_lowest_common_ancestor2(self):
        """Test that the one-pair function works on identity."""
        G = nx.DiGraph()
        G.add_node(3)
        assert nx.lowest_common_ancestor(G, 3, 3) == 3
