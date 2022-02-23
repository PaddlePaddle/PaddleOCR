import networkx as nx
from networkx.testing import assert_graphs_equal, assert_edges_equal, assert_nodes_equal

# thanks to numpy for this GenericTest class (numpy/testing/test_utils.py)


class _GenericTest:
    @classmethod
    def _test_equal(cls, a, b):
        cls._assert_func(a, b)

    @classmethod
    def _test_not_equal(cls, a, b):
        try:
            cls._assert_func(a, b)
            passed = True
        except AssertionError:
            pass
        else:
            raise AssertionError("a and b are found equal but are not")


class TestNodesEqual(_GenericTest):
    _assert_func = assert_nodes_equal

    def test_nodes_equal(self):
        a = [1, 2, 5, 4]
        b = [4, 5, 1, 2]
        self._test_equal(a, b)

    def test_nodes_not_equal(self):
        a = [1, 2, 5, 4]
        b = [4, 5, 1, 3]
        self._test_not_equal(a, b)

    def test_nodes_with_data_equal(self):
        G = nx.Graph()
        G.add_nodes_from([1, 2, 3], color="red")
        H = nx.Graph()
        H.add_nodes_from([1, 2, 3], color="red")
        self._test_equal(G.nodes(data=True), H.nodes(data=True))

    def test_edges_with_data_not_equal(self):
        G = nx.Graph()
        G.add_nodes_from([1, 2, 3], color="red")
        H = nx.Graph()
        H.add_nodes_from([1, 2, 3], color="blue")
        self._test_not_equal(G.nodes(data=True), H.nodes(data=True))


class TestEdgesEqual(_GenericTest):
    _assert_func = assert_edges_equal

    def test_edges_equal(self):
        a = [(1, 2), (5, 4)]
        b = [(4, 5), (1, 2)]
        self._test_equal(a, b)

    def test_edges_not_equal(self):
        a = [(1, 2), (5, 4)]
        b = [(4, 5), (1, 3)]
        self._test_not_equal(a, b)

    def test_edges_with_data_equal(self):
        G = nx.MultiGraph()
        nx.add_path(G, [0, 1, 2], weight=1)
        H = nx.MultiGraph()
        nx.add_path(H, [0, 1, 2], weight=1)
        self._test_equal(G.edges(data=True, keys=True), H.edges(data=True, keys=True))

    def test_edges_with_data_not_equal(self):
        G = nx.MultiGraph()
        nx.add_path(G, [0, 1, 2], weight=1)
        H = nx.MultiGraph()
        nx.add_path(H, [0, 1, 2], weight=2)
        self._test_not_equal(
            G.edges(data=True, keys=True), H.edges(data=True, keys=True)
        )

    def test_no_edges(self):
        G = nx.MultiGraph()
        H = nx.MultiGraph()
        self._test_equal(G.edges(data=True, keys=True), H.edges(data=True, keys=True))

    def test_duplicate_edges(self):
        a = [(1, 2), (5, 4), (1, 2)]
        b = [(4, 5), (1, 2)]
        self._test_not_equal(a, b)

    def test_duplicate_edges_with_data(self):
        a = [(1, 2, {"weight": 10}), (5, 4), (1, 2, {"weight": 1})]
        b = [(4, 5), (1, 2), (1, 2, {"weight": 1})]
        self._test_not_equal(a, b)

    def test_order_of_edges_with_data(self):
        a = [(1, 2, {"weight": 10}), (1, 2, {"weight": 1})]
        b = [(1, 2, {"weight": 1}), (1, 2, {"weight": 10})]
        self._test_equal(a, b)

    def test_order_of_multiedges(self):
        wt1 = {"weight": 1}
        wt2 = {"weight": 2}
        a = [(1, 2, wt1), (1, 2, wt1), (1, 2, wt2)]
        b = [(1, 2, wt1), (1, 2, wt2), (1, 2, wt2)]
        self._test_not_equal(a, b)

    def test_order_of_edges_with_keys(self):
        a = [(1, 2, 0, {"weight": 10}), (1, 2, 1, {"weight": 1}), (1, 2, 2)]
        b = [(1, 2, 1, {"weight": 1}), (1, 2, 2), (1, 2, 0, {"weight": 10})]
        self._test_equal(a, b)
        a = [(1, 2, 1, {"weight": 10}), (1, 2, 0, {"weight": 1}), (1, 2, 2)]
        b = [(1, 2, 1, {"weight": 1}), (1, 2, 2), (1, 2, 0, {"weight": 10})]
        self._test_not_equal(a, b)


class TestGraphsEqual(_GenericTest):
    _assert_func = assert_graphs_equal

    def test_graphs_equal(self):
        G = nx.path_graph(4)
        H = nx.Graph()
        nx.add_path(H, range(4))
        self._test_equal(G, H)

    def test_digraphs_equal(self):
        G = nx.path_graph(4, create_using=nx.DiGraph())
        H = nx.DiGraph()
        nx.add_path(H, range(4))
        self._test_equal(G, H)

    def test_multigraphs_equal(self):
        G = nx.path_graph(4, create_using=nx.MultiGraph())
        H = nx.MultiGraph()
        nx.add_path(H, range(4))
        self._test_equal(G, H)

    def test_multidigraphs_equal(self):
        G = nx.path_graph(4, create_using=nx.MultiDiGraph())
        H = nx.MultiDiGraph()
        nx.add_path(H, range(4))
        self._test_equal(G, H)

    def test_graphs_not_equal(self):
        G = nx.path_graph(4)
        H = nx.Graph()
        nx.add_cycle(H, range(4))
        self._test_not_equal(G, H)

    def test_graphs_not_equal2(self):
        G = nx.path_graph(4)
        H = nx.Graph()
        nx.add_path(H, range(3))
        self._test_not_equal(G, H)

    def test_graphs_not_equal3(self):
        G = nx.path_graph(4)
        H = nx.Graph()
        nx.add_path(H, range(4))
        H.name = "path_graph(4)"
        self._test_not_equal(G, H)
