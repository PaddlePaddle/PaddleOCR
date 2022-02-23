from random import Random

import pytest


import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti


class TestDistance:
    def setup_method(self):
        G = cnlti(nx.grid_2d_graph(4, 4), first_label=1, ordering="sorted")
        self.G = G

    def test_eccentricity(self):
        assert nx.eccentricity(self.G, 1) == 6
        e = nx.eccentricity(self.G)
        assert e[1] == 6

        sp = dict(nx.shortest_path_length(self.G))
        e = nx.eccentricity(self.G, sp=sp)
        assert e[1] == 6

        e = nx.eccentricity(self.G, v=1)
        assert e == 6

        # This behavior changed in version 1.8 (ticket #739)
        e = nx.eccentricity(self.G, v=[1, 1])
        assert e[1] == 6
        e = nx.eccentricity(self.G, v=[1, 2])
        assert e[1] == 6

        # test against graph with one node
        G = nx.path_graph(1)
        e = nx.eccentricity(G)
        assert e[0] == 0
        e = nx.eccentricity(G, v=0)
        assert e == 0
        pytest.raises(nx.NetworkXError, nx.eccentricity, G, 1)

        # test against empty graph
        G = nx.empty_graph()
        e = nx.eccentricity(G)
        assert e == {}

    def test_diameter(self):
        assert nx.diameter(self.G) == 6

    def test_radius(self):
        assert nx.radius(self.G) == 4

    def test_periphery(self):
        assert set(nx.periphery(self.G)) == {1, 4, 13, 16}

    def test_center(self):
        assert set(nx.center(self.G)) == {6, 7, 10, 11}

    def test_bound_diameter(self):
        assert nx.diameter(self.G, usebounds=True) == 6

    def test_bound_radius(self):
        assert nx.radius(self.G, usebounds=True) == 4

    def test_bound_periphery(self):
        result = {1, 4, 13, 16}
        assert set(nx.periphery(self.G, usebounds=True)) == result

    def test_bound_center(self):
        result = {6, 7, 10, 11}
        assert set(nx.center(self.G, usebounds=True)) == result

    def test_radius_exception(self):
        G = nx.Graph()
        G.add_edge(1, 2)
        G.add_edge(3, 4)
        pytest.raises(nx.NetworkXError, nx.diameter, G)

    def test_eccentricity_infinite(self):
        with pytest.raises(nx.NetworkXError):
            G = nx.Graph([(1, 2), (3, 4)])
            e = nx.eccentricity(G)

    def test_eccentricity_undirected_not_connected(self):
        with pytest.raises(nx.NetworkXError):
            G = nx.Graph([(1, 2), (3, 4)])
            e = nx.eccentricity(G, sp=1)

    def test_eccentricity_directed_weakly_connected(self):
        with pytest.raises(nx.NetworkXError):
            DG = nx.DiGraph([(1, 2), (1, 3)])
            nx.eccentricity(DG)


class TestResistanceDistance:
    @classmethod
    def setup_class(cls):
        global np
        global sp
        np = pytest.importorskip("numpy")
        sp = pytest.importorskip("scipy")

    def setup_method(self):
        G = nx.Graph()
        G.add_edge(1, 2, weight=2)
        G.add_edge(2, 3, weight=4)
        G.add_edge(3, 4, weight=1)
        G.add_edge(1, 4, weight=3)
        self.G = G

    def test_laplacian_submatrix(self):
        from networkx.algorithms.distance_measures import _laplacian_submatrix

        M = sp.sparse.csr_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        N = sp.sparse.csr_matrix([[5, 6], [8, 9]], dtype=np.float32)
        Mn, Mn_nodelist = _laplacian_submatrix(1, M, [1, 2, 3])
        assert Mn_nodelist == [2, 3]
        assert np.allclose(Mn.toarray(), N.toarray())

    def test_laplacian_submatrix_square(self):
        with pytest.raises(nx.NetworkXError):
            from networkx.algorithms.distance_measures import _laplacian_submatrix

            M = sp.sparse.csr_matrix([[1, 2], [4, 5], [7, 8]], dtype=np.float32)
            _laplacian_submatrix(1, M, [1, 2, 3])

    def test_laplacian_submatrix_matrix_node_dim(self):
        with pytest.raises(nx.NetworkXError):
            from networkx.algorithms.distance_measures import _laplacian_submatrix

            M = sp.sparse.csr_matrix(
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32
            )
            _laplacian_submatrix(1, M, [1, 2, 3, 4])

    def test_resistance_distance(self):
        rd = nx.resistance_distance(self.G, 1, 3, "weight", True)
        test_data = 1 / (1 / (2 + 4) + 1 / (1 + 3))
        assert round(rd, 5) == round(test_data, 5)

    def test_resistance_distance_noinv(self):
        rd = nx.resistance_distance(self.G, 1, 3, "weight", False)
        test_data = 1 / (1 / (1 / 2 + 1 / 4) + 1 / (1 / 1 + 1 / 3))
        assert round(rd, 5) == round(test_data, 5)

    def test_resistance_distance_no_weight(self):
        rd = nx.resistance_distance(self.G, 1, 3)
        assert round(rd, 5) == 1

    def test_resistance_distance_neg_weight(self):
        self.G[2][3]["weight"] = -4
        rd = nx.resistance_distance(self.G, 1, 3, "weight", True)
        test_data = 1 / (1 / (2 + -4) + 1 / (1 + 3))
        assert round(rd, 5) == round(test_data, 5)

    def test_multigraph(self):
        G = nx.MultiGraph()
        G.add_edge(1, 2, weight=2)
        G.add_edge(2, 3, weight=4)
        G.add_edge(3, 4, weight=1)
        G.add_edge(1, 4, weight=3)
        rd = nx.resistance_distance(G, 1, 3, "weight", True)
        assert np.isclose(rd, 1 / (1 / (2 + 4) + 1 / (1 + 3)))

    def test_resistance_distance_div0(self):
        with pytest.raises(ZeroDivisionError):
            self.G[1][2]["weight"] = 0
            nx.resistance_distance(self.G, 1, 3, "weight")

    def test_resistance_distance_not_connected(self):
        with pytest.raises(nx.NetworkXError):
            self.G.add_node(5)
            nx.resistance_distance(self.G, 1, 5)

    def test_resistance_distance_same_node(self):
        with pytest.raises(nx.NetworkXError):
            nx.resistance_distance(self.G, 1, 1)

    def test_resistance_distance_nodeA_not_in_graph(self):
        with pytest.raises(nx.NetworkXError):
            nx.resistance_distance(self.G, 9, 1)

    def test_resistance_distance_nodeB_not_in_graph(self):
        with pytest.raises(nx.NetworkXError):
            nx.resistance_distance(self.G, 1, 9)


class TestBarycenter:
    """Test :func:`networkx.algorithms.distance_measures.barycenter`."""

    def barycenter_as_subgraph(self, g, **kwargs):
        """Return the subgraph induced on the barycenter of g"""
        b = nx.barycenter(g, **kwargs)
        assert isinstance(b, list)
        assert set(b) <= set(g)
        return g.subgraph(b)

    def test_must_be_connected(self):
        pytest.raises(nx.NetworkXNoPath, nx.barycenter, nx.empty_graph(5))

    def test_sp_kwarg(self):
        # Complete graph K_5. Normally it works...
        K_5 = nx.complete_graph(5)
        sp = dict(nx.shortest_path_length(K_5))
        assert nx.barycenter(K_5, sp=sp) == list(K_5)

        # ...but not with the weight argument
        for u, v, data in K_5.edges.data():
            data["weight"] = 1
        pytest.raises(ValueError, nx.barycenter, K_5, sp=sp, weight="weight")

        # ...and a corrupted sp can make it seem like K_5 is disconnected
        del sp[0][1]
        pytest.raises(nx.NetworkXNoPath, nx.barycenter, K_5, sp=sp)

    def test_trees(self):
        """The barycenter of a tree is a single vertex or an edge.

        See [West01]_, p. 78.
        """
        prng = Random(0xDEADBEEF)
        for i in range(50):
            RT = nx.random_tree(prng.randint(1, 75), prng)
            b = self.barycenter_as_subgraph(RT)
            if len(b) == 2:
                assert b.size() == 1
            else:
                assert len(b) == 1
                assert b.size() == 0

    def test_this_one_specific_tree(self):
        """Test the tree pictured at the bottom of [West01]_, p. 78."""
        g = nx.Graph(
            {
                "a": ["b"],
                "b": ["a", "x"],
                "x": ["b", "y"],
                "y": ["x", "z"],
                "z": ["y", 0, 1, 2, 3, 4],
                0: ["z"],
                1: ["z"],
                2: ["z"],
                3: ["z"],
                4: ["z"],
            }
        )
        b = self.barycenter_as_subgraph(g, attr="barycentricity")
        assert list(b) == ["z"]
        assert not b.edges
        expected_barycentricity = {
            0: 23,
            1: 23,
            2: 23,
            3: 23,
            4: 23,
            "a": 35,
            "b": 27,
            "x": 21,
            "y": 17,
            "z": 15,
        }
        for node, barycentricity in expected_barycentricity.items():
            assert g.nodes[node]["barycentricity"] == barycentricity

        # Doubling weights should do nothing but double the barycentricities
        for edge in g.edges:
            g.edges[edge]["weight"] = 2
        b = self.barycenter_as_subgraph(g, weight="weight", attr="barycentricity2")
        assert list(b) == ["z"]
        assert not b.edges
        for node, barycentricity in expected_barycentricity.items():
            assert g.nodes[node]["barycentricity2"] == barycentricity * 2
