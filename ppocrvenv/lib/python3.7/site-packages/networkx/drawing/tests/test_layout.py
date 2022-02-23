"""Unit tests for layout functions."""
import networkx as nx

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")


class TestLayout:
    @classmethod
    def setup_class(cls):
        cls.Gi = nx.grid_2d_graph(5, 5)
        cls.Gs = nx.Graph()
        nx.add_path(cls.Gs, "abcdef")
        cls.bigG = nx.grid_2d_graph(25, 25)  # > 500 nodes for sparse

    @staticmethod
    def collect_node_distances(positions):
        distances = []
        prev_val = None
        for k in positions:
            if prev_val is not None:
                diff = positions[k] - prev_val
                distances.append((diff @ diff) ** 0.5)
            prev_val = positions[k]
        return distances

    def test_spring_fixed_without_pos(self):
        G = nx.path_graph(4)
        pytest.raises(ValueError, nx.spring_layout, G, fixed=[0])
        pos = {0: (1, 1), 2: (0, 0)}
        pytest.raises(ValueError, nx.spring_layout, G, fixed=[0, 1], pos=pos)
        nx.spring_layout(G, fixed=[0, 2], pos=pos)  # No ValueError

    def test_spring_init_pos(self):
        # Tests GH #2448
        import math

        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3)])

        init_pos = {0: (0.0, 0.0)}
        fixed_pos = [0]
        pos = nx.fruchterman_reingold_layout(G, pos=init_pos, fixed=fixed_pos)
        has_nan = any(math.isnan(c) for coords in pos.values() for c in coords)
        assert not has_nan, "values should not be nan"

    def test_smoke_empty_graph(self):
        G = []
        nx.random_layout(G)
        nx.circular_layout(G)
        nx.planar_layout(G)
        nx.spring_layout(G)
        nx.fruchterman_reingold_layout(G)
        nx.spectral_layout(G)
        nx.shell_layout(G)
        nx.bipartite_layout(G, G)
        nx.spiral_layout(G)
        nx.multipartite_layout(G)
        nx.kamada_kawai_layout(G)

    def test_smoke_int(self):
        G = self.Gi
        nx.random_layout(G)
        nx.circular_layout(G)
        nx.planar_layout(G)
        nx.spring_layout(G)
        nx.fruchterman_reingold_layout(G)
        nx.fruchterman_reingold_layout(self.bigG)
        nx.spectral_layout(G)
        nx.spectral_layout(G.to_directed())
        nx.spectral_layout(self.bigG)
        nx.spectral_layout(self.bigG.to_directed())
        nx.shell_layout(G)
        nx.spiral_layout(G)
        nx.kamada_kawai_layout(G)
        nx.kamada_kawai_layout(G, dim=1)
        nx.kamada_kawai_layout(G, dim=3)

    def test_smoke_string(self):
        G = self.Gs
        nx.random_layout(G)
        nx.circular_layout(G)
        nx.planar_layout(G)
        nx.spring_layout(G)
        nx.fruchterman_reingold_layout(G)
        nx.spectral_layout(G)
        nx.shell_layout(G)
        nx.spiral_layout(G)
        nx.kamada_kawai_layout(G)
        nx.kamada_kawai_layout(G, dim=1)
        nx.kamada_kawai_layout(G, dim=3)

    def check_scale_and_center(self, pos, scale, center):
        center = np.array(center)
        low = center - scale
        hi = center + scale
        vpos = np.array(list(pos.values()))
        length = vpos.max(0) - vpos.min(0)
        assert (length <= 2 * scale).all()
        assert (vpos >= low).all()
        assert (vpos <= hi).all()

    def test_scale_and_center_arg(self):
        sc = self.check_scale_and_center
        c = (4, 5)
        G = nx.complete_graph(9)
        G.add_node(9)
        sc(nx.random_layout(G, center=c), scale=0.5, center=(4.5, 5.5))
        # rest can have 2*scale length: [-scale, scale]
        sc(nx.spring_layout(G, scale=2, center=c), scale=2, center=c)
        sc(nx.spectral_layout(G, scale=2, center=c), scale=2, center=c)
        sc(nx.circular_layout(G, scale=2, center=c), scale=2, center=c)
        sc(nx.shell_layout(G, scale=2, center=c), scale=2, center=c)
        sc(nx.spiral_layout(G, scale=2, center=c), scale=2, center=c)
        sc(nx.kamada_kawai_layout(G, scale=2, center=c), scale=2, center=c)

        c = (2, 3, 5)
        sc(nx.kamada_kawai_layout(G, dim=3, scale=2, center=c), scale=2, center=c)

    def test_planar_layout_non_planar_input(self):
        G = nx.complete_graph(9)
        pytest.raises(nx.NetworkXException, nx.planar_layout, G)

    def test_smoke_planar_layout_embedding_input(self):
        embedding = nx.PlanarEmbedding()
        embedding.set_data({0: [1, 2], 1: [0, 2], 2: [0, 1]})
        nx.planar_layout(embedding)

    def test_default_scale_and_center(self):
        sc = self.check_scale_and_center
        c = (0, 0)
        G = nx.complete_graph(9)
        G.add_node(9)
        sc(nx.random_layout(G), scale=0.5, center=(0.5, 0.5))
        sc(nx.spring_layout(G), scale=1, center=c)
        sc(nx.spectral_layout(G), scale=1, center=c)
        sc(nx.circular_layout(G), scale=1, center=c)
        sc(nx.shell_layout(G), scale=1, center=c)
        sc(nx.spiral_layout(G), scale=1, center=c)
        sc(nx.kamada_kawai_layout(G), scale=1, center=c)

        c = (0, 0, 0)
        sc(nx.kamada_kawai_layout(G, dim=3), scale=1, center=c)

    def test_circular_planar_and_shell_dim_error(self):
        G = nx.path_graph(4)
        pytest.raises(ValueError, nx.circular_layout, G, dim=1)
        pytest.raises(ValueError, nx.shell_layout, G, dim=1)
        pytest.raises(ValueError, nx.shell_layout, G, dim=3)
        pytest.raises(ValueError, nx.planar_layout, G, dim=1)
        pytest.raises(ValueError, nx.planar_layout, G, dim=3)

    def test_adjacency_interface_numpy(self):
        A = nx.to_numpy_array(self.Gs)
        pos = nx.drawing.layout._fruchterman_reingold(A)
        assert pos.shape == (6, 2)
        pos = nx.drawing.layout._fruchterman_reingold(A, dim=3)
        assert pos.shape == (6, 3)
        pos = nx.drawing.layout._sparse_fruchterman_reingold(A)
        assert pos.shape == (6, 2)

    def test_adjacency_interface_scipy(self):
        A = nx.to_scipy_sparse_matrix(self.Gs, dtype="d")
        pos = nx.drawing.layout._sparse_fruchterman_reingold(A)
        assert pos.shape == (6, 2)
        pos = nx.drawing.layout._sparse_spectral(A)
        assert pos.shape == (6, 2)
        pos = nx.drawing.layout._sparse_fruchterman_reingold(A, dim=3)
        assert pos.shape == (6, 3)

    def test_single_nodes(self):
        G = nx.path_graph(1)
        vpos = nx.shell_layout(G)
        assert not vpos[0].any()
        G = nx.path_graph(4)
        vpos = nx.shell_layout(G, [[0], [1, 2], [3]])
        assert not vpos[0].any()
        assert vpos[3].any()  # ensure node 3 not at origin (#3188)
        assert np.linalg.norm(vpos[3]) <= 1  # ensure node 3 fits (#3753)
        vpos = nx.shell_layout(G, [[0], [1, 2], [3]], rotate=0)
        assert np.linalg.norm(vpos[3]) <= 1  # ensure node 3 fits (#3753)

    def test_smoke_initial_pos_fruchterman_reingold(self):
        pos = nx.circular_layout(self.Gi)
        npos = nx.fruchterman_reingold_layout(self.Gi, pos=pos)

    def test_fixed_node_fruchterman_reingold(self):
        # Dense version (numpy based)
        pos = nx.circular_layout(self.Gi)
        npos = nx.spring_layout(self.Gi, pos=pos, fixed=[(0, 0)])
        assert tuple(pos[(0, 0)]) == tuple(npos[(0, 0)])
        # Sparse version (scipy based)
        pos = nx.circular_layout(self.bigG)
        npos = nx.spring_layout(self.bigG, pos=pos, fixed=[(0, 0)])
        for axis in range(2):
            assert pos[(0, 0)][axis] == pytest.approx(npos[(0, 0)][axis], abs=1e-7)

    def test_center_parameter(self):
        G = nx.path_graph(1)
        nx.random_layout(G, center=(1, 1))
        vpos = nx.circular_layout(G, center=(1, 1))
        assert tuple(vpos[0]) == (1, 1)
        vpos = nx.planar_layout(G, center=(1, 1))
        assert tuple(vpos[0]) == (1, 1)
        vpos = nx.spring_layout(G, center=(1, 1))
        assert tuple(vpos[0]) == (1, 1)
        vpos = nx.fruchterman_reingold_layout(G, center=(1, 1))
        assert tuple(vpos[0]) == (1, 1)
        vpos = nx.spectral_layout(G, center=(1, 1))
        assert tuple(vpos[0]) == (1, 1)
        vpos = nx.shell_layout(G, center=(1, 1))
        assert tuple(vpos[0]) == (1, 1)
        vpos = nx.spiral_layout(G, center=(1, 1))
        assert tuple(vpos[0]) == (1, 1)

    def test_center_wrong_dimensions(self):
        G = nx.path_graph(1)
        assert id(nx.spring_layout) == id(nx.fruchterman_reingold_layout)
        pytest.raises(ValueError, nx.random_layout, G, center=(1, 1, 1))
        pytest.raises(ValueError, nx.circular_layout, G, center=(1, 1, 1))
        pytest.raises(ValueError, nx.planar_layout, G, center=(1, 1, 1))
        pytest.raises(ValueError, nx.spring_layout, G, center=(1, 1, 1))
        pytest.raises(ValueError, nx.spring_layout, G, dim=3, center=(1, 1))
        pytest.raises(ValueError, nx.spectral_layout, G, center=(1, 1, 1))
        pytest.raises(ValueError, nx.spectral_layout, G, dim=3, center=(1, 1))
        pytest.raises(ValueError, nx.shell_layout, G, center=(1, 1, 1))
        pytest.raises(ValueError, nx.spiral_layout, G, center=(1, 1, 1))
        pytest.raises(ValueError, nx.kamada_kawai_layout, G, center=(1, 1, 1))

    def test_empty_graph(self):
        G = nx.empty_graph()
        vpos = nx.random_layout(G, center=(1, 1))
        assert vpos == {}
        vpos = nx.circular_layout(G, center=(1, 1))
        assert vpos == {}
        vpos = nx.planar_layout(G, center=(1, 1))
        assert vpos == {}
        vpos = nx.bipartite_layout(G, G)
        assert vpos == {}
        vpos = nx.spring_layout(G, center=(1, 1))
        assert vpos == {}
        vpos = nx.fruchterman_reingold_layout(G, center=(1, 1))
        assert vpos == {}
        vpos = nx.spectral_layout(G, center=(1, 1))
        assert vpos == {}
        vpos = nx.shell_layout(G, center=(1, 1))
        assert vpos == {}
        vpos = nx.spiral_layout(G, center=(1, 1))
        assert vpos == {}
        vpos = nx.multipartite_layout(G, center=(1, 1))
        assert vpos == {}
        vpos = nx.kamada_kawai_layout(G, center=(1, 1))
        assert vpos == {}

    def test_bipartite_layout(self):
        G = nx.complete_bipartite_graph(3, 5)
        top, bottom = nx.bipartite.sets(G)

        vpos = nx.bipartite_layout(G, top)
        assert len(vpos) == len(G)

        top_x = vpos[list(top)[0]][0]
        bottom_x = vpos[list(bottom)[0]][0]
        for node in top:
            assert vpos[node][0] == top_x
        for node in bottom:
            assert vpos[node][0] == bottom_x

        vpos = nx.bipartite_layout(
            G, top, align="horizontal", center=(2, 2), scale=2, aspect_ratio=1
        )
        assert len(vpos) == len(G)

        top_y = vpos[list(top)[0]][1]
        bottom_y = vpos[list(bottom)[0]][1]
        for node in top:
            assert vpos[node][1] == top_y
        for node in bottom:
            assert vpos[node][1] == bottom_y

        pytest.raises(ValueError, nx.bipartite_layout, G, top, align="foo")

    def test_multipartite_layout(self):
        sizes = (0, 5, 7, 2, 8)
        G = nx.complete_multipartite_graph(*sizes)

        vpos = nx.multipartite_layout(G)
        assert len(vpos) == len(G)

        start = 0
        for n in sizes:
            end = start + n
            assert all(vpos[start][0] == vpos[i][0] for i in range(start + 1, end))
            start += n

        vpos = nx.multipartite_layout(G, align="horizontal", scale=2, center=(2, 2))
        assert len(vpos) == len(G)

        start = 0
        for n in sizes:
            end = start + n
            assert all(vpos[start][1] == vpos[i][1] for i in range(start + 1, end))
            start += n

        pytest.raises(ValueError, nx.multipartite_layout, G, align="foo")

    def test_kamada_kawai_costfn_1d(self):
        costfn = nx.drawing.layout._kamada_kawai_costfn

        pos = np.array([4.0, 7.0])
        invdist = 1 / np.array([[0.1, 2.0], [2.0, 0.3]])

        cost, grad = costfn(pos, np, invdist, meanweight=0, dim=1)

        assert cost == pytest.approx(((3 / 2.0 - 1) ** 2), abs=1e-7)
        assert grad[0] == pytest.approx((-0.5), abs=1e-7)
        assert grad[1] == pytest.approx(0.5, abs=1e-7)

    def check_kamada_kawai_costfn(self, pos, invdist, meanwt, dim):
        costfn = nx.drawing.layout._kamada_kawai_costfn

        cost, grad = costfn(pos.ravel(), np, invdist, meanweight=meanwt, dim=dim)

        expected_cost = 0.5 * meanwt * np.sum(np.sum(pos, axis=0) ** 2)
        for i in range(pos.shape[0]):
            for j in range(i + 1, pos.shape[0]):
                diff = np.linalg.norm(pos[i] - pos[j])
                expected_cost += (diff * invdist[i][j] - 1.0) ** 2

        assert cost == pytest.approx(expected_cost, abs=1e-7)

        dx = 1e-4
        for nd in range(pos.shape[0]):
            for dm in range(pos.shape[1]):
                idx = nd * pos.shape[1] + dm
                ps = pos.flatten()

                ps[idx] += dx
                cplus = costfn(ps, np, invdist, meanweight=meanwt, dim=pos.shape[1])[0]

                ps[idx] -= 2 * dx
                cminus = costfn(ps, np, invdist, meanweight=meanwt, dim=pos.shape[1])[0]

                assert grad[idx] == pytest.approx((cplus - cminus) / (2 * dx), abs=1e-5)

    def test_kamada_kawai_costfn(self):
        invdist = 1 / np.array([[0.1, 2.1, 1.7], [2.1, 0.2, 0.6], [1.7, 0.6, 0.3]])
        meanwt = 0.3

        # 2d
        pos = np.array([[1.3, -3.2], [2.7, -0.3], [5.1, 2.5]])

        self.check_kamada_kawai_costfn(pos, invdist, meanwt, 2)

        # 3d
        pos = np.array([[0.9, 8.6, -8.7], [-10, -0.5, -7.1], [9.1, -8.1, 1.6]])

        self.check_kamada_kawai_costfn(pos, invdist, meanwt, 3)

    def test_spiral_layout(self):

        G = self.Gs

        # a lower value of resolution should result in a more compact layout
        # intuitively, the total distance from the start and end nodes
        # via each node in between (transiting through each) will be less,
        # assuming rescaling does not occur on the computed node positions
        pos_standard = nx.spiral_layout(G, resolution=0.35)
        pos_tighter = nx.spiral_layout(G, resolution=0.34)
        distances = self.collect_node_distances(pos_standard)
        distances_tighter = self.collect_node_distances(pos_tighter)
        assert sum(distances) > sum(distances_tighter)

        # return near-equidistant points after the first value if set to true
        pos_equidistant = nx.spiral_layout(G, equidistant=True)
        distances_equidistant = self.collect_node_distances(pos_equidistant)
        for d in range(1, len(distances_equidistant) - 1):
            # test similarity to two decimal places
            assert distances_equidistant[d] == pytest.approx(
                distances_equidistant[d + 1], abs=1e-2
            )

    def test_rescale_layout_dict(self):
        G = nx.empty_graph()
        vpos = nx.random_layout(G, center=(1, 1))
        assert nx.rescale_layout_dict(vpos) == {}

        G = nx.empty_graph(2)
        vpos = {0: (0.0, 0.0), 1: (1.0, 1.0)}
        s_vpos = nx.rescale_layout_dict(vpos)
        assert np.linalg.norm([sum(x) for x in zip(*s_vpos.values())]) < 1e-6

        G = nx.empty_graph(3)
        vpos = {0: (0, 0), 1: (1, 1), 2: (0.5, 0.5)}
        s_vpos = nx.rescale_layout_dict(vpos)
        assert s_vpos == {0: (-1, -1), 1: (1, 1), 2: (0, 0)}
        s_vpos = nx.rescale_layout_dict(vpos, scale=2)
        assert s_vpos == {0: (-2, -2), 1: (2, 2), 2: (0, 0)}
