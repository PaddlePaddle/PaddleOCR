import numpy as np
import itertools
from numpy.testing import (assert_equal,
                           assert_almost_equal,
                           assert_array_equal,
                           assert_array_almost_equal,
                           suppress_warnings)
import pytest
from pytest import raises as assert_raises
from pytest import warns as assert_warns
from scipy.spatial import SphericalVoronoi, distance
from scipy.optimize import linear_sum_assignment
from scipy.constants import golden as phi
from scipy.special import gamma


TOL = 1E-10


def _generate_tetrahedron():
    return np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])


def _generate_cube():
    return np.array(list(itertools.product([-1, 1.], repeat=3)))


def _generate_octahedron():
    return np.array([[-1, 0, 0], [+1, 0, 0], [0, -1, 0],
                     [0, +1, 0], [0, 0, -1], [0, 0, +1]])


def _generate_dodecahedron():

    x1 = _generate_cube()
    x2 = np.array([[0, -phi, -1 / phi],
                   [0, -phi, +1 / phi],
                   [0, +phi, -1 / phi],
                   [0, +phi, +1 / phi]])
    x3 = np.array([[-1 / phi, 0, -phi],
                   [+1 / phi, 0, -phi],
                   [-1 / phi, 0, +phi],
                   [+1 / phi, 0, +phi]])
    x4 = np.array([[-phi, -1 / phi, 0],
                   [-phi, +1 / phi, 0],
                   [+phi, -1 / phi, 0],
                   [+phi, +1 / phi, 0]])
    return np.concatenate((x1, x2, x3, x4))


def _generate_icosahedron():
    x = np.array([[0, -1, -phi],
                  [0, -1, +phi],
                  [0, +1, -phi],
                  [0, +1, +phi]])
    return np.concatenate([np.roll(x, i, axis=1) for i in range(3)])


def _generate_polytope(name):
    polygons = ["triangle", "square", "pentagon", "hexagon", "heptagon",
                "octagon", "nonagon", "decagon", "undecagon", "dodecagon"]
    polyhedra = ["tetrahedron", "cube", "octahedron", "dodecahedron",
                 "icosahedron"]
    if name not in polygons and name not in polyhedra:
        raise ValueError("unrecognized polytope")

    if name in polygons:
        n = polygons.index(name) + 3
        thetas = np.linspace(0, 2 * np.pi, n, endpoint=False)
        p = np.vstack([np.cos(thetas), np.sin(thetas)]).T
    elif name == "tetrahedron":
        p = _generate_tetrahedron()
    elif name == "cube":
        p = _generate_cube()
    elif name == "octahedron":
        p = _generate_octahedron()
    elif name == "dodecahedron":
        p = _generate_dodecahedron()
    elif name == "icosahedron":
        p = _generate_icosahedron()

    return p / np.linalg.norm(p, axis=1, keepdims=True)


def _hypersphere_area(dim, radius):
    # https://en.wikipedia.org/wiki/N-sphere#Closed_forms
    return 2 * np.pi**(dim / 2) / gamma(dim / 2) * radius**(dim - 1)


def _sample_sphere(n, dim, seed=None):
    # Sample points uniformly at random from the hypersphere
    rng = np.random.RandomState(seed=seed)
    points = rng.randn(n, dim)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points


class TestSphericalVoronoi:

    def setup_method(self):
        self.points = np.array([
            [-0.78928481, -0.16341094, 0.59188373],
            [-0.66839141, 0.73309634, 0.12578818],
            [0.32535778, -0.92476944, -0.19734181],
            [-0.90177102, -0.03785291, -0.43055335],
            [0.71781344, 0.68428936, 0.12842096],
            [-0.96064876, 0.23492353, -0.14820556],
            [0.73181537, -0.22025898, -0.6449281],
            [0.79979205, 0.54555747, 0.25039913]]
        )

    def test_constructor(self):
        center = np.array([1, 2, 3])
        radius = 2
        s1 = SphericalVoronoi(self.points)
        # user input checks in SphericalVoronoi now require
        # the radius / center to match the generators so adjust
        # accordingly here
        s2 = SphericalVoronoi(self.points * radius, radius)
        s3 = SphericalVoronoi(self.points + center, center=center)
        s4 = SphericalVoronoi(self.points * radius + center, radius, center)
        assert_array_equal(s1.center, np.array([0, 0, 0]))
        assert_equal(s1.radius, 1)
        assert_array_equal(s2.center, np.array([0, 0, 0]))
        assert_equal(s2.radius, 2)
        assert_array_equal(s3.center, center)
        assert_equal(s3.radius, 1)
        assert_array_equal(s4.center, center)
        assert_equal(s4.radius, radius)

    def test_vertices_regions_translation_invariance(self):
        sv_origin = SphericalVoronoi(self.points)
        center = np.array([1, 1, 1])
        sv_translated = SphericalVoronoi(self.points + center, center=center)
        assert_equal(sv_origin.regions, sv_translated.regions)
        assert_array_almost_equal(sv_origin.vertices + center,
                                  sv_translated.vertices)

    def test_vertices_regions_scaling_invariance(self):
        sv_unit = SphericalVoronoi(self.points)
        sv_scaled = SphericalVoronoi(self.points * 2, 2)
        assert_equal(sv_unit.regions, sv_scaled.regions)
        assert_array_almost_equal(sv_unit.vertices * 2,
                                  sv_scaled.vertices)

    def test_old_radius_api(self):
        sv_unit = SphericalVoronoi(self.points, radius=1)
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "`radius` is `None`")
            sv = SphericalVoronoi(self.points, None)
            assert_array_almost_equal(sv_unit.vertices, sv.vertices)

    def test_old_radius_api_warning(self):
        with assert_warns(DeprecationWarning):
            SphericalVoronoi(self.points, None)

    def test_sort_vertices_of_regions(self):
        sv = SphericalVoronoi(self.points)
        unsorted_regions = sv.regions
        sv.sort_vertices_of_regions()
        assert_equal(sorted(sv.regions), sorted(unsorted_regions))

    def test_sort_vertices_of_regions_flattened(self):
        expected = sorted([[0, 6, 5, 2, 3], [2, 3, 10, 11, 8, 7], [0, 6, 4, 1],
                           [4, 8, 7, 5, 6], [9, 11, 10], [2, 7, 5],
                           [1, 4, 8, 11, 9], [0, 3, 10, 9, 1]])
        expected = list(itertools.chain(*sorted(expected)))  # type: ignore
        sv = SphericalVoronoi(self.points)
        sv.sort_vertices_of_regions()
        actual = list(itertools.chain(*sorted(sv.regions)))
        assert_array_equal(actual, expected)

    def test_sort_vertices_of_regions_dimensionality(self):
        points = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1],
                           [0.5, 0.5, 0.5, 0.5]])
        with pytest.raises(TypeError, match="three-dimensional"):
            sv = SphericalVoronoi(points)
            sv.sort_vertices_of_regions()

    def test_num_vertices(self):
        # for any n >= 3, a spherical Voronoi diagram has 2n - 4
        # vertices; this is a direct consequence of Euler's formula
        # as explained by Dinis and Mamede (2010) Proceedings of the
        # 2010 International Symposium on Voronoi Diagrams in Science
        # and Engineering
        sv = SphericalVoronoi(self.points)
        expected = self.points.shape[0] * 2 - 4
        actual = sv.vertices.shape[0]
        assert_equal(actual, expected)

    def test_voronoi_circles(self):
        sv = SphericalVoronoi(self.points)
        for vertex in sv.vertices:
            distances = distance.cdist(sv.points, np.array([vertex]))
            closest = np.array(sorted(distances)[0:3])
            assert_almost_equal(closest[0], closest[1], 7, str(vertex))
            assert_almost_equal(closest[0], closest[2], 7, str(vertex))

    def test_duplicate_point_handling(self):
        # an exception should be raised for degenerate generators
        # related to Issue# 7046
        self.degenerate = np.concatenate((self.points, self.points))
        with assert_raises(ValueError):
            SphericalVoronoi(self.degenerate)

    def test_incorrect_radius_handling(self):
        # an exception should be raised if the radius provided
        # cannot possibly match the input generators
        with assert_raises(ValueError):
            SphericalVoronoi(self.points, radius=0.98)

    def test_incorrect_center_handling(self):
        # an exception should be raised if the center provided
        # cannot possibly match the input generators
        with assert_raises(ValueError):
            SphericalVoronoi(self.points, center=[0.1, 0, 0])

    @pytest.mark.parametrize("dim", range(2, 6))
    @pytest.mark.parametrize("shift", [False, True])
    def test_single_hemisphere_handling(self, dim, shift):
        n = 10
        points = _sample_sphere(n, dim, seed=0)
        points[:, 0] = np.abs(points[:, 0])
        center = (np.arange(dim) + 1) * shift
        sv = SphericalVoronoi(points + center, center=center)
        dots = np.einsum('ij,ij->i', sv.vertices - center,
                                     sv.points[sv._simplices[:, 0]] - center)
        circumradii = np.arccos(np.clip(dots, -1, 1))
        assert np.max(circumradii) > np.pi / 2

    @pytest.mark.parametrize("n", [1, 2, 10])
    @pytest.mark.parametrize("dim", range(2, 6))
    @pytest.mark.parametrize("shift", [False, True])
    def test_rank_deficient(self, n, dim, shift):
        center = (np.arange(dim) + 1) * shift
        points = _sample_sphere(n, dim - 1, seed=0)
        points = np.hstack([points, np.zeros((n, 1))])
        with pytest.raises(ValueError, match="Rank of input points"):
            SphericalVoronoi(points + center, center=center)

    @pytest.mark.parametrize("dim", range(2, 6))
    def test_higher_dimensions(self, dim):
        n = 100
        points = _sample_sphere(n, dim, seed=0)
        sv = SphericalVoronoi(points)
        assert sv.vertices.shape[1] == dim
        assert len(sv.regions) == n

        # verify Euler characteristic
        cell_counts = []
        simplices = np.sort(sv._simplices)
        for i in range(1, dim + 1):
            cells = []
            for indices in itertools.combinations(range(dim), i):
                cells.append(simplices[:, list(indices)])
            cells = np.unique(np.concatenate(cells), axis=0)
            cell_counts.append(len(cells))
        expected_euler = 1 + (-1)**(dim-1)
        actual_euler = sum([(-1)**i * e for i, e in enumerate(cell_counts)])
        assert expected_euler == actual_euler

    @pytest.mark.parametrize("dim", range(2, 6))
    def test_cross_polytope_regions(self, dim):
        # The hypercube is the dual of the cross-polytope, so the voronoi
        # vertices of the cross-polytope lie on the points of the hypercube.

        # generate points of the cross-polytope
        points = np.concatenate((-np.eye(dim), np.eye(dim)))
        sv = SphericalVoronoi(points)
        assert all([len(e) == 2**(dim - 1) for e in sv.regions])

        # generate points of the hypercube
        expected = np.vstack(list(itertools.product([-1, 1], repeat=dim)))
        expected = expected.astype(np.float64) / np.sqrt(dim)

        # test that Voronoi vertices are correctly placed
        dist = distance.cdist(sv.vertices, expected)
        res = linear_sum_assignment(dist)
        assert dist[res].sum() < TOL

    @pytest.mark.parametrize("dim", range(2, 6))
    def test_hypercube_regions(self, dim):
        # The cross-polytope is the dual of the hypercube, so the voronoi
        # vertices of the hypercube lie on the points of the cross-polytope.

        # generate points of the hypercube
        points = np.vstack(list(itertools.product([-1, 1], repeat=dim)))
        points = points.astype(np.float64) / np.sqrt(dim)
        sv = SphericalVoronoi(points)

        # generate points of the cross-polytope
        expected = np.concatenate((-np.eye(dim), np.eye(dim)))

        # test that Voronoi vertices are correctly placed
        dist = distance.cdist(sv.vertices, expected)
        res = linear_sum_assignment(dist)
        assert dist[res].sum() < TOL

    @pytest.mark.parametrize("n", [10, 500])
    @pytest.mark.parametrize("dim", [2, 3])
    @pytest.mark.parametrize("radius", [0.5, 1, 2])
    @pytest.mark.parametrize("shift", [False, True])
    @pytest.mark.parametrize("single_hemisphere", [False, True])
    def test_area_reconstitution(self, n, dim, radius, shift,
                                 single_hemisphere):
        points = _sample_sphere(n, dim, seed=0)

        # move all points to one side of the sphere for single-hemisphere test
        if single_hemisphere:
            points[:, 0] = np.abs(points[:, 0])

        center = (np.arange(dim) + 1) * shift
        points = radius * points + center

        sv = SphericalVoronoi(points, radius=radius, center=center)
        areas = sv.calculate_areas()
        assert_almost_equal(areas.sum(), _hypersphere_area(dim, radius))

    @pytest.mark.parametrize("poly", ["triangle", "dodecagon",
                                      "tetrahedron", "cube", "octahedron",
                                      "dodecahedron", "icosahedron"])
    def test_equal_area_reconstitution(self, poly):
        points = _generate_polytope(poly)
        n, dim = points.shape
        sv = SphericalVoronoi(points)
        areas = sv.calculate_areas()
        assert_almost_equal(areas, _hypersphere_area(dim, 1) / n)

    def test_area_unsupported_dimension(self):
        dim = 4
        points = np.concatenate((-np.eye(dim), np.eye(dim)))
        sv = SphericalVoronoi(points)
        with pytest.raises(TypeError, match="Only supported"):
            sv.calculate_areas()

    @pytest.mark.parametrize("radius", [1, 1.])
    @pytest.mark.parametrize("center", [None, (1, 2, 3), (1., 2., 3.)])
    def test_attribute_types(self, radius, center):
        points = radius * self.points
        if center is not None:
            points += center

        sv = SphericalVoronoi(points, radius=radius, center=center)
        assert sv.points.dtype is np.dtype(np.float64)
        assert sv.center.dtype is np.dtype(np.float64)
        assert isinstance(sv.radius, float)

    def test_region_types(self):
        # Tests that region integer type does not change
        # See Issue #13412
        sv = SphericalVoronoi(self.points)
        dtype = type(sv.regions[0][0])
        sv.sort_vertices_of_regions()
        assert type(sv.regions[0][0]) == dtype
        sv.sort_vertices_of_regions()
        assert type(sv.regions[0][0]) == dtype
