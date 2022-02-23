import os
import copy

import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
                           assert_, assert_allclose, assert_array_equal)
import pytest
from pytest import raises as assert_raises

import scipy.spatial.qhull as qhull
from scipy.spatial import cKDTree as KDTree
from scipy.spatial import Voronoi

import itertools

def sorted_tuple(x):
    return tuple(sorted(x))


def sorted_unique_tuple(x):
    return tuple(np.unique(x))


def assert_unordered_tuple_list_equal(a, b, tpl=tuple):
    if isinstance(a, np.ndarray):
        a = a.tolist()
    if isinstance(b, np.ndarray):
        b = b.tolist()
    a = list(map(tpl, a))
    a.sort()
    b = list(map(tpl, b))
    b.sort()
    assert_equal(a, b)


np.random.seed(1234)

points = [(0,0), (0,1), (1,0), (1,1), (0.5, 0.5), (0.5, 1.5)]

pathological_data_1 = np.array([
    [-3.14,-3.14], [-3.14,-2.36], [-3.14,-1.57], [-3.14,-0.79],
    [-3.14,0.0], [-3.14,0.79], [-3.14,1.57], [-3.14,2.36],
    [-3.14,3.14], [-2.36,-3.14], [-2.36,-2.36], [-2.36,-1.57],
    [-2.36,-0.79], [-2.36,0.0], [-2.36,0.79], [-2.36,1.57],
    [-2.36,2.36], [-2.36,3.14], [-1.57,-0.79], [-1.57,0.79],
    [-1.57,-1.57], [-1.57,0.0], [-1.57,1.57], [-1.57,-3.14],
    [-1.57,-2.36], [-1.57,2.36], [-1.57,3.14], [-0.79,-1.57],
    [-0.79,1.57], [-0.79,-3.14], [-0.79,-2.36], [-0.79,-0.79],
    [-0.79,0.0], [-0.79,0.79], [-0.79,2.36], [-0.79,3.14],
    [0.0,-3.14], [0.0,-2.36], [0.0,-1.57], [0.0,-0.79], [0.0,0.0],
    [0.0,0.79], [0.0,1.57], [0.0,2.36], [0.0,3.14], [0.79,-3.14],
    [0.79,-2.36], [0.79,-0.79], [0.79,0.0], [0.79,0.79],
    [0.79,2.36], [0.79,3.14], [0.79,-1.57], [0.79,1.57],
    [1.57,-3.14], [1.57,-2.36], [1.57,2.36], [1.57,3.14],
    [1.57,-1.57], [1.57,0.0], [1.57,1.57], [1.57,-0.79],
    [1.57,0.79], [2.36,-3.14], [2.36,-2.36], [2.36,-1.57],
    [2.36,-0.79], [2.36,0.0], [2.36,0.79], [2.36,1.57],
    [2.36,2.36], [2.36,3.14], [3.14,-3.14], [3.14,-2.36],
    [3.14,-1.57], [3.14,-0.79], [3.14,0.0], [3.14,0.79],
    [3.14,1.57], [3.14,2.36], [3.14,3.14],
])

pathological_data_2 = np.array([
    [-1, -1], [-1, 0], [-1, 1],
    [0, -1], [0, 0], [0, 1],
    [1, -1 - np.finfo(np.float_).eps], [1, 0], [1, 1],
])

bug_2850_chunks = [np.random.rand(10, 2),
                   np.array([[0,0], [0,1], [1,0], [1,1]])  # add corners
                   ]

# same with some additional chunks
bug_2850_chunks_2 = (bug_2850_chunks +
                     [np.random.rand(10, 2),
                      0.25 + np.array([[0,0], [0,1], [1,0], [1,1]])])

DATASETS = {
    'some-points': np.asarray(points),
    'random-2d': np.random.rand(30, 2),
    'random-3d': np.random.rand(30, 3),
    'random-4d': np.random.rand(30, 4),
    'random-5d': np.random.rand(30, 5),
    'random-6d': np.random.rand(10, 6),
    'random-7d': np.random.rand(10, 7),
    'random-8d': np.random.rand(10, 8),
    'pathological-1': pathological_data_1,
    'pathological-2': pathological_data_2
}

INCREMENTAL_DATASETS = {
    'bug-2850': (bug_2850_chunks, None),
    'bug-2850-2': (bug_2850_chunks_2, None),
}


def _add_inc_data(name, chunksize):
    """
    Generate incremental datasets from basic data sets
    """
    points = DATASETS[name]
    ndim = points.shape[1]

    opts = None
    nmin = ndim + 2

    if name == 'some-points':
        # since Qz is not allowed, use QJ
        opts = 'QJ Pp'
    elif name == 'pathological-1':
        # include enough points so that we get different x-coordinates
        nmin = 12

    chunks = [points[:nmin]]
    for j in range(nmin, len(points), chunksize):
        chunks.append(points[j:j+chunksize])

    new_name = "%s-chunk-%d" % (name, chunksize)
    assert new_name not in INCREMENTAL_DATASETS
    INCREMENTAL_DATASETS[new_name] = (chunks, opts)


for name in DATASETS:
    for chunksize in 1, 4, 16:
        _add_inc_data(name, chunksize)


class Test_Qhull:
    def test_swapping(self):
        # Check that Qhull state swapping works

        x = qhull._Qhull(b'v',
                         np.array([[0,0],[0,1],[1,0],[1,1.],[0.5,0.5]]),
                         b'Qz')
        xd = copy.deepcopy(x.get_voronoi_diagram())

        y = qhull._Qhull(b'v',
                         np.array([[0,0],[0,1],[1,0],[1,2.]]),
                         b'Qz')
        yd = copy.deepcopy(y.get_voronoi_diagram())

        xd2 = copy.deepcopy(x.get_voronoi_diagram())
        x.close()
        yd2 = copy.deepcopy(y.get_voronoi_diagram())
        y.close()

        assert_raises(RuntimeError, x.get_voronoi_diagram)
        assert_raises(RuntimeError, y.get_voronoi_diagram)

        assert_allclose(xd[0], xd2[0])
        assert_unordered_tuple_list_equal(xd[1], xd2[1], tpl=sorted_tuple)
        assert_unordered_tuple_list_equal(xd[2], xd2[2], tpl=sorted_tuple)
        assert_unordered_tuple_list_equal(xd[3], xd2[3], tpl=sorted_tuple)
        assert_array_equal(xd[4], xd2[4])

        assert_allclose(yd[0], yd2[0])
        assert_unordered_tuple_list_equal(yd[1], yd2[1], tpl=sorted_tuple)
        assert_unordered_tuple_list_equal(yd[2], yd2[2], tpl=sorted_tuple)
        assert_unordered_tuple_list_equal(yd[3], yd2[3], tpl=sorted_tuple)
        assert_array_equal(yd[4], yd2[4])

        x.close()
        assert_raises(RuntimeError, x.get_voronoi_diagram)
        y.close()
        assert_raises(RuntimeError, y.get_voronoi_diagram)

    def test_issue_8051(self):
        points = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2],[2, 0], [2, 1], [2, 2]])
        Voronoi(points)


class TestUtilities:
    """
    Check that utility functions work.

    """

    def test_find_simplex(self):
        # Simple check that simplex finding works
        points = np.array([(0,0), (0,1), (1,1), (1,0)], dtype=np.double)
        tri = qhull.Delaunay(points)

        # +---+
        # |\ 0|
        # | \ |
        # |1 \|
        # +---+

        assert_equal(tri.vertices, [[1, 3, 2], [3, 1, 0]])

        for p in [(0.25, 0.25, 1),
                  (0.75, 0.75, 0),
                  (0.3, 0.2, 1)]:
            i = tri.find_simplex(p[:2])
            assert_equal(i, p[2], err_msg='%r' % (p,))
            j = qhull.tsearch(tri, p[:2])
            assert_equal(i, j)

    def test_plane_distance(self):
        # Compare plane distance from hyperplane equations obtained from Qhull
        # to manually computed plane equations
        x = np.array([(0,0), (1, 1), (1, 0), (0.99189033, 0.37674127),
                      (0.99440079, 0.45182168)], dtype=np.double)
        p = np.array([0.99966555, 0.15685619], dtype=np.double)

        tri = qhull.Delaunay(x)

        z = tri.lift_points(x)
        pz = tri.lift_points(p)

        dist = tri.plane_distance(p)

        for j, v in enumerate(tri.vertices):
            x1 = z[v[0]]
            x2 = z[v[1]]
            x3 = z[v[2]]

            n = np.cross(x1 - x3, x2 - x3)
            n /= np.sqrt(np.dot(n, n))
            n *= -np.sign(n[2])

            d = np.dot(n, pz - x3)

            assert_almost_equal(dist[j], d)

    def test_convex_hull(self):
        # Simple check that the convex hull seems to works
        points = np.array([(0,0), (0,1), (1,1), (1,0)], dtype=np.double)
        tri = qhull.Delaunay(points)

        # +---+
        # |\ 0|
        # | \ |
        # |1 \|
        # +---+

        assert_equal(tri.convex_hull, [[3, 2], [1, 2], [1, 0], [3, 0]])

    def test_volume_area(self):
        #Basic check that we get back the correct volume and area for a cube
        points = np.array([(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0),
                           (0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)])
        hull = qhull.ConvexHull(points)

        assert_allclose(hull.volume, 1., rtol=1e-14,
                        err_msg="Volume of cube is incorrect")
        assert_allclose(hull.area, 6., rtol=1e-14,
                        err_msg="Area of cube is incorrect")

    def test_random_volume_area(self):
        #Test that the results for a random 10-point convex are
        #coherent with the output of qconvex Qt s FA
        points = np.array([(0.362568364506, 0.472712355305, 0.347003084477),
                           (0.733731893414, 0.634480295684, 0.950513180209),
                           (0.511239955611, 0.876839441267, 0.418047827863),
                           (0.0765906233393, 0.527373281342, 0.6509863541),
                           (0.146694972056, 0.596725793348, 0.894860986685),
                           (0.513808585741, 0.069576205858, 0.530890338876),
                           (0.512343805118, 0.663537132612, 0.037689295973),
                           (0.47282965018, 0.462176697655, 0.14061843691),
                           (0.240584597123, 0.778660020591, 0.722913476339),
                           (0.951271745935, 0.967000673944, 0.890661319684)])

        hull = qhull.ConvexHull(points)
        assert_allclose(hull.volume, 0.14562013, rtol=1e-07,
                        err_msg="Volume of random polyhedron is incorrect")
        assert_allclose(hull.area, 1.6670425, rtol=1e-07,
                        err_msg="Area of random polyhedron is incorrect")

    def test_incremental_volume_area_random_input(self):
        """Test that incremental mode gives the same volume/area as
        non-incremental mode and incremental mode with restart"""
        nr_points = 20
        dim = 3
        points = np.random.random((nr_points, dim))
        inc_hull = qhull.ConvexHull(points[:dim+1, :], incremental=True)
        inc_restart_hull = qhull.ConvexHull(points[:dim+1, :], incremental=True)
        for i in range(dim+1, nr_points):
            hull = qhull.ConvexHull(points[:i+1, :])
            inc_hull.add_points(points[i:i+1, :])
            inc_restart_hull.add_points(points[i:i+1, :], restart=True)
            assert_allclose(hull.volume, inc_hull.volume, rtol=1e-7)
            assert_allclose(hull.volume, inc_restart_hull.volume, rtol=1e-7)
            assert_allclose(hull.area, inc_hull.area, rtol=1e-7)
            assert_allclose(hull.area, inc_restart_hull.area, rtol=1e-7)

    def _check_barycentric_transforms(self, tri, err_msg="",
                                      unit_cube=False,
                                      unit_cube_tol=0):
        """Check that a triangulation has reasonable barycentric transforms"""
        vertices = tri.points[tri.vertices]
        sc = 1/(tri.ndim + 1.0)
        centroids = vertices.sum(axis=1) * sc

        # Either: (i) the simplex has a `nan` barycentric transform,
        # or, (ii) the centroid is in the simplex

        def barycentric_transform(tr, x):
            r = tr[:,-1,:]
            Tinv = tr[:,:-1,:]
            return np.einsum('ijk,ik->ij', Tinv, x - r)

        eps = np.finfo(float).eps

        c = barycentric_transform(tri.transform, centroids)
        with np.errstate(invalid="ignore"):
            ok = np.isnan(c).all(axis=1) | (abs(c - sc)/sc < 0.1).all(axis=1)

        assert_(ok.all(), "%s %s" % (err_msg, np.nonzero(~ok)))

        # Invalid simplices must be (nearly) zero volume
        q = vertices[:,:-1,:] - vertices[:,-1,None,:]
        volume = np.array([np.linalg.det(q[k,:,:])
                           for k in range(tri.nsimplex)])
        ok = np.isfinite(tri.transform[:,0,0]) | (volume < np.sqrt(eps))
        assert_(ok.all(), "%s %s" % (err_msg, np.nonzero(~ok)))

        # Also, find_simplex for the centroid should end up in some
        # simplex for the non-degenerate cases
        j = tri.find_simplex(centroids)
        ok = (j != -1) | np.isnan(tri.transform[:,0,0])
        assert_(ok.all(), "%s %s" % (err_msg, np.nonzero(~ok)))

        if unit_cube:
            # If in unit cube, no interior point should be marked out of hull
            at_boundary = (centroids <= unit_cube_tol).any(axis=1)
            at_boundary |= (centroids >= 1 - unit_cube_tol).any(axis=1)

            ok = (j != -1) | at_boundary
            assert_(ok.all(), "%s %s" % (err_msg, np.nonzero(~ok)))

    def test_degenerate_barycentric_transforms(self):
        # The triangulation should not produce invalid barycentric
        # transforms that stump the simplex finding
        data = np.load(os.path.join(os.path.dirname(__file__), 'data',
                                    'degenerate_pointset.npz'))
        points = data['c']
        data.close()

        tri = qhull.Delaunay(points)

        # Check that there are not too many invalid simplices
        bad_count = np.isnan(tri.transform[:,0,0]).sum()
        assert_(bad_count < 23, bad_count)

        # Check the transforms
        self._check_barycentric_transforms(tri)

    @pytest.mark.slow
    def test_more_barycentric_transforms(self):
        # Triangulate some "nasty" grids

        eps = np.finfo(float).eps

        npoints = {2: 70, 3: 11, 4: 5, 5: 3}

        for ndim in range(2, 6):
            # Generate an uniform grid in n-d unit cube
            x = np.linspace(0, 1, npoints[ndim])
            grid = np.c_[list(map(np.ravel, np.broadcast_arrays(*np.ix_(*([x]*ndim)))))].T

            err_msg = "ndim=%d" % ndim

            # Check using regular grid
            tri = qhull.Delaunay(grid)
            self._check_barycentric_transforms(tri, err_msg=err_msg,
                                               unit_cube=True)

            # Check with eps-perturbations
            np.random.seed(1234)
            m = (np.random.rand(grid.shape[0]) < 0.2)
            grid[m,:] += 2*eps*(np.random.rand(*grid[m,:].shape) - 0.5)

            tri = qhull.Delaunay(grid)
            self._check_barycentric_transforms(tri, err_msg=err_msg,
                                               unit_cube=True,
                                               unit_cube_tol=2*eps)

            # Check with duplicated data
            tri = qhull.Delaunay(np.r_[grid, grid])
            self._check_barycentric_transforms(tri, err_msg=err_msg,
                                               unit_cube=True,
                                               unit_cube_tol=2*eps)


class TestVertexNeighborVertices:
    def _check(self, tri):
        expected = [set() for j in range(tri.points.shape[0])]
        for s in tri.simplices:
            for a in s:
                for b in s:
                    if a != b:
                        expected[a].add(b)

        indptr, indices = tri.vertex_neighbor_vertices

        got = [set(map(int, indices[indptr[j]:indptr[j+1]]))
               for j in range(tri.points.shape[0])]

        assert_equal(got, expected, err_msg="%r != %r" % (got, expected))

    def test_triangle(self):
        points = np.array([(0,0), (0,1), (1,0)], dtype=np.double)
        tri = qhull.Delaunay(points)
        self._check(tri)

    def test_rectangle(self):
        points = np.array([(0,0), (0,1), (1,1), (1,0)], dtype=np.double)
        tri = qhull.Delaunay(points)
        self._check(tri)

    def test_complicated(self):
        points = np.array([(0,0), (0,1), (1,1), (1,0),
                           (0.5, 0.5), (0.9, 0.5)], dtype=np.double)
        tri = qhull.Delaunay(points)
        self._check(tri)


class TestDelaunay:
    """
    Check that triangulation works.

    """
    def test_masked_array_fails(self):
        masked_array = np.ma.masked_all(1)
        assert_raises(ValueError, qhull.Delaunay, masked_array)

    def test_array_with_nans_fails(self):
        points_with_nan = np.array([(0,0), (0,1), (1,1), (1,np.nan)], dtype=np.double)
        assert_raises(ValueError, qhull.Delaunay, points_with_nan)

    def test_nd_simplex(self):
        # simple smoke test: triangulate a n-dimensional simplex
        for nd in range(2, 8):
            points = np.zeros((nd+1, nd))
            for j in range(nd):
                points[j,j] = 1.0
            points[-1,:] = 1.0

            tri = qhull.Delaunay(points)

            tri.vertices.sort()

            assert_equal(tri.vertices, np.arange(nd+1, dtype=int)[None,:])
            assert_equal(tri.neighbors, -1 + np.zeros((nd+1), dtype=int)[None,:])

    def test_2d_square(self):
        # simple smoke test: 2d square
        points = np.array([(0,0), (0,1), (1,1), (1,0)], dtype=np.double)
        tri = qhull.Delaunay(points)

        assert_equal(tri.vertices, [[1, 3, 2], [3, 1, 0]])
        assert_equal(tri.neighbors, [[-1, -1, 1], [-1, -1, 0]])

    def test_duplicate_points(self):
        x = np.array([0, 1, 0, 1], dtype=np.float64)
        y = np.array([0, 0, 1, 1], dtype=np.float64)

        xp = np.r_[x, x]
        yp = np.r_[y, y]

        # shouldn't fail on duplicate points
        qhull.Delaunay(np.c_[x, y])
        qhull.Delaunay(np.c_[xp, yp])

    def test_pathological(self):
        # both should succeed
        points = DATASETS['pathological-1']
        tri = qhull.Delaunay(points)
        assert_equal(tri.points[tri.vertices].max(), points.max())
        assert_equal(tri.points[tri.vertices].min(), points.min())

        points = DATASETS['pathological-2']
        tri = qhull.Delaunay(points)
        assert_equal(tri.points[tri.vertices].max(), points.max())
        assert_equal(tri.points[tri.vertices].min(), points.min())

    def test_joggle(self):
        # Check that the option QJ indeed guarantees that all input points
        # occur as vertices of the triangulation

        points = np.random.rand(10, 2)
        points = np.r_[points, points]  # duplicate input data

        tri = qhull.Delaunay(points, qhull_options="QJ Qbb Pp")
        assert_array_equal(np.unique(tri.simplices.ravel()),
                           np.arange(len(points)))

    def test_coplanar(self):
        # Check that the coplanar point output option indeed works
        points = np.random.rand(10, 2)
        points = np.r_[points, points]  # duplicate input data

        tri = qhull.Delaunay(points)

        assert_(len(np.unique(tri.simplices.ravel())) == len(points)//2)
        assert_(len(tri.coplanar) == len(points)//2)

        assert_(len(np.unique(tri.coplanar[:,2])) == len(points)//2)

        assert_(np.all(tri.vertex_to_simplex >= 0))

    def test_furthest_site(self):
        points = [(0, 0), (0, 1), (1, 0), (0.5, 0.5), (1.1, 1.1)]
        tri = qhull.Delaunay(points, furthest_site=True)

        expected = np.array([(1, 4, 0), (4, 2, 0)])  # from Qhull
        assert_array_equal(tri.simplices, expected)

    @pytest.mark.parametrize("name", sorted(INCREMENTAL_DATASETS))
    def test_incremental(self, name):
        # Test incremental construction of the triangulation

        chunks, opts = INCREMENTAL_DATASETS[name]
        points = np.concatenate(chunks, axis=0)

        obj = qhull.Delaunay(chunks[0], incremental=True,
                             qhull_options=opts)
        for chunk in chunks[1:]:
            obj.add_points(chunk)

        obj2 = qhull.Delaunay(points)

        obj3 = qhull.Delaunay(chunks[0], incremental=True,
                              qhull_options=opts)
        if len(chunks) > 1:
            obj3.add_points(np.concatenate(chunks[1:], axis=0),
                            restart=True)

        # Check that the incremental mode agrees with upfront mode
        if name.startswith('pathological'):
            # XXX: These produce valid but different triangulations.
            #      They look OK when plotted, but how to check them?

            assert_array_equal(np.unique(obj.simplices.ravel()),
                               np.arange(points.shape[0]))
            assert_array_equal(np.unique(obj2.simplices.ravel()),
                               np.arange(points.shape[0]))
        else:
            assert_unordered_tuple_list_equal(obj.simplices, obj2.simplices,
                                              tpl=sorted_tuple)

        assert_unordered_tuple_list_equal(obj2.simplices, obj3.simplices,
                                          tpl=sorted_tuple)


def assert_hulls_equal(points, facets_1, facets_2):
    # Check that two convex hulls constructed from the same point set
    # are equal

    facets_1 = set(map(sorted_tuple, facets_1))
    facets_2 = set(map(sorted_tuple, facets_2))

    if facets_1 != facets_2 and points.shape[1] == 2:
        # The direct check fails for the pathological cases
        # --- then the convex hull from Delaunay differs (due
        # to rounding error etc.) from the hull computed
        # otherwise, by the question whether (tricoplanar)
        # points that lie almost exactly on the hull are
        # included as vertices of the hull or not.
        #
        # So we check the result, and accept it if the Delaunay
        # hull line segments are a subset of the usual hull.

        eps = 1000 * np.finfo(float).eps

        for a, b in facets_1:
            for ap, bp in facets_2:
                t = points[bp] - points[ap]
                t /= np.linalg.norm(t)       # tangent
                n = np.array([-t[1], t[0]])  # normal

                # check that the two line segments are parallel
                # to the same line
                c1 = np.dot(n, points[b] - points[ap])
                c2 = np.dot(n, points[a] - points[ap])
                if not np.allclose(np.dot(c1, n), 0):
                    continue
                if not np.allclose(np.dot(c2, n), 0):
                    continue

                # Check that the segment (a, b) is contained in (ap, bp)
                c1 = np.dot(t, points[a] - points[ap])
                c2 = np.dot(t, points[b] - points[ap])
                c3 = np.dot(t, points[bp] - points[ap])
                if c1 < -eps or c1 > c3 + eps:
                    continue
                if c2 < -eps or c2 > c3 + eps:
                    continue

                # OK:
                break
            else:
                raise AssertionError("comparison fails")

        # it was OK
        return

    assert_equal(facets_1, facets_2)


class TestConvexHull:
    def test_masked_array_fails(self):
        masked_array = np.ma.masked_all(1)
        assert_raises(ValueError, qhull.ConvexHull, masked_array)

    def test_array_with_nans_fails(self):
        points_with_nan = np.array([(0,0), (1,1), (2,np.nan)], dtype=np.double)
        assert_raises(ValueError, qhull.ConvexHull, points_with_nan)

    @pytest.mark.parametrize("name", sorted(DATASETS))
    def test_hull_consistency_tri(self, name):
        # Check that a convex hull returned by qhull in ndim
        # and the hull constructed from ndim delaunay agree
        points = DATASETS[name]

        tri = qhull.Delaunay(points)
        hull = qhull.ConvexHull(points)

        assert_hulls_equal(points, tri.convex_hull, hull.simplices)

        # Check that the hull extremes are as expected
        if points.shape[1] == 2:
            assert_equal(np.unique(hull.simplices), np.sort(hull.vertices))
        else:
            assert_equal(np.unique(hull.simplices), hull.vertices)

    @pytest.mark.parametrize("name", sorted(INCREMENTAL_DATASETS))
    def test_incremental(self, name):
        # Test incremental construction of the convex hull
        chunks, _ = INCREMENTAL_DATASETS[name]
        points = np.concatenate(chunks, axis=0)

        obj = qhull.ConvexHull(chunks[0], incremental=True)
        for chunk in chunks[1:]:
            obj.add_points(chunk)

        obj2 = qhull.ConvexHull(points)

        obj3 = qhull.ConvexHull(chunks[0], incremental=True)
        if len(chunks) > 1:
            obj3.add_points(np.concatenate(chunks[1:], axis=0),
                            restart=True)

        # Check that the incremental mode agrees with upfront mode
        assert_hulls_equal(points, obj.simplices, obj2.simplices)
        assert_hulls_equal(points, obj.simplices, obj3.simplices)

    def test_vertices_2d(self):
        # The vertices should be in counterclockwise order in 2-D
        np.random.seed(1234)
        points = np.random.rand(30, 2)

        hull = qhull.ConvexHull(points)
        assert_equal(np.unique(hull.simplices), np.sort(hull.vertices))

        # Check counterclockwiseness
        x, y = hull.points[hull.vertices].T
        angle = np.arctan2(y - y.mean(), x - x.mean())
        assert_(np.all(np.diff(np.unwrap(angle)) > 0))

    def test_volume_area(self):
        # Basic check that we get back the correct volume and area for a cube
        points = np.array([(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0),
                           (0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)])
        tri = qhull.ConvexHull(points)

        assert_allclose(tri.volume, 1., rtol=1e-14)
        assert_allclose(tri.area, 6., rtol=1e-14)

    @pytest.mark.parametrize("incremental", [False, True])
    def test_good2d(self, incremental):
        # Make sure the QGn option gives the correct value of "good".
        points = np.array([[0.2, 0.2],
                           [0.2, 0.4],
                           [0.4, 0.4],
                           [0.4, 0.2],
                           [0.3, 0.6]])
        hull = qhull.ConvexHull(points=points,
                                incremental=incremental,
                                qhull_options='QG4')
        expected = np.array([False, True, False, False], dtype=bool)
        actual = hull.good
        assert_equal(actual, expected)

    @pytest.mark.parametrize("visibility", [
                              "QG4",  # visible=True
                              "QG-4",  # visible=False
                              ])
    @pytest.mark.parametrize("new_gen, expected", [
        # add generator that places QG4 inside hull
        # so all facets are invisible
        (np.array([[0.3, 0.7]]),
         np.array([False, False, False, False, False], dtype=bool)),
        # adding a generator on the opposite side of the square
        # should preserve the single visible facet & add one invisible
        # facet
        (np.array([[0.3, -0.7]]),
         np.array([False, True, False, False, False], dtype=bool)),
        # split the visible facet on top of the square into two
        # visible facets, with visibility at the end of the array
        # because add_points concatenates
        (np.array([[0.3, 0.41]]),
         np.array([False, False, False, True, True], dtype=bool)),
        # with our current Qhull options, coplanarity will not count
        # for visibility; this case shifts one visible & one invisible
        # facet & adds a coplanar facet
        # simplex at index position 2 is the shifted visible facet
        # the final simplex is the coplanar facet
        (np.array([[0.5, 0.6], [0.6, 0.6]]),
         np.array([False, False, True, False, False], dtype=bool)),
        # place the new generator such that it envelops the query
        # point within the convex hull, but only just barely within
        # the double precision limit
        # NOTE: testing exact degeneracy is less predictable than this
        # scenario, perhaps because of the default Qt option we have
        # enabled for Qhull to handle precision matters
        (np.array([[0.3, 0.6 + 1e-16]]),
         np.array([False, False, False, False, False], dtype=bool)),
        ])
    def test_good2d_incremental_changes(self, new_gen, expected,
                                        visibility):
        # use the usual square convex hull
        # generators from test_good2d
        points = np.array([[0.2, 0.2],
                           [0.2, 0.4],
                           [0.4, 0.4],
                           [0.4, 0.2],
                           [0.3, 0.6]])
        hull = qhull.ConvexHull(points=points,
                                incremental=True,
                                qhull_options=visibility)
        hull.add_points(new_gen)
        actual = hull.good
        if '-' in visibility:
            expected = np.invert(expected)
        assert_equal(actual, expected)

    @pytest.mark.parametrize("incremental", [False, True])
    def test_good2d_no_option(self, incremental):
        # handle case where good attribue doesn't exist
        # because Qgn or Qg-n wasn't specified
        points = np.array([[0.2, 0.2],
                           [0.2, 0.4],
                           [0.4, 0.4],
                           [0.4, 0.2],
                           [0.3, 0.6]])
        hull = qhull.ConvexHull(points=points,
                                incremental=incremental)
        actual = hull.good
        assert actual is None
        # preserve None after incremental addition
        if incremental:
            hull.add_points(np.zeros((1, 2)))
            actual = hull.good
            assert actual is None

    @pytest.mark.parametrize("incremental", [False, True])
    def test_good2d_inside(self, incremental):
        # Make sure the QGn option gives the correct value of "good".
        # When point n is inside the convex hull of the rest, good is
        # all False.
        points = np.array([[0.2, 0.2],
                           [0.2, 0.4],
                           [0.4, 0.4],
                           [0.4, 0.2],
                           [0.3, 0.3]])
        hull = qhull.ConvexHull(points=points,
                                incremental=incremental,
                                qhull_options='QG4')
        expected = np.array([False, False, False, False], dtype=bool)
        actual = hull.good
        assert_equal(actual, expected)

    @pytest.mark.parametrize("incremental", [False, True])
    def test_good3d(self, incremental):
        # Make sure the QGn option gives the correct value of "good"
        # for a 3d figure
        points = np.array([[0.0, 0.0, 0.0],
                           [0.90029516, -0.39187448, 0.18948093],
                           [0.48676420, -0.72627633, 0.48536925],
                           [0.57651530, -0.81179274, -0.09285832],
                           [0.67846893, -0.71119562, 0.18406710]])
        hull = qhull.ConvexHull(points=points,
                                incremental=incremental,
                                qhull_options='QG0')
        expected = np.array([True, False, False, False], dtype=bool)
        assert_equal(hull.good, expected)

class TestVoronoi:
    def test_masked_array_fails(self):
        masked_array = np.ma.masked_all(1)
        assert_raises(ValueError, qhull.Voronoi, masked_array)

    def test_simple(self):
        # Simple case with known Voronoi diagram
        points = [(0, 0), (0, 1), (0, 2),
                  (1, 0), (1, 1), (1, 2),
                  (2, 0), (2, 1), (2, 2)]

        # qhull v o Fv Qbb Qc Qz < dat
        output = """
        2
        5 10 1
        -10.101 -10.101
           0.5    0.5
           0.5    1.5
           1.5    0.5
           1.5    1.5
        2 0 1
        3 2 0 1
        2 0 2
        3 3 0 1
        4 1 2 4 3
        3 4 0 2
        2 0 3
        3 4 0 3
        2 0 4
        0
        12
        4 0 3 0 1
        4 0 1 0 1
        4 1 4 1 2
        4 1 2 0 2
        4 2 5 0 2
        4 3 4 1 3
        4 3 6 0 3
        4 4 5 2 4
        4 4 7 3 4
        4 5 8 0 4
        4 6 7 0 3
        4 7 8 0 4
        """
        self._compare_qvoronoi(points, output)

    def _compare_qvoronoi(self, points, output, **kw):
        """Compare to output from 'qvoronoi o Fv < data' to Voronoi()"""

        # Parse output
        output = [list(map(float, x.split())) for x in output.strip().splitlines()]
        nvertex = int(output[1][0])
        vertices = list(map(tuple, output[3:2+nvertex]))  # exclude inf
        nregion = int(output[1][1])
        regions = [[int(y)-1 for y in x[1:]]
                   for x in output[2+nvertex:2+nvertex+nregion]]
        ridge_points = [[int(y) for y in x[1:3]]
                        for x in output[3+nvertex+nregion:]]
        ridge_vertices = [[int(y)-1 for y in x[3:]]
                          for x in output[3+nvertex+nregion:]]

        # Compare results
        vor = qhull.Voronoi(points, **kw)

        def sorttuple(x):
            return tuple(sorted(x))

        assert_allclose(vor.vertices, vertices)
        assert_equal(set(map(tuple, vor.regions)),
                     set(map(tuple, regions)))

        p1 = list(zip(list(map(sorttuple, ridge_points)), list(map(sorttuple, ridge_vertices))))
        p2 = list(zip(list(map(sorttuple, vor.ridge_points.tolist())),
                 list(map(sorttuple, vor.ridge_vertices))))
        p1.sort()
        p2.sort()

        assert_equal(p1, p2)

    @pytest.mark.parametrize("name", sorted(DATASETS))
    def test_ridges(self, name):
        # Check that the ridges computed by Voronoi indeed separate
        # the regions of nearest neighborhood, by comparing the result
        # to KDTree.

        points = DATASETS[name]

        tree = KDTree(points)
        vor = qhull.Voronoi(points)

        for p, v in vor.ridge_dict.items():
            # consider only finite ridges
            if not np.all(np.asarray(v) >= 0):
                continue

            ridge_midpoint = vor.vertices[v].mean(axis=0)
            d = 1e-6 * (points[p[0]] - ridge_midpoint)

            dist, k = tree.query(ridge_midpoint + d, k=1)
            assert_equal(k, p[0])

            dist, k = tree.query(ridge_midpoint - d, k=1)
            assert_equal(k, p[1])

    def test_furthest_site(self):
        points = [(0, 0), (0, 1), (1, 0), (0.5, 0.5), (1.1, 1.1)]

        # qhull v o Fv Qbb Qc Qu < dat
        output = """
        2
        3 5 1
        -10.101 -10.101
        0.6000000000000001    0.5
           0.5 0.6000000000000001
        3 0 2 1
        2 0 1
        2 0 2
        0
        3 0 2 1
        5
        4 0 2 0 2
        4 0 4 1 2
        4 0 1 0 1
        4 1 4 0 1
        4 2 4 0 2
        """
        self._compare_qvoronoi(points, output, furthest_site=True)

    def test_furthest_site_flag(self):
        points = [(0, 0), (0, 1), (1, 0), (0.5, 0.5), (1.1, 1.1)]

        vor = Voronoi(points)
        assert_equal(vor.furthest_site,False)
        vor = Voronoi(points,furthest_site=True)
        assert_equal(vor.furthest_site,True)

    @pytest.mark.parametrize("name", sorted(INCREMENTAL_DATASETS))
    def test_incremental(self, name):
        # Test incremental construction of the triangulation

        if INCREMENTAL_DATASETS[name][0][0].shape[1] > 3:
            # too slow (testing of the result --- qhull is still fast)
            return

        chunks, opts = INCREMENTAL_DATASETS[name]
        points = np.concatenate(chunks, axis=0)

        obj = qhull.Voronoi(chunks[0], incremental=True,
                             qhull_options=opts)
        for chunk in chunks[1:]:
            obj.add_points(chunk)

        obj2 = qhull.Voronoi(points)

        obj3 = qhull.Voronoi(chunks[0], incremental=True,
                             qhull_options=opts)
        if len(chunks) > 1:
            obj3.add_points(np.concatenate(chunks[1:], axis=0),
                            restart=True)

        # -- Check that the incremental mode agrees with upfront mode
        assert_equal(len(obj.point_region), len(obj2.point_region))
        assert_equal(len(obj.point_region), len(obj3.point_region))

        # The vertices may be in different order or duplicated in
        # the incremental map
        for objx in obj, obj3:
            vertex_map = {-1: -1}
            for i, v in enumerate(objx.vertices):
                for j, v2 in enumerate(obj2.vertices):
                    if np.allclose(v, v2):
                        vertex_map[i] = j

            def remap(x):
                if hasattr(x, '__len__'):
                    return tuple(set([remap(y) for y in x]))
                try:
                    return vertex_map[x]
                except KeyError as e:
                    raise AssertionError("incremental result has spurious vertex at %r"
                                         % (objx.vertices[x],)) from e

            def simplified(x):
                items = set(map(sorted_tuple, x))
                if () in items:
                    items.remove(())
                items = [x for x in items if len(x) > 1]
                items.sort()
                return items

            assert_equal(
                simplified(remap(objx.regions)),
                simplified(obj2.regions)
                )
            assert_equal(
                simplified(remap(objx.ridge_vertices)),
                simplified(obj2.ridge_vertices)
                )

            # XXX: compare ridge_points --- not clear exactly how to do this


class Test_HalfspaceIntersection:
    def assert_unordered_allclose(self, arr1, arr2, rtol=1e-7):
        """Check that every line in arr1 is only once in arr2"""
        assert_equal(arr1.shape, arr2.shape)

        truths = np.zeros((arr1.shape[0],), dtype=bool)
        for l1 in arr1:
            indexes = np.nonzero((abs(arr2 - l1) < rtol).all(axis=1))[0]
            assert_equal(indexes.shape, (1,))
            truths[indexes[0]] = True
        assert_(truths.all())

    def test_cube_halfspace_intersection(self):
        halfspaces = np.array([[-1.0, 0.0, 0.0],
                               [0.0, -1.0, 0.0],
                               [1.0, 0.0, -1.0],
                               [0.0, 1.0, -1.0]])
        feasible_point = np.array([0.5, 0.5])

        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        hull = qhull.HalfspaceIntersection(halfspaces, feasible_point)

        assert_allclose(hull.intersections, points)

    def test_self_dual_polytope_intersection(self):
        fname = os.path.join(os.path.dirname(__file__), 'data',
                             'selfdual-4d-polytope.txt')
        ineqs = np.genfromtxt(fname)
        halfspaces = -np.hstack((ineqs[:, 1:], ineqs[:, :1]))

        feas_point = np.array([0., 0., 0., 0.])
        hs = qhull.HalfspaceIntersection(halfspaces, feas_point)

        assert_equal(hs.intersections.shape, (24, 4))

        assert_almost_equal(hs.dual_volume, 32.0)
        assert_equal(len(hs.dual_facets), 24)
        for facet in hs.dual_facets:
            assert_equal(len(facet), 6)

        dists = halfspaces[:, -1] + halfspaces[:, :-1].dot(feas_point)
        self.assert_unordered_allclose((halfspaces[:, :-1].T/dists).T, hs.dual_points)

        points = itertools.permutations([0., 0., 0.5, -0.5])
        for point in points:
            assert_equal(np.sum((hs.intersections == point).all(axis=1)), 1)

    def test_wrong_feasible_point(self):
        halfspaces = np.array([[-1.0, 0.0, 0.0],
                               [0.0, -1.0, 0.0],
                               [1.0, 0.0, -1.0],
                               [0.0, 1.0, -1.0]])
        feasible_point = np.array([0.5, 0.5, 0.5])
        #Feasible point is (ndim,) instead of (ndim-1,)
        assert_raises(ValueError, qhull.HalfspaceIntersection, halfspaces, feasible_point)
        feasible_point = np.array([[0.5], [0.5]])
        #Feasible point is (ndim-1, 1) instead of (ndim-1,)
        assert_raises(ValueError, qhull.HalfspaceIntersection, halfspaces, feasible_point)
        feasible_point = np.array([[0.5, 0.5]])
        #Feasible point is (1, ndim-1) instead of (ndim-1,)
        assert_raises(ValueError, qhull.HalfspaceIntersection, halfspaces, feasible_point)

        feasible_point = np.array([-0.5, -0.5])
        #Feasible point is outside feasible region
        assert_raises(qhull.QhullError, qhull.HalfspaceIntersection, halfspaces, feasible_point)

    def test_incremental(self):
        #Cube
        halfspaces = np.array([[0., 0., -1., -0.5],
                               [0., -1., 0., -0.5],
                               [-1., 0., 0., -0.5],
                               [1., 0., 0., -0.5],
                               [0., 1., 0., -0.5],
                               [0., 0., 1., -0.5]])
        #Cut each summit
        extra_normals = np.array([[1., 1., 1.],
                                  [1., 1., -1.],
                                  [1., -1., 1.],
                                  [1, -1., -1.]])
        offsets = np.array([[-1.]]*8)
        extra_halfspaces = np.hstack((np.vstack((extra_normals, -extra_normals)),
                                      offsets))

        feas_point = np.array([0., 0., 0.])

        inc_hs = qhull.HalfspaceIntersection(halfspaces, feas_point, incremental=True)

        inc_res_hs = qhull.HalfspaceIntersection(halfspaces, feas_point, incremental=True)

        for i, ehs in enumerate(extra_halfspaces):
            inc_hs.add_halfspaces(ehs[np.newaxis, :])

            inc_res_hs.add_halfspaces(ehs[np.newaxis, :], restart=True)

            total = np.vstack((halfspaces, extra_halfspaces[:i+1, :]))

            hs = qhull.HalfspaceIntersection(total, feas_point)

            assert_allclose(inc_hs.halfspaces, inc_res_hs.halfspaces)
            assert_allclose(inc_hs.halfspaces, hs.halfspaces)

            #Direct computation and restart should have points in same order
            assert_allclose(hs.intersections, inc_res_hs.intersections)
            #Incremental will have points in different order than direct computation
            self.assert_unordered_allclose(inc_hs.intersections, hs.intersections)

        inc_hs.close()

    def test_cube(self):
        # Halfspaces of the cube:
        halfspaces = np.array([[-1., 0., 0., 0.],  # x >= 0
                               [1., 0., 0., -1.],  # x <= 1
                               [0., -1., 0., 0.],  # y >= 0
                               [0., 1., 0., -1.],  # y <= 1
                               [0., 0., -1., 0.],  # z >= 0
                               [0., 0., 1., -1.]])  # z <= 1
        point = np.array([0.5, 0.5, 0.5])

        hs = qhull.HalfspaceIntersection(halfspaces, point)

        # qhalf H0.5,0.5,0.5 o < input.txt
        qhalf_points = np.array([
            [-2, 0, 0],
            [2, 0, 0],
            [0, -2, 0],
            [0, 2, 0],
            [0, 0, -2],
            [0, 0, 2]])
        qhalf_facets = [
            [2, 4, 0],
            [4, 2, 1],
            [5, 2, 0],
            [2, 5, 1],
            [3, 4, 1],
            [4, 3, 0],
            [5, 3, 1],
            [3, 5, 0]]

        assert len(qhalf_facets) == len(hs.dual_facets)
        for a, b in zip(qhalf_facets, hs.dual_facets):
            assert set(a) == set(b)  # facet orientation can differ

        assert_allclose(hs.dual_points, qhalf_points)
