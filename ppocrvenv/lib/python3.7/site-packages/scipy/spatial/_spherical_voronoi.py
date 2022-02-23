"""
Spherical Voronoi Code

.. versionadded:: 0.18.0

"""
#
# Copyright (C)  Tyler Reddy, Ross Hemsley, Edd Edmondson,
#                Nikolai Nowaczyk, Joe Pitt-Francis, 2015.
#
# Distributed under the same BSD license as SciPy.
#

import warnings
import numpy as np
import scipy
from . import _voronoi
from scipy.spatial import cKDTree

__all__ = ['SphericalVoronoi']


def calculate_solid_angles(R):
    """Calculates the solid angles of plane triangles. Implements the method of
    Van Oosterom and Strackee [VanOosterom]_ with some modifications. Assumes
    that input points have unit norm."""
    # Original method uses a triple product `R1 . (R2 x R3)` for the numerator.
    # This is equal to the determinant of the matrix [R1 R2 R3], which can be
    # computed with better stability.
    numerator = np.linalg.det(R)
    denominator = 1 + (np.einsum('ij,ij->i', R[:, 0], R[:, 1]) +
                       np.einsum('ij,ij->i', R[:, 1], R[:, 2]) +
                       np.einsum('ij,ij->i', R[:, 2], R[:, 0]))
    return np.abs(2 * np.arctan2(numerator, denominator))


class SphericalVoronoi:
    """ Voronoi diagrams on the surface of a sphere.

    .. versionadded:: 0.18.0

    Parameters
    ----------
    points : ndarray of floats, shape (npoints, ndim)
        Coordinates of points from which to construct a spherical
        Voronoi diagram.
    radius : float, optional
        Radius of the sphere (Default: 1)
    center : ndarray of floats, shape (ndim,)
        Center of sphere (Default: origin)
    threshold : float
        Threshold for detecting duplicate points and
        mismatches between points and sphere parameters.
        (Default: 1e-06)

    Attributes
    ----------
    points : double array of shape (npoints, ndim)
        the points in `ndim` dimensions to generate the Voronoi diagram from
    radius : double
        radius of the sphere
    center : double array of shape (ndim,)
        center of the sphere
    vertices : double array of shape (nvertices, ndim)
        Voronoi vertices corresponding to points
    regions : list of list of integers of shape (npoints, _ )
        the n-th entry is a list consisting of the indices
        of the vertices belonging to the n-th point in points

    Methods
    -------
    calculate_areas
        Calculates the areas of the Voronoi regions. For 2D point sets, the
        regions are circular arcs. The sum of the areas is `2 * pi * radius`.
        For 3D point sets, the regions are spherical polygons. The sum of the
        areas is `4 * pi * radius**2`.

    Raises
    ------
    ValueError
        If there are duplicates in `points`.
        If the provided `radius` is not consistent with `points`.

    Notes
    -----
    The spherical Voronoi diagram algorithm proceeds as follows. The Convex
    Hull of the input points (generators) is calculated, and is equivalent to
    their Delaunay triangulation on the surface of the sphere [Caroli]_.
    The Convex Hull neighbour information is then used to
    order the Voronoi region vertices around each generator. The latter
    approach is substantially less sensitive to floating point issues than
    angle-based methods of Voronoi region vertex sorting.

    Empirical assessment of spherical Voronoi algorithm performance suggests
    quadratic time complexity (loglinear is optimal, but algorithms are more
    challenging to implement).

    References
    ----------
    .. [Caroli] Caroli et al. Robust and Efficient Delaunay triangulations of
                points on or close to a sphere. Research Report RR-7004, 2009.

    .. [VanOosterom] Van Oosterom and Strackee. The solid angle of a plane
                     triangle. IEEE Transactions on Biomedical Engineering,
                     2, 1983, pp 125--126.

    See Also
    --------
    Voronoi : Conventional Voronoi diagrams in N dimensions.

    Examples
    --------
    Do some imports and take some points on a cube:

    >>> import matplotlib.pyplot as plt
    >>> from scipy.spatial import SphericalVoronoi, geometric_slerp
    >>> from mpl_toolkits.mplot3d import proj3d
    >>> # set input data
    >>> points = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0],
    ...                    [0, 1, 0], [0, -1, 0], [-1, 0, 0], ])

    Calculate the spherical Voronoi diagram:

    >>> radius = 1
    >>> center = np.array([0, 0, 0])
    >>> sv = SphericalVoronoi(points, radius, center)

    Generate plot:

    >>> # sort vertices (optional, helpful for plotting)
    >>> sv.sort_vertices_of_regions()
    >>> t_vals = np.linspace(0, 1, 2000)
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> # plot the unit sphere for reference (optional)
    >>> u = np.linspace(0, 2 * np.pi, 100)
    >>> v = np.linspace(0, np.pi, 100)
    >>> x = np.outer(np.cos(u), np.sin(v))
    >>> y = np.outer(np.sin(u), np.sin(v))
    >>> z = np.outer(np.ones(np.size(u)), np.cos(v))
    >>> ax.plot_surface(x, y, z, color='y', alpha=0.1)
    >>> # plot generator points
    >>> ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b')
    >>> # plot Voronoi vertices
    >>> ax.scatter(sv.vertices[:, 0], sv.vertices[:, 1], sv.vertices[:, 2],
    ...                    c='g')
    >>> # indicate Voronoi regions (as Euclidean polygons)
    >>> for region in sv.regions:
    ...    n = len(region)
    ...    for i in range(n):
    ...        start = sv.vertices[region][i]
    ...        end = sv.vertices[region][(i + 1) % n]
    ...        result = geometric_slerp(start, end, t_vals)
    ...        ax.plot(result[..., 0],
    ...                result[..., 1],
    ...                result[..., 2],
    ...                c='k')
    >>> ax.azim = 10
    >>> ax.elev = 40
    >>> _ = ax.set_xticks([])
    >>> _ = ax.set_yticks([])
    >>> _ = ax.set_zticks([])
    >>> fig.set_size_inches(4, 4)
    >>> plt.show()

    """
    def __init__(self, points, radius=1, center=None, threshold=1e-06):

        if radius is None:
            radius = 1.
            warnings.warn('`radius` is `None`. '
                          'This will raise an error in a future version. '
                          'Please provide a floating point number '
                          '(i.e. `radius=1`).',
                          DeprecationWarning)

        self.radius = float(radius)
        self.points = np.array(points).astype(np.double)
        self._dim = len(points[0])
        if center is None:
            self.center = np.zeros(self._dim)
        else:
            self.center = np.array(center, dtype=float)

        # test degenerate input
        self._rank = np.linalg.matrix_rank(self.points - self.points[0],
                                           tol=threshold * self.radius)
        if self._rank < self._dim:
            raise ValueError("Rank of input points must be at least {0}".format(self._dim))

        if cKDTree(self.points).query_pairs(threshold * self.radius):
            raise ValueError("Duplicate generators present.")

        radii = np.linalg.norm(self.points - self.center, axis=1)
        max_discrepancy = np.abs(radii - self.radius).max()
        if max_discrepancy >= threshold * self.radius:
            raise ValueError("Radius inconsistent with generators.")

        self._calc_vertices_regions()

    def _calc_vertices_regions(self):
        """
        Calculates the Voronoi vertices and regions of the generators stored
        in self.points. The vertices will be stored in self.vertices and the
        regions in self.regions.

        This algorithm was discussed at PyData London 2015 by
        Tyler Reddy, Ross Hemsley and Nikolai Nowaczyk
        """
        # get Convex Hull
        conv = scipy.spatial.ConvexHull(self.points)
        # get circumcenters of Convex Hull triangles from facet equations
        # for 3D input circumcenters will have shape: (2N-4, 3)
        self.vertices = self.radius * conv.equations[:, :-1] + self.center
        self._simplices = conv.simplices
        # calculate regions from triangulation
        # for 3D input simplex_indices will have shape: (2N-4,)
        simplex_indices = np.arange(len(self._simplices))
        # for 3D input tri_indices will have shape: (6N-12,)
        tri_indices = np.column_stack([simplex_indices] * self._dim).ravel()
        # for 3D input point_indices will have shape: (6N-12,)
        point_indices = self._simplices.ravel()
        # for 3D input indices will have shape: (6N-12,)
        indices = np.argsort(point_indices, kind='mergesort')
        # for 3D input flattened_groups will have shape: (6N-12,)
        flattened_groups = tri_indices[indices].astype(np.intp)
        # intervals will have shape: (N+1,)
        intervals = np.cumsum(np.bincount(point_indices + 1))
        # split flattened groups to get nested list of unsorted regions
        groups = [list(flattened_groups[intervals[i]:intervals[i + 1]])
                  for i in range(len(intervals) - 1)]
        self.regions = groups

    def sort_vertices_of_regions(self):
        """Sort indices of the vertices to be (counter-)clockwise ordered.

        Raises
        ------
        TypeError
            If the points are not three-dimensional.

        Notes
        -----
        For each region in regions, it sorts the indices of the Voronoi
        vertices such that the resulting points are in a clockwise or
        counterclockwise order around the generator point.

        This is done as follows: Recall that the n-th region in regions
        surrounds the n-th generator in points and that the k-th
        Voronoi vertex in vertices is the circumcenter of the k-th triangle
        in self._simplices.  For each region n, we choose the first triangle
        (=Voronoi vertex) in self._simplices and a vertex of that triangle
        not equal to the center n. These determine a unique neighbor of that
        triangle, which is then chosen as the second triangle. The second
        triangle will have a unique vertex not equal to the current vertex or
        the center. This determines a unique neighbor of the second triangle,
        which is then chosen as the third triangle and so forth. We proceed
        through all the triangles (=Voronoi vertices) belonging to the
        generator in points and obtain a sorted version of the vertices
        of its surrounding region.
        """
        if self._dim != 3:
            raise TypeError("Only supported for three-dimensional point sets")
        _voronoi.sort_vertices_of_regions(self._simplices, self.regions)

    def _calculate_areas_3d(self):
        self.sort_vertices_of_regions()
        sizes = [len(region) for region in self.regions]
        csizes = np.cumsum(sizes)
        num_regions = csizes[-1]

        # We create a set of triangles consisting of one point and two Voronoi
        # vertices. The vertices of each triangle are adjacent in the sorted
        # regions list.
        point_indices = [i for i, size in enumerate(sizes)
                         for j in range(size)]

        nbrs1 = np.array([r for region in self.regions for r in region])

        # The calculation of nbrs2 is a vectorized version of:
        # np.array([r for region in self.regions for r in np.roll(region, 1)])
        nbrs2 = np.roll(nbrs1, 1)
        indices = np.roll(csizes, 1)
        indices[0] = 0
        nbrs2[indices] = nbrs1[csizes - 1]

        # Normalize points and vertices.
        pnormalized = (self.points - self.center) / self.radius
        vnormalized = (self.vertices - self.center) / self.radius

        # Create the complete set of triangles and calculate their solid angles
        triangles = np.hstack([pnormalized[point_indices],
                               vnormalized[nbrs1],
                               vnormalized[nbrs2]
                               ]).reshape((num_regions, 3, 3))
        triangle_solid_angles = calculate_solid_angles(triangles)

        # Sum the solid angles of the triangles in each region
        solid_angles = np.cumsum(triangle_solid_angles)[csizes - 1]
        solid_angles[1:] -= solid_angles[:-1]

        # Get polygon areas using A = omega * r**2
        return solid_angles * self.radius**2

    def _calculate_areas_2d(self):
        # Find start and end points of arcs
        arcs = self.points[self._simplices] - self.center

        # Calculate the angle subtended by arcs
        cosine = np.einsum('ij,ij->i', arcs[:, 0], arcs[:, 1])
        sine = np.abs(np.linalg.det(arcs))
        theta = np.arctan2(sine, cosine)

        # Get areas using A = r * theta
        areas = self.radius * theta

        # Correct arcs which go the wrong way (single-hemisphere inputs)
        signs = np.sign(np.einsum('ij,ij->i', arcs[:, 0],
                                              self.vertices - self.center))
        indices = np.where(signs < 0)
        areas[indices] = 2 * np.pi * self.radius - areas[indices]
        return areas

    def calculate_areas(self):
        """Calculates the areas of the Voronoi regions.

        For 2D point sets, the regions are circular arcs. The sum of the areas
        is `2 * pi * radius`.

        For 3D point sets, the regions are spherical polygons. The sum of the
        areas is `4 * pi * radius**2`.

        .. versionadded:: 1.5.0

        Returns
        -------
        areas : double array of shape (npoints,)
            The areas of the Voronoi regions.
        """
        if self._dim == 2:
            return self._calculate_areas_2d()
        elif self._dim == 3:
            return self._calculate_areas_3d()
        else:
            raise TypeError("Only supported for 2D and 3D point sets")
