"""Convex Hull."""
from itertools import product
import numpy as np
from scipy.spatial import ConvexHull
from ..measure.pnpoly import grid_points_in_poly
from ._convex_hull import possible_hull
from ..measure._label import label
from ..util import unique_rows
from .._shared.utils import warn

try:
    # Should be public API of scipy spatial once #15003 is released
    # see https://github.com/scipy/scipy/pull/15003
    from scipy.spatial import QhullError
except ImportError:
    from scipy.spatial.qhull import QhullError

__all__ = ['convex_hull_image', 'convex_hull_object']


def _offsets_diamond(ndim):
    offsets = np.zeros((2 * ndim, ndim))
    for vertex, (axis, offset) in enumerate(product(range(ndim), (-0.5, 0.5))):
        offsets[vertex, axis] = offset
    return offsets


def _check_coords_in_hull(gridcoords, hull_equations, tolerance):
    r"""Checks all the coordinates for inclusiveness in the convex hull.

    Parameters
    ----------
    gridcoords : (M, N) ndarray
        Coordinates of ``N`` points in ``M`` dimensions.
    hull_equations : (M, N) ndarray
        Hyperplane equations of the facets of the convex hull.
    tolerance : float
        Tolerance when determining whether a point is inside the hull. Due
        to numerical floating point errors, a tolerance of 0 can result in
        some points erroneously being classified as being outside the hull.

    Returns
    -------
    coords_in_hull : ndarray of bool
        Binary 1D ndarray representing points in n-dimensional space
        with value ``True`` set for points inside the convex hull.

    Notes
    -----
    Checking the inclusiveness of coordinates in a convex hull requires
    intermediate calculations of dot products which are memory-intensive.
    Thus, the convex hull equations are checked individually with all
    coordinates to keep within the memory limit.

    References
    ----------
    .. [1] https://github.com/scikit-image/scikit-image/issues/5019

    """
    ndim, n_coords = gridcoords.shape
    n_hull_equations = hull_equations.shape[0]
    coords_in_hull = np.ones(n_coords, dtype=bool)

    # Pre-allocate arrays to cache intermediate results for reducing overheads
    dot_array = np.empty(n_coords, dtype=np.float64)
    test_ineq_temp = np.empty(n_coords, dtype=np.float64)
    coords_single_ineq = np.empty(n_coords, dtype=bool)

    # A point is in the hull if it satisfies all of the hull's inequalities
    for idx in range(n_hull_equations):
        # Tests a hyperplane equation on all coordinates of volume
        np.dot(hull_equations[idx, :ndim], gridcoords, out=dot_array)
        np.add(dot_array, hull_equations[idx, ndim:], out=test_ineq_temp)
        np.less(test_ineq_temp, tolerance, out=coords_single_ineq)
        coords_in_hull *= coords_single_ineq

    return coords_in_hull


def convex_hull_image(image, offset_coordinates=True, tolerance=1e-10):
    """Compute the convex hull image of a binary image.

    The convex hull is the set of pixels included in the smallest convex
    polygon that surround all white pixels in the input image.

    Parameters
    ----------
    image : array
        Binary input image. This array is cast to bool before processing.
    offset_coordinates : bool, optional
        If ``True``, a pixel at coordinate, e.g., (4, 7) will be represented
        by coordinates (3.5, 7), (4.5, 7), (4, 6.5), and (4, 7.5). This adds
        some "extent" to a pixel when computing the hull.
    tolerance : float, optional
        Tolerance when determining whether a point is inside the hull. Due
        to numerical floating point errors, a tolerance of 0 can result in
        some points erroneously being classified as being outside the hull.

    Returns
    -------
    hull : (M, N) array of bool
        Binary image with pixels in convex hull set to True.

    References
    ----------
    .. [1] https://blogs.mathworks.com/steve/2011/10/04/binary-image-convex-hull-algorithm-notes/

    """
    ndim = image.ndim
    if np.count_nonzero(image) == 0:
        warn("Input image is entirely zero, no valid convex hull. "
             "Returning empty image", UserWarning)
        return np.zeros(image.shape, dtype=bool)
    # In 2D, we do an optimisation by choosing only pixels that are
    # the starting or ending pixel of a row or column.  This vastly
    # limits the number of coordinates to examine for the virtual hull.
    if ndim == 2:
        coords = possible_hull(np.ascontiguousarray(image, dtype=np.uint8))
    else:
        coords = np.transpose(np.nonzero(image))
        if offset_coordinates:
            # when offsetting, we multiply number of vertices by 2 * ndim.
            # therefore, we reduce the number of coordinates by using a
            # convex hull on the original set, before offsetting.
            try:
                hull0 = ConvexHull(coords)
            except QhullError as err:
                warn(f"Failed to get convex hull image. "
                     f"Returning empty image, see error message below:\n"
                     f"{err}")
                return np.zeros(image.shape, dtype=bool)
            coords = hull0.points[hull0.vertices]

    # Add a vertex for the middle of each pixel edge
    if offset_coordinates:
        offsets = _offsets_diamond(image.ndim)
        coords = (coords[:, np.newaxis, :] + offsets).reshape(-1, ndim)

    # repeated coordinates can *sometimes* cause problems in
    # scipy.spatial.ConvexHull, so we remove them.
    coords = unique_rows(coords)

    # Find the convex hull
    try:
        hull = ConvexHull(coords)
    except QhullError as err:
        warn(f"Failed to get convex hull image. "
             f"Returning empty image, see error message below:\n"
             f"{err}")
        return np.zeros(image.shape, dtype=bool)
    vertices = hull.points[hull.vertices]

    # If 2D, use fast Cython function to locate convex hull pixels
    if ndim == 2:
        mask = grid_points_in_poly(image.shape, vertices)
    else:
        gridcoords = np.reshape(np.mgrid[tuple(map(slice, image.shape))],
                                (ndim, -1))

        coords_in_hull = _check_coords_in_hull(gridcoords,
                                               hull.equations, tolerance)
        mask = np.reshape(coords_in_hull, image.shape)

    return mask


def convex_hull_object(image, *, connectivity=2):
    r"""Compute the convex hull image of individual objects in a binary image.

    The convex hull is the set of pixels included in the smallest convex
    polygon that surround all white pixels in the input image.

    Parameters
    ----------
    image : (M, N) ndarray
        Binary input image.
    connectivity : {1, 2}, int, optional
        Determines the neighbors of each pixel. Adjacent elements
        within a squared distance of ``connectivity`` from pixel center
        are considered neighbors.::

            1-connectivity      2-connectivity
                  [ ]           [ ]  [ ]  [ ]
                   |               \  |  /
             [ ]--[x]--[ ]      [ ]--[x]--[ ]
                   |               /  |  \
                  [ ]           [ ]  [ ]  [ ]

    Returns
    -------
    hull : ndarray of bool
        Binary image with pixels inside convex hull set to ``True``.

    Notes
    -----
    This function uses ``skimage.morphology.label`` to define unique objects,
    finds the convex hull of each using ``convex_hull_image``, and combines
    these regions with logical OR. Be aware the convex hulls of unconnected
    objects may overlap in the result. If this is suspected, consider using
    convex_hull_image separately on each object or adjust ``connectivity``.
    """
    if image.ndim > 2:
        raise ValueError("Input must be a 2D image")

    if connectivity not in (1, 2):
        raise ValueError('`connectivity` must be either 1 or 2.')

    labeled_im = label(image, connectivity=connectivity, background=0)
    convex_obj = np.zeros(image.shape, dtype=bool)
    convex_img = np.zeros(image.shape, dtype=bool)

    for i in range(1, labeled_im.max() + 1):
        convex_obj = convex_hull_image(labeled_im == i)
        convex_img = np.logical_or(convex_img, convex_obj)

    return convex_img
