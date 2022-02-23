"""flood_fill.py - in place flood fill algorithm

This module provides a function to fill all equal (or within tolerance) values
connected to a given seed point with a different value.
"""

import numpy as np

from .._shared.utils import deprecate_kwarg
from ._flood_fill_cy import _flood_fill_equal, _flood_fill_tolerance
from ._util import (_offsets_to_raveled_neighbors, _resolve_neighborhood,
                    _set_border_values,)


@deprecate_kwarg(kwarg_mapping={'selem': 'footprint'}, removed_version="1.0",
                 deprecated_version="0.19")
def flood_fill(image, seed_point, new_value, *, footprint=None,
               connectivity=None, tolerance=None, in_place=False):
    """Perform flood filling on an image.

    Starting at a specific `seed_point`, connected points equal or within
    `tolerance` of the seed value are found, then set to `new_value`.

    Parameters
    ----------
    image : ndarray
        An n-dimensional array.
    seed_point : tuple or int
        The point in `image` used as the starting point for the flood fill.  If
        the image is 1D, this point may be given as an integer.
    new_value : `image` type
        New value to set the entire fill.  This must be chosen in agreement
        with the dtype of `image`.
    footprint : ndarray, optional
        The footprint (structuring element) used to determine the neighborhood
        of each evaluated pixel. It must contain only 1's and 0's, have the
        same number of dimensions as `image`. If not given, all adjacent pixels
        are considered as part of the neighborhood (fully connected).
    connectivity : int, optional
        A number used to determine the neighborhood of each evaluated pixel.
        Adjacent pixels whose squared distance from the center is less than or
        equal to `connectivity` are considered neighbors. Ignored if
        `footprint` is not None.
    tolerance : float or int, optional
        If None (default), adjacent values must be strictly equal to the
        value of `image` at `seed_point` to be filled.  This is fastest.
        If a tolerance is provided, adjacent points with values within plus or
        minus tolerance from the seed point are filled (inclusive).
    in_place : bool, optional
        If True, flood filling is applied to `image` in place.  If False, the
        flood filled result is returned without modifying the input `image`
        (default).

    Returns
    -------
    filled : ndarray
        An array with the same shape as `image` is returned, with values in
        areas connected to and equal (or within tolerance of) the seed point
        replaced with `new_value`.

    Notes
    -----
    The conceptual analogy of this operation is the 'paint bucket' tool in many
    raster graphics programs.

    Examples
    --------
    >>> from skimage.morphology import flood_fill
    >>> image = np.zeros((4, 7), dtype=int)
    >>> image[1:3, 1:3] = 1
    >>> image[3, 0] = 1
    >>> image[1:3, 4:6] = 2
    >>> image[3, 6] = 3
    >>> image
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 0, 2, 2, 0],
           [0, 1, 1, 0, 2, 2, 0],
           [1, 0, 0, 0, 0, 0, 3]])

    Fill connected ones with 5, with full connectivity (diagonals included):

    >>> flood_fill(image, (1, 1), 5)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 5, 5, 0, 2, 2, 0],
           [0, 5, 5, 0, 2, 2, 0],
           [5, 0, 0, 0, 0, 0, 3]])

    Fill connected ones with 5, excluding diagonal points (connectivity 1):

    >>> flood_fill(image, (1, 1), 5, connectivity=1)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 5, 5, 0, 2, 2, 0],
           [0, 5, 5, 0, 2, 2, 0],
           [1, 0, 0, 0, 0, 0, 3]])

    Fill with a tolerance:

    >>> flood_fill(image, (0, 0), 5, tolerance=1)
    array([[5, 5, 5, 5, 5, 5, 5],
           [5, 5, 5, 5, 2, 2, 5],
           [5, 5, 5, 5, 2, 2, 5],
           [5, 5, 5, 5, 5, 5, 3]])
    """
    mask = flood(image, seed_point, footprint=footprint,
                 connectivity=connectivity, tolerance=tolerance)

    if not in_place:
        image = image.copy()

    image[mask] = new_value
    return image


@deprecate_kwarg(kwarg_mapping={'selem': 'footprint'}, removed_version="1.0",
                 deprecated_version="0.19")
def flood(image, seed_point, *, footprint=None, connectivity=None,
          tolerance=None):
    """Mask corresponding to a flood fill.

    Starting at a specific `seed_point`, connected points equal or within
    `tolerance` of the seed value are found.

    Parameters
    ----------
    image : ndarray
        An n-dimensional array.
    seed_point : tuple or int
        The point in `image` used as the starting point for the flood fill.  If
        the image is 1D, this point may be given as an integer.
    footprint : ndarray, optional
        The footprint (structuring element) used to determine the neighborhood
        of each evaluated pixel. It must contain only 1's and 0's, have the
        same number of dimensions as `image`. If not given, all adjacent pixels
        are considered as part of the neighborhood (fully connected).
    connectivity : int, optional
        A number used to determine the neighborhood of each evaluated pixel.
        Adjacent pixels whose squared distance from the center is larger or
        equal to `connectivity` are considered neighbors. Ignored if
        `footprint` is not None.
    tolerance : float or int, optional
        If None (default), adjacent values must be strictly equal to the
        initial value of `image` at `seed_point`.  This is fastest.  If a value
        is given, a comparison will be done at every point and if within
        tolerance of the initial value will also be filled (inclusive).

    Returns
    -------
    mask : ndarray
        A Boolean array with the same shape as `image` is returned, with True
        values for areas connected to and equal (or within tolerance of) the
        seed point.  All other values are False.

    Notes
    -----
    The conceptual analogy of this operation is the 'paint bucket' tool in many
    raster graphics programs.  This function returns just the mask
    representing the fill.

    If indices are desired rather than masks for memory reasons, the user can
    simply run `numpy.nonzero` on the result, save the indices, and discard
    this mask.

    Examples
    --------
    >>> from skimage.morphology import flood
    >>> image = np.zeros((4, 7), dtype=int)
    >>> image[1:3, 1:3] = 1
    >>> image[3, 0] = 1
    >>> image[1:3, 4:6] = 2
    >>> image[3, 6] = 3
    >>> image
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 0, 2, 2, 0],
           [0, 1, 1, 0, 2, 2, 0],
           [1, 0, 0, 0, 0, 0, 3]])

    Fill connected ones with 5, with full connectivity (diagonals included):

    >>> mask = flood(image, (1, 1))
    >>> image_flooded = image.copy()
    >>> image_flooded[mask] = 5
    >>> image_flooded
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 5, 5, 0, 2, 2, 0],
           [0, 5, 5, 0, 2, 2, 0],
           [5, 0, 0, 0, 0, 0, 3]])

    Fill connected ones with 5, excluding diagonal points (connectivity 1):

    >>> mask = flood(image, (1, 1), connectivity=1)
    >>> image_flooded = image.copy()
    >>> image_flooded[mask] = 5
    >>> image_flooded
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 5, 5, 0, 2, 2, 0],
           [0, 5, 5, 0, 2, 2, 0],
           [1, 0, 0, 0, 0, 0, 3]])

    Fill with a tolerance:

    >>> mask = flood(image, (0, 0), tolerance=1)
    >>> image_flooded = image.copy()
    >>> image_flooded[mask] = 5
    >>> image_flooded
    array([[5, 5, 5, 5, 5, 5, 5],
           [5, 5, 5, 5, 2, 2, 5],
           [5, 5, 5, 5, 2, 2, 5],
           [5, 5, 5, 5, 5, 5, 3]])
    """
    # Correct start point in ravelled image - only copy if non-contiguous
    image = np.asarray(image)
    if image.flags.f_contiguous is True:
        order = 'F'
    elif image.flags.c_contiguous is True:
        order = 'C'
    else:
        image = np.ascontiguousarray(image)
        order = 'C'

    # Shortcut for rank zero
    if 0 in image.shape:
        return np.zeros(image.shape, dtype=bool)

    # Convenience for 1d input
    try:
        iter(seed_point)
    except TypeError:
        seed_point = (seed_point,)

    seed_value = image[seed_point]
    seed_point = tuple(np.asarray(seed_point) % image.shape)

    footprint = _resolve_neighborhood(footprint, connectivity, image.ndim)

    # Must annotate borders
    working_image = np.pad(image, 1, mode='constant',
                           constant_values=image.min())

    # Stride-aware neighbors - works for both C- and Fortran-contiguity
    ravelled_seed_idx = np.ravel_multi_index([i + 1 for i in seed_point],
                                             working_image.shape, order=order)
    neighbor_offsets = _offsets_to_raveled_neighbors(
        working_image.shape, footprint, center=((1,) * image.ndim),
        order=order)

    # Use a set of flags; see _flood_fill_cy.pyx for meanings
    flags = np.zeros(working_image.shape, dtype=np.uint8, order=order)
    _set_border_values(flags, value=2)

    try:
        if tolerance is not None:
            # Check if tolerance could create overflow problems
            try:
                max_value = np.finfo(working_image.dtype).max
                min_value = np.finfo(working_image.dtype).min
            except ValueError:
                max_value = np.iinfo(working_image.dtype).max
                min_value = np.iinfo(working_image.dtype).min

            high_tol = min(max_value, seed_value + tolerance)
            low_tol = max(min_value, seed_value - tolerance)

            _flood_fill_tolerance(working_image.ravel(order),
                                  flags.ravel(order),
                                  neighbor_offsets,
                                  ravelled_seed_idx,
                                  seed_value,
                                  low_tol,
                                  high_tol)
        else:
            _flood_fill_equal(working_image.ravel(order),
                              flags.ravel(order),
                              neighbor_offsets,
                              ravelled_seed_idx,
                              seed_value)
    except TypeError:
        if working_image.dtype == np.float16:
            # Provide the user with clearer error message
            raise TypeError("dtype of `image` is float16 which is not "
                            "supported, try upcasting to float32")
        else:
            raise

    # Output what the user requested; view does not create a new copy.
    return flags[(slice(1, -1),) * image.ndim].view(bool)
