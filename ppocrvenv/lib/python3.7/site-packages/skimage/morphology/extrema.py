"""extrema.py - local minima and maxima

This module provides functions to find local maxima and minima of an image.
Here, local maxima (minima) are defined as connected sets of pixels with equal
gray level which is strictly greater (smaller) than the gray level of all
pixels in direct neighborhood of the connected set. In addition, the module
provides the related functions h-maxima and h-minima.

Soille, P. (2003). Morphological Image Analysis: Principles and Applications
(2nd ed.), Chapter 6. Springer-Verlag New York, Inc.
"""
import numpy as np

from .._shared.utils import deprecate_kwarg, warn
from ..util import dtype_limits, invert, crop
from . import grayreconstruct, _util
from ._extrema_cy import _local_maxima


def _add_constant_clip(image, const_value):
    """Add constant to the image while handling overflow issues gracefully.
    """
    min_dtype, max_dtype = dtype_limits(image, clip_negative=False)

    if const_value > (max_dtype - min_dtype):
        raise ValueError("The added constant is not compatible"
                         "with the image data type.")

    result = image + const_value
    result[image > max_dtype-const_value] = max_dtype
    return(result)


def _subtract_constant_clip(image, const_value):
    """Subtract constant from image while handling underflow issues.
    """
    min_dtype, max_dtype = dtype_limits(image, clip_negative=False)

    if const_value > (max_dtype-min_dtype):
        raise ValueError("The subtracted constant is not compatible"
                         "with the image data type.")

    result = image - const_value
    result[image < (const_value + min_dtype)] = min_dtype
    return(result)


@deprecate_kwarg(kwarg_mapping={'selem': 'footprint'}, removed_version="1.0",
                 deprecated_version="0.19")
def h_maxima(image, h, footprint=None):
    """Determine all maxima of the image with height >= h.

    The local maxima are defined as connected sets of pixels with equal
    gray level strictly greater than the gray level of all pixels in direct
    neighborhood of the set.

    A local maximum M of height h is a local maximum for which
    there is at least one path joining M with an equal or higher local maximum
    on which the minimal value is f(M) - h (i.e. the values along the path
    are not decreasing by more than h with respect to the maximum's value)
    and no path to an equal or higher local maximum for which the minimal
    value is greater.

    The global maxima of the image are also found by this function.

    Parameters
    ----------
    image : ndarray
        The input image for which the maxima are to be calculated.
    h : unsigned integer
        The minimal height of all extracted maxima.
    footprint : ndarray, optional
        The neighborhood expressed as an n-D array of 1's and 0's.
        Default is the ball of radius 1 according to the maximum norm
        (i.e. a 3x3 square for 2D images, a 3x3x3 cube for 3D images, etc.)

    Returns
    -------
    h_max : ndarray
        The local maxima of height >= h and the global maxima.
        The resulting image is a binary image, where pixels belonging to
        the determined maxima take value 1, the others take value 0.

    See Also
    --------
    skimage.morphology.extrema.h_minima
    skimage.morphology.extrema.local_maxima
    skimage.morphology.extrema.local_minima

    References
    ----------
    .. [1] Soille, P., "Morphological Image Analysis: Principles and
           Applications" (Chapter 6), 2nd edition (2003), ISBN 3540429883.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.morphology import extrema

    We create an image (quadratic function with a maximum in the center and
    4 additional constant maxima.
    The heights of the maxima are: 1, 21, 41, 61, 81

    >>> w = 10
    >>> x, y = np.mgrid[0:w,0:w]
    >>> f = 20 - 0.2*((x - w/2)**2 + (y-w/2)**2)
    >>> f[2:4,2:4] = 40; f[2:4,7:9] = 60; f[7:9,2:4] = 80; f[7:9,7:9] = 100
    >>> f = f.astype(int)

    We can calculate all maxima with a height of at least 40:

    >>> maxima = extrema.h_maxima(f, 40)

    The resulting image will contain 3 local maxima.
    """

    # Check for h value that is larger then range of the image. If this
    # is True then there are no h-maxima in the image.
    if h > np.ptp(image):
        return np.zeros(image.shape, dtype=np.uint8)

    # Check for floating point h value. For this to work properly
    # we need to explicitly convert image to float64.
    #
    # FIXME: This could give incorrect results if image is int64 and
    #        has a very high dynamic range. The dtype of image is
    #        changed to float64, and different integer values could
    #        become the same float due to rounding.
    #
    #   >>> ii64 = np.iinfo(np.int64)
    #   >>> a = np.array([ii64.max, ii64.max - 2])
    #   >>> a[0] == a[1]
    #   False
    #   >>> b = a.astype(np.float64)
    #   >>> b[0] == b[1]
    #   True
    #
    if np.issubdtype(type(h), np.floating) and \
       np.issubdtype(image.dtype, np.integer):
        if ((h % 1) != 0):
            warn('possible precision loss converting image to '
                 'floating point. To silence this warning, '
                 'ensure image and h have same data type.',
                 stacklevel=2)
            image = image.astype(float)
        else:
            h = image.dtype.type(h)

    if (h == 0):
        raise ValueError("h = 0 is ambiguous, use local_maxima() "
                         "instead?")

    if np.issubdtype(image.dtype, np.floating):
        # The purpose of the resolution variable is to allow for the
        # small rounding errors that inevitably occur when doing
        # floating point arithmetic. We want shifted_img to be
        # guaranteed to be h less than image. If we only subtract h
        # there may be pixels were shifted_img ends up being
        # slightly greater than image - h.
        #
        # The resolution is scaled based on the pixel values in the
        # image because floating point precision is relative. A
        # very large value of 1.0e10 will have a large precision,
        # say +-1.0e4, and a very small value of 1.0e-10 will have
        # a very small precision, say +-1.0e-16.
        #
        resolution = 2 * np.finfo(image.dtype).resolution * np.abs(image)
        shifted_img = image - h - resolution
    else:
        shifted_img = _subtract_constant_clip(image, h)

    rec_img = grayreconstruct.reconstruction(shifted_img, image,
                                             method='dilation',
                                             footprint=footprint)
    residue_img = image - rec_img
    return (residue_img >= h).astype(np.uint8)


@deprecate_kwarg(kwarg_mapping={'selem': 'footprint'}, removed_version="1.0",
                 deprecated_version="0.19")
def h_minima(image, h, footprint=None):
    """Determine all minima of the image with depth >= h.

    The local minima are defined as connected sets of pixels with equal
    gray level strictly smaller than the gray levels of all pixels in direct
    neighborhood of the set.

    A local minimum M of depth h is a local minimum for which
    there is at least one path joining M with an equal or lower local minimum
    on which the maximal value is f(M) + h (i.e. the values along the path
    are not increasing by more than h with respect to the minimum's value)
    and no path to an equal or lower local minimum for which the maximal
    value is smaller.

    The global minima of the image are also found by this function.

    Parameters
    ----------
    image : ndarray
        The input image for which the minima are to be calculated.
    h : unsigned integer
        The minimal depth of all extracted minima.
    footprint : ndarray, optional
        The neighborhood expressed as an n-D array of 1's and 0's.
        Default is the ball of radius 1 according to the maximum norm
        (i.e. a 3x3 square for 2D images, a 3x3x3 cube for 3D images, etc.)

    Returns
    -------
    h_min : ndarray
        The local minima of depth >= h and the global minima.
        The resulting image is a binary image, where pixels belonging to
        the determined minima take value 1, the others take value 0.

    See Also
    --------
    skimage.morphology.extrema.h_maxima
    skimage.morphology.extrema.local_maxima
    skimage.morphology.extrema.local_minima

    References
    ----------
    .. [1] Soille, P., "Morphological Image Analysis: Principles and
           Applications" (Chapter 6), 2nd edition (2003), ISBN 3540429883.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.morphology import extrema

    We create an image (quadratic function with a minimum in the center and
    4 additional constant maxima.
    The depth of the minima are: 1, 21, 41, 61, 81

    >>> w = 10
    >>> x, y = np.mgrid[0:w,0:w]
    >>> f = 180 + 0.2*((x - w/2)**2 + (y-w/2)**2)
    >>> f[2:4,2:4] = 160; f[2:4,7:9] = 140; f[7:9,2:4] = 120; f[7:9,7:9] = 100
    >>> f = f.astype(int)

    We can calculate all minima with a depth of at least 40:

    >>> minima = extrema.h_minima(f, 40)

    The resulting image will contain 3 local minima.
    """
    if h > np.ptp(image):
        return np.zeros(image.shape, dtype=np.uint8)

    if np.issubdtype(type(h), np.floating) and \
       np.issubdtype(image.dtype, np.integer):
        if ((h % 1) != 0):
            warn('possible precision loss converting image to '
                 'floating point. To silence this warning, '
                 'ensure image and h have same data type.',
                 stacklevel=2)
            image = image.astype(float)
        else:
            h = image.dtype.type(h)

    if (h == 0):
        raise ValueError("h = 0 is ambiguous, use local_minima() "
                         "instead?")

    if np.issubdtype(image.dtype, np.floating):
        resolution = 2 * np.finfo(image.dtype).resolution * np.abs(image)
        shifted_img = image + h + resolution
    else:
        shifted_img = _add_constant_clip(image, h)

    rec_img = grayreconstruct.reconstruction(shifted_img, image,
                                             method='erosion',
                                             footprint=footprint)
    residue_img = rec_img - image
    return (residue_img >= h).astype(np.uint8)


@deprecate_kwarg(kwarg_mapping={'selem': 'footprint'}, removed_version="1.0",
                 deprecated_version="0.19")
def local_maxima(image, footprint=None, connectivity=None, indices=False,
                 allow_borders=True):
    """Find local maxima of n-dimensional array.

    The local maxima are defined as connected sets of pixels with equal gray
    level (plateaus) strictly greater than the gray levels of all pixels in the
    neighborhood.

    Parameters
    ----------
    image : ndarray
        An n-dimensional array.
    footprint : ndarray, optional
        The footprint (structuring element) used to determine the neighborhood
        of each evaluated pixel (``True`` denotes a connected pixel). It must
        be a boolean array and have the same number of dimensions as `image`.
        If neither `footprint` nor `connectivity` are given, all adjacent
        pixels are considered as part of the neighborhood.
    connectivity : int, optional
        A number used to determine the neighborhood of each evaluated pixel.
        Adjacent pixels whose squared distance from the center is less than or
        equal to `connectivity` are considered neighbors. Ignored if
        `footprint` is not None.
    indices : bool, optional
        If True, the output will be a tuple of one-dimensional arrays
        representing the indices of local maxima in each dimension. If False,
        the output will be a boolean array with the same shape as `image`.
    allow_borders : bool, optional
        If true, plateaus that touch the image border are valid maxima.

    Returns
    -------
    maxima : ndarray or tuple[ndarray]
        If `indices` is false, a boolean array with the same shape as `image`
        is returned with ``True`` indicating the position of local maxima
        (``False`` otherwise). If `indices` is true, a tuple of one-dimensional
        arrays containing the coordinates (indices) of all found maxima.

    Warns
    -----
    UserWarning
        If `allow_borders` is false and any dimension of the given `image` is
        shorter than 3 samples, maxima can't exist and a warning is shown.

    See Also
    --------
    skimage.morphology.local_minima
    skimage.morphology.h_maxima
    skimage.morphology.h_minima

    Notes
    -----
    This function operates on the following ideas:

    1. Make a first pass over the image's last dimension and flag candidates
       for local maxima by comparing pixels in only one direction.
       If the pixels aren't connected in the last dimension all pixels are
       flagged as candidates instead.

    For each candidate:

    2. Perform a flood-fill to find all connected pixels that have the same
       gray value and are part of the plateau.
    3. Consider the connected neighborhood of a plateau: if no bordering sample
       has a higher gray level, mark the plateau as a definite local maximum.

    Examples
    --------
    >>> from skimage.morphology import local_maxima
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

    Find local maxima by comparing to all neighboring pixels (maximal
    connectivity):

    >>> local_maxima(image)
    array([[False, False, False, False, False, False, False],
           [False,  True,  True, False, False, False, False],
           [False,  True,  True, False, False, False, False],
           [ True, False, False, False, False, False,  True]])
    >>> local_maxima(image, indices=True)
    (array([1, 1, 2, 2, 3, 3]), array([1, 2, 1, 2, 0, 6]))

    Find local maxima without comparing to diagonal pixels (connectivity 1):

    >>> local_maxima(image, connectivity=1)
    array([[False, False, False, False, False, False, False],
           [False,  True,  True, False,  True,  True, False],
           [False,  True,  True, False,  True,  True, False],
           [ True, False, False, False, False, False,  True]])

    and exclude maxima that border the image edge:

    >>> local_maxima(image, connectivity=1, allow_borders=False)
    array([[False, False, False, False, False, False, False],
           [False,  True,  True, False,  True,  True, False],
           [False,  True,  True, False,  True,  True, False],
           [False, False, False, False, False, False, False]])
    """
    image = np.asarray(image, order="C")
    if image.size == 0:
        # Return early for empty input
        if indices:
            # Make sure that output is a tuple of 1 empty array per dimension
            return np.nonzero(image)
        else:
            return np.zeros(image.shape, dtype=bool)

    if allow_borders:
        # Ensure that local maxima are always at least one smaller sample away
        # from the image border
        image = np.pad(image, 1, mode='constant', constant_values=image.min())

    # Array of flags used to store the state of each pixel during evaluation.
    # See _extrema_cy.pyx for their meaning
    flags = np.zeros(image.shape, dtype=np.uint8)
    _util._set_border_values(flags, value=3)

    if any(s < 3 for s in image.shape):
        # Warn and skip if any dimension is smaller than 3
        # -> no maxima can exist & footprint can't be applied
        warn(
            "maxima can't exist for an image with any dimension smaller 3 "
            "if borders aren't allowed",
            stacklevel=3
        )
    else:
        footprint = _util._resolve_neighborhood(footprint, connectivity,
                                                image.ndim)
        neighbor_offsets = _util._offsets_to_raveled_neighbors(
            image.shape, footprint, center=((1,) * image.ndim)
        )

        try:
            _local_maxima(image.ravel(), flags.ravel(), neighbor_offsets)
        except TypeError:
            if image.dtype == np.float16:
                # Provide the user with clearer error message
                raise TypeError("dtype of `image` is float16 which is not "
                                "supported, try upcasting to float32")
            else:
                raise  # Otherwise raise original message

    if allow_borders:
        # Revert padding performed at the beginning of the function
        flags = crop(flags, 1)
    else:
        # No padding was performed but set edge values back to 0
        _util._set_border_values(flags, value=0)

    if indices:
        return np.nonzero(flags)
    else:
        return flags.view(bool)


@deprecate_kwarg(kwarg_mapping={'selem': 'footprint'}, removed_version="1.0",
                 deprecated_version="0.19")
def local_minima(image, footprint=None, connectivity=None, indices=False,
                 allow_borders=True):
    """Find local minima of n-dimensional array.

    The local minima are defined as connected sets of pixels with equal gray
    level (plateaus) strictly smaller than the gray levels of all pixels in the
    neighborhood.

    Parameters
    ----------
    image : ndarray
        An n-dimensional array.
    footprint : ndarray, optional
        The footprint (structuring element) used to determine the neighborhood
        of each evaluated pixel (``True`` denotes a connected pixel). It must
        be a boolean array and have the same number of dimensions as `image`.
        If neither `footprint` nor `connectivity` are given, all adjacent
        pixels are considered as part of the neighborhood.
    connectivity : int, optional
        A number used to determine the neighborhood of each evaluated pixel.
        Adjacent pixels whose squared distance from the center is less than or
        equal to `connectivity` are considered neighbors. Ignored if
        `footprint` is not None.
    indices : bool, optional
        If True, the output will be a tuple of one-dimensional arrays
        representing the indices of local minima in each dimension. If False,
        the output will be a boolean array with the same shape as `image`.
    allow_borders : bool, optional
        If true, plateaus that touch the image border are valid minima.

    Returns
    -------
    minima : ndarray or tuple[ndarray]
        If `indices` is false, a boolean array with the same shape as `image`
        is returned with ``True`` indicating the position of local minima
        (``False`` otherwise). If `indices` is true, a tuple of one-dimensional
        arrays containing the coordinates (indices) of all found minima.

    See Also
    --------
    skimage.morphology.local_maxima
    skimage.morphology.h_maxima
    skimage.morphology.h_minima

    Notes
    -----
    This function operates on the following ideas:

    1. Make a first pass over the image's last dimension and flag candidates
       for local minima by comparing pixels in only one direction.
       If the pixels aren't connected in the last dimension all pixels are
       flagged as candidates instead.

    For each candidate:

    2. Perform a flood-fill to find all connected pixels that have the same
       gray value and are part of the plateau.
    3. Consider the connected neighborhood of a plateau: if no bordering sample
       has a smaller gray level, mark the plateau as a definite local minimum.

    Examples
    --------
    >>> from skimage.morphology import local_minima
    >>> image = np.zeros((4, 7), dtype=int)
    >>> image[1:3, 1:3] = -1
    >>> image[3, 0] = -1
    >>> image[1:3, 4:6] = -2
    >>> image[3, 6] = -3
    >>> image
    array([[ 0,  0,  0,  0,  0,  0,  0],
           [ 0, -1, -1,  0, -2, -2,  0],
           [ 0, -1, -1,  0, -2, -2,  0],
           [-1,  0,  0,  0,  0,  0, -3]])

    Find local minima by comparing to all neighboring pixels (maximal
    connectivity):

    >>> local_minima(image)
    array([[False, False, False, False, False, False, False],
           [False,  True,  True, False, False, False, False],
           [False,  True,  True, False, False, False, False],
           [ True, False, False, False, False, False,  True]])
    >>> local_minima(image, indices=True)
    (array([1, 1, 2, 2, 3, 3]), array([1, 2, 1, 2, 0, 6]))

    Find local minima without comparing to diagonal pixels (connectivity 1):

    >>> local_minima(image, connectivity=1)
    array([[False, False, False, False, False, False, False],
           [False,  True,  True, False,  True,  True, False],
           [False,  True,  True, False,  True,  True, False],
           [ True, False, False, False, False, False,  True]])

    and exclude minima that border the image edge:

    >>> local_minima(image, connectivity=1, allow_borders=False)
    array([[False, False, False, False, False, False, False],
           [False,  True,  True, False,  True,  True, False],
           [False,  True,  True, False,  True,  True, False],
           [False, False, False, False, False, False, False]])
    """
    return local_maxima(
        image=invert(image),
        footprint=footprint,
        connectivity=connectivity,
        indices=indices,
        allow_borders=allow_borders
    )
