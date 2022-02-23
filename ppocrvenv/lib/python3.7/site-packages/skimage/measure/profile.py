import numpy as np
from scipy import ndimage as ndi

from .._shared.utils import _validate_interpolation_order, _fix_ndimage_mode


def profile_line(image, src, dst, linewidth=1,
                 order=None, mode='reflect', cval=0.0,
                 *, reduce_func=np.mean):
    """Return the intensity profile of an image measured along a scan line.

    Parameters
    ----------
    image : ndarray, shape (M, N[, C])
        The image, either grayscale (2D array) or multichannel
        (3D array, where the final axis contains the channel
        information).
    src : array_like, shape (2, )
        The coordinates of the start point of the scan line.
    dst : array_like, shape (2, )
        The coordinates of the end point of the scan
        line. The destination point is *included* in the profile, in
        contrast to standard numpy indexing.
    linewidth : int, optional
        Width of the scan, perpendicular to the line
    order : int in {0, 1, 2, 3, 4, 5}, optional
        The order of the spline interpolation, default is 0 if
        image.dtype is bool and 1 otherwise. The order has to be in
        the range 0-5. See `skimage.transform.warp` for detail.
    mode : {'constant', 'nearest', 'reflect', 'mirror', 'wrap'}, optional
        How to compute any values falling outside of the image.
    cval : float, optional
        If `mode` is 'constant', what constant value to use outside the image.
    reduce_func : callable, optional
        Function used to calculate the aggregation of pixel values
        perpendicular to the profile_line direction when `linewidth` > 1.
        If set to None the unreduced array will be returned.

    Returns
    -------
    return_value : array
        The intensity profile along the scan line. The length of the profile
        is the ceil of the computed length of the scan line.

    Examples
    --------
    >>> x = np.array([[1, 1, 1, 2, 2, 2]])
    >>> img = np.vstack([np.zeros_like(x), x, x, x, np.zeros_like(x)])
    >>> img
    array([[0, 0, 0, 0, 0, 0],
           [1, 1, 1, 2, 2, 2],
           [1, 1, 1, 2, 2, 2],
           [1, 1, 1, 2, 2, 2],
           [0, 0, 0, 0, 0, 0]])
    >>> profile_line(img, (2, 1), (2, 4))
    array([1., 1., 2., 2.])
    >>> profile_line(img, (1, 0), (1, 6), cval=4)
    array([1., 1., 1., 2., 2., 2., 2.])

    The destination point is included in the profile, in contrast to
    standard numpy indexing.
    For example:

    >>> profile_line(img, (1, 0), (1, 6))  # The final point is out of bounds
    array([1., 1., 1., 2., 2., 2., 2.])
    >>> profile_line(img, (1, 0), (1, 5))  # This accesses the full first row
    array([1., 1., 1., 2., 2., 2.])

    For different reduce_func inputs:

    >>> profile_line(img, (1, 0), (1, 3), linewidth=3, reduce_func=np.mean)
    array([0.66666667, 0.66666667, 0.66666667, 1.33333333])
    >>> profile_line(img, (1, 0), (1, 3), linewidth=3, reduce_func=np.max)
    array([1, 1, 1, 2])
    >>> profile_line(img, (1, 0), (1, 3), linewidth=3, reduce_func=np.sum)
    array([2, 2, 2, 4])

    The unreduced array will be returned when `reduce_func` is None or when
    `reduce_func` acts on each pixel value individually.

    >>> profile_line(img, (1, 2), (4, 2), linewidth=3, order=0,
    ...     reduce_func=None)
    array([[1, 1, 2],
           [1, 1, 2],
           [1, 1, 2],
           [0, 0, 0]])
    >>> profile_line(img, (1, 0), (1, 3), linewidth=3, reduce_func=np.sqrt)
    array([[1.        , 1.        , 0.        ],
           [1.        , 1.        , 0.        ],
           [1.        , 1.        , 0.        ],
           [1.41421356, 1.41421356, 0.        ]])
    """

    order = _validate_interpolation_order(image.dtype, order)
    mode = _fix_ndimage_mode(mode)

    perp_lines = _line_profile_coordinates(src, dst, linewidth=linewidth)
    if image.ndim == 3:
        pixels = [ndi.map_coordinates(image[..., i], perp_lines,
                                      prefilter=order > 1,
                                      order=order, mode=mode,
                                      cval=cval) for i in
                  range(image.shape[2])]
        pixels = np.transpose(np.asarray(pixels), (1, 2, 0))
    else:
        pixels = ndi.map_coordinates(image, perp_lines, prefilter=order > 1,
                                     order=order, mode=mode, cval=cval)
    # The outputted array with reduce_func=None gives an array where the
    # row values (axis=1) are flipped. Here, we make this consistent.
    pixels = np.flip(pixels, axis=1)

    if reduce_func is None:
        intensities = pixels
    else:
        try:
            intensities = reduce_func(pixels, axis=1)
        except TypeError:  # function doesn't allow axis kwarg
            intensities = np.apply_along_axis(reduce_func, arr=pixels, axis=1)

    return intensities


def _line_profile_coordinates(src, dst, linewidth=1):
    """Return the coordinates of the profile of an image along a scan line.

    Parameters
    ----------
    src : 2-tuple of numeric scalar (float or int)
        The start point of the scan line.
    dst : 2-tuple of numeric scalar (float or int)
        The end point of the scan line.
    linewidth : int, optional
        Width of the scan, perpendicular to the line

    Returns
    -------
    coords : array, shape (2, N, C), float
        The coordinates of the profile along the scan line. The length of the
        profile is the ceil of the computed length of the scan line.

    Notes
    -----
    This is a utility method meant to be used internally by skimage functions.
    The destination point is included in the profile, in contrast to
    standard numpy indexing.
    """
    src_row, src_col = src = np.asarray(src, dtype=float)
    dst_row, dst_col = dst = np.asarray(dst, dtype=float)
    d_row, d_col = dst - src
    theta = np.arctan2(d_row, d_col)

    length = int(np.ceil(np.hypot(d_row, d_col) + 1))
    # we add one above because we include the last point in the profile
    # (in contrast to standard numpy indexing)
    line_col = np.linspace(src_col, dst_col, length)
    line_row = np.linspace(src_row, dst_row, length)

    # we subtract 1 from linewidth to change from pixel-counting
    # (make this line 3 pixels wide) to point distances (the
    # distance between pixel centers)
    col_width = (linewidth - 1) * np.sin(-theta) / 2
    row_width = (linewidth - 1) * np.cos(theta) / 2
    perp_rows = np.stack([np.linspace(row_i - row_width, row_i + row_width,
                                      linewidth) for row_i in line_row])
    perp_cols = np.stack([np.linspace(col_i - col_width, col_i + col_width,
                                      linewidth) for col_i in line_col])
    return np.stack([perp_rows, perp_cols])
