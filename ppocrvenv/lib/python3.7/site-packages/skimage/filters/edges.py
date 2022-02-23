"""

Sobel and Prewitt filters originally part of CellProfiler, code licensed under
both GPL and BSD licenses.
Website: http://www.cellprofiler.org
Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.
Original author: Lee Kamentsky

"""
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import binary_erosion, convolve

from .._shared.utils import _supported_float_type, check_nD
from ..restoration.uft import laplacian
from ..util.dtype import img_as_float

# n-dimensional filter weights
SOBEL_EDGE = np.array([1, 0, -1])
SOBEL_SMOOTH = np.array([1, 2, 1]) / 4
HSOBEL_WEIGHTS = SOBEL_EDGE.reshape((3, 1)) * SOBEL_SMOOTH.reshape((1, 3))
VSOBEL_WEIGHTS = HSOBEL_WEIGHTS.T

SCHARR_EDGE = np.array([1, 0, -1])
SCHARR_SMOOTH = np.array([3, 10, 3]) / 16
HSCHARR_WEIGHTS = SCHARR_EDGE.reshape((3, 1)) * SCHARR_SMOOTH.reshape((1, 3))
VSCHARR_WEIGHTS = HSCHARR_WEIGHTS.T

PREWITT_EDGE = np.array([1, 0, -1])
PREWITT_SMOOTH = np.full((3,), 1/3)
HPREWITT_WEIGHTS = (PREWITT_EDGE.reshape((3, 1))
                    * PREWITT_SMOOTH.reshape((1, 3)))
VPREWITT_WEIGHTS = HPREWITT_WEIGHTS.T

# 2D-only filter weights
ROBERTS_PD_WEIGHTS = np.array([[1, 0],
                               [0, -1]], dtype=np.double)
ROBERTS_ND_WEIGHTS = np.array([[0, 1],
                               [-1, 0]], dtype=np.double)

# These filter weights can be found in Farid & Simoncelli (2004),
# Table 1 (3rd and 4th row). Additional decimal places were computed
# using the code found at https://www.cs.dartmouth.edu/farid/
p = np.array([[0.0376593171958126, 0.249153396177344, 0.426374573253687,
               0.249153396177344, 0.0376593171958126]])
d1 = np.array([[0.109603762960254, 0.276690988455557, 0, -0.276690988455557,
                -0.109603762960254]])
HFARID_WEIGHTS = d1.T * p
VFARID_WEIGHTS = np.copy(HFARID_WEIGHTS.T)


def _mask_filter_result(result, mask):
    """Return result after masking.

    Input masks are eroded so that mask areas in the original image don't
    affect values in the result.
    """
    if mask is not None:
        erosion_footprint = ndi.generate_binary_structure(mask.ndim, mask.ndim)
        mask = binary_erosion(mask, erosion_footprint, border_value=0)
        result *= mask
    return result


def _kernel_shape(ndim, dim):
    """Return list of `ndim` 1s except at position `dim`, where value is -1.

    Parameters
    ----------
    ndim : int
        The number of dimensions of the kernel shape.
    dim : int
        The axis of the kernel to expand to shape -1.

    Returns
    -------
    shape : list of int
        The requested shape.

    Examples
    --------
    >>> _kernel_shape(2, 0)
    [-1, 1]
    >>> _kernel_shape(3, 1)
    [1, -1, 1]
    >>> _kernel_shape(4, -1)
    [1, 1, 1, -1]
    """
    shape = [1, ] * ndim
    shape[dim] = -1
    return shape


def _reshape_nd(arr, ndim, dim):
    """Reshape a 1D array to have n dimensions, all singletons but one.

    Parameters
    ----------
    arr : array, shape (N,)
        Input array
    ndim : int
        Number of desired dimensions of reshaped array.
    dim : int
        Which dimension/axis will not be singleton-sized.

    Returns
    -------
    arr_reshaped : array, shape ([1, ...], N, [1,...])
        View of `arr` reshaped to the desired shape.

    Examples
    --------
    >>> rng = np.random.default_rng()
    >>> arr = rng.random(7)
    >>> _reshape_nd(arr, 2, 0).shape
    (7, 1)
    >>> _reshape_nd(arr, 3, 1).shape
    (1, 7, 1)
    >>> _reshape_nd(arr, 4, -1).shape
    (1, 1, 1, 7)
    """
    kernel_shape = _kernel_shape(ndim, dim)
    return np.reshape(arr, kernel_shape)


def _generic_edge_filter(image, *, smooth_weights, edge_weights=[1, 0, -1],
                         axis=None, mode='reflect', cval=0.0, mask=None):
    """Apply a generic, n-dimensional edge filter.

    The filter is computed by applying the edge weights along one dimension
    and the smoothing weights along all other dimensions. If no axis is given,
    or a tuple of axes is given the filter is computed along all axes in turn,
    and the magnitude is computed as the square root of the average square
    magnitude of all the axes.

    Parameters
    ----------
    image : array
        The input image.
    smooth_weights : array of float
        The smoothing weights for the filter. These are applied to dimensions
        orthogonal to the edge axis.
    edge_weights : 1D array of float, optional
        The weights to compute the edge along the chosen axes.
    axis : int or sequence of int, optional
        Compute the edge filter along this axis. If not provided, the edge
        magnitude is computed. This is defined as::

            edge_mag = np.sqrt(sum([_generic_edge_filter(image, ..., axis=i)**2
                                    for i in range(image.ndim)]) / image.ndim)

        The magnitude is also computed if axis is a sequence.
    mode : str or sequence of str, optional
        The boundary mode for the convolution. See `scipy.ndimage.convolve`
        for a description of the modes. This can be either a single boundary
        mode or one boundary mode per axis.
    cval : float, optional
        When `mode` is ``'constant'``, this is the constant used in values
        outside the boundary of the image data.
    """
    ndim = image.ndim
    if axis is None:
        axes = list(range(ndim))
    elif np.isscalar(axis):
        axes = [axis]
    else:
        axes = axis
    return_magnitude = (len(axes) > 1)

    if image.dtype.kind == 'f':
        float_dtype = _supported_float_type(image.dtype)
        image = image.astype(float_dtype, copy=False)
    else:
        image = img_as_float(image)
    output = np.zeros(image.shape, dtype=image.dtype)

    for edge_dim in axes:
        kernel = _reshape_nd(edge_weights, ndim, edge_dim)
        smooth_axes = list(set(range(ndim)) - {edge_dim})
        for smooth_dim in smooth_axes:
            kernel = kernel * _reshape_nd(smooth_weights, ndim, smooth_dim)
        ax_output = ndi.convolve(image, kernel, mode=mode)
        if return_magnitude:
            ax_output *= ax_output
        output += ax_output

    if return_magnitude:
        output = np.sqrt(output) / np.sqrt(ndim)
    return output


def sobel(image, mask=None, *, axis=None, mode='reflect', cval=0.0):
    """Find edges in an image using the Sobel filter.

    Parameters
    ----------
    image : array
        The input image.
    mask : array of bool, optional
        Clip the output image to this mask. (Values where mask=0 will be set
        to 0.)
    axis : int or sequence of int, optional
        Compute the edge filter along this axis. If not provided, the edge
        magnitude is computed. This is defined as::

            sobel_mag = np.sqrt(sum([sobel(image, axis=i)**2
                                     for i in range(image.ndim)]) / image.ndim)

        The magnitude is also computed if axis is a sequence.
    mode : str or sequence of str, optional
        The boundary mode for the convolution. See `scipy.ndimage.convolve`
        for a description of the modes. This can be either a single boundary
        mode or one boundary mode per axis.
    cval : float, optional
        When `mode` is ``'constant'``, this is the constant used in values
        outside the boundary of the image data.

    Returns
    -------
    output : array of float
        The Sobel edge map.

    See also
    --------
    sobel_h, sobel_v : horizontal and vertical edge detection.
    scharr, prewitt, farid, skimage.feature.canny

    References
    ----------
    .. [1] D. Kroon, 2009, Short Paper University Twente, Numerical
           Optimization of Kernel Based Image Derivatives.

    .. [2] https://en.wikipedia.org/wiki/Sobel_operator

    Examples
    --------
    >>> from skimage import data
    >>> from skimage import filters
    >>> camera = data.camera()
    >>> edges = filters.sobel(camera)
    """
    output = _generic_edge_filter(image, smooth_weights=SOBEL_SMOOTH,
                                  axis=axis, mode=mode, cval=cval)
    output = _mask_filter_result(output, mask)
    return output


def sobel_h(image, mask=None):
    """Find the horizontal edges of an image using the Sobel transform.

    Parameters
    ----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : 2-D array
        The Sobel edge map.

    Notes
    -----
    We use the following kernel::

      1   2   1
      0   0   0
     -1  -2  -1

    """
    check_nD(image, 2)
    return sobel(image, mask=mask, axis=0)


def sobel_v(image, mask=None):
    """Find the vertical edges of an image using the Sobel transform.

    Parameters
    ----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : 2-D array
        The Sobel edge map.

    Notes
    -----
    We use the following kernel::

      1   0  -1
      2   0  -2
      1   0  -1

    """
    check_nD(image, 2)
    return sobel(image, mask=mask, axis=1)


def scharr(image, mask=None, *, axis=None, mode='reflect', cval=0.0):
    """Find the edge magnitude using the Scharr transform.

    Parameters
    ----------
    image : array
        The input image.
    mask : array of bool, optional
        Clip the output image to this mask. (Values where mask=0 will be set
        to 0.)
    axis : int or sequence of int, optional
        Compute the edge filter along this axis. If not provided, the edge
        magnitude is computed. This is defined as::

            sch_mag = np.sqrt(sum([scharr(image, axis=i)**2
                                   for i in range(image.ndim)]) / image.ndim)

        The magnitude is also computed if axis is a sequence.
    mode : str or sequence of str, optional
        The boundary mode for the convolution. See `scipy.ndimage.convolve`
        for a description of the modes. This can be either a single boundary
        mode or one boundary mode per axis.
    cval : float, optional
        When `mode` is ``'constant'``, this is the constant used in values
        outside the boundary of the image data.

    Returns
    -------
    output : array of float
        The Scharr edge map.

    See also
    --------
    scharr_h, scharr_v : horizontal and vertical edge detection.
    sobel, prewitt, farid, skimage.feature.canny

    Notes
    -----
    The Scharr operator has a better rotation invariance than
    other edge filters such as the Sobel or the Prewitt operators.

    References
    ----------
    .. [1] D. Kroon, 2009, Short Paper University Twente, Numerical
           Optimization of Kernel Based Image Derivatives.

    .. [2] https://en.wikipedia.org/wiki/Sobel_operator#Alternative_operators

    Examples
    --------
    >>> from skimage import data
    >>> from skimage import filters
    >>> camera = data.camera()
    >>> edges = filters.scharr(camera)
    """
    output = _generic_edge_filter(image, smooth_weights=SCHARR_SMOOTH,
                                  axis=axis, mode=mode, cval=cval)
    output = _mask_filter_result(output, mask)
    return output


def scharr_h(image, mask=None):
    """Find the horizontal edges of an image using the Scharr transform.

    Parameters
    ----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : 2-D array
        The Scharr edge map.

    Notes
    -----
    We use the following kernel::

      3   10   3
      0    0   0
     -3  -10  -3

    References
    ----------
    .. [1] D. Kroon, 2009, Short Paper University Twente, Numerical
           Optimization of Kernel Based Image Derivatives.

    """
    check_nD(image, 2)
    return scharr(image, mask=mask, axis=0)


def scharr_v(image, mask=None):
    """Find the vertical edges of an image using the Scharr transform.

    Parameters
    ----------
    image : 2-D array
        Image to process
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : 2-D array
        The Scharr edge map.

    Notes
    -----
    We use the following kernel::

       3   0   -3
      10   0  -10
       3   0   -3

    References
    ----------
    .. [1] D. Kroon, 2009, Short Paper University Twente, Numerical
           Optimization of Kernel Based Image Derivatives.
    """
    check_nD(image, 2)
    return scharr(image, mask=mask, axis=1)


def prewitt(image, mask=None, *, axis=None, mode='reflect', cval=0.0):
    """Find the edge magnitude using the Prewitt transform.

    Parameters
    ----------
    image : array
        The input image.
    mask : array of bool, optional
        Clip the output image to this mask. (Values where mask=0 will be set
        to 0.)
    axis : int or sequence of int, optional
        Compute the edge filter along this axis. If not provided, the edge
        magnitude is computed. This is defined as::

            prw_mag = np.sqrt(sum([prewitt(image, axis=i)**2
                                   for i in range(image.ndim)]) / image.ndim)

        The magnitude is also computed if axis is a sequence.
    mode : str or sequence of str, optional
        The boundary mode for the convolution. See `scipy.ndimage.convolve`
        for a description of the modes. This can be either a single boundary
        mode or one boundary mode per axis.
    cval : float, optional
        When `mode` is ``'constant'``, this is the constant used in values
        outside the boundary of the image data.

    Returns
    -------
    output : array of float
        The Prewitt edge map.

    See also
    --------
    prewitt_h, prewitt_v : horizontal and vertical edge detection.
    sobel, scharr, farid, skimage.feature.canny

    Notes
    -----
    The edge magnitude depends slightly on edge directions, since the
    approximation of the gradient operator by the Prewitt operator is not
    completely rotation invariant. For a better rotation invariance, the Scharr
    operator should be used. The Sobel operator has a better rotation
    invariance than the Prewitt operator, but a worse rotation invariance than
    the Scharr operator.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage import filters
    >>> camera = data.camera()
    >>> edges = filters.prewitt(camera)
    """
    output = _generic_edge_filter(image, smooth_weights=PREWITT_SMOOTH,
                                  axis=axis, mode=mode, cval=cval)
    output = _mask_filter_result(output, mask)
    return output


def prewitt_h(image, mask=None):
    """Find the horizontal edges of an image using the Prewitt transform.

    Parameters
    ----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : 2-D array
        The Prewitt edge map.

    Notes
    -----
    We use the following kernel::

      1/3   1/3   1/3
       0     0     0
     -1/3  -1/3  -1/3

    """
    check_nD(image, 2)
    return prewitt(image, mask=mask, axis=0)


def prewitt_v(image, mask=None):
    """Find the vertical edges of an image using the Prewitt transform.

    Parameters
    ----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : 2-D array
        The Prewitt edge map.

    Notes
    -----
    We use the following kernel::

      1/3   0  -1/3
      1/3   0  -1/3
      1/3   0  -1/3

    """
    check_nD(image, 2)
    return prewitt(image, mask=mask, axis=1)


def roberts(image, mask=None):
    """Find the edge magnitude using Roberts' cross operator.

    Parameters
    ----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : 2-D array
        The Roberts' Cross edge map.

    See also
    --------
    roberts_pos_diag, roberts_neg_diag : diagonal edge detection.
    sobel, scharr, prewitt, skimage.feature.canny

    Examples
    --------
    >>> from skimage import data
    >>> camera = data.camera()
    >>> from skimage import filters
    >>> edges = filters.roberts(camera)

    """
    check_nD(image, 2)
    out = np.sqrt(roberts_pos_diag(image, mask) ** 2 +
                  roberts_neg_diag(image, mask) ** 2)
    out /= np.sqrt(2)
    return out


def roberts_pos_diag(image, mask=None):
    """Find the cross edges of an image using Roberts' cross operator.

    The kernel is applied to the input image to produce separate measurements
    of the gradient component one orientation.

    Parameters
    ----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : 2-D array
        The Robert's edge map.

    Notes
    -----
    We use the following kernel::

      1   0
      0  -1

    """
    check_nD(image, 2)
    if image.dtype.kind == 'f':
        float_dtype = _supported_float_type(image.dtype)
        image = image.astype(float_dtype, copy=False)
    else:
        image = img_as_float(image)
    result = convolve(image, ROBERTS_PD_WEIGHTS)
    return _mask_filter_result(result, mask)


def roberts_neg_diag(image, mask=None):
    """Find the cross edges of an image using the Roberts' Cross operator.

    The kernel is applied to the input image to produce separate measurements
    of the gradient component one orientation.

    Parameters
    ----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : 2-D array
        The Robert's edge map.

    Notes
    -----
    We use the following kernel::

      0   1
     -1   0

    """
    check_nD(image, 2)
    if image.dtype.kind == 'f':
        float_dtype = _supported_float_type(image.dtype)
        image = image.astype(float_dtype, copy=False)
    else:
        image = img_as_float(image)
    result = convolve(image, ROBERTS_ND_WEIGHTS)
    return _mask_filter_result(result, mask)


def laplace(image, ksize=3, mask=None):
    """Find the edges of an image using the Laplace operator.

    Parameters
    ----------
    image : ndarray
        Image to process.
    ksize : int, optional
        Define the size of the discrete Laplacian operator such that it
        will have a size of (ksize,) * image.ndim.
    mask : ndarray, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : ndarray
        The Laplace edge map.

    Notes
    -----
    The Laplacian operator is generated using the function
    skimage.restoration.uft.laplacian().

    """
    if image.dtype.kind == 'f':
        float_dtype = _supported_float_type(image.dtype)
        image = image.astype(float_dtype, copy=False)
    else:
        image = img_as_float(image)
    # Create the discrete Laplacian operator - We keep only the real part of
    # the filter
    _, laplace_op = laplacian(image.ndim, (ksize,) * image.ndim)
    result = convolve(image, laplace_op)
    return _mask_filter_result(result, mask)


def farid(image, *, mask=None):
    """Find the edge magnitude using the Farid transform.

    Parameters
    ----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : 2-D array
        The Farid edge map.

    See also
    --------
    farid_h, farid_v : horizontal and vertical edge detection.
    sobel, prewitt, farid, skimage.feature.canny

    Notes
    -----
    Take the square root of the sum of the squares of the horizontal and
    vertical derivatives to get a magnitude that is somewhat insensitive to
    direction. Similar to the Scharr operator, this operator is designed with
    a rotation invariance constraint.

    References
    ----------
    .. [1] Farid, H. and Simoncelli, E. P., "Differentiation of discrete
           multidimensional signals", IEEE Transactions on Image Processing
           13(4): 496-508, 2004. :DOI:`10.1109/TIP.2004.823819`
    .. [2] Wikipedia, "Farid and Simoncelli Derivatives." Available at:
           <https://en.wikipedia.org/wiki/Image_derivatives#Farid_and_Simoncelli_Derivatives>

    Examples
    --------
    >>> from skimage import data
    >>> camera = data.camera()
    >>> from skimage import filters
    >>> edges = filters.farid(camera)
    """
    check_nD(image, 2)
    out = np.sqrt(farid_h(image, mask=mask) ** 2
                  + farid_v(image, mask=mask) ** 2)
    out /= np.sqrt(2)
    return out


def farid_h(image, *, mask=None):
    """Find the horizontal edges of an image using the Farid transform.

    Parameters
    ----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : 2-D array
        The Farid edge map.

    Notes
    -----
    The kernel was constructed using the 5-tap weights from [1].

    References
    ----------
    .. [1] Farid, H. and Simoncelli, E. P., "Differentiation of discrete
           multidimensional signals", IEEE Transactions on Image Processing
           13(4): 496-508, 2004. :DOI:`10.1109/TIP.2004.823819`
    .. [2] Farid, H. and Simoncelli, E. P. "Optimally rotation-equivariant
           directional derivative kernels", In: 7th International Conference on
           Computer Analysis of Images and Patterns, Kiel, Germany. Sep, 1997.
    """
    check_nD(image, 2)
    if image.dtype.kind == 'f':
        float_dtype = _supported_float_type(image.dtype)
        image = image.astype(float_dtype, copy=False)
    else:
        image = img_as_float(image)
    result = convolve(image, HFARID_WEIGHTS)
    return _mask_filter_result(result, mask)


def farid_v(image, *, mask=None):
    """Find the vertical edges of an image using the Farid transform.

    Parameters
    ----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : 2-D array
        The Farid edge map.

    Notes
    -----
    The kernel was constructed using the 5-tap weights from [1].

    References
    ----------
    .. [1] Farid, H. and Simoncelli, E. P., "Differentiation of discrete
           multidimensional signals", IEEE Transactions on Image Processing
           13(4): 496-508, 2004. :DOI:`10.1109/TIP.2004.823819`
    """
    check_nD(image, 2)
    if image.dtype.kind == 'f':
        float_dtype = _supported_float_type(image.dtype)
        image = image.astype(float_dtype, copy=False)
    else:
        image = img_as_float(image)
    result = convolve(image, VFARID_WEIGHTS)
    return _mask_filter_result(result, mask)
