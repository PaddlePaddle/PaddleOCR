from itertools import combinations_with_replacement
from warnings import warn

import numpy as np
from scipy import ndimage as ndi
from scipy import spatial, stats

from .._shared.filters import gaussian
from .._shared.utils import _supported_float_type, safe_as_int
from ..transform import integral_image
from ..util import img_as_float
from ._hessian_det_appx import _hessian_matrix_det
from .corner_cy import _corner_fast, _corner_moravec, _corner_orientations
from .peak import peak_local_max
from .util import _prepare_grayscale_input_2D, _prepare_grayscale_input_nD


def _compute_derivatives(image, mode='constant', cval=0):
    """Compute derivatives in axis directions using the Sobel operator.

    Parameters
    ----------
    image : ndarray
        Input image.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.

    Returns
    -------
    derivatives : list of ndarray
        Derivatives in each axis direction.

    """

    derivatives = [ndi.sobel(image, axis=i, mode=mode, cval=cval)
                   for i in range(image.ndim)]

    return derivatives


def structure_tensor(image, sigma=1, mode='constant', cval=0, order=None):
    """Compute structure tensor using sum of squared differences.

    The (2-dimensional) structure tensor A is defined as::

        A = [Arr Arc]
            [Arc Acc]

    which is approximated by the weighted sum of squared differences in a local
    window around each pixel in the image. This formula can be extended to a
    larger number of dimensions (see [1]_).

    Parameters
    ----------
    image : ndarray
        Input image.
    sigma : float or array-like of float, optional
        Standard deviation used for the Gaussian kernel, which is used as a
        weighting function for the local summation of squared differences.
        If sigma is an iterable, its length must be equal to `image.ndim` and
        each element is used for the Gaussian kernel applied along its
        respective axis.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    order : {'rc', 'xy'}, optional
        NOTE: Only applies in 2D. Higher dimensions must always use 'rc' order.
        This parameter allows for the use of reverse or forward order of
        the image axes in gradient computation. 'rc' indicates the use of
        the first axis initially (Arr, Arc, Acc), whilst 'xy' indicates the
        usage of the last axis initially (Axx, Axy, Ayy).

    Returns
    -------
    A_elems : list of ndarray
        Upper-diagonal elements of the structure tensor for each pixel in the
        input image.

    Examples
    --------
    >>> from skimage.feature import structure_tensor
    >>> square = np.zeros((5, 5))
    >>> square[2, 2] = 1
    >>> Arr, Arc, Acc = structure_tensor(square, sigma=0.1, order='rc')
    >>> Acc
    array([[0., 0., 0., 0., 0.],
           [0., 1., 0., 1., 0.],
           [0., 4., 0., 4., 0.],
           [0., 1., 0., 1., 0.],
           [0., 0., 0., 0., 0.]])

    See also
    --------
    structure_tensor_eigenvalues

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Structure_tensor
    """
    if order == 'xy' and image.ndim > 2:
        raise ValueError('Only "rc" order is supported for dim > 2.')

    if order is None:
        if image.ndim == 2:
            # The legacy 2D code followed (x, y) convention, so we swap the
            # axis order to maintain compatibility with old code
            warn('deprecation warning: the default order of the structure '
                 'tensor values will be "row-column" instead of "xy" starting '
                 'in skimage version 0.20. Use order="rc" or order="xy" to '
                 'set this explicitly.  (Specify order="xy" to maintain the '
                 'old behavior.)', category=FutureWarning, stacklevel=2)
            order = 'xy'
        else:
            order = 'rc'

    if not np.isscalar(sigma):
        sigma = tuple(sigma)
        if len(sigma) != image.ndim:
            raise ValueError('sigma must have as many elements as image '
                             'has axes')

    image = _prepare_grayscale_input_nD(image)

    derivatives = _compute_derivatives(image, mode=mode, cval=cval)

    if order == 'xy':
        derivatives = reversed(derivatives)

    # structure tensor
    A_elems = [gaussian(der0 * der1, sigma, mode=mode, cval=cval)
               for der0, der1 in combinations_with_replacement(derivatives, 2)]

    return A_elems


def hessian_matrix(image, sigma=1, mode='constant', cval=0, order='rc'):
    """Compute the Hessian matrix.

    In 2D, the Hessian matrix is defined as::

        H = [Hrr Hrc]
            [Hrc Hcc]

    which is computed by convolving the image with the second derivatives
    of the Gaussian kernel in the respective r- and c-directions.

    The implementation here also supports n-dimensional data.

    Parameters
    ----------
    image : ndarray
        Input image.
    sigma : float
        Standard deviation used for the Gaussian kernel, which is used as
        weighting function for the auto-correlation matrix.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    order : {'rc', 'xy'}, optional
        This parameter allows for the use of reverse or forward order of
        the image axes in gradient computation. 'rc' indicates the use of
        the first axis initially (Hrr, Hrc, Hcc), whilst 'xy' indicates the
        usage of the last axis initially (Hxx, Hxy, Hyy)

    Returns
    -------
    H_elems : list of ndarray
        Upper-diagonal elements of the hessian matrix for each pixel in the
        input image. In 2D, this will be a three element list containing [Hrr,
        Hrc, Hcc]. In nD, the list will contain ``(n**2 + n) / 2`` arrays.

    Examples
    --------
    >>> from skimage.feature import hessian_matrix
    >>> square = np.zeros((5, 5))
    >>> square[2, 2] = 4
    >>> Hrr, Hrc, Hcc = hessian_matrix(square, sigma=0.1, order='rc')
    >>> Hrc
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  0., -1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0., -1.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    """

    image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)

    gaussian_filtered = gaussian(image, sigma=sigma, mode=mode, cval=cval)

    gradients = np.gradient(gaussian_filtered)
    axes = range(image.ndim)

    if order == 'rc':
        axes = reversed(axes)

    H_elems = [np.gradient(gradients[ax0], axis=ax1)
               for ax0, ax1 in combinations_with_replacement(axes, 2)]
    return H_elems


def hessian_matrix_det(image, sigma=1, approximate=True):
    """Compute the approximate Hessian Determinant over an image.

    The 2D approximate method uses box filters over integral images to
    compute the approximate Hessian Determinant.

    Parameters
    ----------
    image : ndarray
        The image over which to compute the Hessian Determinant.
    sigma : float, optional
        Standard deviation of the Gaussian kernel used for the Hessian
        matrix.
    approximate : bool, optional
        If ``True`` and the image is 2D, use a much faster approximate
        computation. This argument has no effect on 3D and higher images.

    Returns
    -------
    out : array
        The array of the Determinant of Hessians.

    References
    ----------
    .. [1] Herbert Bay, Andreas Ess, Tinne Tuytelaars, Luc Van Gool,
           "SURF: Speeded Up Robust Features"
           ftp://ftp.vision.ee.ethz.ch/publications/articles/eth_biwi_00517.pdf

    Notes
    -----
    For 2D images when ``approximate=True``, the running time of this method
    only depends on size of the image. It is independent of `sigma` as one
    would expect. The downside is that the result for `sigma` less than `3`
    is not accurate, i.e., not similar to the result obtained if someone
    computed the Hessian and took its determinant.
    """
    image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    if image.ndim == 2 and approximate:
        integral = integral_image(image)
        return np.array(_hessian_matrix_det(integral, sigma))
    else:  # slower brute-force implementation for nD images
        hessian_mat_array = _symmetric_image(hessian_matrix(image, sigma))
        return np.linalg.det(hessian_mat_array)


def _image_orthogonal_matrix22_eigvals(M00, M01, M11):
    l1 = (M00 + M11) / 2 + np.sqrt(4 * M01 ** 2 + (M00 - M11) ** 2) / 2
    l2 = (M00 + M11) / 2 - np.sqrt(4 * M01 ** 2 + (M00 - M11) ** 2) / 2
    return l1, l2


def _symmetric_compute_eigenvalues(S_elems):
    """Compute eigenvalues from the upperdiagonal entries of a symmetric matrix

    Parameters
    ----------
    S_elems : list of ndarray
        The upper-diagonal elements of the matrix, as returned by
        `hessian_matrix` or `structure_tensor`.

    Returns
    -------
    eigs : ndarray
        The eigenvalues of the matrix, in decreasing order. The eigenvalues are
        the leading dimension. That is, ``eigs[i, j, k]`` contains the
        ith-largest eigenvalue at position (j, k).
    """

    if len(S_elems) == 3:  # Use fast Cython code for 2D
        eigs = np.stack(_image_orthogonal_matrix22_eigvals(*S_elems))
    else:
        matrices = _symmetric_image(S_elems)
        # eigvalsh returns eigenvalues in increasing order. We want decreasing
        eigs = np.linalg.eigvalsh(matrices)[..., ::-1]
        leading_axes = tuple(range(eigs.ndim - 1))
        eigs = np.transpose(eigs, (eigs.ndim - 1,) + leading_axes)
    return eigs


def _symmetric_image(S_elems):
    """Convert the upper-diagonal elements of a matrix to the full
    symmetric matrix.

    Parameters
    ----------
    S_elems : list of array
        The upper-diagonal elements of the matrix, as returned by
        `hessian_matrix` or `structure_tensor`.

    Returns
    -------
    image : array
        An array of shape ``(M, N[, ...], image.ndim, image.ndim)``,
        containing the matrix corresponding to each coordinate.
    """
    image = S_elems[0]
    symmetric_image = np.zeros(image.shape + (image.ndim, image.ndim),
                               dtype=S_elems[0].dtype)
    for idx, (row, col) in \
            enumerate(combinations_with_replacement(range(image.ndim), 2)):
        symmetric_image[..., row, col] = S_elems[idx]
        symmetric_image[..., col, row] = S_elems[idx]
    return symmetric_image


def structure_tensor_eigenvalues(A_elems):
    """Compute eigenvalues of structure tensor.

    Parameters
    ----------
    A_elems : list of ndarray
        The upper-diagonal elements of the structure tensor, as returned
        by `structure_tensor`.

    Returns
    -------
    ndarray
        The eigenvalues of the structure tensor, in decreasing order. The
        eigenvalues are the leading dimension. That is, the coordinate
        [i, j, k] corresponds to the ith-largest eigenvalue at position (j, k).

    Examples
    --------
    >>> from skimage.feature import structure_tensor
    >>> from skimage.feature import structure_tensor_eigenvalues
    >>> square = np.zeros((5, 5))
    >>> square[2, 2] = 1
    >>> A_elems = structure_tensor(square, sigma=0.1, order='rc')
    >>> structure_tensor_eigenvalues(A_elems)[0]
    array([[0., 0., 0., 0., 0.],
           [0., 2., 4., 2., 0.],
           [0., 4., 0., 4., 0.],
           [0., 2., 4., 2., 0.],
           [0., 0., 0., 0., 0.]])

    See also
    --------
    structure_tensor
    """
    return _symmetric_compute_eigenvalues(A_elems)


def structure_tensor_eigvals(Axx, Axy, Ayy):
    """Compute eigenvalues of structure tensor.

    Parameters
    ----------
    Axx : ndarray
        Element of the structure tensor for each pixel in the input image.
    Axy : ndarray
        Element of the structure tensor for each pixel in the input image.
    Ayy : ndarray
        Element of the structure tensor for each pixel in the input image.

    Returns
    -------
    l1 : ndarray
        Larger eigen value for each input matrix.
    l2 : ndarray
        Smaller eigen value for each input matrix.

    Examples
    --------
    >>> from skimage.feature import structure_tensor, structure_tensor_eigvals
    >>> square = np.zeros((5, 5))
    >>> square[2, 2] = 1
    >>> Arr, Arc, Acc = structure_tensor(square, sigma=0.1, order='rc')
    >>> structure_tensor_eigvals(Acc, Arc, Arr)[0]  # doctest: +SKIP
    array([[0., 0., 0., 0., 0.],
           [0., 2., 4., 2., 0.],
           [0., 4., 0., 4., 0.],
           [0., 2., 4., 2., 0.],
           [0., 0., 0., 0., 0.]])

    """
    warn('deprecation warning: the function structure_tensor_eigvals is '
         'deprecated and will be removed in version 0.20. Please use '
         'structure_tensor_eigenvalues instead.',
         category=FutureWarning, stacklevel=2)

    return _image_orthogonal_matrix22_eigvals(Axx, Axy, Ayy)


def hessian_matrix_eigvals(H_elems):
    """Compute eigenvalues of Hessian matrix.

    Parameters
    ----------
    H_elems : list of ndarray
        The upper-diagonal elements of the Hessian matrix, as returned
        by `hessian_matrix`.

    Returns
    -------
    eigs : ndarray
        The eigenvalues of the Hessian matrix, in decreasing order. The
        eigenvalues are the leading dimension. That is, ``eigs[i, j, k]``
        contains the ith-largest eigenvalue at position (j, k).

    Examples
    --------
    >>> from skimage.feature import hessian_matrix, hessian_matrix_eigvals
    >>> square = np.zeros((5, 5))
    >>> square[2, 2] = 4
    >>> H_elems = hessian_matrix(square, sigma=0.1, order='rc')
    >>> hessian_matrix_eigvals(H_elems)[0]
    array([[ 0.,  0.,  2.,  0.,  0.],
           [ 0.,  1.,  0.,  1.,  0.],
           [ 2.,  0., -2.,  0.,  2.],
           [ 0.,  1.,  0.,  1.,  0.],
           [ 0.,  0.,  2.,  0.,  0.]])
    """
    return _symmetric_compute_eigenvalues(H_elems)


def shape_index(image, sigma=1, mode='constant', cval=0):
    """Compute the shape index.

    The shape index, as defined by Koenderink & van Doorn [1]_, is a
    single valued measure of local curvature, assuming the image as a 3D plane
    with intensities representing heights.

    It is derived from the eigenvalues of the Hessian, and its
    value ranges from -1 to 1 (and is undefined (=NaN) in *flat* regions),
    with following ranges representing following shapes:

    .. table:: Ranges of the shape index and corresponding shapes.

      ===================  =============
      Interval (s in ...)  Shape
      ===================  =============
      [  -1, -7/8)         Spherical cup
      [-7/8, -5/8)         Through
      [-5/8, -3/8)         Rut
      [-3/8, -1/8)         Saddle rut
      [-1/8, +1/8)         Saddle
      [+1/8, +3/8)         Saddle ridge
      [+3/8, +5/8)         Ridge
      [+5/8, +7/8)         Dome
      [+7/8,   +1]         Spherical cap
      ===================  =============

    Parameters
    ----------
    image : (M, N) ndarray
        Input image.
    sigma : float, optional
        Standard deviation used for the Gaussian kernel, which is used for
        smoothing the input data before Hessian eigen value calculation.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.

    Returns
    -------
    s : ndarray
        Shape index

    References
    ----------
    .. [1] Koenderink, J. J. & van Doorn, A. J.,
           "Surface shape and curvature scales",
           Image and Vision Computing, 1992, 10, 557-564.
           :DOI:`10.1016/0262-8856(92)90076-F`

    Examples
    --------
    >>> from skimage.feature import shape_index
    >>> square = np.zeros((5, 5))
    >>> square[2, 2] = 4
    >>> s = shape_index(square, sigma=0.1)
    >>> s
    array([[ nan,  nan, -0.5,  nan,  nan],
           [ nan, -0. ,  nan, -0. ,  nan],
           [-0.5,  nan, -1. ,  nan, -0.5],
           [ nan, -0. ,  nan, -0. ,  nan],
           [ nan,  nan, -0.5,  nan,  nan]])
    """

    H = hessian_matrix(image, sigma=sigma, mode=mode, cval=cval, order='rc')
    l1, l2 = hessian_matrix_eigvals(H)

    # don't warn on divide by 0 as occurs in the docstring example
    with np.errstate(divide='ignore', invalid='ignore'):
        return (2.0 / np.pi) * np.arctan((l2 + l1) / (l2 - l1))


def corner_kitchen_rosenfeld(image, mode='constant', cval=0):
    """Compute Kitchen and Rosenfeld corner measure response image.

    The corner measure is calculated as follows::

        (imxx * imy**2 + imyy * imx**2 - 2 * imxy * imx * imy)
            / (imx**2 + imy**2)

    Where imx and imy are the first and imxx, imxy, imyy the second
    derivatives.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.

    Returns
    -------
    response : ndarray
        Kitchen and Rosenfeld response image.

    References
    ----------
    .. [1] Kitchen, L., & Rosenfeld, A. (1982). Gray-level corner detection.
           Pattern recognition letters, 1(2), 95-102.
           :DOI:`10.1016/0167-8655(82)90020-4`
    """

    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)

    imy, imx = _compute_derivatives(image, mode=mode, cval=cval)
    imxy, imxx = _compute_derivatives(imx, mode=mode, cval=cval)
    imyy, imyx = _compute_derivatives(imy, mode=mode, cval=cval)

    numerator = (imxx * imy ** 2 + imyy * imx ** 2 - 2 * imxy * imx * imy)
    denominator = (imx ** 2 + imy ** 2)

    response = np.zeros_like(image, dtype=float_dtype)

    mask = denominator != 0
    response[mask] = numerator[mask] / denominator[mask]

    return response


def corner_harris(image, method='k', k=0.05, eps=1e-6, sigma=1):
    """Compute Harris corner measure response image.

    This corner detector uses information from the auto-correlation matrix A::

        A = [(imx**2)   (imx*imy)] = [Axx Axy]
            [(imx*imy)   (imy**2)]   [Axy Ayy]

    Where imx and imy are first derivatives, averaged with a gaussian filter.
    The corner measure is then defined as::

        det(A) - k * trace(A)**2

    or::

        2 * det(A) / (trace(A) + eps)

    Parameters
    ----------
    image : (M, N) ndarray
        Input image.
    method : {'k', 'eps'}, optional
        Method to compute the response image from the auto-correlation matrix.
    k : float, optional
        Sensitivity factor to separate corners from edges, typically in range
        `[0, 0.2]`. Small values of k result in detection of sharp corners.
    eps : float, optional
        Normalisation factor (Noble's corner measure).
    sigma : float, optional
        Standard deviation used for the Gaussian kernel, which is used as
        weighting function for the auto-correlation matrix.

    Returns
    -------
    response : ndarray
        Harris response image.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Corner_detection

    Examples
    --------
    >>> from skimage.feature import corner_harris, corner_peaks
    >>> square = np.zeros([10, 10])
    >>> square[2:8, 2:8] = 1
    >>> square.astype(int)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> corner_peaks(corner_harris(square), min_distance=1)
    array([[2, 2],
           [2, 7],
           [7, 2],
           [7, 7]])

    """

    Arr, Arc, Acc = structure_tensor(image, sigma, order='rc')

    # determinant
    detA = Arr * Acc - Arc ** 2
    # trace
    traceA = Arr + Acc

    if method == 'k':
        response = detA - k * traceA ** 2
    else:
        response = 2 * detA / (traceA + eps)

    return response


def corner_shi_tomasi(image, sigma=1):
    """Compute Shi-Tomasi (Kanade-Tomasi) corner measure response image.

    This corner detector uses information from the auto-correlation matrix A::

        A = [(imx**2)   (imx*imy)] = [Axx Axy]
            [(imx*imy)   (imy**2)]   [Axy Ayy]

    Where imx and imy are first derivatives, averaged with a gaussian filter.
    The corner measure is then defined as the smaller eigenvalue of A::

        ((Axx + Ayy) - sqrt((Axx - Ayy)**2 + 4 * Axy**2)) / 2

    Parameters
    ----------
    image : (M, N) ndarray
        Input image.
    sigma : float, optional
        Standard deviation used for the Gaussian kernel, which is used as
        weighting function for the auto-correlation matrix.

    Returns
    -------
    response : ndarray
        Shi-Tomasi response image.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Corner_detection

    Examples
    --------
    >>> from skimage.feature import corner_shi_tomasi, corner_peaks
    >>> square = np.zeros([10, 10])
    >>> square[2:8, 2:8] = 1
    >>> square.astype(int)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> corner_peaks(corner_shi_tomasi(square), min_distance=1)
    array([[2, 2],
           [2, 7],
           [7, 2],
           [7, 7]])

    """

    Arr, Arc, Acc = structure_tensor(image, sigma, order='rc')

    # minimum eigenvalue of A
    response = ((Arr + Acc) - np.sqrt((Arr - Acc) ** 2 + 4 * Arc ** 2)) / 2

    return response


def corner_foerstner(image, sigma=1):
    """Compute Foerstner corner measure response image.

    This corner detector uses information from the auto-correlation matrix A::

        A = [(imx**2)   (imx*imy)] = [Axx Axy]
            [(imx*imy)   (imy**2)]   [Axy Ayy]

    Where imx and imy are first derivatives, averaged with a gaussian filter.
    The corner measure is then defined as::

        w = det(A) / trace(A)           (size of error ellipse)
        q = 4 * det(A) / trace(A)**2    (roundness of error ellipse)

    Parameters
    ----------
    image : (M, N) ndarray
        Input image.
    sigma : float, optional
        Standard deviation used for the Gaussian kernel, which is used as
        weighting function for the auto-correlation matrix.

    Returns
    -------
    w : ndarray
        Error ellipse sizes.
    q : ndarray
        Roundness of error ellipse.

    References
    ----------
    .. [1] Förstner, W., & Gülch, E. (1987, June). A fast operator for detection and
           precise location of distinct points, corners and centres of circular
           features. In Proc. ISPRS intercommission conference on fast processing of
           photogrammetric data (pp. 281-305).
           https://cseweb.ucsd.edu/classes/sp02/cse252/foerstner/foerstner.pdf
    .. [2] https://en.wikipedia.org/wiki/Corner_detection

    Examples
    --------
    >>> from skimage.feature import corner_foerstner, corner_peaks
    >>> square = np.zeros([10, 10])
    >>> square[2:8, 2:8] = 1
    >>> square.astype(int)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> w, q = corner_foerstner(square)
    >>> accuracy_thresh = 0.5
    >>> roundness_thresh = 0.3
    >>> foerstner = (q > roundness_thresh) * (w > accuracy_thresh) * w
    >>> corner_peaks(foerstner, min_distance=1)
    array([[2, 2],
           [2, 7],
           [7, 2],
           [7, 7]])

    """

    Arr, Arc, Acc = structure_tensor(image, sigma, order='rc')

    # determinant
    detA = Arr * Acc - Arc ** 2
    # trace
    traceA = Arr + Acc

    w = np.zeros_like(image, dtype=detA.dtype)
    q = np.zeros_like(w)

    mask = traceA != 0

    w[mask] = detA[mask] / traceA[mask]
    q[mask] = 4 * detA[mask] / traceA[mask] ** 2

    return w, q


def corner_fast(image, n=12, threshold=0.15):
    """Extract FAST corners for a given image.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image.
    n : int, optional
        Minimum number of consecutive pixels out of 16 pixels on the circle
        that should all be either brighter or darker w.r.t testpixel.
        A point c on the circle is darker w.r.t test pixel p if
        `Ic < Ip - threshold` and brighter if `Ic > Ip + threshold`. Also
        stands for the n in `FAST-n` corner detector.
    threshold : float, optional
        Threshold used in deciding whether the pixels on the circle are
        brighter, darker or similar w.r.t. the test pixel. Decrease the
        threshold when more corners are desired and vice-versa.

    Returns
    -------
    response : ndarray
        FAST corner response image.

    References
    ----------
    .. [1] Rosten, E., & Drummond, T. (2006, May). Machine learning for high-speed
           corner detection. In European conference on computer vision (pp. 430-443).
           Springer, Berlin, Heidelberg.
           :DOI:`10.1007/11744023_34`
           http://www.edwardrosten.com/work/rosten_2006_machine.pdf
    .. [2] Wikipedia, "Features from accelerated segment test",
           https://en.wikipedia.org/wiki/Features_from_accelerated_segment_test

    Examples
    --------
    >>> from skimage.feature import corner_fast, corner_peaks
    >>> square = np.zeros((12, 12))
    >>> square[3:9, 3:9] = 1
    >>> square.astype(int)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> corner_peaks(corner_fast(square, 9), min_distance=1)
    array([[3, 3],
           [3, 8],
           [8, 3],
           [8, 8]])

    """
    image = _prepare_grayscale_input_2D(image)

    image = np.ascontiguousarray(image)
    response = _corner_fast(image, n, threshold)
    return response


def corner_subpix(image, corners, window_size=11, alpha=0.99):
    """Determine subpixel position of corners.

    A statistical test decides whether the corner is defined as the
    intersection of two edges or a single peak. Depending on the classification
    result, the subpixel corner location is determined based on the local
    covariance of the grey-values. If the significance level for either
    statistical test is not sufficient, the corner cannot be classified, and
    the output subpixel position is set to NaN.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image.
    corners : (K, 2) ndarray
        Corner coordinates `(row, col)`.
    window_size : int, optional
        Search window size for subpixel estimation.
    alpha : float, optional
        Significance level for corner classification.

    Returns
    -------
    positions : (K, 2) ndarray
        Subpixel corner positions. NaN for "not classified" corners.

    References
    ----------
    .. [1] Förstner, W., & Gülch, E. (1987, June). A fast operator for detection and
           precise location of distinct points, corners and centres of circular
           features. In Proc. ISPRS intercommission conference on fast processing of
           photogrammetric data (pp. 281-305).
           https://cseweb.ucsd.edu/classes/sp02/cse252/foerstner/foerstner.pdf
    .. [2] https://en.wikipedia.org/wiki/Corner_detection

    Examples
    --------
    >>> from skimage.feature import corner_harris, corner_peaks, corner_subpix
    >>> img = np.zeros((10, 10))
    >>> img[:5, :5] = 1
    >>> img[5:, 5:] = 1
    >>> img.astype(int)
    array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])
    >>> coords = corner_peaks(corner_harris(img), min_distance=2)
    >>> coords_subpix = corner_subpix(img, coords, window_size=7)
    >>> coords_subpix
    array([[4.5, 4.5]])

    """

    # window extent in one direction
    wext = (window_size - 1) // 2

    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    image = np.pad(image, pad_width=wext, mode='constant', constant_values=0)

    # add pad width, make sure to not modify the input values in-place
    corners = safe_as_int(corners + wext)

    # normal equation arrays
    N_dot = np.zeros((2, 2), dtype=float_dtype)
    N_edge = np.zeros((2, 2), dtype=float_dtype)
    b_dot = np.zeros((2, ), dtype=float_dtype)
    b_edge = np.zeros((2, ), dtype=float_dtype)

    # critical statistical test values
    redundancy = window_size ** 2 - 2
    t_crit_dot = stats.f.isf(1 - alpha, redundancy, redundancy)
    t_crit_edge = stats.f.isf(alpha, redundancy, redundancy)

    # coordinates of pixels within window
    y, x = np.mgrid[- wext:wext + 1, - wext:wext + 1]

    corners_subpix = np.zeros_like(corners, dtype=float_dtype)

    for i, (y0, x0) in enumerate(corners):

        # crop window around corner + border for sobel operator
        miny = y0 - wext - 1
        maxy = y0 + wext + 2
        minx = x0 - wext - 1
        maxx = x0 + wext + 2
        window = image[miny:maxy, minx:maxx]

        winy, winx = _compute_derivatives(window, mode='constant', cval=0)

        # compute gradient squares and remove border
        winx_winx = (winx * winx)[1:-1, 1:-1]
        winx_winy = (winx * winy)[1:-1, 1:-1]
        winy_winy = (winy * winy)[1:-1, 1:-1]

        # sum of squared differences (mean instead of gaussian filter)
        Axx = np.sum(winx_winx)
        Axy = np.sum(winx_winy)
        Ayy = np.sum(winy_winy)

        # sum of squared differences weighted with coordinates
        # (mean instead of gaussian filter)
        bxx_x = np.sum(winx_winx * x)
        bxx_y = np.sum(winx_winx * y)
        bxy_x = np.sum(winx_winy * x)
        bxy_y = np.sum(winx_winy * y)
        byy_x = np.sum(winy_winy * x)
        byy_y = np.sum(winy_winy * y)

        # normal equations for subpixel position
        N_dot[0, 0] = Axx
        N_dot[0, 1] = N_dot[1, 0] = - Axy
        N_dot[1, 1] = Ayy

        N_edge[0, 0] = Ayy
        N_edge[0, 1] = N_edge[1, 0] = Axy
        N_edge[1, 1] = Axx

        b_dot[:] = bxx_y - bxy_x, byy_x - bxy_y
        b_edge[:] = byy_y + bxy_x, bxx_x + bxy_y

        # estimated positions
        try:
            est_dot = np.linalg.solve(N_dot, b_dot)
            est_edge = np.linalg.solve(N_edge, b_edge)
        except np.linalg.LinAlgError:
            # if image is constant the system is singular
            corners_subpix[i, :] = np.nan, np.nan
            continue

        # residuals
        ry_dot = y - est_dot[0]
        rx_dot = x - est_dot[1]
        ry_edge = y - est_edge[0]
        rx_edge = x - est_edge[1]
        # squared residuals
        rxx_dot = rx_dot * rx_dot
        rxy_dot = rx_dot * ry_dot
        ryy_dot = ry_dot * ry_dot
        rxx_edge = rx_edge * rx_edge
        rxy_edge = rx_edge * ry_edge
        ryy_edge = ry_edge * ry_edge

        # determine corner class (dot or edge)
        # variance for different models
        var_dot = np.sum(winx_winx * ryy_dot - 2 * winx_winy * rxy_dot
                         + winy_winy * rxx_dot)
        var_edge = np.sum(winy_winy * ryy_edge + 2 * winx_winy * rxy_edge
                          + winx_winx * rxx_edge)

        # test value (F-distributed)
        if var_dot < np.spacing(1) and var_edge < np.spacing(1):
            t = np.nan
        elif var_dot == 0:
            t = np.inf
        else:
            t = var_edge / var_dot

        # 1 for edge, -1 for dot, 0 for "not classified"
        corner_class = int(t < t_crit_edge) - int(t > t_crit_dot)

        if corner_class == -1:
            corners_subpix[i, :] = y0 + est_dot[0], x0 + est_dot[1]
        elif corner_class == 0:
            corners_subpix[i, :] = np.nan, np.nan
        elif corner_class == 1:
            corners_subpix[i, :] = y0 + est_edge[0], x0 + est_edge[1]

    # subtract pad width
    corners_subpix -= wext

    return corners_subpix


def corner_peaks(image, min_distance=1, threshold_abs=None, threshold_rel=None,
                 exclude_border=True, indices=True, num_peaks=np.inf,
                 footprint=None, labels=None, *, num_peaks_per_label=np.inf,
                 p_norm=np.inf):
    """Find peaks in corner measure response image.

    This differs from `skimage.feature.peak_local_max` in that it suppresses
    multiple connected peaks with the same accumulator value.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image.
    min_distance : int, optional
        The minimal allowed distance separating peaks.
    * : *
        See :py:meth:`skimage.feature.peak_local_max`.
    p_norm : float
        Which Minkowski p-norm to use. Should be in the range [1, inf].
        A finite large p may cause a ValueError if overflow can occur.
        ``inf`` corresponds to the Chebyshev distance and 2 to the
        Euclidean distance.

    Returns
    -------
    output : ndarray or ndarray of bools

        * If `indices = True`  : (row, column, ...) coordinates of peaks.
        * If `indices = False` : Boolean array shaped like `image`, with peaks
          represented by True values.

    See also
    --------
    skimage.feature.peak_local_max

    Notes
    -----
    .. versionchanged:: 0.18
        The default value of `threshold_rel` has changed to None, which
        corresponds to letting `skimage.feature.peak_local_max` decide on the
        default. This is equivalent to `threshold_rel=0`.

    The `num_peaks` limit is applied before suppression of connected peaks.
    To limit the number of peaks after suppression, set `num_peaks=np.inf` and
    post-process the output of this function.

    Examples
    --------
    >>> from skimage.feature import peak_local_max
    >>> response = np.zeros((5, 5))
    >>> response[2:4, 2:4] = 1
    >>> response
    array([[0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 1., 1., 0.],
           [0., 0., 1., 1., 0.],
           [0., 0., 0., 0., 0.]])
    >>> peak_local_max(response)
    array([[2, 2],
           [2, 3],
           [3, 2],
           [3, 3]])
    >>> corner_peaks(response)
    array([[2, 2]])

    """
    if np.isinf(num_peaks):
        num_peaks = None

    # Get the coordinates of the detected peaks
    coords = peak_local_max(image, min_distance=min_distance,
                            threshold_abs=threshold_abs,
                            threshold_rel=threshold_rel,
                            exclude_border=exclude_border,
                            num_peaks=np.inf,
                            footprint=footprint, labels=labels,
                            num_peaks_per_label=num_peaks_per_label)

    if len(coords):
        # Use KDtree to find the peaks that are too close to each other
        tree = spatial.cKDTree(coords)

        rejected_peaks_indices = set()
        for idx, point in enumerate(coords):
            if idx not in rejected_peaks_indices:
                candidates = tree.query_ball_point(point, r=min_distance,
                                                   p=p_norm)
                candidates.remove(idx)
                rejected_peaks_indices.update(candidates)

        # Remove the peaks that are too close to each other
        coords = np.delete(coords, tuple(rejected_peaks_indices),
                           axis=0)[:num_peaks]

    if indices:
        return coords

    peaks = np.zeros_like(image, dtype=bool)
    peaks[tuple(coords.T)] = True

    return peaks


def corner_moravec(image, window_size=1):
    """Compute Moravec corner measure response image.

    This is one of the simplest corner detectors and is comparatively fast but
    has several limitations (e.g. not rotation invariant).

    Parameters
    ----------
    image : (M, N) ndarray
        Input image.
    window_size : int, optional
        Window size.

    Returns
    -------
    response : ndarray
        Moravec response image.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Corner_detection

    Examples
    --------
    >>> from skimage.feature import corner_moravec
    >>> square = np.zeros([7, 7])
    >>> square[3, 3] = 1
    >>> square.astype(int)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> corner_moravec(square).astype(int)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 2, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    """
    image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    return _corner_moravec(np.ascontiguousarray(image), window_size)


def corner_orientations(image, corners, mask):
    """Compute the orientation of corners.

    The orientation of corners is computed using the first order central moment
    i.e. the center of mass approach. The corner orientation is the angle of
    the vector from the corner coordinate to the intensity centroid in the
    local neighborhood around the corner calculated using first order central
    moment.

    Parameters
    ----------
    image : (M, N) array
        Input grayscale image.
    corners : (K, 2) array
        Corner coordinates as ``(row, col)``.
    mask : 2D array
        Mask defining the local neighborhood of the corner used for the
        calculation of the central moment.

    Returns
    -------
    orientations : (K, 1) array
        Orientations of corners in the range [-pi, pi].

    References
    ----------
    .. [1] Ethan Rublee, Vincent Rabaud, Kurt Konolige and Gary Bradski
          "ORB : An efficient alternative to SIFT and SURF"
          http://www.vision.cs.chubu.ac.jp/CV-R/pdf/Rublee_iccv2011.pdf
    .. [2] Paul L. Rosin, "Measuring Corner Properties"
          http://users.cs.cf.ac.uk/Paul.Rosin/corner2.pdf

    Examples
    --------
    >>> from skimage.morphology import octagon
    >>> from skimage.feature import (corner_fast, corner_peaks,
    ...                              corner_orientations)
    >>> square = np.zeros((12, 12))
    >>> square[3:9, 3:9] = 1
    >>> square.astype(int)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> corners = corner_peaks(corner_fast(square, 9), min_distance=1)
    >>> corners
    array([[3, 3],
           [3, 8],
           [8, 3],
           [8, 8]])
    >>> orientations = corner_orientations(square, corners, octagon(3, 2))
    >>> np.rad2deg(orientations)
    array([  45.,  135.,  -45., -135.])

    """
    image = _prepare_grayscale_input_2D(image)
    return _corner_orientations(image, corners, mask)
