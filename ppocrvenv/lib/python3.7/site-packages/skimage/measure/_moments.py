import numpy as np
from .._shared.utils import _supported_float_type, check_nD
from . import _moments_cy
import itertools


def moments_coords(coords, order=3):
    """Calculate all raw image moments up to a certain order.

    The following properties can be calculated from raw image moments:
     * Area as: ``M[0, 0]``.
     * Centroid as: {``M[1, 0] / M[0, 0]``, ``M[0, 1] / M[0, 0]``}.

    Note that raw moments are neither translation, scale nor rotation
    invariant.

    Parameters
    ----------
    coords : (N, D) double or uint8 array
        Array of N points that describe an image of D dimensionality in
        Cartesian space.
    order : int, optional
        Maximum order of moments. Default is 3.

    Returns
    -------
    M : (``order + 1``, ``order + 1``, ...) array
        Raw image moments. (D dimensions)

    References
    ----------
    .. [1] Johannes Kilian. Simple Image Analysis By Moments. Durham
           University, version 0.2, Durham, 2001.

    Examples
    --------
    >>> coords = np.array([[row, col]
    ...                    for row in range(13, 17)
    ...                    for col in range(14, 18)], dtype=np.double)
    >>> M = moments_coords(coords)
    >>> centroid = (M[1, 0] / M[0, 0], M[0, 1] / M[0, 0])
    >>> centroid
    (14.5, 15.5)
    """
    return moments_coords_central(coords, 0, order=order)


def moments_coords_central(coords, center=None, order=3):
    """Calculate all central image moments up to a certain order.

    The following properties can be calculated from raw image moments:
     * Area as: ``M[0, 0]``.
     * Centroid as: {``M[1, 0] / M[0, 0]``, ``M[0, 1] / M[0, 0]``}.

    Note that raw moments are neither translation, scale nor rotation
    invariant.

    Parameters
    ----------
    coords : (N, D) double or uint8 array
        Array of N points that describe an image of D dimensionality in
        Cartesian space. A tuple of coordinates as returned by
        ``np.nonzero`` is also accepted as input.
    center : tuple of float, optional
        Coordinates of the image centroid. This will be computed if it
        is not provided.
    order : int, optional
        Maximum order of moments. Default is 3.

    Returns
    -------
    Mc : (``order + 1``, ``order + 1``, ...) array
        Central image moments. (D dimensions)

    References
    ----------
    .. [1] Johannes Kilian. Simple Image Analysis By Moments. Durham
           University, version 0.2, Durham, 2001.

    Examples
    --------
    >>> coords = np.array([[row, col]
    ...                    for row in range(13, 17)
    ...                    for col in range(14, 18)])
    >>> moments_coords_central(coords)
    array([[16.,  0., 20.,  0.],
           [ 0.,  0.,  0.,  0.],
           [20.,  0., 25.,  0.],
           [ 0.,  0.,  0.,  0.]])

    As seen above, for symmetric objects, odd-order moments (columns 1 and 3,
    rows 1 and 3) are zero when centered on the centroid, or center of mass,
    of the object (the default). If we break the symmetry by adding a new
    point, this no longer holds:

    >>> coords2 = np.concatenate((coords, [[17, 17]]), axis=0)
    >>> np.round(moments_coords_central(coords2),
    ...          decimals=2)  # doctest: +NORMALIZE_WHITESPACE
    array([[17.  ,  0.  , 22.12, -2.49],
           [ 0.  ,  3.53,  1.73,  7.4 ],
           [25.88,  6.02, 36.63,  8.83],
           [ 4.15, 19.17, 14.8 , 39.6 ]])

    Image moments and central image moments are equivalent (by definition)
    when the center is (0, 0):

    >>> np.allclose(moments_coords(coords),
    ...             moments_coords_central(coords, (0, 0)))
    True
    """
    if isinstance(coords, tuple):
        # This format corresponds to coordinate tuples as returned by
        # e.g. np.nonzero: (row_coords, column_coords).
        # We represent them as an npoints x ndim array.
        coords = np.stack(coords, axis=-1)
    check_nD(coords, 2)
    ndim = coords.shape[1]

    float_type = _supported_float_type(coords.dtype)
    if center is None:
        center = np.mean(coords, axis=0, dtype=float)

    # center the coordinates
    coords = coords.astype(float_type, copy=False) - center

    # generate all possible exponents for each axis in the given set of points
    # produces a matrix of shape (N, D, order + 1)
    coords = np.stack([coords ** c for c in range(order + 1)], axis=-1)

    # add extra dimensions for proper broadcasting
    coords = coords.reshape(coords.shape + (1,) * (ndim - 1))

    calc = 1

    for axis in range(ndim):
        # isolate each point's axis
        isolated_axis = coords[:, axis]

        # rotate orientation of matrix for proper broadcasting
        isolated_axis = np.moveaxis(isolated_axis, 1, 1 + axis)

        # calculate the moments for each point, one axis at a time
        calc = calc * isolated_axis

    # sum all individual point moments to get our final answer
    Mc = np.sum(calc, axis=0)

    return Mc


def moments(image, order=3):
    """Calculate all raw image moments up to a certain order.

    The following properties can be calculated from raw image moments:
     * Area as: ``M[0, 0]``.
     * Centroid as: {``M[1, 0] / M[0, 0]``, ``M[0, 1] / M[0, 0]``}.

    Note that raw moments are neither translation, scale nor rotation
    invariant.

    Parameters
    ----------
    image : nD double or uint8 array
        Rasterized shape as image.
    order : int, optional
        Maximum order of moments. Default is 3.

    Returns
    -------
    m : (``order + 1``, ``order + 1``) array
        Raw image moments.

    References
    ----------
    .. [1] Wilhelm Burger, Mark Burge. Principles of Digital Image Processing:
           Core Algorithms. Springer-Verlag, London, 2009.
    .. [2] B. Jähne. Digital Image Processing. Springer-Verlag,
           Berlin-Heidelberg, 6. edition, 2005.
    .. [3] T. H. Reiss. Recognizing Planar Objects Using Invariant Image
           Features, from Lecture notes in computer science, p. 676. Springer,
           Berlin, 1993.
    .. [4] https://en.wikipedia.org/wiki/Image_moment

    Examples
    --------
    >>> image = np.zeros((20, 20), dtype=np.double)
    >>> image[13:17, 13:17] = 1
    >>> M = moments(image)
    >>> centroid = (M[1, 0] / M[0, 0], M[0, 1] / M[0, 0])
    >>> centroid
    (14.5, 14.5)
    """
    return moments_central(image, (0,) * image.ndim, order=order)


def moments_central(image, center=None, order=3, **kwargs):
    """Calculate all central image moments up to a certain order.

    The center coordinates (cr, cc) can be calculated from the raw moments as:
    {``M[1, 0] / M[0, 0]``, ``M[0, 1] / M[0, 0]``}.

    Note that central moments are translation invariant but not scale and
    rotation invariant.

    Parameters
    ----------
    image : nD double or uint8 array
        Rasterized shape as image.
    center : tuple of float, optional
        Coordinates of the image centroid. This will be computed if it
        is not provided.
    order : int, optional
        The maximum order of moments computed.

    Returns
    -------
    mu : (``order + 1``, ``order + 1``) array
        Central image moments.

    References
    ----------
    .. [1] Wilhelm Burger, Mark Burge. Principles of Digital Image Processing:
           Core Algorithms. Springer-Verlag, London, 2009.
    .. [2] B. Jähne. Digital Image Processing. Springer-Verlag,
           Berlin-Heidelberg, 6. edition, 2005.
    .. [3] T. H. Reiss. Recognizing Planar Objects Using Invariant Image
           Features, from Lecture notes in computer science, p. 676. Springer,
           Berlin, 1993.
    .. [4] https://en.wikipedia.org/wiki/Image_moment

    Examples
    --------
    >>> image = np.zeros((20, 20), dtype=np.double)
    >>> image[13:17, 13:17] = 1
    >>> M = moments(image)
    >>> centroid = (M[1, 0] / M[0, 0], M[0, 1] / M[0, 0])
    >>> moments_central(image, centroid)
    array([[16.,  0., 20.,  0.],
           [ 0.,  0.,  0.,  0.],
           [20.,  0., 25.,  0.],
           [ 0.,  0.,  0.,  0.]])
    """
    if center is None:
        center = centroid(image)
    float_dtype = _supported_float_type(image.dtype)
    calc = image.astype(float_dtype, copy=False)
    for dim, dim_length in enumerate(image.shape):
        delta = np.arange(dim_length, dtype=float_dtype) - center[dim]
        powers_of_delta = (
            delta[:, np.newaxis] ** np.arange(order + 1, dtype=float_dtype)
        )
        calc = np.rollaxis(calc, dim, image.ndim)
        calc = np.dot(calc, powers_of_delta)
        calc = np.rollaxis(calc, -1, dim)
    return calc


def moments_normalized(mu, order=3):
    """Calculate all normalized central image moments up to a certain order.

    Note that normalized central moments are translation and scale invariant
    but not rotation invariant.

    Parameters
    ----------
    mu : (M,[ ...,] M) array
        Central image moments, where M must be greater than or equal
        to ``order``.
    order : int, optional
        Maximum order of moments. Default is 3.

    Returns
    -------
    nu : (``order + 1``,[ ...,] ``order + 1``) array
        Normalized central image moments.

    References
    ----------
    .. [1] Wilhelm Burger, Mark Burge. Principles of Digital Image Processing:
           Core Algorithms. Springer-Verlag, London, 2009.
    .. [2] B. Jähne. Digital Image Processing. Springer-Verlag,
           Berlin-Heidelberg, 6. edition, 2005.
    .. [3] T. H. Reiss. Recognizing Planar Objects Using Invariant Image
           Features, from Lecture notes in computer science, p. 676. Springer,
           Berlin, 1993.
    .. [4] https://en.wikipedia.org/wiki/Image_moment

    Examples
    --------
    >>> image = np.zeros((20, 20), dtype=np.double)
    >>> image[13:17, 13:17] = 1
    >>> m = moments(image)
    >>> centroid = (m[0, 1] / m[0, 0], m[1, 0] / m[0, 0])
    >>> mu = moments_central(image, centroid)
    >>> moments_normalized(mu)
    array([[       nan,        nan, 0.078125  , 0.        ],
           [       nan, 0.        , 0.        , 0.        ],
           [0.078125  , 0.        , 0.00610352, 0.        ],
           [0.        , 0.        , 0.        , 0.        ]])
    """
    if np.any(np.array(mu.shape) <= order):
        raise ValueError("Shape of image moments must be >= `order`")
    nu = np.zeros_like(mu)
    mu0 = mu.ravel()[0]
    for powers in itertools.product(range(order + 1), repeat=mu.ndim):
        if sum(powers) < 2:
            nu[powers] = np.nan
        else:
            nu[powers] = mu[powers] / (mu0 ** (sum(powers) / nu.ndim + 1))
    return nu


def moments_hu(nu):
    """Calculate Hu's set of image moments (2D-only).

    Note that this set of moments is proofed to be translation, scale and
    rotation invariant.

    Parameters
    ----------
    nu : (M, M) array
        Normalized central image moments, where M must be >= 4.

    Returns
    -------
    nu : (7,) array
        Hu's set of image moments.

    References
    ----------
    .. [1] M. K. Hu, "Visual Pattern Recognition by Moment Invariants",
           IRE Trans. Info. Theory, vol. IT-8, pp. 179-187, 1962
    .. [2] Wilhelm Burger, Mark Burge. Principles of Digital Image Processing:
           Core Algorithms. Springer-Verlag, London, 2009.
    .. [3] B. Jähne. Digital Image Processing. Springer-Verlag,
           Berlin-Heidelberg, 6. edition, 2005.
    .. [4] T. H. Reiss. Recognizing Planar Objects Using Invariant Image
           Features, from Lecture notes in computer science, p. 676. Springer,
           Berlin, 1993.
    .. [5] https://en.wikipedia.org/wiki/Image_moment

    Examples
    --------
    >>> image = np.zeros((20, 20), dtype=np.double)
    >>> image[13:17, 13:17] = 0.5
    >>> image[10:12, 10:12] = 1
    >>> mu = moments_central(image)
    >>> nu = moments_normalized(mu)
    >>> moments_hu(nu)
    array([7.45370370e-01, 3.51165981e-01, 1.04049179e-01, 4.06442107e-02,
           2.64312299e-03, 2.40854582e-02, 4.33680869e-19])
    """
    dtype = np.float32 if nu.dtype == 'float32' else np.float64
    return _moments_cy.moments_hu(nu.astype(dtype, copy=False))


def centroid(image):
    """Return the (weighted) centroid of an image.

    Parameters
    ----------
    image : array
        The input image.

    Returns
    -------
    center : tuple of float, length ``image.ndim``
        The centroid of the (nonzero) pixels in ``image``.

    Examples
    --------
    >>> image = np.zeros((20, 20), dtype=np.double)
    >>> image[13:17, 13:17] = 0.5
    >>> image[10:12, 10:12] = 1
    >>> centroid(image)
    array([13.16666667, 13.16666667])
    """
    M = moments_central(image, center=(0,) * image.ndim, order=1)
    center = (M[tuple(np.eye(image.ndim, dtype=int))]  # array of weighted sums
                                                       # for each axis
              / M[(0,) * image.ndim])  # weighted sum of all points
    return center


def inertia_tensor(image, mu=None):
    """Compute the inertia tensor of the input image.

    Parameters
    ----------
    image : array
        The input image.
    mu : array, optional
        The pre-computed central moments of ``image``. The inertia tensor
        computation requires the central moments of the image. If an
        application requires both the central moments and the inertia tensor
        (for example, `skimage.measure.regionprops`), then it is more
        efficient to pre-compute them and pass them to the inertia tensor
        call.

    Returns
    -------
    T : array, shape ``(image.ndim, image.ndim)``
        The inertia tensor of the input image. :math:`T_{i, j}` contains
        the covariance of image intensity along axes :math:`i` and :math:`j`.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
    .. [2] Bernd Jähne. Spatio-Temporal Image Processing: Theory and
           Scientific Applications. (Chapter 8: Tensor Methods) Springer, 1993.
    """
    if mu is None:
        mu = moments_central(image, order=2)  # don't need higher-order moments
    mu0 = mu[(0,) * image.ndim]
    result = np.zeros((image.ndim, image.ndim), dtype=mu.dtype)

    # nD expression to get coordinates ([2, 0], [0, 2]) (2D),
    # ([2, 0, 0], [0, 2, 0], [0, 0, 2]) (3D), etc.
    corners2 = tuple(2 * np.eye(image.ndim, dtype=int))
    d = np.diag(result)
    d.flags.writeable = True
    # See https://ocw.mit.edu/courses/aeronautics-and-astronautics/
    #             16-07-dynamics-fall-2009/lecture-notes/MIT16_07F09_Lec26.pdf
    # Iii is the sum of second-order moments of every axis *except* i, not the
    # second order moment of axis i.
    # See also https://github.com/scikit-image/scikit-image/issues/3229
    d[:] = (np.sum(mu[corners2]) - mu[corners2]) / mu0

    for dims in itertools.combinations(range(image.ndim), 2):
        mu_index = np.zeros(image.ndim, dtype=int)
        mu_index[list(dims)] = 1
        result[dims] = -mu[tuple(mu_index)] / mu0
        result.T[dims] = -mu[tuple(mu_index)] / mu0
    return result


def inertia_tensor_eigvals(image, mu=None, T=None):
    """Compute the eigenvalues of the inertia tensor of the image.

    The inertia tensor measures covariance of the image intensity along
    the image axes. (See `inertia_tensor`.) The relative magnitude of the
    eigenvalues of the tensor is thus a measure of the elongation of a
    (bright) object in the image.

    Parameters
    ----------
    image : array
        The input image.
    mu : array, optional
        The pre-computed central moments of ``image``.
    T : array, shape ``(image.ndim, image.ndim)``
        The pre-computed inertia tensor. If ``T`` is given, ``mu`` and
        ``image`` are ignored.

    Returns
    -------
    eigvals : list of float, length ``image.ndim``
        The eigenvalues of the inertia tensor of ``image``, in descending
        order.

    Notes
    -----
    Computing the eigenvalues requires the inertia tensor of the input image.
    This is much faster if the central moments (``mu``) are provided, or,
    alternatively, one can provide the inertia tensor (``T``) directly.
    """
    if T is None:
        T = inertia_tensor(image, mu)
    eigvals = np.linalg.eigvalsh(T)
    # Floating point precision problems could make a positive
    # semidefinite matrix have an eigenvalue that is very slightly
    # negative. This can cause problems down the line, so set values
    # very near zero to zero.
    eigvals = np.clip(eigvals, 0, None, out=eigvals)
    return sorted(eigvals, reverse=True)
