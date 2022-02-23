"""
Ridge filters.

Ridge filters can be used to detect continuous edges, such as vessels,
neurites, wrinkles, rivers, and other tube-like structures. The present
class of ridge filters relies on the eigenvalues of the Hessian matrix of
image intensities to detect tube-like structures where the intensity changes
perpendicular but not along the structure.
"""


from warnings import warn

import numpy as np

from .._shared.utils import _supported_float_type, check_nD
from ..feature.corner import hessian_matrix, hessian_matrix_eigvals
from ..util import img_as_float, invert


def _divide_nonzero(array1, array2, cval=1e-10):
    """
    Divides two arrays.

    Denominator is set to small value where zero to avoid ZeroDivisionError and
    return finite float array.

    Parameters
    ----------
    array1 : (N, ..., M) ndarray
        Array 1 in the enumerator.
    array2 : (N, ..., M) ndarray
        Array 2 in the denominator.
    cval : float, optional
        Value used to replace zero entries in the denominator.

    Returns
    -------
    array : (N, ..., M) ndarray
        Quotient of the array division.
    """

    # Copy denominator
    denominator = np.copy(array2)

    # Set zero entries of denominator to small value
    denominator[denominator == 0] = cval

    # Return quotient
    return np.divide(array1, denominator)


def _sortbyabs(array, axis=0):
    """
    Sort array along a given axis by absolute values.

    Parameters
    ----------
    array : (N, ..., M) ndarray
        Array with input image data.
    axis : int
        Axis along which to sort.

    Returns
    -------
    array : (N, ..., M) ndarray
        Array sorted along a given axis by absolute values.

    Notes
    -----
    Modified from: http://stackoverflow.com/a/11253931/4067734
    """

    # Create auxiliary array for indexing
    index = list(np.ix_(*[np.arange(i) for i in array.shape]))

    # Get indices of abs sorted array
    index[axis] = np.abs(array).argsort(axis)

    # Return abs sorted array
    return array[tuple(index)]


def _check_sigmas(sigmas):
    """Check sigma values for ridges filters.

    Parameters
    ----------
    sigmas : iterable of floats
        Sigmas argument to be checked

    Returns
    -------
    sigmas : ndarray
        input iterable converted to ndarray

    Raises
    ------
    ValueError if any input value is negative

    """
    sigmas = np.asarray(sigmas).ravel()
    if np.any(sigmas < 0.0):
        raise ValueError('Sigma values should be equal to or greater '
                         'than zero.')
    return sigmas


def compute_hessian_eigenvalues(image, sigma, sorting='none',
                                mode='constant', cval=0):
    """
    Compute Hessian eigenvalues of nD images.

    For 2D images, the computation uses a more efficient, skimage-based
    algorithm.

    Parameters
    ----------
    image : (N, ..., M) ndarray
        Array with input image data.
    sigma : float
        Smoothing factor of image for detection of structures at different
        (sigma) scales.
    sorting : {'val', 'abs', 'none'}, optional
        Sorting of eigenvalues by values ('val') or absolute values ('abs'),
        or without sorting ('none'). Default is 'none'.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.

    Returns
    -------
    eigenvalues : (D, N, ..., M) ndarray
        Array with (sorted) eigenvalues of Hessian eigenvalues for each pixel
        of the input image.
    """

    # Convert image to float
    float_dtype = _supported_float_type(image.dtype)
    # rescales integer images to [-1, 1]
    image = img_as_float(image)
    # make sure float16 gets promoted to float32
    image = image.astype(float_dtype, copy=False)

    # Make nD hessian
    hessian_elements = hessian_matrix(image, sigma=sigma, order='rc',
                                      mode=mode, cval=cval)

    # Correct for scale
    hessian_elements = [(sigma ** 2) * e for e in hessian_elements]

    # Compute Hessian eigenvalues
    hessian_eigenvalues = hessian_matrix_eigvals(hessian_elements)

    if sorting == 'abs':

        # Sort eigenvalues by absolute values in ascending order
        hessian_eigenvalues = _sortbyabs(hessian_eigenvalues, axis=0)

    elif sorting == 'val':

        # Sort eigenvalues by values in ascending order
        hessian_eigenvalues = np.sort(hessian_eigenvalues, axis=0)

    # Return Hessian eigenvalues
    return hessian_eigenvalues


def meijering(image, sigmas=range(1, 10, 2), alpha=None,
              black_ridges=True, mode='reflect', cval=0):
    """
    Filter an image with the Meijering neuriteness filter.

    This filter can be used to detect continuous ridges, e.g. neurites,
    wrinkles, rivers. It can be used to calculate the fraction of the
    whole image containing such objects.

    Calculates the eigenvectors of the Hessian to compute the similarity of
    an image region to neurites, according to the method described in [1]_.

    Parameters
    ----------
    image : (N, M[, ..., P]) ndarray
        Array with input image data.
    sigmas : iterable of floats, optional
        Sigmas used as scales of filter
    alpha : float, optional
        Frangi correction constant that adjusts the filter's
        sensitivity to deviation from a plate-like structure.
    black_ridges : boolean, optional
        When True (the default), the filter detects black ridges; when
        False, it detects white ridges.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.

    Returns
    -------
    out : (N, M[, ..., P]) ndarray
        Filtered image (maximum of pixels across all scales).

    See also
    --------
    sato
    frangi
    hessian

    References
    ----------
    .. [1] Meijering, E., Jacob, M., Sarria, J. C., Steiner, P., Hirling, H.,
        Unser, M. (2004). Design and validation of a tool for neurite tracing
        and analysis in fluorescence microscopy images. Cytometry Part A,
        58(2), 167-176.
        :DOI:`10.1002/cyto.a.20022`
    """

    # Check (sigma) scales
    sigmas = _check_sigmas(sigmas)

    # Get image dimensions
    ndim = image.ndim

    # Set parameters
    if alpha is None:
        alpha = 1.0 / ndim

    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)

    # Invert image to detect dark ridges on bright background
    if black_ridges:
        image = invert(image)

    # Generate empty (n+1)D arrays for storing auxiliary images filtered at
    # different (sigma) scales
    filtered_array = np.zeros(sigmas.shape + image.shape, float_dtype)

    # Filtering for all (sigma) scales
    for i, sigma in enumerate(sigmas):

        # Calculate (sorted) eigenvalues
        eigenvalues = compute_hessian_eigenvalues(image, sigma, sorting='abs',
                                                  mode=mode, cval=cval)

        if ndim > 1:

            # Set coefficients for scaling eigenvalues
            coefficients = [alpha] * ndim
            coefficients[0] = 1

            # Compute normalized eigenvalues l_i = e_i + sum_{j!=i} alpha * e_j
            auxiliary = [np.sum([eigenvalues[i] * np.roll(coefficients, j)[i]
                         for j in range(ndim)], axis=0) for i in range(ndim)]

            # Get maximum eigenvalues by magnitude
            auxiliary = auxiliary[-1]

            # Rescale image intensity and avoid ZeroDivisionError
            filtered = _divide_nonzero(auxiliary, np.min(auxiliary))

            # Remove background
            filtered = np.where(auxiliary < 0, filtered, 0)

            # Store results in (n+1)D matrices
            filtered_array[i] = filtered

    # Return for every pixel the maximum value over all (sigma) scales
    return np.max(filtered_array, axis=0)


def sato(image, sigmas=range(1, 10, 2), black_ridges=True,
         mode='reflect', cval=0):
    """
    Filter an image with the Sato tubeness filter.

    This filter can be used to detect continuous ridges, e.g. tubes,
    wrinkles, rivers. It can be used to calculate the fraction of the
    whole image containing such objects.

    Defined only for 2-D and 3-D images. Calculates the eigenvectors of the
    Hessian to compute the similarity of an image region to tubes, according to
    the method described in [1]_.

    Parameters
    ----------
    image : (N, M[, P]) ndarray
        Array with input image data.
    sigmas : iterable of floats, optional
        Sigmas used as scales of filter.
    black_ridges : boolean, optional
        When True (the default), the filter detects black ridges; when
        False, it detects white ridges.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.

    Returns
    -------
    out : (N, M[, P]) ndarray
        Filtered image (maximum of pixels across all scales).

    See also
    --------
    meijering
    frangi
    hessian

    References
    ----------
    .. [1] Sato, Y., Nakajima, S., Shiraga, N., Atsumi, H., Yoshida, S.,
        Koller, T., ..., Kikinis, R. (1998). Three-dimensional multi-scale line
        filter for segmentation and visualization of curvilinear structures in
        medical images. Medical image analysis, 2(2), 143-168.
        :DOI:`10.1016/S1361-8415(98)80009-1`
    """

    # Check image dimensions
    check_nD(image, [2, 3])

    # Check (sigma) scales
    sigmas = _check_sigmas(sigmas)

    # Invert image to detect bright ridges on dark background
    if not black_ridges:
        image = invert(image)

    float_dtype = _supported_float_type(image.dtype)

    # Generate empty (n+1)D arrays for storing auxiliary images filtered
    # at different (sigma) scales
    filtered_array = np.zeros(sigmas.shape + image.shape, dtype=float_dtype)

    # Filtering for all (sigma) scales
    for i, sigma in enumerate(sigmas):

        # Calculate (sorted) eigenvalues
        lamba1, *lambdas = compute_hessian_eigenvalues(image, sigma,
                                                       sorting='val',
                                                       mode=mode, cval=cval)

        # Compute tubeness, see  equation (9) in reference [1]_.
        # np.abs(lambda2) in 2D, np.sqrt(np.abs(lambda2 * lambda3)) in 3D
        filtered = np.abs(np.multiply.reduce(lambdas)) ** (1/len(lambdas))

        # Remove background and store results in (n+1)D matrices
        filtered_array[i] = np.where(lambdas[-1] > 0, filtered, 0)

    # Return for every pixel the maximum value over all (sigma) scales
    return np.max(filtered_array, axis=0)


def frangi(image, sigmas=range(1, 10, 2), scale_range=None,
           scale_step=None, alpha=0.5, beta=0.5, gamma=15,
           black_ridges=True, mode='reflect', cval=0):
    """
    Filter an image with the Frangi vesselness filter.

    This filter can be used to detect continuous ridges, e.g. vessels,
    wrinkles, rivers. It can be used to calculate the fraction of the
    whole image containing such objects.

    Defined only for 2-D and 3-D images. Calculates the eigenvectors of the
    Hessian to compute the similarity of an image region to vessels, according
    to the method described in [1]_.

    Parameters
    ----------
    image : (N, M[, P]) ndarray
        Array with input image data.
    sigmas : iterable of floats, optional
        Sigmas used as scales of filter, i.e.,
        np.arange(scale_range[0], scale_range[1], scale_step)
    scale_range : 2-tuple of floats, optional
        The range of sigmas used.
    scale_step : float, optional
        Step size between sigmas.
    alpha : float, optional
        Frangi correction constant that adjusts the filter's
        sensitivity to deviation from a plate-like structure.
    beta : float, optional
        Frangi correction constant that adjusts the filter's
        sensitivity to deviation from a blob-like structure.
    gamma : float, optional
        Frangi correction constant that adjusts the filter's
        sensitivity to areas of high variance/texture/structure.
    black_ridges : boolean, optional
        When True (the default), the filter detects black ridges; when
        False, it detects white ridges.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.

    Returns
    -------
    out : (N, M[, P]) ndarray
        Filtered image (maximum of pixels across all scales).

    Notes
    -----
    Written by Marc Schrijver, November 2001
    Re-Written by D. J. Kroon, University of Twente, May 2009, [2]_
    Adoption of 3D version from D. G. Ellis, Januar 20017, [3]_

    See also
    --------
    meijering
    sato
    hessian

    References
    ----------
    .. [1] Frangi, A. F., Niessen, W. J., Vincken, K. L., & Viergever, M. A.
        (1998,). Multiscale vessel enhancement filtering. In International
        Conference on Medical Image Computing and Computer-Assisted
        Intervention (pp. 130-137). Springer Berlin Heidelberg.
        :DOI:`10.1007/BFb0056195`
    .. [2] Kroon, D. J.: Hessian based Frangi vesselness filter.
    .. [3] Ellis, D. G.: https://github.com/ellisdg/frangi3d/tree/master/frangi
    """
    if scale_range is not None and scale_step is not None:
        warn('Use keyword parameter `sigmas` instead of `scale_range` and '
             '`scale_range` which will be removed in version 0.17.',
             stacklevel=2)
        sigmas = np.arange(scale_range[0], scale_range[1], scale_step)

    # Check image dimensions
    check_nD(image, [2, 3])

    # Check (sigma) scales
    sigmas = _check_sigmas(sigmas)

    # Rescale filter parameters
    alpha_sq = 2 * alpha ** 2
    beta_sq = 2 * beta ** 2
    gamma_sq = 2 * gamma ** 2

    # Get image dimensions
    ndim = image.ndim

    # Invert image to detect dark ridges on light background
    if black_ridges:
        image = invert(image)

    float_dtype = _supported_float_type(image.dtype)

    # Generate empty (n+1)D arrays for storing auxiliary images filtered
    # at different (sigma) scales
    filtered_array = np.zeros(sigmas.shape + image.shape, dtype=float_dtype)
    lambdas_array = np.zeros_like(filtered_array, dtype=float_dtype)

    # Filtering for all (sigma) scales
    for i, sigma in enumerate(sigmas):

        # Calculate (abs sorted) eigenvalues
        lambda1, *lambdas = compute_hessian_eigenvalues(image, sigma,
                                                        sorting='abs',
                                                        mode=mode, cval=cval)

        # Compute sensitivity to deviation from a plate-like
        # structure see equations (11) and (15) in reference [1]_
        r_a = np.inf if ndim == 2 else _divide_nonzero(*lambdas) ** 2

        # Compute sensitivity to deviation from a blob-like structure,
        # see equations (10) and (15) in reference [1]_,
        # np.abs(lambda2) in 2D, np.sqrt(np.abs(lambda2 * lambda3)) in 3D
        filtered_raw = np.abs(np.multiply.reduce(lambdas)) ** (1/len(lambdas))
        r_b = _divide_nonzero(lambda1, filtered_raw) ** 2

        # Compute sensitivity to areas of high variance/texture/structure,
        # see equation (12)in reference [1]_
        r_g = sum([lambda1 ** 2] + [lambdai ** 2 for lambdai in lambdas])

        # Compute output image for given (sigma) scale and store results in
        # (n+1)D matrices, see equations (13) and (15) in reference [1]_
        filtered_array[i] = ((1 - np.exp(-r_a / alpha_sq))
                             * np.exp(-r_b / beta_sq)
                             * (1 - np.exp(-r_g / gamma_sq)))

        lambdas_array[i] = np.max(lambdas, axis=0)

    # Remove background
    filtered_array[lambdas_array > 0] = 0

    # Return for every pixel the maximum value over all (sigma) scales
    return np.max(filtered_array, axis=0)


def hessian(image, sigmas=range(1, 10, 2), scale_range=None, scale_step=None,
            alpha=0.5, beta=0.5, gamma=15, black_ridges=True, mode='reflect',
            cval=0):
    """Filter an image with the Hybrid Hessian filter.

    This filter can be used to detect continuous edges, e.g. vessels,
    wrinkles, rivers. It can be used to calculate the fraction of the whole
    image containing such objects.

    Defined only for 2-D and 3-D images. Almost equal to Frangi filter, but
    uses alternative method of smoothing. Refer to [1]_ to find the differences
    between Frangi and Hessian filters.

    Parameters
    ----------
    image : (N, M[, P]) ndarray
        Array with input image data.
    sigmas : iterable of floats, optional
        Sigmas used as scales of filter, i.e.,
        np.arange(scale_range[0], scale_range[1], scale_step)
    scale_range : 2-tuple of floats, optional
        The range of sigmas used.
    scale_step : float, optional
        Step size between sigmas.
    beta : float, optional
        Frangi correction constant that adjusts the filter's
        sensitivity to deviation from a blob-like structure.
    gamma : float, optional
        Frangi correction constant that adjusts the filter's
        sensitivity to areas of high variance/texture/structure.
    black_ridges : boolean, optional
        When True (the default), the filter detects black ridges; when
        False, it detects white ridges.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.

    Returns
    -------
    out : (N, M[, P]) ndarray
        Filtered image (maximum of pixels across all scales).

    Notes
    -----
    Written by Marc Schrijver (November 2001)
    Re-Written by D. J. Kroon University of Twente (May 2009) [2]_

    See also
    --------
    meijering
    sato
    frangi

    References
    ----------
    .. [1] Ng, C. C., Yap, M. H., Costen, N., & Li, B. (2014,). Automatic
        wrinkle detection using hybrid Hessian filter. In Asian Conference on
        Computer Vision (pp. 609-622). Springer International Publishing.
        :DOI:`10.1007/978-3-319-16811-1_40`
    .. [2] Kroon, D. J.: Hessian based Frangi vesselness filter.
    """
    filtered = frangi(image, sigmas=sigmas, scale_range=scale_range,
                      scale_step=scale_step, alpha=alpha, beta=beta,
                      gamma=gamma, black_ridges=black_ridges, mode=mode,
                      cval=cval)

    filtered[filtered <= 0] = 1
    return filtered
