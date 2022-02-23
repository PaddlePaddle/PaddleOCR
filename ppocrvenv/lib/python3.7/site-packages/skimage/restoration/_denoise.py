import functools
from math import ceil
import numbers

import scipy.stats
import numpy as np
import pywt

from ..util.dtype import img_as_float
from .._shared import utils
from .._shared.utils import _supported_float_type, warn
from ._denoise_cy import _denoise_bilateral, _denoise_tv_bregman
from .. import color
from ..color.colorconv import ycbcr_from_rgb


def _gaussian_weight(array, sigma_squared, *, dtype=float):
    """Helping function. Define a Gaussian weighting from array and
    sigma_square.

    Parameters
    ----------
    array : ndarray
        Input array.
    sigma_squared : float
        The squared standard deviation used in the filter.
    dtype : data type object, optional (default : float)
        The type and size of the data to be returned.

    Returns
    -------
    gaussian : ndarray
        The input array filtered by the Gaussian.
    """
    return np.exp(-0.5 * (array ** 2 / sigma_squared), dtype=dtype)


def _compute_color_lut(bins, sigma, max_value, *, dtype=float):
    """Helping function. Define a lookup table containing Gaussian filter
    values using the color distance sigma.

    Parameters
    ----------
    bins : int
        Number of discrete values for Gaussian weights of color filtering.
        A larger value results in improved accuracy.
    sigma : float
        Standard deviation for grayvalue/color distance (radiometric
        similarity). A larger value results in averaging of pixels with larger
        radiometric differences. Note, that the image will be converted using
        the `img_as_float` function and thus the standard deviation is in
        respect to the range ``[0, 1]``. If the value is ``None`` the standard
        deviation of the ``image`` will be used.
    max_value : float
        Maximum value of the input image.
    dtype : data type object, optional (default : float)
        The type and size of the data to be returned.

    Returns
    -------
    color_lut : ndarray
        Lookup table for the color distance sigma.
    """
    values = np.linspace(0, max_value, bins, endpoint=False)
    return _gaussian_weight(values, sigma**2, dtype=dtype)


def _compute_spatial_lut(win_size, sigma, *, dtype=float):
    """Helping function. Define a lookup table containing Gaussian filter
    values using the spatial sigma.

    Parameters
    ----------
    win_size : int
        Window size for filtering.
        If win_size is not specified, it is calculated as
        ``max(5, 2 * ceil(3 * sigma_spatial) + 1)``.
    sigma : float
        Standard deviation for range distance. A larger value results in
        averaging of pixels with larger spatial differences.
    dtype : data type object
        The type and size of the data to be returned.

    Returns
    -------
    spatial_lut : ndarray
        Lookup table for the spatial sigma.
    """
    grid_points = np.arange(-win_size // 2, win_size // 2 + 1)
    rr, cc = np.meshgrid(grid_points, grid_points, indexing='ij')
    distances = np.hypot(rr, cc)
    return _gaussian_weight(distances, sigma**2, dtype=dtype).ravel()


@utils.channel_as_last_axis()
@utils.deprecate_multichannel_kwarg(multichannel_position=7)
def denoise_bilateral(image, win_size=None, sigma_color=None, sigma_spatial=1,
                      bins=10000, mode='constant', cval=0, multichannel=False,
                      *, channel_axis=None):
    """Denoise image using bilateral filter.

    Parameters
    ----------
    image : ndarray, shape (M, N[, 3])
        Input image, 2D grayscale or RGB.
    win_size : int
        Window size for filtering.
        If win_size is not specified, it is calculated as
        ``max(5, 2 * ceil(3 * sigma_spatial) + 1)``.
    sigma_color : float
        Standard deviation for grayvalue/color distance (radiometric
        similarity). A larger value results in averaging of pixels with larger
        radiometric differences. If ``None``, the standard deviation of
        ``image`` will be used.
    sigma_spatial : float
        Standard deviation for range distance. A larger value results in
        averaging of pixels with larger spatial differences.
    bins : int
        Number of discrete values for Gaussian weights of color filtering.
        A larger value results in improved accuracy.
    mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}
        How to handle values outside the image borders. See
        `numpy.pad` for detail.
    cval : string
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    multichannel : bool
        Whether the last axis of the image is to be interpreted as multiple
        channels or another spatial dimension. This argument is deprecated:
        specify `channel_axis` instead.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    denoised : ndarray
        Denoised image.

    Notes
    -----
    This is an edge-preserving, denoising filter. It averages pixels based on
    their spatial closeness and radiometric similarity [1]_.

    Spatial closeness is measured by the Gaussian function of the Euclidean
    distance between two pixels and a certain standard deviation
    (`sigma_spatial`).

    Radiometric similarity is measured by the Gaussian function of the
    Euclidean distance between two color values and a certain standard
    deviation (`sigma_color`).

    Note that, if the image is of any `int` dtype, ``image`` will be
    converted using the `img_as_float` function and thus the standard
    deviation (`sigma_color`) will be in range ``[0, 1]``.

    For more information on scikit-image's data type conversions and how
    images are rescaled in these conversions,
    see: https://scikit-image.org/docs/stable/user_guide/data_types.html.

    References
    ----------
    .. [1] C. Tomasi and R. Manduchi. "Bilateral Filtering for Gray and Color
           Images." IEEE International Conference on Computer Vision (1998)
           839-846. :DOI:`10.1109/ICCV.1998.710815`

    Examples
    --------
    >>> from skimage import data, img_as_float
    >>> astro = img_as_float(data.astronaut())
    >>> astro = astro[220:300, 220:320]
    >>> rng = np.random.default_rng()
    >>> noisy = astro + 0.6 * astro.std() * rng.random(astro.shape)
    >>> noisy = np.clip(noisy, 0, 1)
    >>> denoised = denoise_bilateral(noisy, sigma_color=0.05, sigma_spatial=15,
    ...                              channel_axis=-1)
    """
    if channel_axis is not None:
        if image.ndim != 3:
            if image.ndim == 2:
                raise ValueError("Use ``multichannel=False`` for 2D grayscale "
                                 "images. The last axis of the input image "
                                 "must be multiple color channels not another "
                                 "spatial dimension.")
            else:
                raise ValueError(f'Bilateral filter is only implemented for '
                                 f'2D grayscale images (image.ndim == 2) and '
                                 f'2D multichannel (image.ndim == 3) images, '
                                 f'but the input image has {image.ndim} dimensions.')
        elif image.shape[2] not in (3, 4):
            if image.shape[2] > 4:
                msg = f'The last axis of the input image is ' \
                      f'interpreted as channels. Input image with '\
                      f'shape {image.shape} has {image.shape[2]} channels '\
                      f'in last axis. ``denoise_bilateral``is implemented ' \
                      f'for 2D grayscale and color images only.'
                warn(msg)
            else:
                msg = f'Input image must be grayscale, RGB, or RGBA; ' \
                      f'but has shape {image.shape}.'
                warn(msg)
    else:
        if image.ndim > 2:
            raise ValueError(f'Bilateral filter is not implemented for '
                             f'grayscale images of 3 or more dimensions, '
                             f'but input image has {image.shape} shape. Use '
                             f'``channel_axis=-1`` for 2D RGB images.')

    if win_size is None:
        win_size = max(5, 2 * int(ceil(3 * sigma_spatial)) + 1)

    min_value = image.min()
    max_value = image.max()

    if min_value == max_value:
        return image

    # if image.max() is 0, then dist_scale can have an unverified value
    # and color_lut[<int>(dist * dist_scale)] may cause a segmentation fault
    # so we verify we have a positive image and that the max is not 0.0.


    image = np.atleast_3d(img_as_float(image))
    image = np.ascontiguousarray(image)

    sigma_color = sigma_color or image.std()

    color_lut = _compute_color_lut(bins, sigma_color, max_value,
                                   dtype=image.dtype)

    range_lut = _compute_spatial_lut(win_size, sigma_spatial,
                                     dtype=image.dtype)

    out = np.empty(image.shape, dtype=image.dtype)

    dims = image.shape[2]

    # There are a number of arrays needed in the Cython function.
    # It's easier to allocate them outside of Cython so that all
    # arrays are in the same type, then just copy the empty array
    # where needed within Cython.
    empty_dims = np.empty(dims, dtype=image.dtype)

    if min_value < 0:
        image = image - min_value
        max_value -= min_value
    _denoise_bilateral(image, max_value, win_size, sigma_color, sigma_spatial,
                       bins, mode, cval, color_lut, range_lut, empty_dims, out)
    # need to drop the added channels axis for grayscale images
    out = np.squeeze(out)
    if min_value < 0:
        out += min_value
    return out


@utils.channel_as_last_axis()
@utils.deprecate_multichannel_kwarg()
@utils.deprecate_kwarg({'max_iter': 'max_num_iter'}, removed_version="1.0",
                       deprecated_version="0.19")
def denoise_tv_bregman(image, weight=5.0, max_num_iter=100, eps=1e-3,
                       isotropic=True, *, channel_axis=None,
                       multichannel=False):
    """Perform total-variation denoising using split-Bregman optimization.

    Total-variation denoising (also know as total-variation regularization)
    tries to find an image with less total-variation under the constraint
    of being similar to the input image, which is controlled by the
    regularization parameter ([1]_, [2]_, [3]_, [4]_).

    Parameters
    ----------
    image : ndarray
        Input data to be denoised (converted using img_as_float`).
    weight : float
        Denoising weight. The smaller the `weight`, the more denoising (at
        the expense of less similarity to the `input`). The regularization
        parameter `lambda` is chosen as `2 * weight`.
    eps : float, optional
        Relative difference of the value of the cost function that determines
        the stop criterion. The algorithm stops when::

            SUM((u(n) - u(n-1))**2) < eps

    max_num_iter : int, optional
        Maximal number of iterations used for the optimization.
    isotropic : boolean, optional
        Switch between isotropic and anisotropic TV denoising.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.
    multichannel : bool, optional
        Apply total-variation denoising separately for each channel. This
        option should be true for color images, otherwise the denoising is
        also applied in the channels dimension. This argument is deprecated:
        specify `channel_axis` instead.

    Returns
    -------
    u : ndarray
        Denoised image.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Total_variation_denoising
    .. [2] Tom Goldstein and Stanley Osher, "The Split Bregman Method For L1
           Regularized Problems",
           ftp://ftp.math.ucla.edu/pub/camreport/cam08-29.pdf
    .. [3] Pascal Getreuer, "Rudin–Osher–Fatemi Total Variation Denoising
           using Split Bregman" in Image Processing On Line on 2012–05–19,
           https://www.ipol.im/pub/art/2012/g-tvd/article_lr.pdf
    .. [4] https://web.math.ucsb.edu/~cgarcia/UGProjects/BregmanAlgorithms_JacquelineBush.pdf

    """
    image = np.atleast_3d(img_as_float(image))

    rows = image.shape[0]
    cols = image.shape[1]
    dims = image.shape[2]

    shape_ext = (rows + 2, cols + 2, dims)

    out = np.zeros(shape_ext, image.dtype)

    if channel_axis is not None:
        channel_out = np.zeros(shape_ext[:2] + (1,), dtype=out.dtype)
        for c in range(image.shape[-1]):
            # the algorithm below expects 3 dimensions to always be present.
            # slicing the array in this fashion preserves the channel dimension
            # for us
            channel_in = np.ascontiguousarray(image[..., c:c+1])

            _denoise_tv_bregman(channel_in, image.dtype.type(weight),
                                max_num_iter, eps, isotropic, channel_out)

            out[..., c] = channel_out[..., 0]

    else:
        image = np.ascontiguousarray(image)

        _denoise_tv_bregman(image, image.dtype.type(weight), max_num_iter, eps,
                            isotropic, out)

    return np.squeeze(out[1:-1, 1:-1])


def _denoise_tv_chambolle_nd(image, weight=0.1, eps=2.e-4, n_iter_max=200):
    """Perform total-variation denoising on n-dimensional images.

    Parameters
    ----------
    image : ndarray
        n-D input data to be denoised.
    weight : float, optional
        Denoising weight. The greater `weight`, the more denoising (at
        the expense of fidelity to `input`).
    eps : float, optional
        Relative difference of the value of the cost function that determines
        the stop criterion. The algorithm stops when:

            (E_(n-1) - E_n) < eps * E_0

    n_iter_max : int, optional
        Maximal number of iterations used for the optimization.

    Returns
    -------
    out : ndarray
        Denoised array of floats.

    Notes
    -----
    Rudin, Osher and Fatemi algorithm.
    """

    ndim = image.ndim
    p = np.zeros((image.ndim, ) + image.shape, dtype=image.dtype)
    g = np.zeros_like(p)
    d = np.zeros_like(image)
    i = 0
    while i < n_iter_max:
        if i > 0:
            # d will be the (negative) divergence of p
            d = -p.sum(0)
            slices_d = [slice(None), ] * ndim
            slices_p = [slice(None), ] * (ndim + 1)
            for ax in range(ndim):
                slices_d[ax] = slice(1, None)
                slices_p[ax+1] = slice(0, -1)
                slices_p[0] = ax
                d[tuple(slices_d)] += p[tuple(slices_p)]
                slices_d[ax] = slice(None)
                slices_p[ax+1] = slice(None)
            out = image + d
        else:
            out = image
        E = (d ** 2).sum()

        # g stores the gradients of out along each axis
        # e.g. g[0] is the first order finite difference along axis 0
        slices_g = [slice(None), ] * (ndim + 1)
        for ax in range(ndim):
            slices_g[ax+1] = slice(0, -1)
            slices_g[0] = ax
            g[tuple(slices_g)] = np.diff(out, axis=ax)
            slices_g[ax+1] = slice(None)

        norm = np.sqrt((g ** 2).sum(axis=0))[np.newaxis, ...]
        E += weight * norm.sum()
        tau = 1. / (2.*ndim)
        norm *= tau / weight
        norm += 1.
        p -= tau * g
        p /= norm
        E /= float(image.size)
        if i == 0:
            E_init = E
            E_previous = E
        else:
            if np.abs(E_previous - E) < eps * E_init:
                break
            else:
                E_previous = E
        i += 1
    return out


@utils.deprecate_multichannel_kwarg(multichannel_position=4)
def denoise_tv_chambolle(image, weight=0.1, eps=2.e-4, n_iter_max=200,
                         multichannel=False, *, channel_axis=None):
    """Perform total-variation denoising on n-dimensional images.

    Parameters
    ----------
    image : ndarray of ints, uints or floats
        Input data to be denoised. `image` can be of any numeric type,
        but it is cast into an ndarray of floats for the computation
        of the denoised image.
    weight : float, optional
        Denoising weight. The greater `weight`, the more denoising (at
        the expense of fidelity to `input`).
    eps : float, optional
        Relative difference of the value of the cost function that
        determines the stop criterion. The algorithm stops when:

            (E_(n-1) - E_n) < eps * E_0

    n_iter_max : int, optional
        Maximal number of iterations used for the optimization.
    multichannel : bool, optional
        Apply total-variation denoising separately for each channel. This
        option should be true for color images, otherwise the denoising is
        also applied in the channels dimension. This argument is deprecated:
        specify `channel_axis` instead.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    out : ndarray
        Denoised image.

    Notes
    -----
    Make sure to set the multichannel parameter appropriately for color images.

    The principle of total variation denoising is explained in
    https://en.wikipedia.org/wiki/Total_variation_denoising

    The principle of total variation denoising is to minimize the
    total variation of the image, which can be roughly described as
    the integral of the norm of the image gradient. Total variation
    denoising tends to produce "cartoon-like" images, that is,
    piecewise-constant images.

    This code is an implementation of the algorithm of Rudin, Fatemi and Osher
    that was proposed by Chambolle in [1]_.

    References
    ----------
    .. [1] A. Chambolle, An algorithm for total variation minimization and
           applications, Journal of Mathematical Imaging and Vision,
           Springer, 2004, 20, 89-97.

    Examples
    --------
    2D example on astronaut image:

    >>> from skimage import color, data
    >>> img = color.rgb2gray(data.astronaut())[:50, :50]
    >>> rng = np.random.default_rng()
    >>> img += 0.5 * img.std() * rng.standard_normal(img.shape)
    >>> denoised_img = denoise_tv_chambolle(img, weight=60)

    3D example on synthetic data:

    >>> x, y, z = np.ogrid[0:20, 0:20, 0:20]
    >>> mask = (x - 22)**2 + (y - 20)**2 + (z - 17)**2 < 8**2
    >>> mask = mask.astype(float)
    >>> rng = np.random.default_rng()
    >>> mask += 0.2 * rng.standard_normal(mask.shape)
    >>> res = denoise_tv_chambolle(mask, weight=100)

    """

    im_type = image.dtype
    if not im_type.kind == 'f':
        image = img_as_float(image)

    # enforce float16->float32 and float128->float64
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)

    if channel_axis is not None:
        channel_axis = channel_axis % image.ndim
        _at = functools.partial(utils.slice_at_axis, axis=channel_axis)
        out = np.zeros_like(image)
        for c in range(image.shape[channel_axis]):
            out[_at(c)] = _denoise_tv_chambolle_nd(image[_at(c)], weight, eps,
                                                   n_iter_max)
    else:
        out = _denoise_tv_chambolle_nd(image, weight, eps, n_iter_max)
    return out


def _bayes_thresh(details, var):
    """BayesShrink threshold for a zero-mean details coeff array."""
    # Equivalent to:  dvar = np.var(details) for 0-mean details array
    dvar = np.mean(details*details)
    eps = np.finfo(details.dtype).eps
    thresh = var / np.sqrt(max(dvar - var, eps))
    return thresh


def _universal_thresh(img, sigma):
    """ Universal threshold used by the VisuShrink method """
    return sigma*np.sqrt(2*np.log(img.size))


def _sigma_est_dwt(detail_coeffs, distribution='Gaussian'):
    """Calculate the robust median estimator of the noise standard deviation.

    Parameters
    ----------
    detail_coeffs : ndarray
        The detail coefficients corresponding to the discrete wavelet
        transform of an image.
    distribution : str
        The underlying noise distribution.

    Returns
    -------
    sigma : float
        The estimated noise standard deviation (see section 4.2 of [1]_).

    References
    ----------
    .. [1] D. L. Donoho and I. M. Johnstone. "Ideal spatial adaptation
       by wavelet shrinkage." Biometrika 81.3 (1994): 425-455.
       :DOI:`10.1093/biomet/81.3.425`
    """
    # Consider regions with detail coefficients exactly zero to be masked out
    detail_coeffs = detail_coeffs[np.nonzero(detail_coeffs)]

    if distribution.lower() == 'gaussian':
        # 75th quantile of the underlying, symmetric noise distribution
        denom = scipy.stats.norm.ppf(0.75)
        sigma = np.median(np.abs(detail_coeffs)) / denom
    else:
        raise ValueError("Only Gaussian noise estimation is currently "
                         "supported")
    return sigma


def _wavelet_threshold(image, wavelet, method=None, threshold=None,
                       sigma=None, mode='soft', wavelet_levels=None):
    """Perform wavelet thresholding.

    Parameters
    ----------
    image : ndarray (2d or 3d) of ints, uints or floats
        Input data to be denoised. `image` can be of any numeric type,
        but it is cast into an ndarray of floats for the computation
        of the denoised image.
    wavelet : string
        The type of wavelet to perform. Can be any of the options
        pywt.wavelist outputs. For example, this may be any of ``{db1, db2,
        db3, db4, haar}``.
    method : {'BayesShrink', 'VisuShrink'}, optional
        Thresholding method to be used. The currently supported methods are
        "BayesShrink" [1]_ and "VisuShrink" [2]_. If it is set to None, a
        user-specified ``threshold`` must be supplied instead.
    threshold : float, optional
        The thresholding value to apply during wavelet coefficient
        thresholding. The default value (None) uses the selected ``method`` to
        estimate appropriate threshold(s) for noise removal.
    sigma : float, optional
        The standard deviation of the noise. The noise is estimated when sigma
        is None (the default) by the method in [2]_.
    mode : {'soft', 'hard'}, optional
        An optional argument to choose the type of denoising performed. It
        noted that choosing soft thresholding given additive noise finds the
        best approximation of the original image.
    wavelet_levels : int or None, optional
        The number of wavelet decomposition levels to use.  The default is
        three less than the maximum number of possible decomposition levels
        (see Notes below).

    Returns
    -------
    out : ndarray
        Denoised image.

    References
    ----------
    .. [1] Chang, S. Grace, Bin Yu, and Martin Vetterli. "Adaptive wavelet
           thresholding for image denoising and compression." Image Processing,
           IEEE Transactions on 9.9 (2000): 1532-1546.
           :DOI:`10.1109/83.862633`
    .. [2] D. L. Donoho and I. M. Johnstone. "Ideal spatial adaptation
           by wavelet shrinkage." Biometrika 81.3 (1994): 425-455.
           :DOI:`10.1093/biomet/81.3.425`
    """
    wavelet = pywt.Wavelet(wavelet)
    if not wavelet.orthogonal:
        warn(f'Wavelet thresholding was designed for '
             f'use with orthogonal wavelets. For nonorthogonal '
             f'wavelets such as {wavelet.name},results are '
             f'likely to be suboptimal.')

    # original_extent is used to workaround PyWavelets issue #80
    # odd-sized input results in an image with 1 extra sample after waverecn
    original_extent = tuple(slice(s) for s in image.shape)

    # Determine the number of wavelet decomposition levels
    if wavelet_levels is None:
        # Determine the maximum number of possible levels for image
        wavelet_levels = pywt.dwtn_max_level(image.shape, wavelet)

        # Skip coarsest wavelet scales (see Notes in docstring).
        wavelet_levels = max(wavelet_levels - 3, 1)

    coeffs = pywt.wavedecn(image, wavelet=wavelet, level=wavelet_levels)
    # Detail coefficients at each decomposition level
    dcoeffs = coeffs[1:]

    if sigma is None:
        # Estimate the noise via the method in [2]_
        detail_coeffs = dcoeffs[-1]['d' * image.ndim]
        sigma = _sigma_est_dwt(detail_coeffs, distribution='Gaussian')

    if method is not None and threshold is not None:
        warn(f'Thresholding method {method} selected. The '
             f'user-specified threshold will be ignored.')

    if threshold is None:
        var = sigma**2
        if method is None:
            raise ValueError(
                "If method is None, a threshold must be provided.")
        elif method == "BayesShrink":
            # The BayesShrink thresholds from [1]_ in docstring
            threshold = [{key: _bayes_thresh(level[key], var) for key in level}
                         for level in dcoeffs]
        elif method == "VisuShrink":
            # The VisuShrink thresholds from [2]_ in docstring
            threshold = _universal_thresh(image, sigma)
        else:
            raise ValueError(f'Unrecognized method: {method}')

    if np.isscalar(threshold):
        # A single threshold for all coefficient arrays
        denoised_detail = [{key: pywt.threshold(level[key],
                                                value=threshold,
                                                mode=mode) for key in level}
                           for level in dcoeffs]
    else:
        # Dict of unique threshold coefficients for each detail coeff. array
        denoised_detail = [{key: pywt.threshold(level[key],
                                                value=thresh[key],
                                                mode=mode) for key in level}
                           for thresh, level in zip(threshold, dcoeffs)]
    denoised_coeffs = [coeffs[0]] + denoised_detail
    return pywt.waverecn(denoised_coeffs, wavelet)[original_extent]


def _scale_sigma_and_image_consistently(image, sigma, multichannel,
                                        rescale_sigma):
    """If the ``image`` is rescaled, also rescale ``sigma`` consistently.

    Images that are not floating point will be rescaled via ``img_as_float``.
    Half-precision images will be promoted to single precision.
    """
    if multichannel:
        if isinstance(sigma, numbers.Number) or sigma is None:
            sigma = [sigma] * image.shape[-1]
        elif len(sigma) != image.shape[-1]:
            raise ValueError(
                "When multichannel is True, sigma must be a scalar or have "
                "length equal to the number of channels")
    if image.dtype.kind != 'f':
        if rescale_sigma:
            range_pre = image.max() - image.min()
        image = img_as_float(image)
        if rescale_sigma:
            range_post = image.max() - image.min()
            # apply the same magnitude scaling to sigma
            scale_factor = range_post / range_pre
            if multichannel:
                sigma = [s * scale_factor if s is not None else s
                         for s in sigma]
            elif sigma is not None:
                sigma *= scale_factor
    elif image.dtype == np.float16:
        image = image.astype(np.float32)
    return image, sigma


def _rescale_sigma_rgb2ycbcr(sigmas):
    """Convert user-provided noise standard deviations to YCbCr space.

    Notes
    -----
    If R, G, B are linearly independent random variables and a1, a2, a3 are
    scalars, then random variable C:
        C = a1 * R + a2 * G + a3 * B
    has variance, var_C, given by:
        var_C = a1**2 * var_R + a2**2 * var_G + a3**2 * var_B
    """
    if sigmas[0] is None:
        return sigmas
    sigmas = np.asarray(sigmas)
    rgv_variances = sigmas * sigmas
    for i in range(3):
        scalars = ycbcr_from_rgb[i, :]
        var_channel = np.sum(scalars * scalars * rgv_variances)
        sigmas[i] = np.sqrt(var_channel)
    return sigmas


@utils.channel_as_last_axis()
@utils.deprecate_multichannel_kwarg(multichannel_position=5)
def denoise_wavelet(image, sigma=None, wavelet='db1', mode='soft',
                    wavelet_levels=None, multichannel=False,
                    convert2ycbcr=False, method='BayesShrink',
                    rescale_sigma=True, *, channel_axis=None):
    """Perform wavelet denoising on an image.

    Parameters
    ----------
    image : ndarray ([M[, N[, ...P]][, C]) of ints, uints or floats
        Input data to be denoised. `image` can be of any numeric type,
        but it is cast into an ndarray of floats for the computation
        of the denoised image.
    sigma : float or list, optional
        The noise standard deviation used when computing the wavelet detail
        coefficient threshold(s). When None (default), the noise standard
        deviation is estimated via the method in [2]_.
    wavelet : string, optional
        The type of wavelet to perform and can be any of the options
        ``pywt.wavelist`` outputs. The default is `'db1'`. For example,
        ``wavelet`` can be any of ``{'db2', 'haar', 'sym9'}`` and many more.
    mode : {'soft', 'hard'}, optional
        An optional argument to choose the type of denoising performed. It
        noted that choosing soft thresholding given additive noise finds the
        best approximation of the original image.
    wavelet_levels : int or None, optional
        The number of wavelet decomposition levels to use.  The default is
        three less than the maximum number of possible decomposition levels.
    multichannel : bool, optional
        Apply wavelet denoising separately for each channel (where channels
        correspond to the final axis of the array). This argument is
        deprecated: specify `channel_axis` instead.
    convert2ycbcr : bool, optional
        If True and multichannel True, do the wavelet denoising in the YCbCr
        colorspace instead of the RGB color space. This typically results in
        better performance for RGB images.
    method : {'BayesShrink', 'VisuShrink'}, optional
        Thresholding method to be used. The currently supported methods are
        "BayesShrink" [1]_ and "VisuShrink" [2]_. Defaults to "BayesShrink".
    rescale_sigma : bool, optional
        If False, no rescaling of the user-provided ``sigma`` will be
        performed. The default of ``True`` rescales sigma appropriately if the
        image is rescaled internally.

        .. versionadded:: 0.16
           ``rescale_sigma`` was introduced in 0.16
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    out : ndarray
        Denoised image.

    Notes
    -----
    The wavelet domain is a sparse representation of the image, and can be
    thought of similarly to the frequency domain of the Fourier transform.
    Sparse representations have most values zero or near-zero and truly random
    noise is (usually) represented by many small values in the wavelet domain.
    Setting all values below some threshold to 0 reduces the noise in the
    image, but larger thresholds also decrease the detail present in the image.

    If the input is 3D, this function performs wavelet denoising on each color
    plane separately.

    .. versionchanged:: 0.16
       For floating point inputs, the original input range is maintained and
       there is no clipping applied to the output. Other input types will be
       converted to a floating point value in the range [-1, 1] or [0, 1]
       depending on the input image range. Unless ``rescale_sigma = False``,
       any internal rescaling applied to the ``image`` will also be applied
       to ``sigma`` to maintain the same relative amplitude.

    Many wavelet coefficient thresholding approaches have been proposed. By
    default, ``denoise_wavelet`` applies BayesShrink, which is an adaptive
    thresholding method that computes separate thresholds for each wavelet
    sub-band as described in [1]_.

    If ``method == "VisuShrink"``, a single "universal threshold" is applied to
    all wavelet detail coefficients as described in [2]_. This threshold
    is designed to remove all Gaussian noise at a given ``sigma`` with high
    probability, but tends to produce images that appear overly smooth.

    Although any of the wavelets from ``PyWavelets`` can be selected, the
    thresholding methods assume an orthogonal wavelet transform and may not
    choose the threshold appropriately for biorthogonal wavelets. Orthogonal
    wavelets are desirable because white noise in the input remains white noise
    in the subbands. Biorthogonal wavelets lead to colored noise in the
    subbands. Additionally, the orthogonal wavelets in PyWavelets are
    orthonormal so that noise variance in the subbands remains identical to the
    noise variance of the input. Example orthogonal wavelets are the Daubechies
    (e.g. 'db2') or symmlet (e.g. 'sym2') families.

    References
    ----------
    .. [1] Chang, S. Grace, Bin Yu, and Martin Vetterli. "Adaptive wavelet
           thresholding for image denoising and compression." Image Processing,
           IEEE Transactions on 9.9 (2000): 1532-1546.
           :DOI:`10.1109/83.862633`
    .. [2] D. L. Donoho and I. M. Johnstone. "Ideal spatial adaptation
           by wavelet shrinkage." Biometrika 81.3 (1994): 425-455.
           :DOI:`10.1093/biomet/81.3.425`

    Examples
    --------
    >>> from skimage import color, data
    >>> img = img_as_float(data.astronaut())
    >>> img = color.rgb2gray(img)
    >>> rng = np.random.default_rng()
    >>> img += 0.1 * rng.standard_normal(img.shape)
    >>> img = np.clip(img, 0, 1)
    >>> denoised_img = denoise_wavelet(img, sigma=0.1, rescale_sigma=True)

    """
    multichannel = channel_axis is not None
    if method not in ["BayesShrink", "VisuShrink"]:
        raise ValueError(f'Invalid method: {method}. The currently supported '
                         f'methods are "BayesShrink" and "VisuShrink".')

    # floating-point inputs are not rescaled, so don't clip their output.
    clip_output = image.dtype.kind != 'f'

    if convert2ycbcr and not multichannel:
        raise ValueError("convert2ycbcr requires multichannel == True")

    image, sigma = _scale_sigma_and_image_consistently(image,
                                                       sigma,
                                                       multichannel,
                                                       rescale_sigma)
    if multichannel:
        if convert2ycbcr:
            out = color.rgb2ycbcr(image)
            # convert user-supplied sigmas to the new colorspace as well
            if rescale_sigma:
                sigma = _rescale_sigma_rgb2ycbcr(sigma)
            for i in range(3):
                # renormalizing this color channel to live in [0, 1]
                _min, _max = out[..., i].min(), out[..., i].max()
                scale_factor = _max - _min
                if scale_factor == 0:
                    # skip any channel containing only zeros!
                    continue
                channel = out[..., i] - _min
                channel /= scale_factor
                sigma_channel = sigma[i]
                if sigma_channel is not None:
                    sigma_channel /= scale_factor
                out[..., i] = denoise_wavelet(channel,
                                              wavelet=wavelet,
                                              method=method,
                                              sigma=sigma_channel,
                                              mode=mode,
                                              wavelet_levels=wavelet_levels,
                                              rescale_sigma=rescale_sigma)
                out[..., i] = out[..., i] * scale_factor
                out[..., i] += _min
            out = color.ycbcr2rgb(out)
        else:
            out = np.empty_like(image)
            for c in range(image.shape[-1]):
                out[..., c] = _wavelet_threshold(image[..., c],
                                                 wavelet=wavelet,
                                                 method=method,
                                                 sigma=sigma[c], mode=mode,
                                                 wavelet_levels=wavelet_levels)
    else:
        out = _wavelet_threshold(image, wavelet=wavelet, method=method,
                                 sigma=sigma, mode=mode,
                                 wavelet_levels=wavelet_levels)

    if clip_output:
        clip_range = (-1, 1) if image.min() < 0 else (0, 1)
        out = np.clip(out, *clip_range, out=out)
    return out


@utils.deprecate_multichannel_kwarg(multichannel_position=2)
def estimate_sigma(image, average_sigmas=False, multichannel=False, *,
                   channel_axis=None):
    """
    Robust wavelet-based estimator of the (Gaussian) noise standard deviation.

    Parameters
    ----------
    image : ndarray
        Image for which to estimate the noise standard deviation.
    average_sigmas : bool, optional
        If true, average the channel estimates of `sigma`.  Otherwise return
        a list of sigmas corresponding to each channel.
    multichannel : bool
        Estimate sigma separately for each channel. This argument is
        deprecated: specify `channel_axis` instead.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    sigma : float or list
        Estimated noise standard deviation(s).  If `multichannel` is True and
        `average_sigmas` is False, a separate noise estimate for each channel
        is returned.  Otherwise, the average of the individual channel
        estimates is returned.

    Notes
    -----
    This function assumes the noise follows a Gaussian distribution. The
    estimation algorithm is based on the median absolute deviation of the
    wavelet detail coefficients as described in section 4.2 of [1]_.

    References
    ----------
    .. [1] D. L. Donoho and I. M. Johnstone. "Ideal spatial adaptation
       by wavelet shrinkage." Biometrika 81.3 (1994): 425-455.
       :DOI:`10.1093/biomet/81.3.425`

    Examples
    --------
    >>> import skimage.data
    >>> from skimage import img_as_float
    >>> img = img_as_float(skimage.data.camera())
    >>> sigma = 0.1
    >>> rng = np.random.default_rng()
    >>> img = img + sigma * rng.standard_normal(img.shape)
    >>> sigma_hat = estimate_sigma(img, channel_axis=None)
    """
    if channel_axis is not None:
        channel_axis = channel_axis % image.ndim
        _at = functools.partial(utils.slice_at_axis, axis=channel_axis)
        nchannels = image.shape[channel_axis]
        sigmas = [estimate_sigma(
            image[_at(c)], channel_axis=None) for c in range(nchannels)]
        if average_sigmas:
            sigmas = np.mean(sigmas)
        return sigmas
    elif image.shape[-1] <= 4:
        msg = f'image is size {image.shape[-1]} on the last axis, '\
              f'but channel_axis is None. If this is a color image, '\
              f'please set channel_axis=-1 for proper noise estimation.'
        warn(msg)
    coeffs = pywt.dwtn(image, wavelet='db2')
    detail_coeffs = coeffs['d' * image.ndim]
    return _sigma_est_dwt(detail_coeffs, distribution='Gaussian')
