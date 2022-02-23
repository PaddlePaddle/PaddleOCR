"""Implementations restoration functions"""
import warnings

import numpy as np
from scipy.signal import convolve

from .._shared.utils import _supported_float_type, deprecate_kwarg
from . import uft


def wiener(image, psf, balance, reg=None, is_real=True, clip=True):
    r"""Wiener-Hunt deconvolution

    Return the deconvolution with a Wiener-Hunt approach (i.e. with
    Fourier diagonalisation).

    Parameters
    ----------
    image : (M, N) ndarray
       Input degraded image
    psf : ndarray
       Point Spread Function. This is assumed to be the impulse
       response (input image space) if the data-type is real, or the
       transfer function (Fourier space) if the data-type is
       complex. There is no constraints on the shape of the impulse
       response. The transfer function must be of shape `(M, N)` if
       `is_real is True`, `(M, N // 2 + 1)` otherwise (see
       `np.fft.rfftn`).
    balance : float
       The regularisation parameter value that tunes the balance
       between the data adequacy that improve frequency restoration
       and the prior adequacy that reduce frequency restoration (to
       avoid noise artifacts).
    reg : ndarray, optional
       The regularisation operator. The Laplacian by default. It can
       be an impulse response or a transfer function, as for the
       psf. Shape constraint is the same as for the `psf` parameter.
    is_real : boolean, optional
       True by default. Specify if ``psf`` and ``reg`` are provided
       with hermitian hypothesis, that is only half of the frequency
       plane is provided (due to the redundancy of Fourier transform
       of real signal). It's apply only if ``psf`` and/or ``reg`` are
       provided as transfer function.  For the hermitian property see
       ``uft`` module or ``np.fft.rfftn``.
    clip : boolean, optional
       True by default. If True, pixel values of the result above 1 or
       under -1 are thresholded for skimage pipeline compatibility.

    Returns
    -------
    im_deconv : (M, N) ndarray
       The deconvolved image.

    Examples
    --------
    >>> from skimage import color, data, restoration
    >>> img = color.rgb2gray(data.astronaut())
    >>> from scipy.signal import convolve2d
    >>> psf = np.ones((5, 5)) / 25
    >>> img = convolve2d(img, psf, 'same')
    >>> rng = np.random.default_rng()
    >>> img += 0.1 * img.std() * rng.standard_normal(img.shape)
    >>> deconvolved_img = restoration.wiener(img, psf, 1100)

    Notes
    -----
    This function applies the Wiener filter to a noisy and degraded
    image by an impulse response (or PSF). If the data model is

    .. math:: y = Hx + n

    where :math:`n` is noise, :math:`H` the PSF and :math:`x` the
    unknown original image, the Wiener filter is

    .. math::
       \hat x = F^\dagger (|\Lambda_H|^2 + \lambda |\Lambda_D|^2)
       \Lambda_H^\dagger F y

    where :math:`F` and :math:`F^\dagger` are the Fourier and inverse
    Fourier transforms respectively, :math:`\Lambda_H` the transfer
    function (or the Fourier transform of the PSF, see [Hunt] below)
    and :math:`\Lambda_D` the filter to penalize the restored image
    frequencies (Laplacian by default, that is penalization of high
    frequency). The parameter :math:`\lambda` tunes the balance
    between the data (that tends to increase high frequency, even
    those coming from noise), and the regularization.

    These methods are then specific to a prior model. Consequently,
    the application or the true image nature must corresponds to the
    prior model. By default, the prior model (Laplacian) introduce
    image smoothness or pixel correlation. It can also be interpreted
    as high-frequency penalization to compensate the instability of
    the solution with respect to the data (sometimes called noise
    amplification or "explosive" solution).

    Finally, the use of Fourier space implies a circulant property of
    :math:`H`, see [Hunt].

    References
    ----------
    .. [1] François Orieux, Jean-François Giovannelli, and Thomas
           Rodet, "Bayesian estimation of regularization and point
           spread function parameters for Wiener-Hunt deconvolution",
           J. Opt. Soc. Am. A 27, 1593-1607 (2010)

           https://www.osapublishing.org/josaa/abstract.cfm?URI=josaa-27-7-1593

           http://research.orieux.fr/files/papers/OGR-JOSA10.pdf

    .. [2] B. R. Hunt "A matrix theory proof of the discrete
           convolution theorem", IEEE Trans. on Audio and
           Electroacoustics, vol. au-19, no. 4, pp. 285-288, dec. 1971
    """
    if reg is None:
        reg, _ = uft.laplacian(image.ndim, image.shape, is_real=is_real)
    if not np.iscomplexobj(reg):
        reg = uft.ir2tf(reg, image.shape, is_real=is_real)
    float_type = _supported_float_type(image.dtype)
    image = image.astype(float_type, copy=False)
    psf = psf.real.astype(float_type, copy=False)
    reg = reg.real.astype(float_type, copy=False)

    if psf.shape != reg.shape:
        trans_func = uft.ir2tf(psf, image.shape, is_real=is_real)
    else:
        trans_func = psf

    wiener_filter = np.conj(trans_func) / (np.abs(trans_func) ** 2 +
                                           balance * np.abs(reg) ** 2)
    if is_real:
        deconv = uft.uirfft2(wiener_filter * uft.urfft2(image),
                             shape=image.shape)
    else:
        deconv = uft.uifft2(wiener_filter * uft.ufft2(image))

    if clip:
        deconv[deconv > 1] = 1
        deconv[deconv < -1] = -1

    return deconv


def unsupervised_wiener(image, psf, reg=None, user_params=None, is_real=True,
                        clip=True, *, random_state=None):
    """Unsupervised Wiener-Hunt deconvolution.

    Return the deconvolution with a Wiener-Hunt approach, where the
    hyperparameters are automatically estimated. The algorithm is a
    stochastic iterative process (Gibbs sampler) described in the
    reference below. See also ``wiener`` function.

    Parameters
    ----------
    image : (M, N) ndarray
       The input degraded image.
    psf : ndarray
       The impulse response (input image's space) or the transfer
       function (Fourier space). Both are accepted. The transfer
       function is automatically recognized as being complex
       (``np.iscomplexobj(psf)``).
    reg : ndarray, optional
       The regularisation operator. The Laplacian by default. It can
       be an impulse response or a transfer function, as for the psf.
    user_params : dict, optional
       Dictionary of parameters for the Gibbs sampler. See below.
    clip : boolean, optional
       True by default. If true, pixel values of the result above 1 or
       under -1 are thresholded for skimage pipeline compatibility.
    random_state : {None, int, `numpy.random.Generator`}, optional
        If `random_state` is None the `numpy.random.Generator` singleton is
        used.
        If `random_state` is an int, a new ``Generator`` instance is used,
        seeded with `random_state`.
        If `random_state` is already a ``Generator`` instance then that
        instance is used.

        .. versionadded:: 0.19

    Returns
    -------
    x_postmean : (M, N) ndarray
       The deconvolved image (the posterior mean).
    chains : dict
       The keys ``noise`` and ``prior`` contain the chain list of
       noise and prior precision respectively.

    Other parameters
    ----------------
    The keys of ``user_params`` are:

    threshold : float
       The stopping criterion: the norm of the difference between to
       successive approximated solution (empirical mean of object
       samples, see Notes section). 1e-4 by default.
    burnin : int
       The number of sample to ignore to start computation of the
       mean. 15 by default.
    min_num_iter : int
       The minimum number of iterations. 30 by default.
    max_num_iter : int
       The maximum number of iterations if ``threshold`` is not
       satisfied. 200 by default.
    callback : callable (None by default)
       A user provided callable to which is passed, if the function
       exists, the current image sample for whatever purpose. The user
       can store the sample, or compute other moments than the
       mean. It has no influence on the algorithm execution and is
       only for inspection.

    Examples
    --------
    >>> from skimage import color, data, restoration
    >>> img = color.rgb2gray(data.astronaut())
    >>> from scipy.signal import convolve2d
    >>> psf = np.ones((5, 5)) / 25
    >>> img = convolve2d(img, psf, 'same')
    >>> rng = np.random.default_rng()
    >>> img += 0.1 * img.std() * rng.standard_normal(img.shape)
    >>> deconvolved_img = restoration.unsupervised_wiener(img, psf)

    Notes
    -----
    The estimated image is design as the posterior mean of a
    probability law (from a Bayesian analysis). The mean is defined as
    a sum over all the possible images weighted by their respective
    probability. Given the size of the problem, the exact sum is not
    tractable. This algorithm use of MCMC to draw image under the
    posterior law. The practical idea is to only draw highly probable
    images since they have the biggest contribution to the mean. At the
    opposite, the less probable images are drawn less often since
    their contribution is low. Finally the empirical mean of these
    samples give us an estimation of the mean, and an exact
    computation with an infinite sample set.

    References
    ----------
    .. [1] François Orieux, Jean-François Giovannelli, and Thomas
           Rodet, "Bayesian estimation of regularization and point
           spread function parameters for Wiener-Hunt deconvolution",
           J. Opt. Soc. Am. A 27, 1593-1607 (2010)

           https://www.osapublishing.org/josaa/abstract.cfm?URI=josaa-27-7-1593

           http://research.orieux.fr/files/papers/OGR-JOSA10.pdf
    """

    if user_params is not None:
        for s in ('max', 'min'):
            if (s + '_iter') in user_params:
                warning_msg = (
                    f'`{s}_iter` is a deprecated key for `user_params`. '
                    f'It will be removed in version 1.0. '
                    f'Use `{s}_num_iter` instead.'
                )
                warnings.warn(warning_msg, FutureWarning)
                user_params[s + '_num_iter'] = user_params.pop(s + '_iter')

    params = {'threshold': 1e-4, 'max_num_iter': 200,
              'min_num_iter': 30, 'burnin': 15, 'callback': None}
    params.update(user_params or {})

    if reg is None:
        reg, _ = uft.laplacian(image.ndim, image.shape, is_real=is_real)
    if not np.iscomplexobj(reg):
        reg = uft.ir2tf(reg, image.shape, is_real=is_real)
    float_type = _supported_float_type(image.dtype)
    image = image.astype(float_type, copy=False)
    psf = psf.real.astype(float_type, copy=False)
    reg = reg.real.astype(float_type, copy=False)

    if psf.shape != reg.shape:
        trans_fct = uft.ir2tf(psf, image.shape, is_real=is_real)
    else:
        trans_fct = psf

    # The mean of the object
    x_postmean = np.zeros(trans_fct.shape, dtype=float_type)
    # The previous computed mean in the iterative loop
    prev_x_postmean = np.zeros(trans_fct.shape, dtype=float_type)

    # Difference between two successive mean
    delta = np.NAN

    # Initial state of the chain
    gn_chain, gx_chain = [1], [1]

    # The correlation of the object in Fourier space (if size is big,
    # this can reduce computation time in the loop)
    areg2 = np.abs(reg) ** 2
    atf2 = np.abs(trans_fct) ** 2

    # The Fourier transform may change the image.size attribute, so we
    # store it.
    if is_real:
        data_spectrum = uft.urfft2(image)
    else:
        data_spectrum = uft.ufft2(image)

    rng = np.random.default_rng(random_state)

    # Gibbs sampling
    for iteration in range(params['max_num_iter']):
        # Sample of Eq. 27 p(circX^k | gn^k-1, gx^k-1, y).

        # weighting (correlation in direct space)
        precision = gn_chain[-1] * atf2 + gx_chain[-1] * areg2  # Eq. 29
        # Note: Use astype instead of dtype argument to standard_normal to get
        #       similar random values across precisions, as needed for
        #       reference data used by test_unsupervised_wiener.
        _rand1 = rng.standard_normal(data_spectrum.shape)
        _rand1 = _rand1.astype(float_type, copy=False)
        _rand2 = rng.standard_normal(data_spectrum.shape)
        _rand2 = _rand2.astype(float_type, copy=False)
        excursion = np.sqrt(0.5 / precision) * (_rand1 + 1j * _rand2)

        # mean Eq. 30 (RLS for fixed gn, gamma0 and gamma1 ...)
        wiener_filter = gn_chain[-1] * np.conj(trans_fct) / precision

        # sample of X in Fourier space
        x_sample = wiener_filter * data_spectrum + excursion
        if params['callback']:
            params['callback'](x_sample)

        # sample of Eq. 31 p(gn | x^k, gx^k, y)
        gn_chain.append(rng.gamma(image.size / 2,
                                  2 / uft.image_quad_norm(data_spectrum
                                                          - x_sample
                                                          * trans_fct)))

        # sample of Eq. 31 p(gx | x^k, gn^k-1, y)
        gx_chain.append(rng.gamma((image.size - 1) / 2,
                                  2 / uft.image_quad_norm(x_sample * reg)))

        # current empirical average
        if iteration > params['burnin']:
            x_postmean = prev_x_postmean + x_sample

        if iteration > (params['burnin'] + 1):
            current = x_postmean / (iteration - params['burnin'])
            previous = prev_x_postmean / (iteration - params['burnin'] - 1)

            delta = (np.sum(np.abs(current - previous))
                     / np.sum(np.abs(x_postmean))
                     / (iteration - params['burnin']))

        prev_x_postmean = x_postmean

        # stop of the algorithm
        if (
            (iteration > params['min_num_iter'])
            and (delta < params['threshold'])
        ):
            break

    # Empirical average \approx POSTMEAN Eq. 44
    x_postmean = x_postmean / (iteration - params['burnin'])
    if is_real:
        x_postmean = uft.uirfft2(x_postmean, shape=image.shape)
    else:
        x_postmean = uft.uifft2(x_postmean)

    if clip:
        x_postmean[x_postmean > 1] = 1
        x_postmean[x_postmean < -1] = -1

    return (x_postmean, {'noise': gn_chain, 'prior': gx_chain})


@deprecate_kwarg({'iterations': 'num_iter'}, removed_version="1.0",
                 deprecated_version="0.19")
def richardson_lucy(image, psf, num_iter=50, clip=True, filter_epsilon=None):
    """Richardson-Lucy deconvolution.

    Parameters
    ----------
    image : ndarray
       Input degraded image (can be N dimensional).
    psf : ndarray
       The point spread function.
    num_iter : int, optional
       Number of iterations. This parameter plays the role of
       regularisation.
    clip : boolean, optional
       True by default. If true, pixel value of the result above 1 or
       under -1 are thresholded for skimage pipeline compatibility.
    filter_epsilon: float, optional
       Value below which intermediate results become 0 to avoid division
       by small numbers.

    Returns
    -------
    im_deconv : ndarray
       The deconvolved image.

    Examples
    --------
    >>> from skimage import img_as_float, data, restoration
    >>> camera = img_as_float(data.camera())
    >>> from scipy.signal import convolve2d
    >>> psf = np.ones((5, 5)) / 25
    >>> camera = convolve2d(camera, psf, 'same')
    >>> rng = np.random.default_rng()
    >>> camera += 0.1 * camera.std() * rng.standard_normal(camera.shape)
    >>> deconvolved = restoration.richardson_lucy(camera, psf, 5)

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
    """
    float_type = _supported_float_type(image.dtype)
    image = image.astype(float_type, copy=False)
    psf = psf.astype(float_type, copy=False)
    im_deconv = np.full(image.shape, 0.5, dtype=float_type)
    psf_mirror = np.flip(psf)

    # Small regularization parameter used to avoid 0 divisions
    eps = 1e-12

    for _ in range(num_iter):
        conv = convolve(im_deconv, psf, mode='same') + eps
        if filter_epsilon:
            relative_blur = np.where(conv < filter_epsilon, 0, image / conv)
        else:
            relative_blur = image / conv
        im_deconv *= convolve(relative_blur, psf_mirror, mode='same')

    if clip:
        im_deconv[im_deconv > 1] = 1
        im_deconv[im_deconv < -1] = -1

    return im_deconv
