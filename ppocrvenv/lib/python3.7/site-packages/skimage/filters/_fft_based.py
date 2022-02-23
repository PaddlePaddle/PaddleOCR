import functools

import numpy as np
import scipy.fft as fft

from .._shared.utils import _supported_float_type


def _get_ND_butterworth_filter(shape, factor, order, high_pass, real,
                               dtype=np.float64):
    """Create a N-dimensional Butterworth mask for an FFT

    Parameters
    ----------
    shape : tuple of int
        Shape of the n-dimensional FFT and mask.
    factor : float
        Fraction of mask dimensions where the cutoff should be.
    order : float
        Controls the slope in the cutoff region.
    high_pass : bool
        Whether the filter is high pass (low frequencies attenuated) or
        low pass (high frequencies are attenuated).
    real : bool
        Whether the FFT is of a real (True) or complex (False) image

    Returns
    -------
    wfilt : ndarray
        The FFT mask.

    """
    ranges = []
    for i, d in enumerate(shape):
        # start and stop ensures center of mask aligns with center of FFT
        axis = np.arange(-(d - 1) // 2, (d - 1) // 2 + 1) / (d * factor)
        ranges.append(fft.ifftshift(axis ** 2))
    # for real image FFT, halve the last axis
    if real:
        limit = d // 2 + 1
        ranges[-1] = ranges[-1][:limit]
    # q2 = squared Euclidian distance grid
    q2 = functools.reduce(
            np.add, np.meshgrid(*ranges, indexing="ij", sparse=True)
            )
    q2 = q2.astype(dtype)
    wfilt = 1 / (1 + np.power(q2, order))
    if high_pass:
        wfilt = 1 - wfilt
    return wfilt


def butterworth(
    image,
    cutoff_frequency_ratio=0.005,
    high_pass=True,
    order=2.0,
    channel_axis=None,
):
    """Apply a Butterworth filter to enhance high or low frequency features.

    This filter is defined in the Fourier domain.

    Parameters
    ----------
    image : (M[, N[, ..., P]][, C]) ndarray
        Input image.
    cutoff_frequency_ratio : float, optional
        Determines the position of the cut-off relative to the shape of the
        FFT.
    high_pass : bool, optional
        Whether to perform a high pass filter. If False, a low pass filter is
        performed.
    order : float, optional
        Order of the filter which affects the slope near the cut-off. Higher
        order means steeper slope in frequency space.
    channel_axis : int, optional
        If there is a channel dimension, provide the index here. If None
        (default) then all axes are assumed to be spatial dimensions.

    Returns
    -------
    result : ndarray
        The Butterworth-filtered image.

    Notes
    -----
    A band-pass filter can be achieved by combining a high pass and low
    pass filter.

    The literature contains multiple conventions for the functional form of
    the Butterworth filter. Here it is implemented as the n-dimensional form of

    .. math::
        \\frac{1}{1 - \\left(\\frac{f}{c*f_{max}}\\right)^{2*n}}

    with :math:`f` the absolute value of the spatial frequency, :math:`c` the
    ``cutoff_frequency_ratio`` and :math:`n` the ``order`` modeled after [2]_

    Examples
    --------
    Apply a high pass and low pass Butterworth filter to a grayscale and
    color image respectively:

    >>> from skimage.data import camera, astronaut
    >>> from skimage.filters import butterworth
    >>> high_pass = butterworth(camera(), 0.07, True, 8)
    >>> low_pass = butterworth(astronaut(), 0.01, False, 4, channel_axis=-1)

    References
    ----------
    .. [1] Butterworth, Stephen. "On the theory of filter amplifiers."
           Wireless Engineer 7.6 (1930): 536-541.
    .. [2] Russ, John C., et al. "The image processing handbook."
           Computers in Physics 8.2 (1994): 177-178.

    """
    fft_shape = (image.shape if channel_axis is None
                 else np.delete(image.shape, channel_axis))
    is_real = np.isrealobj(image)
    float_dtype = _supported_float_type(image.dtype, allow_complex=True)
    wfilt = _get_ND_butterworth_filter(
        fft_shape, cutoff_frequency_ratio, order, high_pass, is_real,
        float_dtype
    )
    axes = np.arange(image.ndim)
    if channel_axis is not None:
        axes = np.delete(axes, channel_axis)
        abs_channel = channel_axis % image.ndim
        post = image.ndim - abs_channel - 1
        sl = ((slice(None),) * abs_channel + (np.newaxis,) +
              (slice(None),) * post)
        wfilt = wfilt[sl]
    if is_real:
        butterfilt = fft.irfftn(wfilt * fft.rfftn(image, axes=axes),
                                s=fft_shape, axes=axes)
    else:
        butterfilt = fft.ifftn(wfilt * fft.fftn(image, axes=axes),
                               s=fft_shape, axes=axes)
    return butterfilt
