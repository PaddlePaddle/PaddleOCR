from collections.abc import Iterable

import numpy as np
from scipy import ndimage as ndi

from .._shared import utils
from .._shared.filters import gaussian
from .._shared.utils import _supported_float_type, convert_to_float, warn
from ..util import img_as_float

__all__ = ['gaussian', 'difference_of_gaussians']


@utils.deprecate_multichannel_kwarg()
def difference_of_gaussians(image, low_sigma, high_sigma=None, *,
                            mode='nearest', cval=0, channel_axis=None,
                            truncate=4.0, multichannel=False):
    """Find features between ``low_sigma`` and ``high_sigma`` in size.

    This function uses the Difference of Gaussians method for applying
    band-pass filters to multi-dimensional arrays. The input array is
    blurred with two Gaussian kernels of differing sigmas to produce two
    intermediate, filtered images. The more-blurred image is then subtracted
    from the less-blurred image. The final output image will therefore have
    had high-frequency components attenuated by the smaller-sigma Gaussian, and
    low frequency components will have been removed due to their presence in
    the more-blurred intermediate.

    Parameters
    ----------
    image : ndarray
        Input array to filter.
    low_sigma : scalar or sequence of scalars
        Standard deviation(s) for the Gaussian kernel with the smaller sigmas
        across all axes. The standard deviations are given for each axis as a
        sequence, or as a single number, in which case the single number is
        used as the standard deviation value for all axes.
    high_sigma : scalar or sequence of scalars, optional (default is None)
        Standard deviation(s) for the Gaussian kernel with the larger sigmas
        across all axes. The standard deviations are given for each axis as a
        sequence, or as a single number, in which case the single number is
        used as the standard deviation value for all axes. If None is given
        (default), sigmas for all axes are calculated as 1.6 * low_sigma.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The ``mode`` parameter determines how the array borders are
        handled, where ``cval`` is the value when mode is equal to
        'constant'. Default is 'nearest'.
    cval : scalar, optional
        Value to fill past edges of input if ``mode`` is 'constant'. Default
        is 0.0
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.
    truncate : float, optional (default is 4.0)
        Truncate the filter at this many standard deviations.
    multichannel : bool, optional (default: False)
        Whether the last axis of the image is to be interpreted as multiple
        channels. If True, each channel is filtered separately (channels are
        not mixed together). This argument is deprecated: specify
        `channel_axis` instead.

    Returns
    -------
    filtered_image : ndarray
        the filtered array.

    See also
    --------
    skimage.feature.blob_dog

    Notes
    -----
    This function will subtract an array filtered with a Gaussian kernel
    with sigmas given by ``high_sigma`` from an array filtered with a
    Gaussian kernel with sigmas provided by ``low_sigma``. The values for
    ``high_sigma`` must always be greater than or equal to the corresponding
    values in ``low_sigma``, or a ``ValueError`` will be raised.

    When ``high_sigma`` is none, the values for ``high_sigma`` will be
    calculated as 1.6x the corresponding values in ``low_sigma``. This ratio
    was originally proposed by Marr and Hildreth (1980) [1]_ and is commonly
    used when approximating the inverted Laplacian of Gaussian, which is used
    in edge and blob detection.

    Input image is converted according to the conventions of ``img_as_float``.

    Except for sigma values, all parameters are used for both filters.

    Examples
    --------
    Apply a simple Difference of Gaussians filter to a color image:

    >>> from skimage.data import astronaut
    >>> from skimage.filters import difference_of_gaussians
    >>> filtered_image = difference_of_gaussians(astronaut(), 2, 10,
    ...                                          channel_axis=-1)

    Apply a Laplacian of Gaussian filter as approximated by the Difference
    of Gaussians filter:

    >>> filtered_image = difference_of_gaussians(astronaut(), 2,
    ...                                          channel_axis=-1)

    Apply a Difference of Gaussians filter to a grayscale image using different
    sigma values for each axis:

    >>> from skimage.data import camera
    >>> filtered_image = difference_of_gaussians(camera(), (2,5), (3,20))

    References
    ----------
    .. [1] Marr, D. and Hildreth, E. Theory of Edge Detection. Proc. R. Soc.
           Lond. Series B 207, 187-217 (1980).
           https://doi.org/10.1098/rspb.1980.0020

    """
    image = img_as_float(image)
    low_sigma = np.array(low_sigma, dtype='float', ndmin=1)
    if high_sigma is None:
        high_sigma = low_sigma * 1.6
    else:
        high_sigma = np.array(high_sigma, dtype='float', ndmin=1)

    if channel_axis is not None:
        spatial_dims = image.ndim - 1
    else:
        spatial_dims = image.ndim

    if len(low_sigma) != 1 and len(low_sigma) != spatial_dims:
        raise ValueError('low_sigma must have length equal to number of'
                         ' spatial dimensions of input')
    if len(high_sigma) != 1 and len(high_sigma) != spatial_dims:
        raise ValueError('high_sigma must have length equal to number of'
                         ' spatial dimensions of input')

    low_sigma = low_sigma * np.ones(spatial_dims)
    high_sigma = high_sigma * np.ones(spatial_dims)

    if any(high_sigma < low_sigma):
        raise ValueError('high_sigma must be equal to or larger than'
                         'low_sigma for all axes')

    im1 = gaussian(image, low_sigma, mode=mode, cval=cval,
                   channel_axis=channel_axis, truncate=truncate,
                   preserve_range=False)

    im2 = gaussian(image, high_sigma, mode=mode, cval=cval,
                   channel_axis=channel_axis, truncate=truncate,
                   preserve_range=False)

    return im1 - im2
