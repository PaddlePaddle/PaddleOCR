import numpy as np

from ..util.dtype import img_as_float
from .._shared import utils
from .._shared.filters import gaussian


def _unsharp_mask_single_channel(image, radius, amount, vrange):
    """Single channel implementation of the unsharp masking filter."""

    blurred = gaussian(image, sigma=radius, mode='reflect')

    result = image + (image - blurred) * amount
    if vrange is not None:
        return np.clip(result, vrange[0], vrange[1], out=result)
    return result


@utils.deprecate_multichannel_kwarg(multichannel_position=3)
def unsharp_mask(image, radius=1.0, amount=1.0, multichannel=False,
                 preserve_range=False, *, channel_axis=None):
    """Unsharp masking filter.

    The sharp details are identified as the difference between the original
    image and its blurred version. These details are then scaled, and added
    back to the original image.

    Parameters
    ----------
    image : [P, ..., ]M[, N][, C] ndarray
        Input image.
    radius : scalar or sequence of scalars, optional
        If a scalar is given, then its value is used for all dimensions.
        If sequence is given, then there must be exactly one radius
        for each dimension except the last dimension for multichannel images.
        Note that 0 radius means no blurring, and negative values are
        not allowed.
    amount : scalar, optional
        The details will be amplified with this factor. The factor could be 0
        or negative. Typically, it is a small positive number, e.g. 1.0.
    multichannel : bool, optional
        If True, the last ``image`` dimension is considered as a color channel,
        otherwise as spatial. Color channels are processed individually.
        This argument is deprecated: specify `channel_axis` instead.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of ``img_as_float``.
        Also see https://scikit-image.org/docs/dev/user_guide/data_types.html
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    output : [P, ..., ]M[, N][, C] ndarray of float
        Image with unsharp mask applied.

    Notes
    -----
    Unsharp masking is an image sharpening technique. It is a linear image
    operation, and numerically stable, unlike deconvolution which is an
    ill-posed problem. Because of this stability, it is often
    preferred over deconvolution.

    The main idea is as follows: sharp details are identified as the
    difference between the original image and its blurred version.
    These details are added back to the original image after a scaling step:

        enhanced image = original + amount * (original - blurred)

    When applying this filter to several color layers independently,
    color bleeding may occur. More visually pleasing result can be
    achieved by processing only the brightness/lightness/intensity
    channel in a suitable color space such as HSV, HSL, YUV, or YCbCr.

    Unsharp masking is described in most introductory digital image
    processing books. This implementation is based on [1]_.

    Examples
    --------
    >>> array = np.ones(shape=(5,5), dtype=np.uint8)*100
    >>> array[2,2] = 120
    >>> array
    array([[100, 100, 100, 100, 100],
           [100, 100, 100, 100, 100],
           [100, 100, 120, 100, 100],
           [100, 100, 100, 100, 100],
           [100, 100, 100, 100, 100]], dtype=uint8)
    >>> np.around(unsharp_mask(array, radius=0.5, amount=2),2)
    array([[0.39, 0.39, 0.39, 0.39, 0.39],
           [0.39, 0.39, 0.38, 0.39, 0.39],
           [0.39, 0.38, 0.53, 0.38, 0.39],
           [0.39, 0.39, 0.38, 0.39, 0.39],
           [0.39, 0.39, 0.39, 0.39, 0.39]])

    >>> array = np.ones(shape=(5,5), dtype=np.int8)*100
    >>> array[2,2] = 127
    >>> np.around(unsharp_mask(array, radius=0.5, amount=2),2)
    array([[0.79, 0.79, 0.79, 0.79, 0.79],
           [0.79, 0.78, 0.75, 0.78, 0.79],
           [0.79, 0.75, 1.  , 0.75, 0.79],
           [0.79, 0.78, 0.75, 0.78, 0.79],
           [0.79, 0.79, 0.79, 0.79, 0.79]])

    >>> np.around(unsharp_mask(array, radius=0.5, amount=2, preserve_range=True), 2)
    array([[100.  , 100.  ,  99.99, 100.  , 100.  ],
           [100.  ,  99.39,  95.48,  99.39, 100.  ],
           [ 99.99,  95.48, 147.59,  95.48,  99.99],
           [100.  ,  99.39,  95.48,  99.39, 100.  ],
           [100.  , 100.  ,  99.99, 100.  , 100.  ]])


    References
    ----------
    .. [1]  Maria Petrou, Costas Petrou
            "Image Processing: The Fundamentals", (2010), ed ii., page 357,
            ISBN 13: 9781119994398  :DOI:`10.1002/9781119994398`
    .. [2]  Wikipedia. Unsharp masking
            https://en.wikipedia.org/wiki/Unsharp_masking

    """
    vrange = None  # Range for valid values; used for clipping.
    float_dtype = utils._supported_float_type(image.dtype)
    if preserve_range:
        fimg = image.astype(float_dtype, copy=False)
    else:
        fimg = img_as_float(image).astype(float_dtype, copy=False)
        negative = np.any(fimg < 0)
        if negative:
            vrange = [-1., 1.]
        else:
            vrange = [0., 1.]

    if channel_axis is not None:
        result = np.empty_like(fimg, dtype=float_dtype)
        for channel in range(image.shape[channel_axis]):
            sl = utils.slice_at_axis(channel, channel_axis)
            result[sl] = _unsharp_mask_single_channel(
                fimg[sl], radius, amount, vrange)
        return result
    else:
        return _unsharp_mask_single_channel(fimg, radius, amount, vrange)
