import math

import numpy as np
from scipy import ndimage as ndi

from .._shared import utils
from .._shared.filters import gaussian
from .._shared.utils import convert_to_float
from ..transform import resize


def _smooth(image, sigma, mode, cval, multichannel=None):
    """Return image with each channel smoothed by the Gaussian filter."""
    smoothed = np.empty_like(image)

    # apply Gaussian filter to all channels independently
    if multichannel:
        sigma = (sigma, ) * (image.ndim - 1) + (0, )
        channel_axis = -1
    else:
        channel_axis = None
    gaussian(image, sigma, output=smoothed, mode=mode, cval=cval,
             channel_axis=channel_axis)
    return smoothed


def _check_factor(factor):
    if factor <= 1:
        raise ValueError('scale factor must be greater than 1')


@utils.channel_as_last_axis()
@utils.deprecate_multichannel_kwarg(multichannel_position=6)
def pyramid_reduce(image, downscale=2, sigma=None, order=1,
                   mode='reflect', cval=0, multichannel=False,
                   preserve_range=False, *, channel_axis=-1):
    """Smooth and then downsample image.

    Parameters
    ----------
    image : ndarray
        Input image.
    downscale : float, optional
        Downscale factor.
    sigma : float, optional
        Sigma for Gaussian filter. Default is `2 * downscale / 6.0` which
        corresponds to a filter mask twice the size of the scale factor that
        covers more than 99% of the Gaussian distribution.
    order : int, optional
        Order of splines used in interpolation of downsampling. See
        `skimage.transform.warp` for detail.
    mode : {'reflect', 'constant', 'edge', 'symmetric', 'wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        cval is the value when mode is equal to 'constant'.
    cval : float, optional
        Value to fill past edges of input if mode is 'constant'.
    multichannel : bool, optional
        Whether the last axis of the image is to be interpreted as multiple
        channels or another spatial dimension. This argument is deprecated:
        specify `channel_axis` instead.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see https://scikit-image.org/docs/dev/user_guide/data_types.html
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    out : array
        Smoothed and downsampled float image.

    References
    ----------
    .. [1] http://persci.mit.edu/pub_pdfs/pyramid83.pdf

    """
    _check_factor(downscale)
    multichannel = channel_axis is not None

    image = convert_to_float(image, preserve_range)

    out_shape = tuple([math.ceil(d / float(downscale)) for d in image.shape])
    if multichannel:
        out_shape = out_shape[:-1]

    if sigma is None:
        # automatically determine sigma which covers > 99% of distribution
        sigma = 2 * downscale / 6.0

    smoothed = _smooth(image, sigma, mode, cval, multichannel)
    out = resize(smoothed, out_shape, order=order, mode=mode, cval=cval,
                 anti_aliasing=False)

    return out


@utils.channel_as_last_axis()
@utils.deprecate_multichannel_kwarg(multichannel_position=6)
def pyramid_expand(image, upscale=2, sigma=None, order=1,
                   mode='reflect', cval=0, multichannel=False,
                   preserve_range=False, *, channel_axis=None):
    """Upsample and then smooth image.

    Parameters
    ----------
    image : ndarray
        Input image.
    upscale : float, optional
        Upscale factor.
    sigma : float, optional
        Sigma for Gaussian filter. Default is `2 * upscale / 6.0` which
        corresponds to a filter mask twice the size of the scale factor that
        covers more than 99% of the Gaussian distribution.
    order : int, optional
        Order of splines used in interpolation of upsampling. See
        `skimage.transform.warp` for detail.
    mode : {'reflect', 'constant', 'edge', 'symmetric', 'wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        cval is the value when mode is equal to 'constant'.
    cval : float, optional
        Value to fill past edges of input if mode is 'constant'.
    multichannel : bool, optional
        Whether the last axis of the image is to be interpreted as multiple
        channels or another spatial dimension. This argument is deprecated:
        specify `channel_axis` instead.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see https://scikit-image.org/docs/dev/user_guide/data_types.html
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    out : array
        Upsampled and smoothed float image.

    References
    ----------
    .. [1] http://persci.mit.edu/pub_pdfs/pyramid83.pdf

    """
    _check_factor(upscale)
    multichannel = channel_axis is not None

    image = convert_to_float(image, preserve_range)

    out_shape = tuple([math.ceil(upscale * d) for d in image.shape])
    if multichannel:
        out_shape = out_shape[:-1]

    if sigma is None:
        # automatically determine sigma which covers > 99% of distribution
        sigma = 2 * upscale / 6.0

    resized = resize(image, out_shape, order=order,
                     mode=mode, cval=cval, anti_aliasing=False)
    out = _smooth(resized, sigma, mode, cval, multichannel)

    return out


@utils.channel_as_last_axis()
@utils.deprecate_multichannel_kwarg(multichannel_position=7)
def pyramid_gaussian(image, max_layer=-1, downscale=2, sigma=None, order=1,
                     mode='reflect', cval=0, multichannel=False,
                     preserve_range=False, *, channel_axis=-1):
    """Yield images of the Gaussian pyramid formed by the input image.

    Recursively applies the `pyramid_reduce` function to the image, and yields
    the downscaled images.

    Note that the first image of the pyramid will be the original, unscaled
    image. The total number of images is `max_layer + 1`. In case all layers
    are computed, the last image is either a one-pixel image or the image where
    the reduction does not change its shape.

    Parameters
    ----------
    image : ndarray
        Input image.
    max_layer : int, optional
        Number of layers for the pyramid. 0th layer is the original image.
        Default is -1 which builds all possible layers.
    downscale : float, optional
        Downscale factor.
    sigma : float, optional
        Sigma for Gaussian filter. Default is `2 * downscale / 6.0` which
        corresponds to a filter mask twice the size of the scale factor that
        covers more than 99% of the Gaussian distribution.
    order : int, optional
        Order of splines used in interpolation of downsampling. See
        `skimage.transform.warp` for detail.
    mode : {'reflect', 'constant', 'edge', 'symmetric', 'wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        cval is the value when mode is equal to 'constant'.
    cval : float, optional
        Value to fill past edges of input if mode is 'constant'.
    multichannel : bool, optional
        Whether the last axis of the image is to be interpreted as multiple
        channels or another spatial dimension. This argument is deprecated:
        specify `channel_axis` instead.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see https://scikit-image.org/docs/dev/user_guide/data_types.html
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    pyramid : generator
        Generator yielding pyramid layers as float images.

    References
    ----------
    .. [1] http://persci.mit.edu/pub_pdfs/pyramid83.pdf

    """
    _check_factor(downscale)

    # cast to float for consistent data type in pyramid
    image = convert_to_float(image, preserve_range)

    layer = 0
    current_shape = image.shape

    prev_layer_image = image
    yield image

    # build downsampled images until max_layer is reached or downscale process
    # does not change image size
    while layer != max_layer:
        layer += 1

        layer_image = pyramid_reduce(prev_layer_image, downscale, sigma, order,
                                     mode, cval, channel_axis=channel_axis)

        prev_shape = np.asarray(current_shape)
        prev_layer_image = layer_image
        current_shape = np.asarray(layer_image.shape)

        # no change to previous pyramid layer
        if np.all(current_shape == prev_shape):
            break

        yield layer_image


@utils.channel_as_last_axis()
@utils.deprecate_multichannel_kwarg(multichannel_position=7)
def pyramid_laplacian(image, max_layer=-1, downscale=2, sigma=None, order=1,
                      mode='reflect', cval=0, multichannel=False,
                      preserve_range=False, *, channel_axis=-1):
    """Yield images of the laplacian pyramid formed by the input image.

    Each layer contains the difference between the downsampled and the
    downsampled, smoothed image::

        layer = resize(prev_layer) - smooth(resize(prev_layer))

    Note that the first image of the pyramid will be the difference between the
    original, unscaled image and its smoothed version. The total number of
    images is `max_layer + 1`. In case all layers are computed, the last image
    is either a one-pixel image or the image where the reduction does not
    change its shape.

    Parameters
    ----------
    image : ndarray
        Input image.
    max_layer : int, optional
        Number of layers for the pyramid. 0th layer is the original image.
        Default is -1 which builds all possible layers.
    downscale : float, optional
        Downscale factor.
    sigma : float, optional
        Sigma for Gaussian filter. Default is `2 * downscale / 6.0` which
        corresponds to a filter mask twice the size of the scale factor that
        covers more than 99% of the Gaussian distribution.
    order : int, optional
        Order of splines used in interpolation of downsampling. See
        `skimage.transform.warp` for detail.
    mode : {'reflect', 'constant', 'edge', 'symmetric', 'wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        cval is the value when mode is equal to 'constant'.
    cval : float, optional
        Value to fill past edges of input if mode is 'constant'.
    multichannel : bool, optional
        Whether the last axis of the image is to be interpreted as multiple
        channels or another spatial dimension. This argument is deprecated:
        specify `channel_axis` instead.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see https://scikit-image.org/docs/dev/user_guide/data_types.html
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    pyramid : generator
        Generator yielding pyramid layers as float images.

    References
    ----------
    .. [1] http://persci.mit.edu/pub_pdfs/pyramid83.pdf
    .. [2] http://sepwww.stanford.edu/data/media/public/sep/morgan/texturematch/paper_html/node3.html

    """
    _check_factor(downscale)
    multichannel = channel_axis is not None

    # cast to float for consistent data type in pyramid
    image = convert_to_float(image, preserve_range)

    if sigma is None:
        # automatically determine sigma which covers > 99% of distribution
        sigma = 2 * downscale / 6.0

    current_shape = image.shape

    smoothed_image = _smooth(image, sigma, mode, cval, multichannel)
    yield image - smoothed_image

    # build downsampled images until max_layer is reached or downscale process
    # does not change image size
    if max_layer == -1:
        max_layer = int(np.ceil(math.log(np.max(current_shape), downscale)))

    for layer in range(max_layer):

        out_shape = tuple(
            [math.ceil(d / float(downscale)) for d in current_shape])

        if multichannel:
            out_shape = out_shape[:-1]

        resized_image = resize(smoothed_image, out_shape, order=order,
                               mode=mode, cval=cval, anti_aliasing=False)
        smoothed_image = _smooth(resized_image, sigma, mode, cval,
                                 multichannel)
        current_shape = np.asarray(resized_image.shape)

        yield resized_image - smoothed_image
