import functools

import numpy as np

from .. import color
from ..util.dtype import _convert


__all__ = ['adapt_rgb', 'hsv_value', 'each_channel']


def is_rgb_like(image, channel_axis=-1):
    """Return True if the image *looks* like it's RGB.

    This function should not be public because it is only intended to be used
    for functions that don't accept volumes as input, since checking an image's
    shape is fragile.
    """
    return (image.ndim == 3) and (image.shape[channel_axis] in (3, 4))


def adapt_rgb(apply_to_rgb):
    """Return decorator that adapts to RGB images to a gray-scale filter.

    This function is only intended to be used for functions that don't accept
    volumes as input, since checking an image's shape is fragile.

    Parameters
    ----------
    apply_to_rgb : function
        Function that returns a filtered image from an image-filter and RGB
        image. This will only be called if the image is RGB-like.
    """
    def decorator(image_filter):
        @functools.wraps(image_filter)
        def image_filter_adapted(image, *args, **kwargs):
            if is_rgb_like(image):
                return apply_to_rgb(image_filter, image, *args, **kwargs)
            else:
                return image_filter(image, *args, **kwargs)
        return image_filter_adapted
    return decorator


def hsv_value(image_filter, image, *args, **kwargs):
    """Return color image by applying `image_filter` on HSV-value of `image`.

    Note that this function is intended for use with `adapt_rgb`.

    Parameters
    ----------
    image_filter : function
        Function that filters a gray-scale image.
    image : array
        Input image. Note that RGBA images are treated as RGB.
    """
    # Slice the first three channels so that we remove any alpha channels.
    hsv = color.rgb2hsv(image[:, :, :3])
    value = hsv[:, :, 2].copy()
    value = image_filter(value, *args, **kwargs)
    hsv[:, :, 2] = _convert(value, hsv.dtype)
    return color.hsv2rgb(hsv)


def each_channel(image_filter, image, *args, **kwargs):
    """Return color image by applying `image_filter` on channels of `image`.

    Note that this function is intended for use with `adapt_rgb`.

    Parameters
    ----------
    image_filter : function
        Function that filters a gray-scale image.
    image : array
        Input image.
    """
    c_new = [image_filter(c, *args, **kwargs)
             for c in np.moveaxis(image, -1, 0)]
    return np.stack(c_new, axis=-1)
