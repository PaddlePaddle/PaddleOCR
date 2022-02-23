import numpy as np
import scipy.ndimage as ndi

from ..color import rgb2gray
from ..util import img_as_float


__all__ = ['blur_effect']


def blur_effect(image, h_size=11, channel_axis=None, reduce_func=np.max):
    """Compute a metric that indicates the strength of blur in an image
    (0 for no blur, 1 for maximal blur).

    Parameters
    ----------
    image : ndarray
        RGB or grayscale nD image. The input image is converted to grayscale
        before computing the blur metric.
    h_size : int, optional
        Size of the re-blurring filter.
    channel_axis : int or None, optional
        If None, the image is assumed to be grayscale (single-channel).
        Otherwise, this parameter indicates which axis of the array
        corresponds to color channels.
    reduce_func : callable, optional
        Function used to calculate the aggregation of blur metrics along all
        axes. If set to None, the entire list is returned, where the i-th
        element is the blur metric along the i-th axis.

    Returns
    -------
    blur : float (0 to 1) or list of floats
        Blur metric: by default, the maximum of blur metrics along all axes.

    Notes
    -----
    `h_size` must keep the same value in order to compare results between
    images. Most of the time, the default size (11) is enough. This means that
    the metric can clearly discriminate blur up to an average 11x11 filter; if
    blur is higher, the metric still gives good results but its values tend
    towards an asymptote.

    References
    ----------
    .. [1] Frederique Crete, Thierry Dolmiere, Patricia Ladret, and Marina
       Nicolas "The blur effect: perception and estimation with a new
       no-reference perceptual blur metric" Proc. SPIE 6492, Human Vision and
       Electronic Imaging XII, 64920I (2007)
       https://hal.archives-ouvertes.fr/hal-00232709
       :DOI:`10.1117/12.702790`
    """

    if channel_axis is not None:
        try:
            # ensure color channels are in the final dimension
            image = np.moveaxis(image, channel_axis, -1)
        except np.AxisError:
            print('channel_axis must be one of the image array dimensions')
            raise
        except TypeError:
            print('channel_axis must be an integer')
            raise
        image = rgb2gray(image)
    n_axes = image.ndim
    image = img_as_float(image)
    shape = image.shape
    B = []

    from ..filters import sobel
    slices = tuple([slice(2, s - 1) for s in shape])
    for ax in range(n_axes):
        filt_im = ndi.uniform_filter1d(image, h_size, axis=ax)
        im_sharp = np.abs(sobel(image, axis=ax))
        im_blur = np.abs(sobel(filt_im, axis=ax))
        T = np.maximum(0, im_sharp - im_blur)
        M1 = np.sum(im_sharp[slices])
        M2 = np.sum(T[slices])
        B.append(np.abs((M1 - M2)) / M1)

    return B if reduce_func is None else reduce_func(B)
