"""Inferior and superior ranks, provided by the user, are passed to the kernel
function to provide a softer version of the rank filters. E.g.
``autolevel_percentile`` will stretch image levels between percentile [p0, p1]
instead of using [min, max]. It means that isolated bright or dark pixels will
not produce halos.

The local histogram is computed using a sliding window similar to the method
described in [1]_.

Input image can be 8-bit or 16-bit, for 16-bit input images, the number of
histogram bins is determined from the maximum value present in the image.

Result image is 8-/16-bit or double with respect to the input image and the
rank filter operation.

References
----------

.. [1] Huang, T. ,Yang, G. ;  Tang, G.. "A fast two-dimensional
       median filtering algorithm", IEEE Transactions on Acoustics, Speech and
       Signal Processing, Feb 1979. Volume: 27 , Issue: 1, Page(s): 13 - 18.

"""

from ..._shared.utils import check_nD, deprecate_kwarg
from . import percentile_cy
from .generic import _preprocess_input

__all__ = ['autolevel_percentile', 'gradient_percentile',
           'mean_percentile', 'subtract_mean_percentile',
           'enhance_contrast_percentile', 'percentile', 'pop_percentile',
           'threshold_percentile']


def _apply(func, image, footprint, out, mask, shift_x, shift_y, p0, p1,
           out_dtype=None):
    check_nD(image, 2)
    image, footprint, out, mask, n_bins = _preprocess_input(
        image, footprint, out, mask, out_dtype
    )

    func(image, footprint, shift_x=shift_x, shift_y=shift_y, mask=mask,
         out=out, n_bins=n_bins, p0=p0, p1=p1)

    return out.reshape(out.shape[:2])


@deprecate_kwarg(kwarg_mapping={'selem': 'footprint'}, removed_version="1.0",
                 deprecated_version="0.19")
def autolevel_percentile(image, footprint, out=None, mask=None, shift_x=False,
                         shift_y=False, p0=0, p1=1):
    """Return grayscale local autolevel of an image.

    This filter locally stretches the histogram of grayvalues to cover the
    entire range of values from "white" to "black".

    Only grayvalues between percentiles [p0, p1] are considered in the filter.

    Parameters
    ----------
    image : 2-D array (uint8, uint16)
        Input image.
    footprint : 2-D array
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).
    p0, p1 : float in [0, ..., 1]
        Define the [p0, p1] percentile interval to be considered for computing
        the value.

    Returns
    -------
    out : 2-D array (same dtype as input image)
        Output image.

    """

    return _apply(percentile_cy._autolevel,
                  image, footprint, out=out, mask=mask, shift_x=shift_x,
                  shift_y=shift_y, p0=p0, p1=p1)


@deprecate_kwarg(kwarg_mapping={'selem': 'footprint'}, removed_version="1.0",
                 deprecated_version="0.19")
def gradient_percentile(image, footprint, out=None, mask=None, shift_x=False,
                        shift_y=False, p0=0, p1=1):
    """Return local gradient of an image (i.e. local maximum - local minimum).

    Only grayvalues between percentiles [p0, p1] are considered in the filter.

    Parameters
    ----------
    image : 2-D array (uint8, uint16)
        Input image.
    footprint : 2-D array
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).
    p0, p1 : float in [0, ..., 1]
        Define the [p0, p1] percentile interval to be considered for computing
        the value.

    Returns
    -------
    out : 2-D array (same dtype as input image)
        Output image.

    """

    return _apply(percentile_cy._gradient,
                  image, footprint, out=out, mask=mask, shift_x=shift_x,
                  shift_y=shift_y, p0=p0, p1=p1)


@deprecate_kwarg(kwarg_mapping={'selem': 'footprint'}, removed_version="1.0",
                 deprecated_version="0.19")
def mean_percentile(image, footprint, out=None, mask=None, shift_x=False,
                    shift_y=False, p0=0, p1=1):
    """Return local mean of an image.

    Only grayvalues between percentiles [p0, p1] are considered in the filter.

    Parameters
    ----------
    image : 2-D array (uint8, uint16)
        Input image.
    footprint : 2-D array
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).
    p0, p1 : float in [0, ..., 1]
        Define the [p0, p1] percentile interval to be considered for computing
        the value.

    Returns
    -------
    out : 2-D array (same dtype as input image)
        Output image.

    """

    return _apply(percentile_cy._mean,
                  image, footprint, out=out, mask=mask, shift_x=shift_x,
                  shift_y=shift_y, p0=p0, p1=p1)


@deprecate_kwarg(kwarg_mapping={'selem': 'footprint'}, removed_version="1.0",
                 deprecated_version="0.19")
def subtract_mean_percentile(image, footprint, out=None, mask=None,
                             shift_x=False, shift_y=False, p0=0, p1=1):
    """Return image subtracted from its local mean.

    Only grayvalues between percentiles [p0, p1] are considered in the filter.

    Parameters
    ----------
    image : 2-D array (uint8, uint16)
        Input image.
    footprint : 2-D array
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).
    p0, p1 : float in [0, ..., 1]
        Define the [p0, p1] percentile interval to be considered for computing
        the value.

    Returns
    -------
    out : 2-D array (same dtype as input image)
        Output image.

    """

    return _apply(percentile_cy._subtract_mean,
                  image, footprint, out=out, mask=mask, shift_x=shift_x,
                  shift_y=shift_y, p0=p0, p1=p1)


@deprecate_kwarg(kwarg_mapping={'selem': 'footprint'}, removed_version="1.0",
                 deprecated_version="0.19")
def enhance_contrast_percentile(image, footprint, out=None, mask=None,
                                shift_x=False, shift_y=False, p0=0, p1=1):
    """Enhance contrast of an image.

    This replaces each pixel by the local maximum if the pixel grayvalue is
    closer to the local maximum than the local minimum. Otherwise it is
    replaced by the local minimum.

    Only grayvalues between percentiles [p0, p1] are considered in the filter.

    Parameters
    ----------
    image : 2-D array (uint8, uint16)
        Input image.
    footprint : 2-D array
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).
    p0, p1 : float in [0, ..., 1]
        Define the [p0, p1] percentile interval to be considered for computing
        the value.

    Returns
    -------
    out : 2-D array (same dtype as input image)
        Output image.

    """

    return _apply(percentile_cy._enhance_contrast,
                  image, footprint, out=out, mask=mask, shift_x=shift_x,
                  shift_y=shift_y, p0=p0, p1=p1)


@deprecate_kwarg(kwarg_mapping={'selem': 'footprint'}, removed_version="1.0",
                 deprecated_version="0.19")
def percentile(image, footprint, out=None, mask=None, shift_x=False,
               shift_y=False, p0=0):
    """Return local percentile of an image.

    Returns the value of the p0 lower percentile of the local grayvalue
    distribution.

    Only grayvalues between percentiles [p0, p1] are considered in the filter.

    Parameters
    ----------
    image : 2-D array (uint8, uint16)
        Input image.
    footprint : 2-D array
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).
    p0 : float in [0, ..., 1]
        Set the percentile value.

    Returns
    -------
    out : 2-D array (same dtype as input image)
        Output image.

    """

    return _apply(percentile_cy._percentile,
                  image, footprint, out=out, mask=mask, shift_x=shift_x,
                  shift_y=shift_y, p0=p0, p1=0.)


@deprecate_kwarg(kwarg_mapping={'selem': 'footprint'}, removed_version="1.0",
                 deprecated_version="0.19")
def pop_percentile(image, footprint, out=None, mask=None, shift_x=False,
                   shift_y=False, p0=0, p1=1):
    """Return the local number (population) of pixels.

    The number of pixels is defined as the number of pixels which are included
    in the footprint and the mask.

    Only grayvalues between percentiles [p0, p1] are considered in the filter.

    Parameters
    ----------
    image : 2-D array (uint8, uint16)
        Input image.
    footprint : 2-D array
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).
    p0, p1 : float in [0, ..., 1]
        Define the [p0, p1] percentile interval to be considered for computing
        the value.

    Returns
    -------
    out : 2-D array (same dtype as input image)
        Output image.

    """

    return _apply(percentile_cy._pop,
                  image, footprint, out=out, mask=mask, shift_x=shift_x,
                  shift_y=shift_y, p0=p0, p1=p1)


@deprecate_kwarg(kwarg_mapping={'selem': 'footprint'}, removed_version="1.0",
                 deprecated_version="0.19")
def sum_percentile(image, footprint, out=None, mask=None, shift_x=False,
                   shift_y=False, p0=0, p1=1):
    """Return the local sum of pixels.

    Only grayvalues between percentiles [p0, p1] are considered in the filter.

    Note that the sum may overflow depending on the data type of the input
    array.

    Parameters
    ----------
    image : 2-D array (uint8, uint16)
        Input image.
    footprint : 2-D array
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).
    p0, p1 : float in [0, ..., 1]
        Define the [p0, p1] percentile interval to be considered for computing
        the value.

    Returns
    -------
    out : 2-D array (same dtype as input image)
        Output image.

    """

    return _apply(percentile_cy._sum,
                  image, footprint, out=out, mask=mask, shift_x=shift_x,
                  shift_y=shift_y, p0=p0, p1=p1)


@deprecate_kwarg(kwarg_mapping={'selem': 'footprint'}, removed_version="1.0",
                 deprecated_version="0.19")
def threshold_percentile(image, footprint, out=None, mask=None, shift_x=False,
                         shift_y=False, p0=0):
    """Local threshold of an image.

    The resulting binary mask is True if the grayvalue of the center pixel is
    greater than the local mean.

    Only grayvalues between percentiles [p0, p1] are considered in the filter.

    Parameters
    ----------
    image : 2-D array (uint8, uint16)
        Input image.
    footprint : 2-D array
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : 2-D array (same dtype as input)
        If None, a new array is allocated.
    mask : ndarray
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int
        Offset added to the footprint center point. Shift is bounded to the
        footprint sizes (center must be inside the given footprint).
    p0 : float in [0, ..., 1]
        Set the percentile value.

    Returns
    -------
    out : 2-D array (same dtype as input image)
        Output image.

    """

    return _apply(percentile_cy._threshold,
                  image, footprint, out=out, mask=mask, shift_x=shift_x,
                  shift_y=shift_y, p0=p0, p1=0)
