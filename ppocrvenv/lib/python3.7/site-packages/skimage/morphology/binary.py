"""
Binary morphological operations
"""
import numpy as np
from scipy import ndimage as ndi

from .._shared.utils import deprecate_kwarg
from .misc import default_footprint


# The default_footprint decorator provides a diamond footprint as
# default with the same dimension as the input image and size 3 along each
# axis.
@default_footprint
@deprecate_kwarg(kwarg_mapping={'selem': 'footprint'}, removed_version="1.0",
                 deprecated_version="0.19")
def binary_erosion(image, footprint=None, out=None):
    """Return fast binary morphological erosion of an image.

    This function returns the same result as grayscale erosion but performs
    faster for binary images.

    Morphological erosion sets a pixel at ``(i,j)`` to the minimum over all
    pixels in the neighborhood centered at ``(i,j)``. Erosion shrinks bright
    regions and enlarges dark regions.

    Parameters
    ----------
    image : ndarray
        Binary input image.
    footprint : ndarray, optional
        The neighborhood expressed as a 2-D array of 1's and 0's.
        If None, use a cross-shaped footprint (connectivity=1).
    out : ndarray of bool, optional
        The array to store the result of the morphology. If None is
        passed, a new array will be allocated.

    Returns
    -------
    eroded : ndarray of bool or uint
        The result of the morphological erosion taking values in
        ``[False, True]``.

    """
    if out is None:
        out = np.empty(image.shape, dtype=bool)
    ndi.binary_erosion(image, structure=footprint, output=out,
                       border_value=True)
    return out


@default_footprint
@deprecate_kwarg(kwarg_mapping={'selem': 'footprint'}, removed_version="1.0",
                 deprecated_version="0.19")
def binary_dilation(image, footprint=None, out=None):
    """Return fast binary morphological dilation of an image.

    This function returns the same result as grayscale dilation but performs
    faster for binary images.

    Morphological dilation sets a pixel at ``(i,j)`` to the maximum over all
    pixels in the neighborhood centered at ``(i,j)``. Dilation enlarges bright
    regions and shrinks dark regions.

    Parameters
    ----------
    image : ndarray
        Binary input image.
    footprint : ndarray, optional
        The neighborhood expressed as a 2-D array of 1's and 0's.
        If None, use a cross-shaped footprint (connectivity=1).
    out : ndarray of bool, optional
        The array to store the result of the morphology. If None is
        passed, a new array will be allocated.

    Returns
    -------
    dilated : ndarray of bool or uint
        The result of the morphological dilation with values in
        ``[False, True]``.
    """
    if out is None:
        out = np.empty(image.shape, dtype=bool)
    ndi.binary_dilation(image, structure=footprint, output=out)
    return out


@default_footprint
@deprecate_kwarg(kwarg_mapping={'selem': 'footprint'}, removed_version="1.0",
                 deprecated_version="0.19")
def binary_opening(image, footprint=None, out=None):
    """Return fast binary morphological opening of an image.

    This function returns the same result as grayscale opening but performs
    faster for binary images.

    The morphological opening on an image is defined as an erosion followed by
    a dilation. Opening can remove small bright spots (i.e. "salt") and connect
    small dark cracks. This tends to "open" up (dark) gaps between (bright)
    features.

    Parameters
    ----------
    image : ndarray
        Binary input image.
    footprint : ndarray, optional
        The neighborhood expressed as a 2-D array of 1's and 0's.
        If None, use a cross-shaped footprint (connectivity=1).
    out : ndarray of bool, optional
        The array to store the result of the morphology. If None
        is passed, a new array will be allocated.

    Returns
    -------
    opening : ndarray of bool
        The result of the morphological opening.

    """
    eroded = binary_erosion(image, footprint)
    out = binary_dilation(eroded, footprint, out=out)
    return out


@default_footprint
@deprecate_kwarg(kwarg_mapping={'selem': 'footprint'}, removed_version="1.0",
                 deprecated_version="0.19")
def binary_closing(image, footprint=None, out=None):
    """Return fast binary morphological closing of an image.

    This function returns the same result as grayscale closing but performs
    faster for binary images.

    The morphological closing on an image is defined as a dilation followed by
    an erosion. Closing can remove small dark spots (i.e. "pepper") and connect
    small bright cracks. This tends to "close" up (dark) gaps between (bright)
    features.

    Parameters
    ----------
    image : ndarray
        Binary input image.
    footprint : ndarray, optional
        The neighborhood expressed as a 2-D array of 1's and 0's.
        If None, use a cross-shaped footprint (connectivity=1).
    out : ndarray of bool, optional
        The array to store the result of the morphology. If None,
        is passed, a new array will be allocated.

    Returns
    -------
    closing : ndarray of bool
        The result of the morphological closing.

    """
    dilated = binary_dilation(image, footprint)
    out = binary_erosion(dilated, footprint, out=out)
    return out
