
from warnings import warn

import numpy as np
from scipy import ndimage as ndi

from .._shared.utils import deprecate_kwarg
from .rank import generic


@deprecate_kwarg(kwarg_mapping={'selem': 'footprint'}, removed_version="1.0",
                 deprecated_version="0.19")
def median(image, footprint=None, out=None, mode='nearest', cval=0.0,
           behavior='ndimage'):
    """Return local median of an image.

    Parameters
    ----------
    image : array-like
        Input image.
    footprint : ndarray, optional
        If ``behavior=='rank'``, ``footprint`` is a 2-D array of 1's and 0's.
        If ``behavior=='ndimage'``, ``footprint`` is a N-D array of 1's and 0's
        with the same number of dimension than ``image``.
        If None, ``footprint`` will be a N-D array with 3 elements for each
        dimension (e.g., vector, square, cube, etc.)
    out : ndarray, (same dtype as image), optional
        If None, a new array is allocated.
    mode : {'reflect', 'constant', 'nearest', 'mirror','‘wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        ``cval`` is the value when mode is equal to 'constant'.
        Default is 'nearest'.

        .. versionadded:: 0.15
           ``mode`` is used when ``behavior='ndimage'``.
    cval : scalar, optional
        Value to fill past edges of input if mode is 'constant'. Default is 0.0

        .. versionadded:: 0.15
           ``cval`` was added in 0.15 is used when ``behavior='ndimage'``.
    behavior : {'ndimage', 'rank'}, optional
        Either to use the old behavior (i.e., < 0.15) or the new behavior.
        The old behavior will call the :func:`skimage.filters.rank.median`.
        The new behavior will call the :func:`scipy.ndimage.median_filter`.
        Default is 'ndimage'.

        .. versionadded:: 0.15
           ``behavior`` is introduced in 0.15
        .. versionchanged:: 0.16
           Default ``behavior`` has been changed from 'rank' to 'ndimage'

    Returns
    -------
    out : 2-D array (same dtype as input image)
        Output image.

    See also
    --------
    skimage.filters.rank.median : Rank-based implementation of the median
        filtering offering more flexibility with additional parameters but
        dedicated for unsigned integer images.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.morphology import disk
    >>> from skimage.filters import median
    >>> img = data.camera()
    >>> med = median(img, disk(5))

    """
    if behavior == 'rank':
        if mode != 'nearest' or not np.isclose(cval, 0.0):
            warn("Change 'behavior' to 'ndimage' if you want to use the "
                 "parameters 'mode' or 'cval'. They will be discarded "
                 "otherwise.")
        return generic.median(image, footprint=footprint, out=out)
    if footprint is None:
        footprint = ndi.generate_binary_structure(image.ndim, image.ndim)
    return ndi.median_filter(image, footprint=footprint, output=out, mode=mode,
                             cval=cval)
