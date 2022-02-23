import functools

import numpy as np
from scipy.signal import get_window

from .._shared.utils import safe_as_int
from ..transform import warp


def window(window_type, shape, warp_kwargs=None):
    """Return an n-dimensional window of a given size and dimensionality.

    Parameters
    ----------
    window_type : string, float, or tuple
        The type of window to be created. Any window type supported by
        ``scipy.signal.get_window`` is allowed here. See notes below for a
        current list, or the SciPy documentation for the version of SciPy
        on your machine.
    shape : tuple of int or int
        The shape of the window along each axis. If an integer is provided,
        a 1D window is generated.
    warp_kwargs : dict
        Keyword arguments passed to `skimage.transform.warp` (e.g.,
        ``warp_kwargs={'order':3}`` to change interpolation method).

    Returns
    -------
    nd_window : ndarray
        A window of the specified ``shape``. ``dtype`` is ``np.double``.

    Notes
    -----
    This function is based on ``scipy.signal.get_window`` and thus can access
    all of the window types available to that function
    (e.g., ``"hann"``, ``"boxcar"``). Note that certain window types require
    parameters that have to be supplied with the window name as a tuple
    (e.g., ``("tukey", 0.8)``). If only a float is supplied, it is interpreted
    as the beta parameter of the Kaiser window.

    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.get_window.html
    for more details.

    Note that this function generates a double precision array of the specified
    ``shape`` and can thus generate very large arrays that consume a large
    amount of available memory.

    The approach taken here to create nD windows is to first calculate the
    Euclidean distance from the center of the intended nD window to each
    position in the array. That distance is used to sample, with
    interpolation, from a 1D window returned from ``scipy.signal.get_window``.
    The method of interpolation can be changed with the ``order`` keyword
    argument passed to `skimage.transform.warp`.

    Some coordinates in the output window will be outside of the original
    signal; these will be filled in with zeros.

    Window types:
    - boxcar
    - triang
    - blackman
    - hamming
    - hann
    - bartlett
    - flattop
    - parzen
    - bohman
    - blackmanharris
    - nuttall
    - barthann
    - kaiser (needs beta)
    - gaussian (needs standard deviation)
    - general_gaussian (needs power, width)
    - slepian (needs width)
    - dpss (needs normalized half-bandwidth)
    - chebwin (needs attenuation)
    - exponential (needs decay scale)
    - tukey (needs taper fraction)

    Examples
    --------
    Return a Hann window with shape (512, 512):

    >>> from skimage.filters import window
    >>> w = window('hann', (512, 512))

    Return a Kaiser window with beta parameter of 16 and shape (256, 256, 35):

    >>> w = window(16, (256, 256, 35))

    Return a Tukey window with an alpha parameter of 0.8 and shape (100, 300):

    >>> w = window(('tukey', 0.8), (100, 300))

    References
    ----------
    .. [1] Two-dimensional window design, Wikipedia,
           https://en.wikipedia.org/wiki/Two_dimensional_window_design
    """

    if np.isscalar(shape):
        shape = safe_as_int(shape),
    else:
        shape = tuple(safe_as_int(shape))
    if any(s < 0 for s in shape):
        raise ValueError("invalid shape")

    ndim = len(shape)
    if ndim <= 0:
        raise ValueError("Number of dimensions must be greater than zero")

    max_size = functools.reduce(max, shape)
    w = get_window(window_type, max_size, fftbins=False)
    w = np.reshape(w, (-1,) + (1,) * (ndim-1))

    # Create coords for warping following `ndimage.map_coordinates` convention.
    L = [np.arange(s, dtype=np.float32) * (max_size / s) for s in shape]

    center = (max_size / 2) - 0.5
    dist = 0
    for g in np.meshgrid(*L, sparse=True, indexing='ij'):
        g -= center
        dist = dist + g * g
    dist = np.sqrt(dist)
    coords = np.zeros((ndim,) + dist.shape, dtype=np.float32)
    coords[0] = dist + center

    if warp_kwargs is None:
        warp_kwargs = {}

    return warp(w, coords, mode='constant', cval=0., **warp_kwargs)
