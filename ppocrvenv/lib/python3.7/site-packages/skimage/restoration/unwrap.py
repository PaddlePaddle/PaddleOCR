import numpy as np

from .._shared.utils import warn

from ._unwrap_1d import unwrap_1d
from ._unwrap_2d import unwrap_2d
from ._unwrap_3d import unwrap_3d


def unwrap_phase(image, wrap_around=False, seed=None):
    '''Recover the original from a wrapped phase image.

    From an image wrapped to lie in the interval [-pi, pi), recover the
    original, unwrapped image.

    Parameters
    ----------
    image : 1D, 2D or 3D ndarray of floats, optionally a masked array
        The values should be in the range [-pi, pi). If a masked array is
        provided, the masked entries will not be changed, and their values
        will not be used to guide the unwrapping of neighboring, unmasked
        values. Masked 1D arrays are not allowed, and will raise a
        `ValueError`.
    wrap_around : bool or sequence of bool, optional
        When an element of the sequence is  `True`, the unwrapping process
        will regard the edges along the corresponding axis of the image to be
        connected and use this connectivity to guide the phase unwrapping
        process. If only a single boolean is given, it will apply to all axes.
        Wrap around is not supported for 1D arrays.
    seed : int, optional
        Unwrapping 2D or 3D images uses random initialization. This sets the
        seed of the PRNG to achieve deterministic behavior.

    Returns
    -------
    image_unwrapped : array_like, double
        Unwrapped image of the same shape as the input. If the input `image`
        was a masked array, the mask will be preserved.

    Raises
    ------
    ValueError
        If called with a masked 1D array or called with a 1D array and
        ``wrap_around=True``.

    Examples
    --------
    >>> c0, c1 = np.ogrid[-1:1:128j, -1:1:128j]
    >>> image = 12 * np.pi * np.exp(-(c0**2 + c1**2))
    >>> image_wrapped = np.angle(np.exp(1j * image))
    >>> image_unwrapped = unwrap_phase(image_wrapped)
    >>> np.std(image_unwrapped - image) < 1e-6   # A constant offset is normal
    True

    References
    ----------
    .. [1] Miguel Arevallilo Herraez, David R. Burton, Michael J. Lalor,
           and Munther A. Gdeisat, "Fast two-dimensional phase-unwrapping
           algorithm based on sorting by reliability following a noncontinuous
           path", Journal Applied Optics, Vol. 41, No. 35 (2002) 7437,
    .. [2] Abdul-Rahman, H., Gdeisat, M., Burton, D., & Lalor, M., "Fast
           three-dimensional phase-unwrapping algorithm based on sorting by
           reliability following a non-continuous path. In W. Osten,
           C. Gorecki, & E. L. Novak (Eds.), Optical Metrology (2005) 32--40,
           International Society for Optics and Photonics.
    '''
    if image.ndim not in (1, 2, 3):
        raise ValueError('Image must be 1, 2, or 3 dimensional')
    if isinstance(wrap_around, bool):
        wrap_around = [wrap_around] * image.ndim
    elif (hasattr(wrap_around, '__getitem__')
          and not isinstance(wrap_around, str)):
        if len(wrap_around) != image.ndim:
            raise ValueError('Length of `wrap_around` must equal the '
                             'dimensionality of image')
        wrap_around = [bool(wa) for wa in wrap_around]
    else:
        raise ValueError('`wrap_around` must be a bool or a sequence with '
                         'length equal to the dimensionality of image')
    if image.ndim == 1:
        if np.ma.isMaskedArray(image):
            raise ValueError('1D masked images cannot be unwrapped')
        if wrap_around[0]:
            raise ValueError('`wrap_around` is not supported for 1D images')
    if image.ndim in (2, 3) and 1 in image.shape:
        warn('Image has a length 1 dimension. Consider using an '
             'array of lower dimensionality to use a more efficient '
             'algorithm')

    if np.ma.isMaskedArray(image):
        mask = np.require(np.ma.getmaskarray(image), np.uint8, ['C'])
    else:
        mask = np.zeros_like(image, dtype=np.uint8, order='C')

    image_not_masked = np.asarray(
        np.ma.getdata(image), dtype=np.double, order='C')
    image_unwrapped = np.empty_like(image, dtype=np.double, order='C',
                                    subok=False)

    if image.ndim == 1:
        unwrap_1d(image_not_masked, image_unwrapped)
    elif image.ndim == 2:
        unwrap_2d(image_not_masked, mask, image_unwrapped,
                  wrap_around, seed)
    elif image.ndim == 3:
        unwrap_3d(image_not_masked, mask, image_unwrapped,
                  wrap_around, seed)

    if np.ma.isMaskedArray(image):
        return np.ma.array(image_unwrapped, mask=mask,
                           fill_value=image.fill_value)
    else:
        return image_unwrapped
