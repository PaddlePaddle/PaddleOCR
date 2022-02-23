import numpy as np

from .._shared.utils import _supported_float_type, _to_np_mode


def _validate_window_size(axis_sizes):
    """Ensure all sizes in ``axis_sizes`` are odd.

    Parameters
    ----------
    axis_sizes : iterable of int

    Raises
    ------
    ValueError
        If any given axis size is even.
    """
    for axis_size in axis_sizes:
        if axis_size % 2 == 0:
            msg = (f'Window size for `threshold_sauvola` or '
                   f'`threshold_niblack` must not be even on any dimension. '
                   f'Got {axis_sizes}')
            raise ValueError(msg)


def _get_view(padded, kernel_shape, idx, val):
    """Get a view into `padded` that is offset by `idx` and scaled by `val`.

    If `padded` was created by padding the original image by `kernel_shape` as
    in correlate_sparse, then the view created here will match the size of the
    original image.
    """
    sl_shift = tuple([slice(c, s - (w_ - 1 - c))
                      for c, w_, s in zip(idx, kernel_shape, padded.shape)])
    v = padded[sl_shift]
    if val == 1:
        return v
    return val * v


def _correlate_sparse(image, kernel_shape, kernel_indices, kernel_values):
    """Perform correlation with a sparse kernel.

    Parameters
    ----------
    image : ndarray
        The (prepadded) image to be correlated.
    kernel_shape : tuple of int
        The shape of the sparse filter kernel.
    kernel_indices : list of coordinate tuples
        The indices of each non-zero kernel entry.
    kernel_values : list of float
        The kernel values at each location in kernel_indices.

    Returns
    -------
    out : ndarray
        The filtered image.

    Notes
    -----
    This function only returns results for the 'valid' region of the
    convolution, and thus `out` will be smaller than `image` by an amount
    equal to the kernel size along each axis.
    """
    idx, val = kernel_indices[0], kernel_values[0]
    # implementation assumes this corner is first in kernel_indices_in_values
    if tuple(idx) != (0,) * image.ndim:
        raise RuntimeError("Unexpected initial index in kernel_indices")
    # make a copy to avoid modifying the input image
    out = _get_view(image, kernel_shape, idx, val).copy()
    for idx, val in zip(kernel_indices[1:], kernel_values[1:]):
        out += _get_view(image, kernel_shape, idx, val)
    return out


def correlate_sparse(image, kernel, mode='reflect'):
    """Compute valid cross-correlation of `padded_array` and `kernel`.

    This function is *fast* when `kernel` is large with many zeros.

    See ``scipy.ndimage.correlate`` for a description of cross-correlation.

    Parameters
    ----------
    image : ndarray, dtype float, shape (M, N,[ ...,] P)
        The input array. If mode is 'valid', this array should already be
        padded, as a margin of the same shape as kernel will be stripped
        off.
    kernel : ndarray, dtype float shape (Q, R,[ ...,] S)
        The kernel to be correlated. Must have the same number of
        dimensions as `padded_array`. For high performance, it should
        be sparse (few nonzero entries).
    mode : string, optional
        See `scipy.ndimage.correlate` for valid modes.
        Additionally, mode 'valid' is accepted, in which case no padding is
        applied and the result is the result for the smaller image for which
        the kernel is entirely inside the original data.

    Returns
    -------
    result : array of float, shape (M, N,[ ...,] P)
        The result of cross-correlating `image` with `kernel`. If mode
        'valid' is used, the resulting shape is (M-Q+1, N-R+1,[ ...,] P-S+1).
    """
    kernel = np.asarray(kernel)

    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)

    if mode == 'valid':
        padded_image = image
    else:
        np_mode = _to_np_mode(mode)
        _validate_window_size(kernel.shape)
        padded_image = np.pad(
            image,
            [(w // 2, w // 2) for w in kernel.shape],
            mode=np_mode,
        )

    # extract the kernel's non-zero indices and corresponding values
    indices = np.nonzero(kernel)
    values = list(kernel[indices].astype(float_dtype, copy=False))
    indices = list(zip(*indices))

    # _correlate_sparse requires an index at (0,) * kernel.ndim to be present
    corner_index = (0,) * kernel.ndim
    if corner_index not in indices:
        indices = [corner_index] + indices
        values = [0.0] + values

    return _correlate_sparse(padded_image, kernel.shape, indices, values)
