import numpy as np

from .._shared import utils
from .. import exposure

__all__ = ['montage']


@utils.channel_as_last_axis(multichannel_output=False)
@utils.deprecate_multichannel_kwarg()
def montage(arr_in, fill='mean', rescale_intensity=False, grid_shape=None,
            padding_width=0, multichannel=False, *, channel_axis=None):
    """Create a montage of several single- or multichannel images.

    Create a rectangular montage from an input array representing an ensemble
    of equally shaped single- (gray) or multichannel (color) images.

    For example, ``montage(arr_in)`` called with the following `arr_in`

    +---+---+---+
    | 1 | 2 | 3 |
    +---+---+---+

    will return

    +---+---+
    | 1 | 2 |
    +---+---+
    | 3 | * |
    +---+---+

    where the '*' patch will be determined by the `fill` parameter.

    Parameters
    ----------
    arr_in : (K, M, N[, C]) ndarray
        An array representing an ensemble of `K` images of equal shape.
    fill : float or array-like of floats or 'mean', optional
        Value to fill the padding areas and/or the extra tiles in
        the output array. Has to be `float` for single channel collections.
        For multichannel collections has to be an array-like of shape of
        number of channels. If `mean`, uses the mean value over all images.
    rescale_intensity : bool, optional
        Whether to rescale the intensity of each image to [0, 1].
    grid_shape : tuple, optional
        The desired grid shape for the montage `(ntiles_row, ntiles_column)`.
        The default aspect ratio is square.
    padding_width : int, optional
        The size of the spacing between the tiles and between the tiles and
        the borders. If non-zero, makes the boundaries of individual images
        easier to perceive.
    multichannel : boolean, optional
        If True, the last `arr_in` dimension is threated as a color channel,
        otherwise as spatial. This argument is deprecated: specify
        `channel_axis` instead.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.



    Returns
    -------
    arr_out : (K*(M+p)+p, K*(N+p)+p[, C]) ndarray
        Output array with input images glued together (including padding `p`).

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.util import montage
    >>> arr_in = np.arange(3 * 2 * 2).reshape(3, 2, 2)
    >>> arr_in  # doctest: +NORMALIZE_WHITESPACE
    array([[[ 0,  1],
            [ 2,  3]],
           [[ 4,  5],
            [ 6,  7]],
           [[ 8,  9],
            [10, 11]]])
    >>> arr_out = montage(arr_in)
    >>> arr_out.shape
    (4, 4)
    >>> arr_out
    array([[ 0,  1,  4,  5],
           [ 2,  3,  6,  7],
           [ 8,  9,  5,  5],
           [10, 11,  5,  5]])
    >>> arr_in.mean()
    5.5
    >>> arr_out_nonsquare = montage(arr_in, grid_shape=(1, 3))
    >>> arr_out_nonsquare
    array([[ 0,  1,  4,  5,  8,  9],
           [ 2,  3,  6,  7, 10, 11]])
    >>> arr_out_nonsquare.shape
    (2, 6)
    """

    if channel_axis is not None:
        arr_in = np.asarray(arr_in)
    else:
        arr_in = np.asarray(arr_in)[..., np.newaxis]

    if arr_in.ndim != 4:
        raise ValueError('Input array has to be 3-dimensional for grayscale '
                         'images, or 4-dimensional with a `channel_axis` '
                         'specified.')

    n_images, n_rows, n_cols, n_chan = arr_in.shape

    if grid_shape:
        ntiles_row, ntiles_col = [int(s) for s in grid_shape]
    else:
        ntiles_row = ntiles_col = int(np.ceil(np.sqrt(n_images)))

    # Rescale intensity if necessary
    if rescale_intensity:
        for i in range(n_images):
            arr_in[i] = exposure.rescale_intensity(arr_in[i])

    # Calculate the fill value
    if fill == 'mean':
        fill = arr_in.mean(axis=(0, 1, 2))
    fill = np.atleast_1d(fill).astype(arr_in.dtype)

    # Pre-allocate an array with padding for montage
    n_pad = padding_width
    arr_out = np.empty(((n_rows + n_pad) * ntiles_row + n_pad,
                        (n_cols + n_pad) * ntiles_col + n_pad,
                        n_chan), dtype=arr_in.dtype)
    for idx_chan in range(n_chan):
        arr_out[..., idx_chan] = fill[idx_chan]

    slices_row = [slice(n_pad + (n_rows + n_pad) * n,
                        n_pad + (n_rows + n_pad) * n + n_rows)
                  for n in range(ntiles_row)]
    slices_col = [slice(n_pad + (n_cols + n_pad) * n,
                        n_pad + (n_cols + n_pad) * n + n_cols)
                  for n in range(ntiles_col)]

    # Copy the data to the output array
    for idx_image, image in enumerate(arr_in):
        idx_sr = idx_image // ntiles_col
        idx_sc = idx_image % ntiles_col
        arr_out[slices_row[idx_sr], slices_col[idx_sc], :] = image

    if channel_axis is not None:
        return arr_out
    else:
        return arr_out[..., 0]
