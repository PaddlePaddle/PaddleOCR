import numpy as np

from .._shared.filters import gaussian


def binary_blobs(length=512, blob_size_fraction=0.1, n_dim=2,
                 volume_fraction=0.5, seed=None):
    """
    Generate synthetic binary image with several rounded blob-like objects.

    Parameters
    ----------
    length : int, optional
        Linear size of output image.
    blob_size_fraction : float, optional
        Typical linear size of blob, as a fraction of ``length``, should be
        smaller than 1.
    n_dim : int, optional
        Number of dimensions of output image.
    volume_fraction : float, default 0.5
        Fraction of image pixels covered by the blobs (where the output is 1).
        Should be in [0, 1].
    seed : {None, int, `numpy.random.Generator`}, optional
        If `seed` is None the `numpy.random.Generator` singleton is used.
        If `seed` is an int, a new ``Generator`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` instance then that instance is
        used.

    Returns
    -------
    blobs : ndarray of bools
        Output binary image

    Examples
    --------
    >>> from skimage import data
    >>> data.binary_blobs(length=5, blob_size_fraction=0.2)  # doctest: +SKIP
    array([[ True, False,  True,  True,  True],
           [ True,  True,  True, False,  True],
           [False,  True, False,  True,  True],
           [ True, False, False,  True,  True],
           [ True, False, False, False,  True]])
    >>> blobs = data.binary_blobs(length=256, blob_size_fraction=0.1)
    >>> # Finer structures
    >>> blobs = data.binary_blobs(length=256, blob_size_fraction=0.05)
    >>> # Blobs cover a smaller volume fraction of the image
    >>> blobs = data.binary_blobs(length=256, volume_fraction=0.3)

    """
    # filters is quite an expensive import since it imports all of scipy.signal
    # We lazy import here

    rs = np.random.default_rng(seed)
    shape = tuple([length] * n_dim)
    mask = np.zeros(shape)
    n_pts = max(int(1. / blob_size_fraction) ** n_dim, 1)
    points = (length * rs.random((n_dim, n_pts))).astype(int)
    mask[tuple(indices for indices in points)] = 1
    mask = gaussian(mask, sigma=0.25 * length * blob_size_fraction,
                    preserve_range=False)
    threshold = np.percentile(mask, 100 * (1 - volume_fraction))
    return np.logical_not(mask < threshold)
