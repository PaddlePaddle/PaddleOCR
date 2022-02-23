import numpy as np


def regular_grid(ar_shape, n_points):
    """Find `n_points` regularly spaced along `ar_shape`.

    The returned points (as slices) should be as close to cubically-spaced as
    possible. Essentially, the points are spaced by the Nth root of the input
    array size, where N is the number of dimensions. However, if an array
    dimension cannot fit a full step size, it is "discarded", and the
    computation is done for only the remaining dimensions.

    Parameters
    ----------
    ar_shape : array-like of ints
        The shape of the space embedding the grid. ``len(ar_shape)`` is the
        number of dimensions.
    n_points : int
        The (approximate) number of points to embed in the space.

    Returns
    -------
    slices : tuple of slice objects
        A slice along each dimension of `ar_shape`, such that the intersection
        of all the slices give the coordinates of regularly spaced points.

        .. versionchanged:: 0.14.1
            In scikit-image 0.14.1 and 0.15, the return type was changed from a
            list to a tuple to ensure `compatibility with Numpy 1.15`_ and
            higher. If your code requires the returned result to be a list, you
            may convert the output of this function to a list with:

            >>> result = list(regular_grid(ar_shape=(3, 20, 40), n_points=8))

            .. _compatibility with NumPy 1.15: https://github.com/numpy/numpy/blob/master/doc/release/1.15.0-notes.rst#deprecations

    Examples
    --------
    >>> ar = np.zeros((20, 40))
    >>> g = regular_grid(ar.shape, 8)
    >>> g
    (slice(5, None, 10), slice(5, None, 10))
    >>> ar[g] = 1
    >>> ar.sum()
    8.0
    >>> ar = np.zeros((20, 40))
    >>> g = regular_grid(ar.shape, 32)
    >>> g
    (slice(2, None, 5), slice(2, None, 5))
    >>> ar[g] = 1
    >>> ar.sum()
    32.0
    >>> ar = np.zeros((3, 20, 40))
    >>> g = regular_grid(ar.shape, 8)
    >>> g
    (slice(1, None, 3), slice(5, None, 10), slice(5, None, 10))
    >>> ar[g] = 1
    >>> ar.sum()
    8.0
    """
    ar_shape = np.asanyarray(ar_shape)
    ndim = len(ar_shape)
    unsort_dim_idxs = np.argsort(np.argsort(ar_shape))
    sorted_dims = np.sort(ar_shape)
    space_size = float(np.prod(ar_shape))
    if space_size <= n_points:
        return (slice(None), ) * ndim
    stepsizes = np.full(ndim, (space_size / n_points) ** (1.0 / ndim),
                        dtype='float64')
    if (sorted_dims < stepsizes).any():
        for dim in range(ndim):
            stepsizes[dim] = sorted_dims[dim]
            space_size = float(np.prod(sorted_dims[dim + 1:]))
            stepsizes[dim + 1:] = ((space_size / n_points) **
                                   (1.0 / (ndim - dim - 1)))
            if (sorted_dims >= stepsizes).all():
                break
    starts = (stepsizes // 2).astype(int)
    stepsizes = np.round(stepsizes).astype(int)
    slices = [slice(start, None, step) for
              start, step in zip(starts, stepsizes)]
    slices = tuple(slices[i] for i in unsort_dim_idxs)
    return slices


def regular_seeds(ar_shape, n_points, dtype=int):
    """Return an image with ~`n_points` regularly-spaced nonzero pixels.

    Parameters
    ----------
    ar_shape : tuple of int
        The shape of the desired output image.
    n_points : int
        The desired number of nonzero points.
    dtype : numpy data type, optional
        The desired data type of the output.

    Returns
    -------
    seed_img : array of int or bool
        The desired image.

    Examples
    --------
    >>> regular_seeds((5, 5), 4)
    array([[0, 0, 0, 0, 0],
           [0, 1, 0, 2, 0],
           [0, 0, 0, 0, 0],
           [0, 3, 0, 4, 0],
           [0, 0, 0, 0, 0]])
    """
    grid = regular_grid(ar_shape, n_points)
    seed_img = np.zeros(ar_shape, dtype=dtype)
    seed_img[grid] = 1 + np.reshape(np.arange(seed_img[grid].size),
                                    seed_img[grid].shape)
    return seed_img
