import numpy as np


def integral_image(image, *, dtype=None):
    r"""Integral image / summed area table.

    The integral image contains the sum of all elements above and to the
    left of it, i.e.:

    .. math::

       S[m, n] = \sum_{i \leq m} \sum_{j \leq n} X[i, j]

    Parameters
    ----------
    image : ndarray
        Input image.

    Returns
    -------
    S : ndarray
        Integral image/summed area table of same shape as input image.

    Notes
    -----
    For better accuracy and to avoid potential overflow, the data type of the
    output may differ from the input's when the default dtype of None is used.
    For inputs with integer dtype, the behavior matches that for
    :func:`numpy.cumsum`. Floating point inputs will be promoted to at least
    double precision. The user can set `dtype` to override this behavior.

    References
    ----------
    .. [1] F.C. Crow, "Summed-area tables for texture mapping,"
           ACM SIGGRAPH Computer Graphics, vol. 18, 1984, pp. 207-212.

    """
    if dtype is None and image.real.dtype.kind == 'f':
        # default to at least double precision cumsum for accuracy
        dtype = np.promote_types(image.dtype, np.float64)

    S = image
    for i in range(image.ndim):
        S = S.cumsum(axis=i, dtype=dtype)
    return S


def integrate(ii, start, end):
    """Use an integral image to integrate over a given window.

    Parameters
    ----------
    ii : ndarray
        Integral image.
    start : List of tuples, each tuple of length equal to dimension of `ii`
        Coordinates of top left corner of window(s).
        Each tuple in the list contains the starting row, col, ... index
        i.e `[(row_win1, col_win1, ...), (row_win2, col_win2,...), ...]`.
    end : List of tuples, each tuple of length equal to dimension of `ii`
        Coordinates of bottom right corner of window(s).
        Each tuple in the list containing the end row, col, ... index i.e
        `[(row_win1, col_win1, ...), (row_win2, col_win2, ...), ...]`.

    Returns
    -------
    S : scalar or ndarray
        Integral (sum) over the given window(s).


    Examples
    --------
    >>> arr = np.ones((5, 6), dtype=float)
    >>> ii = integral_image(arr)
    >>> integrate(ii, (1, 0), (1, 2))  # sum from (1, 0) to (1, 2)
    array([3.])
    >>> integrate(ii, [(3, 3)], [(4, 5)])  # sum from (3, 3) to (4, 5)
    array([6.])
    >>> # sum from (1, 0) to (1, 2) and from (3, 3) to (4, 5)
    >>> integrate(ii, [(1, 0), (3, 3)], [(1, 2), (4, 5)])
    array([3., 6.])
    """
    start = np.atleast_2d(np.array(start))
    end = np.atleast_2d(np.array(end))
    rows = start.shape[0]

    total_shape = ii.shape
    total_shape = np.tile(total_shape, [rows, 1])

    # convert negative indices into equivalent positive indices
    start_negatives = start < 0
    end_negatives = end < 0
    start = (start + total_shape) * start_negatives + \
             start * ~(start_negatives)
    end = (end + total_shape) * end_negatives + \
           end * ~(end_negatives)

    if np.any((end - start) < 0):
        raise IndexError('end coordinates must be greater or equal to start')

    # bit_perm is the total number of terms in the expression
    # of S. For example, in the case of a 4x4 2D image
    # sum of image from (1,1) to (2,2) is given by
    # S = + ii[2, 2]
    #     - ii[0, 2] - ii[2, 0]
    #     + ii[0, 0]
    # The total terms = 4 = 2 ** 2(dims)

    S = np.zeros(rows)
    bit_perm = 2 ** ii.ndim
    width = len(bin(bit_perm - 1)[2:])

    # Sum of a (hyper)cube, from an integral image is computed using
    # values at the corners of the cube. The corners of cube are
    # selected using binary numbers as described in the following example.
    # In a 3D cube there are 8 corners. The corners are selected using
    # binary numbers 000 to 111. Each number is called a permutation, where
    # perm(000) means, select end corner where none of the coordinates
    # is replaced, i.e ii[end_row, end_col, end_depth]. Similarly, perm(001)
    # means replace last coordinate by start - 1, i.e
    # ii[end_row, end_col, start_depth - 1], and so on.
    # Sign of even permutations is positive, while those of odd is negative.
    # If 'start_coord - 1' is -ve it is labeled bad and not considered in
    # the final sum.

    for i in range(bit_perm):  # for all permutations
        # boolean permutation array eg [True, False] for '10'
        binary = bin(i)[2:].zfill(width)
        bool_mask = [bit == '1' for bit in binary]

        sign = (-1)**sum(bool_mask)  # determine sign of permutation

        bad = [np.any(((start[r] - 1) * bool_mask) < 0)
               for r in range(rows)]  # find out bad start rows

        corner_points = (end * (np.invert(bool_mask))) + \
                         ((start - 1) * bool_mask)  # find corner for each row

        S += [sign * ii[tuple(corner_points[r])] if(not bad[r]) else 0
              for r in range(rows)]  # add only good rows
    return S
