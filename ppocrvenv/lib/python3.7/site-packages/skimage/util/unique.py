import numpy as np


def unique_rows(ar):
    """Remove repeated rows from a 2D array.

    In particular, if given an array of coordinates of shape
    (Npoints, Ndim), it will remove repeated points.

    Parameters
    ----------
    ar : 2-D ndarray
        The input array.

    Returns
    -------
    ar_out : 2-D ndarray
        A copy of the input array with repeated rows removed.

    Raises
    ------
    ValueError : if `ar` is not two-dimensional.

    Notes
    -----
    The function will generate a copy of `ar` if it is not
    C-contiguous, which will negatively affect performance for large
    input arrays.

    Examples
    --------
    >>> ar = np.array([[1, 0, 1],
    ...                [0, 1, 0],
    ...                [1, 0, 1]], np.uint8)
    >>> unique_rows(ar)
    array([[0, 1, 0],
           [1, 0, 1]], dtype=uint8)
    """
    if ar.ndim != 2:
        raise ValueError("unique_rows() only makes sense for 2D arrays, "
                         "got %dd" % ar.ndim)
    # the view in the next line only works if the array is C-contiguous
    ar = np.ascontiguousarray(ar)
    # np.unique() finds identical items in a raveled array. To make it
    # see each row as a single item, we create a view of each row as a
    # byte string of length itemsize times number of columns in `ar`
    ar_row_view = ar.view('|S%d' % (ar.itemsize * ar.shape[1]))
    _, unique_row_indices = np.unique(ar_row_view, return_index=True)
    ar_out = ar[unique_row_indices]
    return ar_out
