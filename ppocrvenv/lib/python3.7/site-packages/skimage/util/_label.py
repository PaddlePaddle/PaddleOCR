import numpy as np

__all__ = ["label_points"]


def label_points(coords, output_shape):
    """Assign unique integer labels to coordinates on an image mask

    Parameters
    ----------
    coords: ndarray
        An array of N coordinates with dimension D
    output_shape: tuple
        The shape of the mask on which `coords` are labelled

    Returns
    -------
    labels: ndarray
        A mask of zeroes containing unique integer labels at the `coords`

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.util._label import label_points
    >>> coords = np.array([[0, 1], [2, 2]])
    >>> output_shape = (5, 5)
    >>> mask = label_points(coords, output_shape)
    >>> mask
    array([[0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 2, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]], dtype=uint64)

    Notes
    -----
    - The labels are assigned to coordinates that are converted to
      integer and considered to start from 0.
    - Coordinates that are out of range of the mask raise an IndexError.
    - Negative coordinates raise a ValueError
    """
    if coords.shape[1] != len(output_shape):
        raise ValueError("Dimensionality of points should match the "
                         "output shape")

    if np.any(coords < 0):
        raise ValueError("Coordinates should be positive and start from 0")

    np_indices = tuple(np.transpose(np.round(coords).astype(int, copy=False)))
    labels = np.zeros(output_shape, dtype=np.uint64)
    labels[np_indices] = np.arange(1, coords.shape[0] + 1)
    return labels
