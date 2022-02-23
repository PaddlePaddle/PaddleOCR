import numpy as np
import sys


def lookfor(what):
    """Do a keyword search on scikit-image docstrings.

    Parameters
    ----------
    what : str
        Words to look for.

    Examples
    --------
    >>> import skimage
    >>> skimage.lookfor('regular_grid')  # doctest: +SKIP
    Search results for 'regular_grid'
    ---------------------------------
    skimage.lookfor
        Do a keyword search on scikit-image docstrings.
    skimage.util.regular_grid
        Find `n_points` regularly spaced along `ar_shape`.
    """
    return np.lookfor(what, sys.modules[__name__.split('.')[0]])
