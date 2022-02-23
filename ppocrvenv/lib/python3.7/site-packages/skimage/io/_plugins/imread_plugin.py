__all__ = ['imread', 'imsave']

from ...util.dtype import _convert

try:
    import imread as _imread
except ImportError:
    raise ImportError("Imread could not be found"
                      "Please refer to http://pypi.python.org/pypi/imread/ "
                      "for further instructions.")


def imread(fname, dtype=None):
    """Load an image from file.

    Parameters
    ----------
    fname : str
        Name of input file

    """
    im = _imread.imread(fname)
    if dtype is not None:
        im = _convert(im, dtype)
    return im


def imsave(fname, arr, format_str=None):
    """Save an image to disk.

    Parameters
    ----------
    fname : str
        Name of destination file.
    arr : ndarray of uint8 or uint16
        Array (image) to save.
    format_str : str,optional
        Format to save as.

    Notes
    -----
    Currently, only 8-bit precision is supported.
    """
    return _imread.imsave(fname, arr, formatstr=format_str)
