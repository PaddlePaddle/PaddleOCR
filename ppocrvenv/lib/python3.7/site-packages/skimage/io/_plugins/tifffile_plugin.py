__all__ = ['imread', 'imsave']

from tifffile import imwrite as imsave, imread as tifffile_imread


def imread(fname, **kwargs):
    """Load a tiff image from file.

    Parameters
    ----------
    fname : str or file
        File name or file-like-object.
    kwargs : keyword pairs, optional
        Additional keyword arguments to pass through (see ``tifffile``'s
        ``imread`` function).

    Notes
    -----
    Provided by the tifffile library [1]_, and supports many
    advanced image types including multi-page and floating point.

    References
    ----------
    .. [1] https://pypi.org/project/tifffile/

    """
    if 'img_num' in kwargs:
        kwargs['key'] = kwargs.pop('img_num')

    return tifffile_imread(fname, **kwargs)
