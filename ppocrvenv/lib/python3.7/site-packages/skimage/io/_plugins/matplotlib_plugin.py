from collections import namedtuple
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.image
from ...util import dtype as dtypes
from ...exposure import is_low_contrast
from ..._shared.utils import warn
from math import floor, ceil


_default_colormap = 'gray'
_nonstandard_colormap = 'viridis'
_diverging_colormap = 'RdBu'


ImageProperties = namedtuple('ImageProperties',
                             ['signed', 'out_of_range_float',
                              'low_data_range', 'unsupported_dtype'])


def _get_image_properties(image):
    """Determine nonstandard properties of an input image.

    Parameters
    ----------
    image : array
        The input image.

    Returns
    -------
    ip : ImageProperties named tuple
        The properties of the image:

        - signed: whether the image has negative values.
        - out_of_range_float: if the image has floating point data
          outside of [-1, 1].
        - low_data_range: if the image is in the standard image
          range (e.g. [0, 1] for a floating point image) but its
          data range would be too small to display with standard
          image ranges.
        - unsupported_dtype: if the image data type is not a
          standard skimage type, e.g. ``numpy.uint64``.
    """
    immin, immax = np.min(image), np.max(image)
    imtype = image.dtype.type
    try:
        lo, hi = dtypes.dtype_range[imtype]
    except KeyError:
        lo, hi = immin, immax

    signed = immin < 0
    out_of_range_float = (np.issubdtype(image.dtype, np.floating) and
                          (immin < lo or immax > hi))
    low_data_range = (immin != immax and
                      is_low_contrast(image))
    unsupported_dtype = image.dtype not in dtypes._supported_types

    return ImageProperties(signed, out_of_range_float,
                           low_data_range, unsupported_dtype)


def _raise_warnings(image_properties):
    """Raise the appropriate warning for each nonstandard image type.

    Parameters
    ----------
    image_properties : ImageProperties named tuple
        The properties of the considered image.
    """
    ip = image_properties
    if ip.unsupported_dtype:
        warn("Non-standard image type; displaying image with "
             "stretched contrast.", stacklevel=3)
    if ip.low_data_range:
        warn("Low image data range; displaying image with "
             "stretched contrast.", stacklevel=3)
    if ip.out_of_range_float:
        warn("Float image out of standard range; displaying "
             "image with stretched contrast.", stacklevel=3)


def _get_display_range(image):
    """Return the display range for a given set of image properties.

    Parameters
    ----------
    image : array
        The input image.

    Returns
    -------
    lo, hi : same type as immin, immax
        The display range to be used for the input image.
    cmap : string
        The name of the colormap to use.
    """
    ip = _get_image_properties(image)
    immin, immax = np.min(image), np.max(image)
    if ip.signed:
        magnitude = max(abs(immin), abs(immax))
        lo, hi = -magnitude, magnitude
        cmap = _diverging_colormap
    elif any(ip):
        _raise_warnings(ip)
        lo, hi = immin, immax
        cmap = _nonstandard_colormap
    else:
        lo = 0
        imtype = image.dtype.type
        hi = dtypes.dtype_range[imtype][1]
        cmap = _default_colormap
    return lo, hi, cmap


def imshow(image, ax=None, show_cbar=None, **kwargs):
    """Show the input image and return the current axes.

    By default, the image is displayed in grayscale, rather than
    the matplotlib default colormap.

    Images are assumed to have standard range for their type. For
    example, if a floating point image has values in [0, 0.5], the
    most intense color will be gray50, not white.

    If the image exceeds the standard range, or if the range is too
    small to display, we fall back on displaying exactly the range of
    the input image, along with a colorbar to clearly indicate that
    this range transformation has occurred.

    For signed images, we use a diverging colormap centered at 0.

    Parameters
    ----------
    image : array, shape (M, N[, 3])
        The image to display.
    ax : `matplotlib.axes.Axes`, optional
        The axis to use for the image, defaults to plt.gca().
    show_cbar : boolean, optional.
        Whether to show the colorbar (used to override default behavior).
    **kwargs : Keyword arguments
        These are passed directly to `matplotlib.pyplot.imshow`.

    Returns
    -------
    ax_im : `matplotlib.pyplot.AxesImage`
        The `AxesImage` object returned by `plt.imshow`.
    """
    import matplotlib.pyplot as plt

    lo, hi, cmap = _get_display_range(image)

    kwargs.setdefault('interpolation', 'nearest')
    kwargs.setdefault('cmap', cmap)
    kwargs.setdefault('vmin', lo)
    kwargs.setdefault('vmax', hi)

    ax = ax or plt.gca()
    ax_im = ax.imshow(image, **kwargs)
    if (cmap != _default_colormap and show_cbar is not False) or show_cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(ax_im, cax=cax)
    ax.get_figure().tight_layout()

    return ax_im


def imshow_collection(ic, *args, **kwargs):
    """Display all images in the collection.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        The `Figure` object returned by `plt.subplots`.
    """
    import matplotlib.pyplot as plt

    if len(ic) < 1:
        raise ValueError('Number of images to plot must be greater than 0')

    # The target is to plot images on a grid with aspect ratio 4:3
    num_images = len(ic)
    # Two pairs of `nrows, ncols` are possible
    k = (num_images * 12)**0.5
    r1 = max(1, floor(k / 4))
    r2 = ceil(k / 4)
    c1 = ceil(num_images / r1)
    c2 = ceil(num_images / r2)
    # Select the one which is closer to 4:3
    if abs(r1 / c1 - 0.75) < abs(r2 / c2 - 0.75):
        nrows, ncols = r1, c1
    else:
        nrows, ncols = r2, c2

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    ax = np.asarray(axes).ravel()
    for n, image in enumerate(ic):
        ax[n].imshow(image, *args, **kwargs)
    kwargs['ax'] = axes
    return fig


imread = matplotlib.image.imread


def _app_show():
    from matplotlib.pyplot import show
    show()
