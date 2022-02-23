import numpy as np

from .axes_divider import make_axes_locatable, Size
from .mpl_axes import Axes


def make_rgb_axes(ax, pad=0.01, axes_class=None, **kwargs):
    """
    Parameters
    ----------
    pad : float
        Fraction of the axes height.
    """

    divider = make_axes_locatable(ax)

    pad_size = pad * Size.AxesY(ax)

    xsize = ((1-2*pad)/3) * Size.AxesX(ax)
    ysize = ((1-2*pad)/3) * Size.AxesY(ax)

    divider.set_horizontal([Size.AxesX(ax), pad_size, xsize])
    divider.set_vertical([ysize, pad_size, ysize, pad_size, ysize])

    ax.set_axes_locator(divider.new_locator(0, 0, ny1=-1))

    ax_rgb = []
    if axes_class is None:
        try:
            axes_class = ax._axes_class
        except AttributeError:
            axes_class = type(ax)

    for ny in [4, 2, 0]:
        ax1 = axes_class(ax.get_figure(), ax.get_position(original=True),
                         sharex=ax, sharey=ax, **kwargs)
        locator = divider.new_locator(nx=2, ny=ny)
        ax1.set_axes_locator(locator)
        for t in ax1.yaxis.get_ticklabels() + ax1.xaxis.get_ticklabels():
            t.set_visible(False)
        try:
            for axis in ax1.axis.values():
                axis.major_ticklabels.set_visible(False)
        except AttributeError:
            pass

        ax_rgb.append(ax1)

    fig = ax.get_figure()
    for ax1 in ax_rgb:
        fig.add_axes(ax1)

    return ax_rgb


class RGBAxes:
    """
    4-panel imshow (RGB, R, G, B).

    Layout:

        +---------------+-----+
        |               |  R  |
        +               +-----+
        |      RGB      |  G  |
        +               +-----+
        |               |  B  |
        +---------------+-----+

    Subclasses can override the ``_defaultAxesClass`` attribute.

    Attributes
    ----------
    RGB : ``_defaultAxesClass``
        The axes object for the three-channel imshow.
    R : ``_defaultAxesClass``
        The axes object for the red channel imshow.
    G : ``_defaultAxesClass``
        The axes object for the green channel imshow.
    B : ``_defaultAxesClass``
        The axes object for the blue channel imshow.
    """

    _defaultAxesClass = Axes

    def __init__(self, *args, pad=0, **kwargs):
        """
        Parameters
        ----------
        pad : float, default: 0
            fraction of the axes height to put as padding.
        axes_class : matplotlib.axes.Axes

        *args
            Unpacked into axes_class() init for RGB
        **kwargs
            Unpacked into axes_class() init for RGB, R, G, B axes
        """
        axes_class = kwargs.pop("axes_class", self._defaultAxesClass)
        self.RGB = ax = axes_class(*args, **kwargs)
        ax.get_figure().add_axes(ax)
        self.R, self.G, self.B = make_rgb_axes(
            ax, pad=pad, axes_class=axes_class, **kwargs)
        # Set the line color and ticks for the axes.
        for ax1 in [self.RGB, self.R, self.G, self.B]:
            ax1.axis[:].line.set_color("w")
            ax1.axis[:].major_ticks.set_markeredgecolor("w")

    def imshow_rgb(self, r, g, b, **kwargs):
        """
        Create the four images {rgb, r, g, b}.

        Parameters
        ----------
        r, g, b : array-like
            The red, green, and blue arrays.
        kwargs : imshow kwargs
            kwargs get unpacked into the imshow calls for the four images.

        Returns
        -------
        rgb : matplotlib.image.AxesImage
        r : matplotlib.image.AxesImage
        g : matplotlib.image.AxesImage
        b : matplotlib.image.AxesImage
        """
        if not (r.shape == g.shape == b.shape):
            raise ValueError(
                f'Input shapes ({r.shape}, {g.shape}, {b.shape}) do not match')
        RGB = np.dstack([r, g, b])
        R = np.zeros_like(RGB)
        R[:, :, 0] = r
        G = np.zeros_like(RGB)
        G[:, :, 1] = g
        B = np.zeros_like(RGB)
        B[:, :, 2] = b
        im_rgb = self.RGB.imshow(RGB, **kwargs)
        im_r = self.R.imshow(R, **kwargs)
        im_g = self.G.imshow(G, **kwargs)
        im_b = self.B.imshow(B, **kwargs)
        return im_rgb, im_r, im_g, im_b
