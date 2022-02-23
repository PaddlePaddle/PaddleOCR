import numpy as np
from ..qt import QtWidgets, has_qt, FigureManagerQT, FigureCanvasQTAgg
from ..._shared.utils import warn
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib import _pylab_helpers
from matplotlib.colors import LinearSegmentedColormap

if has_qt and 'agg' not in mpl.get_backend().lower():
    warn("Recommended matplotlib backend is `Agg` for full "
         "skimage.viewer functionality.")


__all__ = ['init_qtapp', 'start_qtapp', 'RequiredAttr', 'figimage',
           'LinearColormap', 'ClearColormap', 'FigureCanvas', 'new_plot',
           'update_axes_image']


QApp = None


def init_qtapp():
    """Initialize QAppliction.

    The QApplication needs to be initialized before creating any QWidgets
    """
    global QApp
    QApp = QtWidgets.QApplication.instance()
    if QApp is None:
        QApp = QtWidgets.QApplication([])
    return QApp


def is_event_loop_running(app=None):
    """Return True if event loop is running."""
    if app is None:
        app = init_qtapp()
    if hasattr(app, '_in_event_loop'):
        return app._in_event_loop
    else:
        return False


def start_qtapp(app=None):
    """Start Qt mainloop"""
    if app is None:
        app = init_qtapp()
    if not is_event_loop_running(app):
        app._in_event_loop = True
        app.exec_()
        app._in_event_loop = False
    else:
        app._in_event_loop = True


class RequiredAttr(object):
    """A class attribute that must be set before use."""

    instances = dict()

    def __init__(self, init_val=None):
        self.instances[self, None] = init_val

    def __get__(self, obj, objtype):
        value = self.instances[self, obj]
        if value is None:
            raise AttributeError('Required attribute not set')
        return value

    def __set__(self, obj, value):
        self.instances[self, obj] = value


class LinearColormap(LinearSegmentedColormap):
    """LinearSegmentedColormap in which color varies smoothly.

    This class is a simplification of LinearSegmentedColormap, which doesn't
    support jumps in color intensities.

    Parameters
    ----------
    name : str
        Name of colormap.

    segmented_data : dict
        Dictionary of 'red', 'green', 'blue', and (optionally) 'alpha' values.
        Each color key contains a list of `x`, `y` tuples. `x` must increase
        monotonically from 0 to 1 and corresponds to input values for a
        mappable object (e.g. an image). `y` corresponds to the color
        intensity.

    """
    def __init__(self, name, segmented_data, **kwargs):
        segmented_data = {key: [(x, y, y) for x, y in value]
                          for key, value in segmented_data.items()}
        LinearSegmentedColormap.__init__(self, name, segmented_data, **kwargs)


class ClearColormap(LinearColormap):
    """Color map that varies linearly from alpha = 0 to 1
    """
    def __init__(self, rgb, max_alpha=1, name='clear_color'):
        r, g, b = rgb
        cg_speq = {'blue':  [(0.0, b), (1.0, b)],
                   'green': [(0.0, g), (1.0, g)],
                   'red':   [(0.0, r), (1.0, r)],
                   'alpha': [(0.0, 0.0), (1.0, max_alpha)]}
        LinearColormap.__init__(self, name, cg_speq)


class FigureCanvas(FigureCanvasQTAgg):
    """Canvas for displaying images."""
    def __init__(self, figure, **kwargs):
        self.fig = figure
        FigureCanvasQTAgg.__init__(self, self.fig)
        FigureCanvasQTAgg.setSizePolicy(self,
                                        QtWidgets.QSizePolicy.Expanding,
                                        QtWidgets.QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)

    def resizeEvent(self, event):
        FigureCanvasQTAgg.resizeEvent(self, event)
        # Call to `resize_event` missing in FigureManagerQT.
        # See https://github.com/matplotlib/matplotlib/pull/1585
        self.resize_event()


def new_canvas(*args, **kwargs):
    """Return a new figure canvas."""
    allnums = _pylab_helpers.Gcf.figs.keys()
    num = max(allnums) + 1 if allnums else 1

    FigureClass = kwargs.pop('FigureClass', Figure)
    figure = FigureClass(*args, **kwargs)
    canvas = FigureCanvas(figure)
    fig_manager = FigureManagerQT(canvas, num)
    return fig_manager.canvas


def new_plot(parent=None, subplot_kw=None, **fig_kw):
    """Return new figure and axes.

    Parameters
    ----------
    parent : QtWidget
        Qt widget that displays the plot objects. If None, you must manually
        call ``canvas.setParent`` and pass the parent widget.
    subplot_kw : dict
        Keyword arguments passed ``matplotlib.figure.Figure.add_subplot``.
    fig_kw : dict
        Keyword arguments passed ``matplotlib.figure.Figure``.
    """
    if subplot_kw is None:
        subplot_kw = {}
    canvas = new_canvas(**fig_kw)
    canvas.setParent(parent)

    fig = canvas.figure
    ax = fig.add_subplot(1, 1, 1, **subplot_kw)
    return fig, ax


def figimage(image, scale=1, dpi=None, **kwargs):
    """Return figure and axes with figure tightly surrounding image.

    Unlike pyplot.figimage, this actually plots onto an axes object, which
    fills the figure. Plotting the image onto an axes allows for subsequent
    overlays of axes artists.

    Parameters
    ----------
    image : array
        image to plot
    scale : float
        If scale is 1, the figure and axes have the same dimension as the
        image.  Smaller values of `scale` will shrink the figure.
    dpi : int
        Dots per inch for figure. If None, use the default rcParam.
    """
    dpi = dpi if dpi is not None else mpl.rcParams['figure.dpi']
    kwargs.setdefault('interpolation', 'nearest')
    kwargs.setdefault('cmap', 'gray')

    h, w, d = np.atleast_3d(image).shape
    figsize = np.array((w, h), dtype=float) / dpi * scale

    fig, ax = new_plot(figsize=figsize, dpi=dpi)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)

    ax.set_axis_off()
    ax.imshow(image, **kwargs)
    ax.figure.canvas.draw()
    return fig, ax


def update_axes_image(image_axes, image):
    """Update the image displayed by an image plot.

    This sets the image plot's array and updates its shape appropriately

    Parameters
    ----------
    image_axes : `matplotlib.image.AxesImage`
        Image axes to update.
    image : array
        Image array.
    """
    image_axes.set_array(image)

    # Adjust size if new image shape doesn't match the original
    h, w = image.shape[:2]
    image_axes.set_extent((0, w, h, 0))
