import numpy as np
from matplotlib import lines

__all__ = ['CanvasToolBase', 'ToolHandles']


def _pass(*args):
    pass


class CanvasToolBase(object):
    """Base canvas tool for matplotlib axes.

    Parameters
    ----------
    manager : Viewer or PlotPlugin.
        Skimage viewer or plot plugin object.
    on_move : function
        Function called whenever a control handle is moved.
        This function must accept the end points of line as the only argument.
    on_release : function
        Function called whenever the control handle is released.
    on_enter : function
        Function called whenever the "enter" key is pressed.
    """

    def __init__(self, manager, on_move=None, on_enter=None, on_release=None,
                 useblit=True, ax=None):
        self.manager = manager
        self.ax = manager.ax
        self.artists = []
        self.active = True

        self.callback_on_move = _pass if on_move is None else on_move
        self.callback_on_enter = _pass if on_enter is None else on_enter
        self.callback_on_release = _pass if on_release is None else on_release

    def ignore(self, event):
        """Return True if event should be ignored.

        This method (or a version of it) should be called at the beginning
        of any event callback.
        """
        return not self.active

    def hit_test(self, event):
        return False

    def redraw(self):
        self.manager.redraw()

    def set_visible(self, val):
        for artist in self.artists:
            artist.set_visible(val)

    def on_key_press(self, event):
        if event.key == 'enter':
            self.callback_on_enter(self.geometry)
            self.set_visible(False)
            self.manager.redraw()

    def on_mouse_press(self, event):
        pass

    def on_mouse_release(self, event):
        pass

    def on_move(self, event):
        pass

    def on_scroll(self, event):
        pass

    def remove(self):
        self.manager.remove_tool(self)

    @property
    def geometry(self):
        """Geometry information that gets passed to callback functions."""
        return None


class ToolHandles(object):
    """Control handles for canvas tools.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        Matplotlib axes where tool handles are displayed.
    x, y : 1D arrays
        Coordinates of control handles.
    marker : str
        Shape of marker used to display handle. See `matplotlib.pyplot.plot`.
    marker_props : dict
        Additional marker properties. See :class:`matplotlib.lines.Line2D`.
    """
    def __init__(self, ax, x, y, marker='o', marker_props=None):
        self.ax = ax

        props = dict(marker=marker, markersize=7, mfc='w', ls='none',
                     alpha=0.5, visible=False)
        props.update(marker_props if marker_props is not None else {})
        self._markers = lines.Line2D(x, y, animated=True, **props)
        self.ax.add_line(self._markers)
        self.artist = self._markers

    @property
    def x(self):
        return self._markers.get_xdata()

    @property
    def y(self):
        return self._markers.get_ydata()

    def set_data(self, pts, y=None):
        """Set x and y positions of handles"""
        if y is not None:
            x = pts
            pts = np.array([x, y])
        self._markers.set_data(pts)

    def set_visible(self, val):
        self._markers.set_visible(val)

    def set_animated(self, val):
        self._markers.set_animated(val)

    def closest(self, x, y):
        """Return index and pixel distance to closest index."""
        pts = np.transpose((self.x, self.y))
        # Transform data coordinates to pixel coordinates.
        pts = self.ax.transData.transform(pts)
        diff = pts - ((x, y))
        dist = np.sqrt(np.sum(diff**2, axis=1))
        return np.argmin(dist), np.min(dist)
