import numpy as np
from matplotlib import lines
from ...viewer.canvastools.base import CanvasToolBase, ToolHandles


__all__ = ['LineTool', 'ThickLineTool']


class LineTool(CanvasToolBase):
    """Widget for line selection in a plot.

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
    maxdist : float
        Maximum pixel distance allowed when selecting control handle.
    line_props : dict
        Properties for :class:`matplotlib.lines.Line2D`.
    handle_props : dict
        Marker properties for the handles (also see
        :class:`matplotlib.lines.Line2D`).

    Attributes
    ----------
    end_points : 2D array
        End points of line ((x1, y1), (x2, y2)).
    """
    def __init__(self, manager, on_move=None, on_release=None, on_enter=None,
                 maxdist=10, line_props=None, handle_props=None,
                 **kwargs):
        super(LineTool, self).__init__(manager, on_move=on_move,
                                       on_enter=on_enter,
                                       on_release=on_release, **kwargs)

        props = dict(color='r', linewidth=1, alpha=0.4, solid_capstyle='butt')
        props.update(line_props if line_props is not None else {})
        self.linewidth = props['linewidth']
        self.maxdist = maxdist
        self._active_pt = None

        x = (0, 0)
        y = (0, 0)
        self._end_pts = np.transpose([x, y])

        self._line = lines.Line2D(x, y, visible=False, animated=True, **props)
        self.ax.add_line(self._line)

        self._handles = ToolHandles(self.ax, x, y,
                                    marker_props=handle_props)
        self._handles.set_visible(False)
        self.artists = [self._line, self._handles.artist]

        if on_enter is None:
            def on_enter(pts):
                x, y = np.transpose(pts)
                print("length = %0.2f" %
                      np.sqrt(np.diff(x)**2 + np.diff(y)**2))
        self.callback_on_enter = on_enter
        self.manager.add_tool(self)

    @property
    def end_points(self):
        return self._end_pts.astype(int)

    @end_points.setter
    def end_points(self, pts):
        self._end_pts = np.asarray(pts)

        self._line.set_data(np.transpose(pts))
        self._handles.set_data(np.transpose(pts))
        self._line.set_linewidth(self.linewidth)

        self.set_visible(True)
        self.redraw()

    def hit_test(self, event):
        if event.button != 1 or not self.ax.in_axes(event):
            return False
        idx, px_dist = self._handles.closest(event.x, event.y)
        if px_dist < self.maxdist:
            self._active_pt = idx
            return True
        else:
            self._active_pt = None
            return False

    def on_mouse_press(self, event):
        self.set_visible(True)
        if self._active_pt is None:
            self._active_pt = 0
            x, y = event.xdata, event.ydata
            self._end_pts = np.array([[x, y], [x, y]])

    def on_mouse_release(self, event):
        if event.button != 1:
            return
        self._active_pt = None
        self.callback_on_release(self.geometry)
        self.redraw()

    def on_move(self, event):
        if event.button != 1 or self._active_pt is None:
            return
        if not self.ax.in_axes(event):
            return
        self.update(event.xdata, event.ydata)
        self.callback_on_move(self.geometry)

    def update(self, x=None, y=None):
        if x is not None:
            self._end_pts[self._active_pt, :] = x, y
        self.end_points = self._end_pts

    @property
    def geometry(self):
        return self.end_points


class ThickLineTool(LineTool):
    """Widget for line selection in a plot.

    The thickness of the line can be varied using the mouse scroll wheel, or
    with the '+' and '-' keys.

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
    on_change : function
        Function called whenever the line thickness is changed.
    maxdist : float
        Maximum pixel distance allowed when selecting control handle.
    line_props : dict
        Properties for :class:`matplotlib.lines.Line2D`.
    handle_props : dict
        Marker properties for the handles (also see
        :class:`matplotlib.lines.Line2D`).

    Attributes
    ----------
    end_points : 2D array
        End points of line ((x1, y1), (x2, y2)).
    """

    def __init__(self, manager, on_move=None, on_enter=None, on_release=None,
                 on_change=None, maxdist=10, line_props=None, handle_props=None):
        super(ThickLineTool, self).__init__(manager,
                                            on_move=on_move,
                                            on_enter=on_enter,
                                            on_release=on_release,
                                            maxdist=maxdist,
                                            line_props=line_props,
                                            handle_props=handle_props)

        if on_change is None:
            def on_change(*args):
                pass
        self.callback_on_change = on_change

    def on_scroll(self, event):
        if not event.inaxes:
            return
        if event.button == 'up':
            self._thicken_scan_line()
        elif event.button == 'down':
            self._shrink_scan_line()

    def on_key_press(self, event):
        if event.key == '+':
            self._thicken_scan_line()
        elif event.key == '-':
            self._shrink_scan_line()

    def _thicken_scan_line(self):
        self.linewidth += 1
        self.update()
        self.callback_on_change(self.geometry)

    def _shrink_scan_line(self):
        if self.linewidth > 1:
            self.linewidth -= 1
            self.update()
            self.callback_on_change(self.geometry)


if __name__ == '__main__':  # pragma: no cover
    from ... import data
    from ...viewer import ImageViewer

    image = data.camera()

    viewer = ImageViewer(image)
    h, w = image.shape

    line_tool = ThickLineTool(viewer)
    line_tool.end_points = ([w/3, h/2], [2*w/3, h/2])
    viewer.show()
