from matplotlib.widgets import RectangleSelector
from ...viewer.canvastools.base import CanvasToolBase
from ...viewer.canvastools.base import ToolHandles


__all__ = ['RectangleTool']


class RectangleTool(CanvasToolBase, RectangleSelector):
    """Widget for selecting a rectangular region in a plot.

    After making the desired selection, press "Enter" to accept the selection
    and call the `on_enter` callback function.

    Parameters
    ----------
    manager : Viewer or PlotPlugin.
        Skimage viewer or plot plugin object.
    on_move : function
        Function called whenever a control handle is moved.
        This function must accept the rectangle extents as the only argument.
    on_release : function
        Function called whenever the control handle is released.
    on_enter : function
        Function called whenever the "enter" key is pressed.
    maxdist : float
        Maximum pixel distance allowed when selecting control handle.
    rect_props : dict
        Properties for :class:`matplotlib.patches.Rectangle`. This class
        redefines defaults in :class:`matplotlib.widgets.RectangleSelector`.

    Attributes
    ----------
    extents : tuple
        Rectangle extents: (xmin, xmax, ymin, ymax).

    Examples
    ----------
    >>> from skimage import data
    >>> from skimage.viewer import ImageViewer
    >>> from skimage.viewer.canvastools import RectangleTool
    >>> from skimage.draw import line
    >>> from skimage.draw import set_color

    >>> viewer = ImageViewer(data.coffee())  # doctest: +SKIP

    >>> def print_the_rect(extents):
    ...     global viewer
    ...     im = viewer.image
    ...     coord = np.int64(extents)
    ...     [rr1, cc1] = line(coord[2],coord[0],coord[2],coord[1])
    ...     [rr2, cc2] = line(coord[2],coord[1],coord[3],coord[1])
    ...     [rr3, cc3] = line(coord[3],coord[1],coord[3],coord[0])
    ...     [rr4, cc4] = line(coord[3],coord[0],coord[2],coord[0])
    ...     set_color(im, (rr1, cc1), [255, 255, 0])
    ...     set_color(im, (rr2, cc2), [0, 255, 255])
    ...     set_color(im, (rr3, cc3), [255, 0, 255])
    ...     set_color(im, (rr4, cc4), [0, 0, 0])
    ...     viewer.image=im

    >>> rect_tool = RectangleTool(viewer, on_enter=print_the_rect) # doctest: +SKIP
    >>> viewer.show() # doctest: +SKIP
    """

    def __init__(self, manager, on_move=None, on_release=None, on_enter=None,
                 maxdist=10, rect_props=None):
        self._rect = None
        props = dict(edgecolor=None, facecolor='r', alpha=0.15)
        props.update(rect_props if rect_props is not None else {})
        if props['edgecolor'] is None:
            props['edgecolor'] = props['facecolor']
        RectangleSelector.__init__(self, manager.ax, lambda *args: None,
                                   rectprops=props)
        CanvasToolBase.__init__(self, manager, on_move=on_move,
                                on_enter=on_enter, on_release=on_release)

        # Events are handled by the viewer
        try:
            self.disconnect_events()
        except AttributeError:
            # disconnect the events manually (hack for older mpl versions)
            [self.canvas.mpl_disconnect(i) for i in range(10)]

        # Alias rectangle attribute, which is initialized in RectangleSelector.
        self._rect = self.to_draw
        self._rect.set_animated(True)

        self.maxdist = maxdist
        self.active_handle = None
        self._extents_on_press = None

        if on_enter is None:
            def on_enter(extents):
                print("(xmin=%.3g, xmax=%.3g, ymin=%.3g, ymax=%.3g)" % extents)
        self.callback_on_enter = on_enter

        props = dict(mec=props['edgecolor'])
        self._corner_order = ['NW', 'NE', 'SE', 'SW']
        xc, yc = self.corners
        self._corner_handles = ToolHandles(self.ax, xc, yc, marker_props=props)

        self._edge_order = ['W', 'N', 'E', 'S']
        xe, ye = self.edge_centers
        self._edge_handles = ToolHandles(self.ax, xe, ye, marker='s',
                                         marker_props=props)

        self.artists = [self._rect,
                        self._corner_handles.artist,
                        self._edge_handles.artist]
        self.manager.add_tool(self)

    @property
    def _rect_bbox(self):
        if not self._rect:
            return 0, 0, 0, 0
        x0 = self._rect.get_x()
        y0 = self._rect.get_y()
        width = self._rect.get_width()
        height = self._rect.get_height()
        return x0, y0, width, height

    @property
    def corners(self):
        """Corners of rectangle from lower left, moving clockwise."""
        x0, y0, width, height = self._rect_bbox
        xc = x0, x0 + width, x0 + width, x0
        yc = y0, y0, y0 + height, y0 + height
        return xc, yc

    @property
    def edge_centers(self):
        """Midpoint of rectangle edges from left, moving clockwise."""
        x0, y0, width, height = self._rect_bbox
        w = width / 2.
        h = height / 2.
        xe = x0, x0 + w, x0 + width, x0 + w
        ye = y0 + h, y0, y0 + h, y0 + height
        return xe, ye

    @property
    def extents(self):
        """Return (xmin, xmax, ymin, ymax)."""
        x0, y0, width, height = self._rect_bbox
        xmin, xmax = sorted([x0, x0 + width])
        ymin, ymax = sorted([y0, y0 + height])
        return xmin, xmax, ymin, ymax

    @extents.setter
    def extents(self, extents):
        x1, x2, y1, y2 = extents
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])
        # Update displayed rectangle
        self._rect.set_x(xmin)
        self._rect.set_y(ymin)
        self._rect.set_width(xmax - xmin)
        self._rect.set_height(ymax - ymin)
        # Update displayed handles
        self._corner_handles.set_data(*self.corners)
        self._edge_handles.set_data(*self.edge_centers)

        self.set_visible(True)
        self.redraw()

    def on_mouse_release(self, event):
        if event.button != 1:
            return
        if not self.ax.in_axes(event):
            self.eventpress = None
            return
        RectangleSelector.release(self, event)
        self._extents_on_press = None
        # Undo hiding of rectangle and redraw.
        self.set_visible(True)
        self.redraw()
        self.callback_on_release(self.geometry)

    def on_mouse_press(self, event):
        if event.button != 1 or not self.ax.in_axes(event):
            return
        self._set_active_handle(event)
        if self.active_handle is None:
            # Clear previous rectangle before drawing new rectangle.
            self.set_visible(False)
            self.redraw()
        self.set_visible(True)
        RectangleSelector.press(self, event)

    def _set_active_handle(self, event):
        """Set active handle based on the location of the mouse event"""
        # Note: event.xdata/ydata in data coordinates, event.x/y in pixels
        c_idx, c_dist = self._corner_handles.closest(event.x, event.y)
        e_idx, e_dist = self._edge_handles.closest(event.x, event.y)

        # Set active handle as closest handle, if mouse click is close enough.
        if c_dist > self.maxdist and e_dist > self.maxdist:
            self.active_handle = None
            return
        elif c_dist < e_dist:
            self.active_handle = self._corner_order[c_idx]
        else:
            self.active_handle = self._edge_order[e_idx]

        # Save coordinates of rectangle at the start of handle movement.
        x1, x2, y1, y2 = self.extents
        # Switch variables so that only x2 and/or y2 are updated on move.
        if self.active_handle in ['W', 'SW', 'NW']:
            x1, x2 = x2, event.xdata
        if self.active_handle in ['N', 'NW', 'NE']:
            y1, y2 = y2, event.ydata
        self._extents_on_press = x1, x2, y1, y2

    def on_move(self, event):
        if self.eventpress is None or not self.ax.in_axes(event):
            return

        if self.active_handle is None:
            # New rectangle
            x1 = self.eventpress.xdata
            y1 = self.eventpress.ydata
            x2, y2 = event.xdata, event.ydata
        else:
            x1, x2, y1, y2 = self._extents_on_press
            if self.active_handle in ['E', 'W'] + self._corner_order:
                x2 = event.xdata
            if self.active_handle in ['N', 'S'] + self._corner_order:
                y2 = event.ydata
        self.extents = (x1, x2, y1, y2)
        self.callback_on_move(self.geometry)

    @property
    def geometry(self):
        return self.extents


if __name__ == '__main__':  # pragma: no cover
    from ...viewer import ImageViewer
    from ... import data

    viewer = ImageViewer(data.camera())

    rect_tool = RectangleTool(viewer)
    viewer.show()
    print("Final selection:")
    rect_tool.callback_on_enter(rect_tool.extents)
