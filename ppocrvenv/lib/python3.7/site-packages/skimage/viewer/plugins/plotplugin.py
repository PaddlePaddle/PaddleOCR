import numpy as np

from ..qt import QtGui
from ..utils import new_plot
from ..utils.canvas import BlitManager, EventManager
from .base import Plugin


__all__ = ['PlotPlugin']


class PlotPlugin(Plugin):
    """Plugin for ImageViewer that contains a plot canvas.

    Base class for plugins that contain a Matplotlib plot canvas, which can,
    for example, display an image histogram.

    See base Plugin class for additional details.
    """

    def __init__(self, image_filter=None, height=150, width=400, **kwargs):
        super(PlotPlugin, self).__init__(image_filter=image_filter,
                                         height=height, width=width, **kwargs)

        self._height = height
        self._width = width
        self._blit_manager = None
        self._tools = []
        self._event_manager = None

    def attach(self, image_viewer):
        super(PlotPlugin, self).attach(image_viewer)
        # Add plot for displaying intensity profile.
        self.add_plot()
        if image_viewer.useblit:
            self._blit_manager = BlitManager(self.ax)
        self._event_manager = EventManager(self.ax)

    def redraw(self):
        """Redraw plot."""
        self.canvas.draw_idle()

    def add_plot(self):
        self.fig, self.ax = new_plot()
        self.fig.set_figwidth(self._width / float(self.fig.dpi))
        self.fig.set_figheight(self._height / float(self.fig.dpi))

        self.canvas = self.fig.canvas
        #TODO: Converted color is slightly different than Qt background.
        qpalette = QtGui.QPalette()
        qcolor = qpalette.color(QtGui.QPalette.Window)
        bgcolor = qcolor.toRgb().value()
        if np.isscalar(bgcolor):
            bgcolor = str(bgcolor / 255.)
        self.fig.patch.set_facecolor(bgcolor)
        self.layout.addWidget(self.canvas, self.row, 0)

    def _update_original_image(self, image):
        super(PlotPlugin, self)._update_original_image(image)
        self.redraw()

    def add_tool(self, tool):
        if self._blit_manager:
            self._blit_manager.add_artists(tool.artists)
        self._tools.append(tool)
        self._event_manager.attach(tool)

    def remove_tool(self, tool):
        if tool not in self._tools:
            return
        if self._blit_manager:
            self._blit_manager.remove_artists(tool.artists)
        self._tools.remove(tool)
        self._event_manager.detach(tool)
