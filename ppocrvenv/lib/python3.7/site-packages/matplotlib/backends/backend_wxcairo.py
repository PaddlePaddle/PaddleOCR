import wx.lib.wxcairo as wxcairo

from .backend_cairo import cairo, FigureCanvasCairo, RendererCairo
from .backend_wx import (
    _BackendWx, _FigureCanvasWxBase, FigureFrameWx,
    NavigationToolbar2Wx as NavigationToolbar2WxCairo)


class FigureFrameWxCairo(FigureFrameWx):
    def get_canvas(self, fig):
        return FigureCanvasWxCairo(self, -1, fig)


class FigureCanvasWxCairo(_FigureCanvasWxBase, FigureCanvasCairo):
    """
    The FigureCanvas contains the figure and does event handling.

    In the wxPython backend, it is derived from wxPanel, and (usually) lives
    inside a frame instantiated by a FigureManagerWx. The parent window
    probably implements a wxSizer to control the displayed control size - but
    we give a hint as to our preferred minimum size.
    """

    def __init__(self, parent, id, figure):
        # _FigureCanvasWxBase should be fixed to have the same signature as
        # every other FigureCanvas and use cooperative inheritance, but in the
        # meantime the following will make do.
        _FigureCanvasWxBase.__init__(self, parent, id, figure)
        FigureCanvasCairo.__init__(self, figure)
        self._renderer = RendererCairo(self.figure.dpi)

    def draw(self, drawDC=None):
        width = int(self.figure.bbox.width)
        height = int(self.figure.bbox.height)
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        self._renderer.set_ctx_from_surface(surface)
        self._renderer.set_width_height(width, height)
        self._renderer.dpi = self.figure.dpi
        self.figure.draw(self._renderer)
        self.bitmap = wxcairo.BitmapFromImageSurface(surface)
        self._isDrawn = True
        self.gui_repaint(drawDC=drawDC)


@_BackendWx.export
class _BackendWxCairo(_BackendWx):
    FigureCanvas = FigureCanvasWxCairo
    _frame_class = FigureFrameWxCairo
