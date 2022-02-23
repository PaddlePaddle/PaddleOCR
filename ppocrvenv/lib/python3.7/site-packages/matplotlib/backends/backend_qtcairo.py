import ctypes

from .backend_cairo import cairo, FigureCanvasCairo, RendererCairo
from .backend_qt import QtCore, QtGui, _BackendQT, FigureCanvasQT
from .qt_compat import QT_API, _enum, _setDevicePixelRatio


class FigureCanvasQTCairo(FigureCanvasQT, FigureCanvasCairo):
    def __init__(self, figure=None):
        super().__init__(figure=figure)
        self._renderer = RendererCairo(self.figure.dpi)
        self._renderer.set_width_height(-1, -1)  # Invalid values.

    def draw(self):
        if hasattr(self._renderer.gc, "ctx"):
            self._renderer.dpi = self.figure.dpi
            self.figure.draw(self._renderer)
        super().draw()

    def paintEvent(self, event):
        width = int(self.device_pixel_ratio * self.width())
        height = int(self.device_pixel_ratio * self.height())
        if (width, height) != self._renderer.get_canvas_width_height():
            surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
            self._renderer.set_ctx_from_surface(surface)
            self._renderer.set_width_height(width, height)
            self._renderer.dpi = self.figure.dpi
            self.figure.draw(self._renderer)
        buf = self._renderer.gc.ctx.get_target().get_data()
        if QT_API == "PyQt6":
            from PyQt6 import sip
            ptr = int(sip.voidptr(buf))
        else:
            ptr = buf
        qimage = QtGui.QImage(
            ptr, width, height,
            _enum("QtGui.QImage.Format").Format_ARGB32_Premultiplied)
        # Adjust the buf reference count to work around a memory leak bug in
        # QImage under PySide.
        if QT_API in ('PySide', 'PySide2'):
            if QtCore.__version_info__ < (5, 12):
                ctypes.c_long.from_address(id(buf)).value = 1
        _setDevicePixelRatio(qimage, self.device_pixel_ratio)
        painter = QtGui.QPainter(self)
        painter.eraseRect(event.rect())
        painter.drawImage(0, 0, qimage)
        self._draw_rect_callback(painter)
        painter.end()


@_BackendQT.export
class _BackendQTCairo(_BackendQT):
    FigureCanvas = FigureCanvasQTCairo
