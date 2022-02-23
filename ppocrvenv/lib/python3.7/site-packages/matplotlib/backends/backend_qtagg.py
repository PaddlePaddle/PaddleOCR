"""
Render to qt from agg.
"""

import ctypes

from matplotlib.transforms import Bbox

from .qt_compat import QT_API, _enum, _setDevicePixelRatio
from .. import cbook
from .backend_agg import FigureCanvasAgg
from .backend_qt import (
    QtCore, QtGui, QtWidgets, _BackendQT, FigureCanvasQT, FigureManagerQT,
    NavigationToolbar2QT, backend_version)


class FigureCanvasQTAgg(FigureCanvasAgg, FigureCanvasQT):

    def __init__(self, figure=None):
        # Must pass 'figure' as kwarg to Qt base class.
        super().__init__(figure=figure)

    def paintEvent(self, event):
        """
        Copy the image from the Agg canvas to the qt.drawable.

        In Qt, all drawing should be done inside of here when a widget is
        shown onscreen.
        """
        self._draw_idle()  # Only does something if a draw is pending.

        # If the canvas does not have a renderer, then give up and wait for
        # FigureCanvasAgg.draw(self) to be called.
        if not hasattr(self, 'renderer'):
            return

        painter = QtGui.QPainter(self)
        try:
            # See documentation of QRect: bottom() and right() are off
            # by 1, so use left() + width() and top() + height().
            rect = event.rect()
            # scale rect dimensions using the screen dpi ratio to get
            # correct values for the Figure coordinates (rather than
            # QT5's coords)
            width = rect.width() * self.device_pixel_ratio
            height = rect.height() * self.device_pixel_ratio
            left, top = self.mouseEventCoords(rect.topLeft())
            # shift the "top" by the height of the image to get the
            # correct corner for our coordinate system
            bottom = top - height
            # same with the right side of the image
            right = left + width
            # create a buffer using the image bounding box
            bbox = Bbox([[left, bottom], [right, top]])
            reg = self.copy_from_bbox(bbox)
            buf = cbook._unmultiplied_rgba8888_to_premultiplied_argb32(
                memoryview(reg))

            # clear the widget canvas
            painter.eraseRect(rect)

            if QT_API == "PyQt6":
                from PyQt6 import sip
                ptr = int(sip.voidptr(buf))
            else:
                ptr = buf
            qimage = QtGui.QImage(
                ptr, buf.shape[1], buf.shape[0],
                _enum("QtGui.QImage.Format").Format_ARGB32_Premultiplied)
            _setDevicePixelRatio(qimage, self.device_pixel_ratio)
            # set origin using original QT coordinates
            origin = QtCore.QPoint(rect.left(), rect.top())
            painter.drawImage(origin, qimage)
            # Adjust the buf reference count to work around a memory
            # leak bug in QImage under PySide.
            if QT_API in ('PySide', 'PySide2'):
                if QtCore.__version_info__ < (5, 12):
                    ctypes.c_long.from_address(id(buf)).value = 1

            self._draw_rect_callback(painter)
        finally:
            painter.end()

    def print_figure(self, *args, **kwargs):
        super().print_figure(*args, **kwargs)
        self.draw()


@_BackendQT.export
class _BackendQTAgg(_BackendQT):
    FigureCanvas = FigureCanvasQTAgg
