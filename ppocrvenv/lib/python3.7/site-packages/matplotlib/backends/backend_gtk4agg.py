import numpy as np

from .. import cbook
try:
    from . import backend_cairo
except ImportError as e:
    raise ImportError('backend Gtk4Agg requires cairo') from e
from . import backend_agg, backend_gtk4
from .backend_cairo import cairo
from .backend_gtk4 import Gtk, _BackendGTK4


class FigureCanvasGTK4Agg(backend_gtk4.FigureCanvasGTK4,
                          backend_agg.FigureCanvasAgg):
    def __init__(self, figure):
        backend_gtk4.FigureCanvasGTK4.__init__(self, figure)

    def on_draw_event(self, widget, ctx):
        scale = self.device_pixel_ratio
        allocation = self.get_allocation()

        Gtk.render_background(
            self.get_style_context(), ctx,
            allocation.x, allocation.y,
            allocation.width, allocation.height)

        ctx = backend_cairo._to_context(ctx)

        buf = cbook._unmultiplied_rgba8888_to_premultiplied_argb32(
            np.asarray(self.get_renderer().buffer_rgba()))
        height, width, _ = buf.shape
        image = cairo.ImageSurface.create_for_data(
            buf.ravel().data, cairo.FORMAT_ARGB32, width, height)
        image.set_device_scale(scale, scale)
        ctx.set_source_surface(image, 0, 0)
        ctx.paint()

        return False

    def draw(self):
        # Call these explicitly because GTK's draw is a GObject method which
        # isn't cooperative with Python class methods.
        backend_agg.FigureCanvasAgg.draw(self)
        backend_gtk4.FigureCanvasGTK4.draw(self)


class FigureManagerGTK4Agg(backend_gtk4.FigureManagerGTK4):
    pass


@_BackendGTK4.export
class _BackendGTK4Agg(_BackendGTK4):
    FigureCanvas = FigureCanvasGTK4Agg
    FigureManager = FigureManagerGTK4Agg
