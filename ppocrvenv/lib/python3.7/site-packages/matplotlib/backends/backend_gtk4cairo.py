from contextlib import nullcontext

from . import backend_cairo, backend_gtk4
from .backend_gtk4 import Gtk, _BackendGTK4


class RendererGTK4Cairo(backend_cairo.RendererCairo):
    def set_context(self, ctx):
        self.gc.ctx = backend_cairo._to_context(ctx)


class FigureCanvasGTK4Cairo(backend_gtk4.FigureCanvasGTK4,
                            backend_cairo.FigureCanvasCairo):
    _context_is_scaled = True

    def __init__(self, figure):
        super().__init__(figure)
        self._renderer = RendererGTK4Cairo(self.figure.dpi)

    def on_draw_event(self, widget, ctx):
        with (self.toolbar._wait_cursor_for_draw_cm() if self.toolbar
              else nullcontext()):
            self._renderer.set_context(ctx)
            scale = self.device_pixel_ratio
            # Scale physical drawing to logical size.
            ctx.scale(1 / scale, 1 / scale)
            allocation = self.get_allocation()
            Gtk.render_background(
                self.get_style_context(), ctx,
                allocation.x, allocation.y,
                allocation.width, allocation.height)
            self._renderer.set_width_height(
                allocation.width * scale, allocation.height * scale)
            self._renderer.dpi = self.figure.dpi
            self.figure.draw(self._renderer)


@_BackendGTK4.export
class _BackendGTK4Cairo(_BackendGTK4):
    FigureCanvas = FigureCanvasGTK4Cairo
