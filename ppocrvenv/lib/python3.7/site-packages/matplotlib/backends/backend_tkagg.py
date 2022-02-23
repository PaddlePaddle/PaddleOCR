from . import _backend_tk
from .backend_agg import FigureCanvasAgg
from ._backend_tk import (
    _BackendTk, FigureCanvasTk, FigureManagerTk, NavigationToolbar2Tk)


class FigureCanvasTkAgg(FigureCanvasAgg, FigureCanvasTk):
    def draw(self):
        super().draw()
        self.blit()

    def blit(self, bbox=None):
        _backend_tk.blit(self._tkphoto, self.renderer.buffer_rgba(),
                         (0, 1, 2, 3), bbox=bbox)


@_BackendTk.export
class _BackendTkAgg(_BackendTk):
    FigureCanvas = FigureCanvasTkAgg
