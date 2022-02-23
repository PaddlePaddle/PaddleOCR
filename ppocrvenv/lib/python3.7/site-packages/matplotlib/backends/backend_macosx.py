import matplotlib as mpl
from matplotlib import cbook
from matplotlib._pylab_helpers import Gcf
from matplotlib.backends import _macosx
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backend_bases import (
    _Backend, FigureCanvasBase, FigureManagerBase, NavigationToolbar2,
    TimerBase)
from matplotlib.figure import Figure
from matplotlib.widgets import SubplotTool


class TimerMac(_macosx.Timer, TimerBase):
    """Subclass of `.TimerBase` using CFRunLoop timer events."""
    # completely implemented at the C-level (in _macosx.Timer)


class FigureCanvasMac(_macosx.FigureCanvas, FigureCanvasAgg):
    # docstring inherited

    # Events such as button presses, mouse movements, and key presses
    # are handled in the C code and the base class methods
    # button_press_event, button_release_event, motion_notify_event,
    # key_press_event, and key_release_event are called from there.

    required_interactive_framework = "macosx"
    _timer_cls = TimerMac

    def __init__(self, figure):
        FigureCanvasBase.__init__(self, figure)
        width, height = self.get_width_height()
        _macosx.FigureCanvas.__init__(self, width, height)

    def set_cursor(self, cursor):
        # docstring inherited
        _macosx.set_cursor(cursor)

    def _draw(self):
        renderer = self.get_renderer(cleared=self.figure.stale)
        if self.figure.stale:
            self.figure.draw(renderer)
        return renderer

    def draw(self):
        # docstring inherited
        self._draw()
        self.flush_events()

    # draw_idle is provided by _macosx.FigureCanvas

    def blit(self, bbox=None):
        self.draw_idle()

    def resize(self, width, height):
        # Size from macOS is logical pixels, dpi is physical.
        scale = self.figure.dpi / self.device_pixel_ratio
        width /= scale
        height /= scale
        self.figure.set_size_inches(width, height, forward=False)
        FigureCanvasBase.resize_event(self)
        self.draw_idle()


class FigureManagerMac(_macosx.FigureManager, FigureManagerBase):
    """
    Wrap everything up into a window for the pylab interface
    """
    def __init__(self, canvas, num):
        _macosx.FigureManager.__init__(self, canvas)
        FigureManagerBase.__init__(self, canvas, num)
        if mpl.rcParams['toolbar'] == 'toolbar2':
            self.toolbar = NavigationToolbar2Mac(canvas)
        else:
            self.toolbar = None
        if self.toolbar is not None:
            self.toolbar.update()

        if mpl.is_interactive():
            self.show()
            self.canvas.draw_idle()

    def close(self):
        Gcf.destroy(self)


class NavigationToolbar2Mac(_macosx.NavigationToolbar2, NavigationToolbar2):

    def __init__(self, canvas):
        self.canvas = canvas  # Needed by the _macosx __init__.
        data_path = cbook._get_data_path('images')
        _, tooltips, image_names, _ = zip(*NavigationToolbar2.toolitems)
        _macosx.NavigationToolbar2.__init__(
            self,
            tuple(str(data_path / image_name) + ".pdf"
                  for image_name in image_names if image_name is not None),
            tuple(tooltip for tooltip in tooltips if tooltip is not None))
        NavigationToolbar2.__init__(self, canvas)

    def draw_rubberband(self, event, x0, y0, x1, y1):
        self.canvas.set_rubberband(int(x0), int(y0), int(x1), int(y1))

    def remove_rubberband(self):
        self.canvas.remove_rubberband()

    def save_figure(self, *args):
        filename = _macosx.choose_save_file('Save the figure',
                                            self.canvas.get_default_filename())
        if filename is None:  # Cancel
            return
        self.canvas.figure.savefig(filename)

    def prepare_configure_subplots(self):
        toolfig = Figure(figsize=(6, 3))
        canvas = FigureCanvasMac(toolfig)
        toolfig.subplots_adjust(top=0.9)
        # Need to keep a reference to the tool.
        _tool = SubplotTool(self.canvas.figure, toolfig)
        return canvas

    def set_message(self, message):
        _macosx.NavigationToolbar2.set_message(self, message.encode('utf-8'))


@_Backend.export
class _BackendMac(_Backend):
    FigureCanvas = FigureCanvasMac
    FigureManager = FigureManagerMac

    @staticmethod
    def mainloop():
        _macosx.show()
