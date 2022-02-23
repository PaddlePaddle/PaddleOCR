"""
Displays Agg images in the browser, with interactivity
"""
# The WebAgg backend is divided into two modules:
#
# - `backend_webagg_core.py` contains code necessary to embed a WebAgg
#   plot inside of a web application, and communicate in an abstract
#   way over a web socket.
#
# - `backend_webagg.py` contains a concrete implementation of a basic
#   application, implemented with asyncio.

import asyncio
import datetime
from io import BytesIO, StringIO
import json
import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image

from matplotlib import _api, backend_bases, backend_tools
from matplotlib.backends import backend_agg
from matplotlib.backend_bases import _Backend

_log = logging.getLogger(__name__)

_SPECIAL_KEYS_LUT = {'Alt': 'alt',
                     'AltGraph': 'alt',
                     'CapsLock': 'caps_lock',
                     'Control': 'control',
                     'Meta': 'meta',
                     'NumLock': 'num_lock',
                     'ScrollLock': 'scroll_lock',
                     'Shift': 'shift',
                     'Super': 'super',
                     'Enter': 'enter',
                     'Tab': 'tab',
                     'ArrowDown': 'down',
                     'ArrowLeft': 'left',
                     'ArrowRight': 'right',
                     'ArrowUp': 'up',
                     'End': 'end',
                     'Home': 'home',
                     'PageDown': 'pagedown',
                     'PageUp': 'pageup',
                     'Backspace': 'backspace',
                     'Delete': 'delete',
                     'Insert': 'insert',
                     'Escape': 'escape',
                     'Pause': 'pause',
                     'Select': 'select',
                     'Dead': 'dead',
                     'F1': 'f1',
                     'F2': 'f2',
                     'F3': 'f3',
                     'F4': 'f4',
                     'F5': 'f5',
                     'F6': 'f6',
                     'F7': 'f7',
                     'F8': 'f8',
                     'F9': 'f9',
                     'F10': 'f10',
                     'F11': 'f11',
                     'F12': 'f12'}


def _handle_key(key):
    """Handle key values"""
    value = key[key.index('k') + 1:]
    if 'shift+' in key:
        if len(value) == 1:
            key = key.replace('shift+', '')
    if value in _SPECIAL_KEYS_LUT:
        value = _SPECIAL_KEYS_LUT[value]
    key = key[:key.index('k')] + value
    return key


class TimerTornado(backend_bases.TimerBase):
    def __init__(self, *args, **kwargs):
        self._timer = None
        super().__init__(*args, **kwargs)

    def _timer_start(self):
        import tornado

        self._timer_stop()
        if self._single:
            ioloop = tornado.ioloop.IOLoop.instance()
            self._timer = ioloop.add_timeout(
                datetime.timedelta(milliseconds=self.interval),
                self._on_timer)
        else:
            self._timer = tornado.ioloop.PeriodicCallback(
                self._on_timer,
                max(self.interval, 1e-6))
            self._timer.start()

    def _timer_stop(self):
        import tornado

        if self._timer is None:
            return
        elif self._single:
            ioloop = tornado.ioloop.IOLoop.instance()
            ioloop.remove_timeout(self._timer)
        else:
            self._timer.stop()
        self._timer = None

    def _timer_set_interval(self):
        # Only stop and restart it if the timer has already been started
        if self._timer is not None:
            self._timer_stop()
            self._timer_start()


class TimerAsyncio(backend_bases.TimerBase):
    def __init__(self, *args, **kwargs):
        self._task = None
        super().__init__(*args, **kwargs)

    async def _timer_task(self, interval):
        while True:
            try:
                await asyncio.sleep(interval)
                self._on_timer()

                if self._single:
                    break
            except asyncio.CancelledError:
                break

    def _timer_start(self):
        self._timer_stop()

        self._task = asyncio.ensure_future(
            self._timer_task(max(self.interval / 1_000., 1e-6))
        )

    def _timer_stop(self):
        if self._task is not None:
            self._task.cancel()
        self._task = None

    def _timer_set_interval(self):
        # Only stop and restart it if the timer has already been started
        if self._task is not None:
            self._timer_stop()
            self._timer_start()


class FigureCanvasWebAggCore(backend_agg.FigureCanvasAgg):
    _timer_cls = TimerAsyncio
    # Webagg and friends having the right methods, but still
    # having bugs in practice.  Do not advertise that it works until
    # we can debug this.
    supports_blit = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set to True when the renderer contains data that is newer
        # than the PNG buffer.
        self._png_is_old = True

        # Set to True by the `refresh` message so that the next frame
        # sent to the clients will be a full frame.
        self._force_full = True

        # Store the current image mode so that at any point, clients can
        # request the information. This should be changed by calling
        # self.set_image_mode(mode) so that the notification can be given
        # to the connected clients.
        self._current_image_mode = 'full'

    def show(self):
        # show the figure window
        from matplotlib.pyplot import show
        show()

    def draw(self):
        self._png_is_old = True
        try:
            super().draw()
        finally:
            self.manager.refresh_all()  # Swap the frames.

    def blit(self, bbox=None):
        self._png_is_old = True
        self.manager.refresh_all()

    def draw_idle(self):
        self.send_event("draw")

    def set_cursor(self, cursor):
        # docstring inherited
        cursor = _api.check_getitem({
            backend_tools.Cursors.HAND: 'pointer',
            backend_tools.Cursors.POINTER: 'default',
            backend_tools.Cursors.SELECT_REGION: 'crosshair',
            backend_tools.Cursors.MOVE: 'move',
            backend_tools.Cursors.WAIT: 'wait',
            backend_tools.Cursors.RESIZE_HORIZONTAL: 'ew-resize',
            backend_tools.Cursors.RESIZE_VERTICAL: 'ns-resize',
        }, cursor=cursor)
        self.send_event('cursor', cursor=cursor)

    def set_image_mode(self, mode):
        """
        Set the image mode for any subsequent images which will be sent
        to the clients. The modes may currently be either 'full' or 'diff'.

        Note: diff images may not contain transparency, therefore upon
        draw this mode may be changed if the resulting image has any
        transparent component.
        """
        _api.check_in_list(['full', 'diff'], mode=mode)
        if self._current_image_mode != mode:
            self._current_image_mode = mode
            self.handle_send_image_mode(None)

    def get_diff_image(self):
        if self._png_is_old:
            renderer = self.get_renderer()

            # The buffer is created as type uint32 so that entire
            # pixels can be compared in one numpy call, rather than
            # needing to compare each plane separately.
            buff = (np.frombuffer(renderer.buffer_rgba(), dtype=np.uint32)
                    .reshape((renderer.height, renderer.width)))

            # If any pixels have transparency, we need to force a full
            # draw as we cannot overlay new on top of old.
            pixels = buff.view(dtype=np.uint8).reshape(buff.shape + (4,))

            if self._force_full or np.any(pixels[:, :, 3] != 255):
                self.set_image_mode('full')
                output = buff
            else:
                self.set_image_mode('diff')
                diff = buff != self._last_buff
                output = np.where(diff, buff, 0)

            # Store the current buffer so we can compute the next diff.
            np.copyto(self._last_buff, buff)
            self._force_full = False
            self._png_is_old = False

            data = output.view(dtype=np.uint8).reshape((*output.shape, 4))
            with BytesIO() as png:
                Image.fromarray(data).save(png, format="png")
                return png.getvalue()

    def get_renderer(self, cleared=None):
        # Mirrors super.get_renderer, but caches the old one so that we can do
        # things such as produce a diff image in get_diff_image.
        w, h = self.figure.bbox.size.astype(int)
        key = w, h, self.figure.dpi
        try:
            self._lastKey, self._renderer
        except AttributeError:
            need_new_renderer = True
        else:
            need_new_renderer = (self._lastKey != key)

        if need_new_renderer:
            self._renderer = backend_agg.RendererAgg(
                w, h, self.figure.dpi)
            self._lastKey = key
            self._last_buff = np.copy(np.frombuffer(
                self._renderer.buffer_rgba(), dtype=np.uint32
            ).reshape((self._renderer.height, self._renderer.width)))

        elif cleared:
            self._renderer.clear()

        return self._renderer

    def handle_event(self, event):
        e_type = event['type']
        handler = getattr(self, 'handle_{0}'.format(e_type),
                          self.handle_unknown_event)
        return handler(event)

    def handle_unknown_event(self, event):
        _log.warning('Unhandled message type {0}. {1}'.format(
                     event['type'], event))

    def handle_ack(self, event):
        # Network latency tends to decrease if traffic is flowing
        # in both directions.  Therefore, the browser sends back
        # an "ack" message after each image frame is received.
        # This could also be used as a simple sanity check in the
        # future, but for now the performance increase is enough
        # to justify it, even if the server does nothing with it.
        pass

    def handle_draw(self, event):
        self.draw()

    def _handle_mouse(self, event):
        x = event['x']
        y = event['y']
        y = self.get_renderer().height - y

        # Javascript button numbers and matplotlib button numbers are
        # off by 1
        button = event['button'] + 1

        e_type = event['type']
        guiEvent = event.get('guiEvent', None)
        if e_type == 'button_press':
            self.button_press_event(x, y, button, guiEvent=guiEvent)
        elif e_type == 'dblclick':
            self.button_press_event(x, y, button, dblclick=True,
                                    guiEvent=guiEvent)
        elif e_type == 'button_release':
            self.button_release_event(x, y, button, guiEvent=guiEvent)
        elif e_type == 'motion_notify':
            self.motion_notify_event(x, y, guiEvent=guiEvent)
        elif e_type == 'figure_enter':
            self.enter_notify_event(xy=(x, y), guiEvent=guiEvent)
        elif e_type == 'figure_leave':
            self.leave_notify_event()
        elif e_type == 'scroll':
            self.scroll_event(x, y, event['step'], guiEvent=guiEvent)
    handle_button_press = handle_button_release = handle_dblclick = \
        handle_figure_enter = handle_figure_leave = handle_motion_notify = \
        handle_scroll = _handle_mouse

    def _handle_key(self, event):
        key = _handle_key(event['key'])
        e_type = event['type']
        guiEvent = event.get('guiEvent', None)
        if e_type == 'key_press':
            self.key_press_event(key, guiEvent=guiEvent)
        elif e_type == 'key_release':
            self.key_release_event(key, guiEvent=guiEvent)
    handle_key_press = handle_key_release = _handle_key

    def handle_toolbar_button(self, event):
        # TODO: Be more suspicious of the input
        getattr(self.toolbar, event['name'])()

    def handle_refresh(self, event):
        figure_label = self.figure.get_label()
        if not figure_label:
            figure_label = "Figure {0}".format(self.manager.num)
        self.send_event('figure_label', label=figure_label)
        self._force_full = True
        if self.toolbar:
            # Normal toolbar init would refresh this, but it happens before the
            # browser canvas is set up.
            self.toolbar.set_history_buttons()
        self.draw_idle()

    def handle_resize(self, event):
        x = int(event.get('width', 800)) * self.device_pixel_ratio
        y = int(event.get('height', 800)) * self.device_pixel_ratio
        fig = self.figure
        # An attempt at approximating the figure size in pixels.
        fig.set_size_inches(x / fig.dpi, y / fig.dpi, forward=False)
        # Acknowledge the resize, and force the viewer to update the
        # canvas size to the figure's new size (which is hopefully
        # identical or within a pixel or so).
        self._png_is_old = True
        self.manager.resize(*fig.bbox.size, forward=False)
        self.resize_event()

    def handle_send_image_mode(self, event):
        # The client requests notification of what the current image mode is.
        self.send_event('image_mode', mode=self._current_image_mode)

    def handle_set_device_pixel_ratio(self, event):
        self._handle_set_device_pixel_ratio(event.get('device_pixel_ratio', 1))

    def handle_set_dpi_ratio(self, event):
        # This handler is for backwards-compatibility with older ipympl.
        self._handle_set_device_pixel_ratio(event.get('dpi_ratio', 1))

    def _handle_set_device_pixel_ratio(self, device_pixel_ratio):
        if self._set_device_pixel_ratio(device_pixel_ratio):
            self._force_full = True
            self.draw_idle()

    def send_event(self, event_type, **kwargs):
        if self.manager:
            self.manager._send_event(event_type, **kwargs)


_ALLOWED_TOOL_ITEMS = {
    'home',
    'back',
    'forward',
    'pan',
    'zoom',
    'download',
    None,
}


class NavigationToolbar2WebAgg(backend_bases.NavigationToolbar2):

    # Use the standard toolbar items + download button
    toolitems = [
        (text, tooltip_text, image_file, name_of_method)
        for text, tooltip_text, image_file, name_of_method
        in (*backend_bases.NavigationToolbar2.toolitems,
            ('Download', 'Download plot', 'filesave', 'download'))
        if name_of_method in _ALLOWED_TOOL_ITEMS
    ]

    cursor = _api.deprecate_privatize_attribute("3.5")

    def __init__(self, canvas):
        self.message = ''
        self._cursor = None  # Remove with deprecation.
        super().__init__(canvas)

    def set_message(self, message):
        if message != self.message:
            self.canvas.send_event("message", message=message)
        self.message = message

    def draw_rubberband(self, event, x0, y0, x1, y1):
        self.canvas.send_event("rubberband", x0=x0, y0=y0, x1=x1, y1=y1)

    def remove_rubberband(self):
        self.canvas.send_event("rubberband", x0=-1, y0=-1, x1=-1, y1=-1)

    def save_figure(self, *args):
        """Save the current figure"""
        self.canvas.send_event('save')

    def pan(self):
        super().pan()
        self.canvas.send_event('navigate_mode', mode=self.mode.name)

    def zoom(self):
        super().zoom()
        self.canvas.send_event('navigate_mode', mode=self.mode.name)

    def set_history_buttons(self):
        can_backward = self._nav_stack._pos > 0
        can_forward = self._nav_stack._pos < len(self._nav_stack._elements) - 1
        self.canvas.send_event('history_buttons',
                               Back=can_backward, Forward=can_forward)


class FigureManagerWebAgg(backend_bases.FigureManagerBase):
    ToolbarCls = NavigationToolbar2WebAgg

    def __init__(self, canvas, num):
        self.web_sockets = set()
        super().__init__(canvas, num)
        self.toolbar = self._get_toolbar(canvas)

    def show(self):
        pass

    def _get_toolbar(self, canvas):
        toolbar = self.ToolbarCls(canvas)
        return toolbar

    def resize(self, w, h, forward=True):
        self._send_event(
            'resize',
            size=(w / self.canvas.device_pixel_ratio,
                  h / self.canvas.device_pixel_ratio),
            forward=forward)

    def set_window_title(self, title):
        self._send_event('figure_label', label=title)

    # The following methods are specific to FigureManagerWebAgg

    def add_web_socket(self, web_socket):
        assert hasattr(web_socket, 'send_binary')
        assert hasattr(web_socket, 'send_json')
        self.web_sockets.add(web_socket)
        self.resize(*self.canvas.figure.bbox.size)
        self._send_event('refresh')

    def remove_web_socket(self, web_socket):
        self.web_sockets.remove(web_socket)

    def handle_json(self, content):
        self.canvas.handle_event(content)

    def refresh_all(self):
        if self.web_sockets:
            diff = self.canvas.get_diff_image()
            if diff is not None:
                for s in self.web_sockets:
                    s.send_binary(diff)

    @classmethod
    def get_javascript(cls, stream=None):
        if stream is None:
            output = StringIO()
        else:
            output = stream

        output.write((Path(__file__).parent / "web_backend/js/mpl.js")
                     .read_text(encoding="utf-8"))

        toolitems = []
        for name, tooltip, image, method in cls.ToolbarCls.toolitems:
            if name is None:
                toolitems.append(['', '', '', ''])
            else:
                toolitems.append([name, tooltip, image, method])
        output.write("mpl.toolbar_items = {0};\n\n".format(
            json.dumps(toolitems)))

        extensions = []
        for filetype, ext in sorted(FigureCanvasWebAggCore.
                                    get_supported_filetypes_grouped().
                                    items()):
            extensions.append(ext[0])
        output.write("mpl.extensions = {0};\n\n".format(
            json.dumps(extensions)))

        output.write("mpl.default_extension = {0};".format(
            json.dumps(FigureCanvasWebAggCore.get_default_filetype())))

        if stream is None:
            return output.getvalue()

    @classmethod
    def get_static_file_path(cls):
        return os.path.join(os.path.dirname(__file__), 'web_backend')

    def _send_event(self, event_type, **kwargs):
        payload = {'type': event_type, **kwargs}
        for s in self.web_sockets:
            s.send_json(payload)


@_Backend.export
class _BackendWebAggCoreAgg(_Backend):
    FigureCanvas = FigureCanvasWebAggCore
    FigureManager = FigureManagerWebAgg
