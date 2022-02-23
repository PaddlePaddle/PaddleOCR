"""
PIL-based formats to take screenshots and grab from the clipboard.
"""

import threading

import numpy as np

from ..core import Format


class BaseGrabFormat(Format):
    """Base format for grab formats."""

    _pillow_imported = False
    _ImageGrab = None

    def __init__(self, *args, **kwargs):
        super(BaseGrabFormat, self).__init__(*args, **kwargs)
        self._lock = threading.RLock()

    def _can_write(self, request):
        return False

    def _init_pillow(self):
        with self._lock:
            if not self._pillow_imported:
                self._pillow_imported = True  # more like tried to import
                import PIL

                if not hasattr(PIL, "__version__"):  # pragma: no cover
                    raise ImportError("Imageio Pillow requires " "Pillow, not PIL!")
                try:
                    from PIL import ImageGrab
                except ImportError:
                    return None
                self._ImageGrab = ImageGrab
        return self._ImageGrab

    class Reader(Format.Reader):
        def _open(self):
            pass

        def _close(self):
            pass

        def _get_data(self, index):
            return self.format._get_data(index)


class ScreenGrabFormat(BaseGrabFormat):
    """The ScreenGrabFormat provided a means to grab screenshots using
    the uri of "<screen>".

    This functionality is provided via Pillow. Note that "<screen>" is
    only supported on Windows and OS X.

    Parameters for reading
    ----------------------
    No parameters.
    """

    def _can_read(self, request):
        if request.mode[1] not in "i?":
            return False
        if request.filename != "<screen>":
            return False
        return bool(self._init_pillow())

    def _get_data(self, index):
        ImageGrab = self._init_pillow()
        assert ImageGrab

        pil_im = ImageGrab.grab()
        assert pil_im is not None
        im = np.asarray(pil_im)
        return im, {}


class ClipboardGrabFormat(BaseGrabFormat):
    """The ClipboardGrabFormat provided a means to grab image data from
    the clipboard, using the uri "<clipboard>"

    This functionality is provided via Pillow. Note that "<clipboard>" is
    only supported on Windows.

    Parameters for reading
    ----------------------
    No parameters.
    """

    def _can_read(self, request):
        if request.mode[1] not in "i?":
            return False
        if request.filename != "<clipboard>":
            return False
        return bool(self._init_pillow())

    def _get_data(self, index):
        ImageGrab = self._init_pillow()
        assert ImageGrab

        pil_im = ImageGrab.grabclipboard()
        if pil_im is None:
            raise RuntimeError(
                "There seems to be no image data on the " "clipboard now."
            )
        im = np.asarray(pil_im)
        return im, {}
