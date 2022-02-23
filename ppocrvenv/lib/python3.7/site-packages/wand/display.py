""":mod:`wand.display` --- Displaying images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :func:`display()` functions shows you the image.  It is useful for
debugging.

If you are in Mac, the image will be opened by your default image application
(:program:`Preview.app` usually).

If you are in Windows, the image will be opened by :program:`imdisplay.exe`,
or your default image application (:program:`Windows Photo Viewer` usually)
if :program:`imdisplay.exe` is unavailable.

You can use it from CLI also.  Execute :mod:`wand.display` module through
:option:`python -m <-m>` option:

.. sourcecode:: console

   $ python -m wand.display wandtests/assets/mona-lisa.jpg

.. versionadded:: 0.1.9

"""
from __future__ import print_function
import ctypes
import os
import platform
import sys
import tempfile

from .image import Image
from .api import library
from .exceptions import BlobError, DelegateError

__all__ = 'display',


def display(image, server_name=':0'):
    """Displays the passed ``image``.

    :param image: an image to display
    :type image: :class:`~wand.image.Image`
    :param server_name: X11 server name to use.  it is ignored and not used
                        for Mac.  default is ``':0'``
    :type server_name: :class:`str`

    """
    if not isinstance(image, Image):
        raise TypeError('image must be a wand.image.Image instance, not ' +
                        repr(image))
    system = platform.system()
    if system == 'Windows':
        try:
            image.save(filename='win:.')
        except DelegateError:
            pass
        else:
            return
    if system in ('Windows', 'Darwin'):
        ext = '.' + image.format.lower()
        if ext in ('miff', 'xc'):
            ext = 'png'
        path = tempfile.mktemp(suffix=ext)
        image.save(filename=path)
        os.system(('start ' if system == 'Windows' else 'open ') + path)
    else:
        library.MagickDisplayImage.argtypes = [ctypes.c_void_p,
                                               ctypes.c_char_p]
        library.MagickDisplayImage(image.wand, str(server_name).encode())


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: python -m wand.display FILE', file=sys.stderr)
        raise SystemExit
    path = sys.argv[1]
    try:
        with Image(filename=path) as image:
            display(image)
    except BlobError:
        print('cannot read the file', path, file=sys.stderr)
