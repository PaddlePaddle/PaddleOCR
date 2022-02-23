""":mod:`wand.cdefs.pixel_iterator` --- Pixel-Iterator definitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 0.5.0
"""
from ctypes import POINTER, c_void_p, c_int, c_size_t
from wand.cdefs.wandtypes import c_ssize_t

__all__ = ('load',)


def load(lib, IM_VERSION):
    """Define Pixel Iterator methods. The ImageMagick version is given as
    a second argument for comparison. This will quick to determine which
    methods are available from the library, and can be implemented as::

        if IM_VERSION < 0x700:
            # ... do ImageMagick-6 methods ...
        else
            # ... do ImageMagick-7 methods ...

    .. seealso::

        #include "wand/pixel-iterator.h"
        // Or
        #include "MagickWand/pixel-iterator.h"

    :param lib: the loaded ``MagickWand`` library
    :type lib: :class:`ctypes.CDLL`
    :param IM_VERSION: the ImageMagick version number (i.e. 0x0689)
    :type IM_VERSION: :class:`numbers.Integral`

    .. versionadded:: 0.5.0

    """
    lib.ClonePixelIterator.argtypes = [c_void_p]
    lib.ClonePixelIterator.restype = c_void_p
    lib.DestroyPixelIterator.argtypes = [c_void_p]
    lib.DestroyPixelIterator.restype = c_void_p
    lib.IsPixelIterator.argtypes = [c_void_p]
    lib.NewPixelIterator.argtypes = [c_void_p]
    lib.NewPixelIterator.restype = c_void_p
    lib.PixelClearIteratorException.argtypes = [c_void_p]
    lib.PixelGetIteratorException.argtypes = [c_void_p, POINTER(c_int)]
    lib.PixelGetIteratorException.restype = c_void_p
    lib.PixelGetNextIteratorRow.argtypes = [c_void_p, POINTER(c_size_t)]
    lib.PixelGetNextIteratorRow.restype = POINTER(c_void_p)
    lib.PixelSetFirstIteratorRow.argtypes = [c_void_p]
    lib.PixelSetIteratorRow.argtypes = [c_void_p, c_ssize_t]
