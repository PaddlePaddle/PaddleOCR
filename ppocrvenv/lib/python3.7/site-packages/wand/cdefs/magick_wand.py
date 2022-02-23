""":mod:`wand.cdefs.magick_wand` --- Magick-Wand definitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 0.5.0
"""
from ctypes import POINTER, c_void_p, c_bool, c_int
from wand.cdefs.wandtypes import c_ssize_t

__all__ = ('load',)


def load(lib, IM_VERSION):
    """Define Magick Wand methods. The ImageMagick version is given as
    a second argument for comparison. This will quick to determine which
    methods are available from the library, and can be implemented as::

        if IM_VERSION < 0x700:
            # ... do ImageMagick-6 methods ...
        else
            # ... do ImageMagick-7 methods ...

    .. seealso::

        #include "wand/magick-wand.h"
        // Or
        #include "MagickWand/magick-wand.h"

    :param lib: the loaded ``MagickWand`` library
    :type lib: :class:`ctypes.CDLL`
    :param IM_VERSION: the ImageMagick version number (i.e. 0x0689)
    :type IM_VERSION: :class:`numbers.Integral`

    .. versionadded:: 0.5.0

    """
    lib.ClearMagickWand.argtypes = [c_void_p]
    lib.CloneMagickWand.argtypes = [c_void_p]
    lib.CloneMagickWand.restype = c_void_p
    lib.DestroyMagickWand.argtypes = [c_void_p]
    lib.DestroyMagickWand.restype = c_void_p
    lib.IsMagickWand.argtypes = [c_void_p]
    try:
        lib.IsMagickWandInstantiated.argtypes = []
        lib.IsMagickWandInstantiated.restype = c_bool
    except AttributeError:
        lib.IsMagickWandInstantiated = None
        pass
    lib.MagickClearException.argtypes = [c_void_p]
    lib.MagickGetException.argtypes = [c_void_p, POINTER(c_int)]
    lib.MagickGetException.restype = c_void_p
    lib.MagickGetExceptionType.argtypes = [c_void_p]
    lib.MagickGetExceptionType.restype = c_int
    lib.MagickGetIteratorIndex.argtypes = [c_void_p]
    lib.MagickGetIteratorIndex.restype = c_ssize_t
    lib.MagickRelinquishMemory.argtypes = [c_void_p]
    lib.MagickRelinquishMemory.restype = c_void_p
    lib.MagickResetIterator.argtypes = [c_void_p]
    lib.MagickSetFirstIterator.argtypes = [c_void_p]
    lib.MagickSetIteratorIndex.argtypes = [c_void_p, c_ssize_t]
    lib.MagickSetIteratorIndex.restype = c_bool
    lib.MagickSetLastIterator.argtypes = [c_void_p]
    lib.MagickWandGenesis.argtypes = []
    lib.MagickWandTerminus.argtypes = []
    lib.NewMagickWandFromImage.argtypes = [c_void_p]
    lib.NewMagickWandFromImage.restype = c_void_p
    lib.NewMagickWand.argtypes = []
    lib.NewMagickWand.restype = c_void_p
