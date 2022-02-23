""":mod:`wand.cdefs.pixel_wand` --- Pixel-Wand definitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 0.5.0
"""
from ctypes import (CDLL, POINTER, c_char_p, c_double,
                    c_float, c_int, c_longdouble, c_size_t,
                    c_ubyte, c_uint, c_ushort, c_void_p)
import numbers
import platform

__all__ = ('load',)


def load(lib, IM_VERSION, IM_QUANTUM_DEPTH, IM_HDRI):
    """Define Pixel Wand methods. The ImageMagick version is given as
    a second argument for comparison. This will quick to determine which
    methods are available from the library, and can be implemented as::

        if IM_VERSION < 0x700:
            # ... do ImageMagick-6 methods ...
        else
            # ... do ImageMagick-7 methods ...

    .. seealso::

        #include "wand/pixel-wand.h"
        // Or
        #include "MagickWand/pixel-wand.h"

    Mapping Pixel methods also requires the wand library to evaluate
    what "Quantum" is to ImageMagick. We must query the library
    to identify if HDRI is enabled, and what the quantum depth is.

    .. seealso::

        MagickCore/magick-type.h

    :param lib: the loaded ``MagickWand`` library.
    :type lib: :class:`ctypes.CDLL`
    :param IM_VERSION: the ImageMagick version number (i.e. 0x0689).
    :type IM_VERSION: :class:`numbers.Integral`
    :param IM_QUANTUM_DEPTH: the ImageMagick Quantum Depth
                             (must be 8, 16, 32, or 64).
    :type IM_QUANTUM_DEPTH: :class:`numbers.Integral`
    :param IM_HDRI: if ImageMagick was compiled with HDRI support.
    :type IM_HDRI: :class:`bool`

    .. versionadded:: 0.5.0

    """
    if not isinstance(lib, CDLL):
        raise AttributeError(repr(lib) + " is not an instanced of ctypes.CDLL")
    if not isinstance(IM_VERSION, numbers.Integral):
        raise AttributeError("Expecting MagickCore version number")
    if IM_QUANTUM_DEPTH not in [8, 16, 32, 65]:
        raise AttributeError("QUANTUM_DEPTH must be one of 8, 16, 32, or 64")
    is_im_6 = IM_VERSION < 0x700
    is_im_7 = IM_VERSION >= 0x700

    # Check for IBM Z Systems, or where `double_t` is defined.
    if platform.machine() in ['s390', 's390x', 'i686']:
        FloatType = c_double
    else:
        FloatType = c_float

    if IM_QUANTUM_DEPTH == 8:
        QuantumType = FloatType if IM_HDRI else c_ubyte
    elif IM_QUANTUM_DEPTH == 16:
        QuantumType = FloatType if IM_HDRI else c_ushort
    elif IM_QUANTUM_DEPTH == 32:
        QuantumType = c_double if IM_HDRI else c_uint
    elif IM_QUANTUM_DEPTH == 64:
        QuantumType = c_longdouble

    lib.ClearPixelWand.argtypes = [c_void_p]
    lib.ClonePixelWand.argtypes = [c_void_p]
    lib.ClonePixelWand.restype = c_void_p
    lib.DestroyPixelWand.argtypes = [c_void_p]
    lib.DestroyPixelWand.restype = c_void_p
    lib.DestroyPixelWands.argtypes = [POINTER(c_void_p), c_size_t]
    lib.DestroyPixelWands.restype = POINTER(c_void_p)
    lib.IsPixelWand.argtypes = [c_void_p]
    lib.IsPixelWandSimilar.argtypes = [c_void_p, c_void_p, c_double]
    lib.NewPixelWand.argtypes = []
    lib.NewPixelWand.restype = c_void_p
    lib.PixelClearException.argtypes = [c_void_p]
    lib.PixelClearException.restype = c_int
    lib.PixelGetAlpha.argtypes = [c_void_p]
    lib.PixelGetAlpha.restype = c_double
    lib.PixelGetAlphaQuantum.argtypes = [c_void_p]
    lib.PixelGetAlphaQuantum.restype = QuantumType
    lib.PixelGetBlack.argtypes = [c_void_p]
    lib.PixelGetBlack.restype = c_double
    lib.PixelGetBlackQuantum.argtypes = [c_void_p]
    lib.PixelGetBlackQuantum.restype = QuantumType
    lib.PixelGetBlue.argtypes = [c_void_p]
    lib.PixelGetBlue.restype = c_double
    lib.PixelGetBlueQuantum.argtypes = [c_void_p]
    lib.PixelGetBlueQuantum.restype = QuantumType
    lib.PixelGetColorAsNormalizedString.argtypes = [c_void_p]
    lib.PixelGetColorAsNormalizedString.restype = c_void_p
    lib.PixelGetColorAsString.argtypes = [c_void_p]
    lib.PixelGetColorAsString.restype = c_void_p
    lib.PixelGetColorCount.argtypes = [c_void_p]
    lib.PixelGetColorCount.restype = c_size_t
    lib.PixelGetCyan.argtypes = [c_void_p]
    lib.PixelGetCyan.restype = c_double
    lib.PixelGetCyanQuantum.argtypes = [c_void_p]
    lib.PixelGetCyanQuantum.restype = QuantumType
    lib.PixelGetException.argtypes = [c_void_p, POINTER(c_int)]
    lib.PixelGetException.restype = c_void_p
    lib.PixelGetExceptionType.argtypes = [c_void_p]
    lib.PixelGetExceptionType.restype = c_int
    lib.PixelGetFuzz.argtypes = [c_void_p]
    lib.PixelGetFuzz.restype = c_double
    lib.PixelGetGreen.argtypes = [c_void_p]
    lib.PixelGetGreen.restype = c_double
    lib.PixelGetGreenQuantum.argtypes = [c_void_p]
    lib.PixelGetGreenQuantum.restype = QuantumType
    lib.PixelGetHSL.argtypes = [c_void_p,
                                POINTER(c_double),
                                POINTER(c_double),
                                POINTER(c_double)]
    lib.PixelGetIndex.argtypes = [c_void_p]
    lib.PixelGetIndex.restype = QuantumType
    lib.PixelGetMagenta.argtypes = [c_void_p]
    lib.PixelGetMagenta.restype = c_double
    lib.PixelGetMagentaQuantum.argtypes = [c_void_p]
    lib.PixelGetMagentaQuantum.restype = QuantumType
    lib.PixelGetMagickColor.argtypes = [c_void_p, c_void_p]
    if is_im_7:
        lib.PixelGetPixel.argtypes = [c_void_p]
        lib.PixelGetPixel.restype = c_void_p
    lib.PixelGetRed.argtypes = [c_void_p]
    lib.PixelGetRed.restype = c_double
    lib.PixelGetRedQuantum.argtypes = [c_void_p]
    lib.PixelGetRedQuantum.restype = QuantumType
    lib.PixelGetYellow.argtypes = [c_void_p]
    lib.PixelGetYellow.restype = c_double
    lib.PixelGetYellowQuantum.argtypes = [c_void_p]
    lib.PixelGetYellowQuantum.restype = QuantumType
    lib.PixelSetAlpha.argtypes = [c_void_p, c_double]
    lib.PixelSetAlphaQuantum.argtypes = [c_void_p, QuantumType]
    lib.PixelSetBlack.argtypes = [c_void_p, c_double]
    lib.PixelSetBlackQuantum.argtypes = [c_void_p, QuantumType]
    lib.PixelSetBlue.argtypes = [c_void_p, c_double]
    lib.PixelSetBlueQuantum.argtypes = [c_void_p, QuantumType]
    lib.PixelSetColor.argtypes = [c_void_p, c_char_p]
    lib.PixelSetColor.restype = c_int
    lib.PixelSetColorCount.argtypes = [c_void_p, c_size_t]
    lib.PixelSetCyan.argtypes = [c_void_p, c_double]
    lib.PixelSetCyanQuantum.argtypes = [c_void_p, QuantumType]
    lib.PixelSetFuzz.argtypes = [c_void_p, c_double]
    lib.PixelSetGreen.argtypes = [c_void_p, c_double]
    lib.PixelSetGreenQuantum.argtypes = [c_void_p, QuantumType]
    lib.PixelSetHSL.argtypes = [c_void_p, c_double, c_double, c_double]
    lib.PixelSetIndex.argtypes = [c_void_p, QuantumType]
    lib.PixelSetMagenta.argtypes = [c_void_p, c_double]
    lib.PixelSetMagentaQuantum.argtypes = [c_void_p, QuantumType]
    if is_im_6:
        lib.PixelSetMagickColor.argtypes = [c_void_p, c_void_p]
    else:
        lib.PixelSetMagickColor = None
    if is_im_7:
        lib.PixelSetPixelColor.argtypes = [c_void_p, c_void_p]
    else:
        lib.PixelSetPixelColor = None
    lib.PixelSetRed.argtypes = [c_void_p, c_double]
    lib.PixelSetRedQuantum.argtypes = [c_void_p, QuantumType]
    lib.PixelSetYellow.argtypes = [c_void_p, c_double]
    lib.PixelSetYellowQuantum.argtypes = [c_void_p, QuantumType]
    if is_im_6:
        lib.PixelSetMagickColor.argtypes = [c_void_p, c_void_p]
        lib.PixelSetPixelColor = None
    if is_im_7:
        lib.PixelSetMagickColor = None
        lib.PixelSetPixelColor.argtypes = [c_void_p, c_void_p]
