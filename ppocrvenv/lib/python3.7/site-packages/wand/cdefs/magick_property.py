""":mod:`wand.cdefs.magick_property` --- Magick-Property definitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 0.5.0
"""
from ctypes import (POINTER, c_void_p, c_char_p, c_size_t, c_ubyte, c_uint,
                    c_int, c_ulong, c_double, c_bool, c_ulonglong)
from wand.cdefs.wandtypes import c_ssize_t

__all__ = ('load',)


def load(lib, IM_VERSION):
    """Define Magick Wand property methods. The ImageMagick version is given as
    a second argument for comparison. This will quick to determine which
    methods are available from the library, and can be implemented as::

        if IM_VERSION < 0x700:
            # ... do ImageMagick-6 methods ...
        else
            # ... do ImageMagick-7 methods ...

    .. seealso::

        #include "wand/magick-property.h"
        // Or
        #include "MagickWand/magick-property.h"

    :param lib: the loaded ``MagickWand`` library
    :type lib: :class:`ctypes.CDLL`
    :param IM_VERSION: the ImageMagick version number (i.e. 0x0689)
    :type IM_VERSION: :class:`numbers.Integral`

    .. versionadded:: 0.5.0

    """
    lib.MagickDeleteImageArtifact.argtypes = [c_void_p, c_char_p]
    lib.MagickDeleteImageArtifact.restype = c_bool
    lib.MagickDeleteImageProperty.argtypes = [c_void_p, c_char_p]
    lib.MagickDeleteImageProperty.restype = c_bool
    lib.MagickDeleteOption.argtypes = [c_void_p, c_char_p]
    lib.MagickDeleteOption.restype = c_bool
    lib.MagickGetAntialias.argtypes = [c_void_p]
    lib.MagickGetAntialias.restype = c_bool
    lib.MagickGetBackgroundColor.argtypes = [c_void_p]
    lib.MagickGetBackgroundColor.restype = c_void_p
    lib.MagickGetColorspace.argtypes = [c_void_p]
    lib.MagickGetColorspace.restype = c_int
    lib.MagickGetCompression.argtypes = [c_void_p]
    lib.MagickGetCompression.restype = c_int
    lib.MagickGetCompressionQuality.argtypes = [c_void_p]
    lib.MagickGetCompressionQuality.restype = c_size_t
    lib.MagickGetFont.argtypes = [c_void_p]
    lib.MagickGetFont.restype = c_void_p
    lib.MagickGetGravity.argtypes = [c_void_p]
    lib.MagickGetGravity.restype = c_int
    lib.MagickGetImageArtifact.argtypes = [c_void_p, c_char_p]
    lib.MagickGetImageArtifact.restype = c_void_p
    lib.MagickGetImageArtifacts.argtypes = [
        c_void_p, c_char_p, POINTER(c_size_t)
    ]
    lib.MagickGetImageArtifacts.restype = POINTER(c_void_p)
    lib.MagickGetImageProfile.argtypes = [
        c_void_p, c_char_p, POINTER(c_size_t)
    ]
    lib.MagickGetImageProfile.restype = POINTER(c_ubyte)
    lib.MagickGetImageProfiles.argtypes = [
        c_void_p, c_char_p, POINTER(c_size_t)
    ]
    lib.MagickGetImageProfiles.restype = POINTER(c_void_p)
    lib.MagickGetImageProperty.argtypes = [c_void_p, c_char_p]
    lib.MagickGetImageProperty.restype = c_void_p
    lib.MagickGetImageProperties.argtypes = [
        c_void_p, c_char_p, POINTER(c_size_t)
    ]
    lib.MagickGetImageProperties.restype = POINTER(c_void_p)
    lib.MagickGetInterlaceScheme.argtypes = [c_void_p]
    lib.MagickGetInterlaceScheme.restype = c_int
    lib.MagickGetOption.argtypes = [c_void_p, c_char_p]
    lib.MagickGetOption.restype = c_void_p
    lib.MagickGetPointsize.argtypes = [c_void_p]
    lib.MagickGetPointsize.restype = c_double
    lib.MagickGetQuantumRange.argtypes = [POINTER(c_size_t)]
    lib.MagickGetResource.argtypes = [c_int]
    lib.MagickGetResource.restype = c_ulonglong
    lib.MagickGetResourceLimit.argtypes = [c_int]
    lib.MagickGetResourceLimit.restype = c_ulonglong
    lib.MagickGetSamplingFactors.argtypes = [c_void_p, POINTER(c_size_t)]
    lib.MagickGetSamplingFactors.restype = POINTER(c_double)
    lib.MagickGetSize.argtypes = [c_void_p, POINTER(c_uint), POINTER(c_uint)]
    lib.MagickGetSize.restype = c_int
    lib.MagickQueryConfigureOption.argtypes = [c_char_p]
    lib.MagickQueryConfigureOption.restype = c_void_p
    lib.MagickQueryConfigureOptions.argtypes = [c_char_p, POINTER(c_size_t)]
    lib.MagickQueryConfigureOptions.restype = POINTER(c_void_p)
    lib.MagickQueryFontMetrics.argtypes = [c_void_p, c_void_p, c_char_p]
    lib.MagickQueryFontMetrics.restype = POINTER(c_double)
    lib.MagickQueryFonts.argtypes = [c_char_p, POINTER(c_size_t)]
    lib.MagickQueryFonts.restype = POINTER(c_void_p)
    lib.MagickQueryFormats.argtypes = [c_char_p, POINTER(c_size_t)]
    lib.MagickQueryFormats.restype = POINTER(c_void_p)
    lib.MagickQueryMultilineFontMetrics.argtypes = [
        c_void_p, c_void_p, c_char_p
    ]
    lib.MagickQueryMultilineFontMetrics.restype = POINTER(c_double)
    lib.MagickRemoveImageProfile.argtypes = [
        c_void_p, c_char_p, POINTER(c_size_t)
    ]
    lib.MagickRemoveImageProfile.restype = POINTER(c_ubyte)
    lib.MagickSetAntialias.argtypes = [c_void_p, c_int]
    lib.MagickSetAntialias.restype = c_bool
    lib.MagickSetBackgroundColor.argtypes = [c_void_p, c_void_p]
    lib.MagickSetBackgroundColor.restype = c_bool
    lib.MagickSetColorspace.argtypes = [c_void_p, c_int]
    lib.MagickSetColorspace.restype = c_bool
    lib.MagickSetCompression.argtypes = [c_void_p, c_int]
    lib.MagickSetCompression.restype = c_bool
    lib.MagickSetCompressionQuality.argtypes = [c_void_p, c_size_t]
    lib.MagickSetCompressionQuality.restype = c_bool
    lib.MagickSetDepth.argtypes = [c_void_p, c_uint]
    lib.MagickSetDepth.restype = c_bool
    lib.MagickSetExtract.argtypes = [c_void_p, c_char_p]
    lib.MagickSetExtract.restype = c_bool
    lib.MagickSetFilename.argtypes = [c_void_p, c_char_p]
    lib.MagickSetFilename.restype = c_bool
    lib.MagickSetFont.argtypes = [c_void_p, c_char_p]
    lib.MagickSetFont.restype = c_bool
    lib.MagickSetFormat.argtypes = [c_void_p, c_char_p]
    lib.MagickSetFormat.restype = c_bool
    lib.MagickSetGravity.argtypes = [c_void_p, c_int]
    lib.MagickSetGravity.restype = c_bool
    lib.MagickSetImageArtifact.argtypes = [c_void_p, c_char_p, c_char_p]
    lib.MagickSetImageProfile.argtypes = [
        c_void_p, c_char_p, c_void_p, c_size_t
    ]
    lib.MagickSetImageProfile.restype = c_bool
    lib.MagickSetImageProperty.argtypes = [c_void_p, c_char_p, c_char_p]
    lib.MagickSetImageProperty.restype = c_bool
    lib.MagickSetInterlaceScheme.argtypes = [c_void_p, c_int]
    lib.MagickSetInterlaceScheme.restype = c_bool
    lib.MagickSetInterpolateMethod.argtypes = [c_void_p, c_int]
    lib.MagickSetInterpolateMethod.restype = c_bool
    lib.MagickSetOption.argtypes = [c_void_p, c_char_p, c_char_p]
    lib.MagickSetOption.restype = c_bool
    lib.MagickSetOrientation.argtypes = [c_void_p, c_int]
    lib.MagickSetOrientation.restype = c_bool
    lib.MagickSetPage.argtypes = [
        c_void_p, c_size_t, c_size_t, c_ssize_t, c_ssize_t
    ]
    lib.MagickSetPage.restype = c_bool
    lib.MagickSetPassphrase.argtypes = [c_void_p, c_char_p]
    lib.MagickSetPassphrase.restype = c_bool
    lib.MagickSetPointsize.argtypes = [c_void_p, c_double]
    lib.MagickSetPointsize.restype = c_bool
    lib.MagickSetResolution.argtypes = [c_void_p, c_double, c_double]
    lib.MagickSetResolution.restype = c_bool
    lib.MagickSetResourceLimit.argtypes = [c_int, c_ulonglong]
    lib.MagickSetResourceLimit.restype = c_bool
    lib.MagickSetSamplingFactors.argtypes = [
        c_void_p, c_size_t, POINTER(c_double)
    ]
    lib.MagickSetSamplingFactors.restype = c_bool
    if IM_VERSION >= 0x708:
        try:
            lib.MagickSetSeed.argtypes = [c_void_p, c_ulong]
        except AttributeError:
            lib.MagickSetSeed = None
    else:
        lib.MagickSetSeed = None
    try:
        lib.MagickSetSecurityPolicy.argtypes = [c_void_p, c_char_p]
        lib.MagickSetSecurityPolicy.restype = c_bool
    except AttributeError:
        lib.MagickSetSecurityPolicy = None
    lib.MagickSetSize.argtypes = [c_void_p, c_uint, c_uint]
    lib.MagickSetSize.restype = c_bool
    lib.MagickSetSizeOffset.argtypes = [
        c_void_p, c_size_t, c_size_t, c_ssize_t
    ]
    lib.MagickSetSizeOffset.restype = c_bool
    lib.MagickSetType.argtypes = [c_void_p, c_int]
    lib.MagickSetType.restype = c_bool
