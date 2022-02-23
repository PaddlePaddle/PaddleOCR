""":mod:`wand.cdefs.core` --- MagickCore definitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 0.5.0
"""
from ctypes import POINTER, c_void_p, c_char_p, c_int, c_size_t, c_ulonglong
from wand.cdefs.wandtypes import c_ssize_t

__all__ = ('load', 'load_with_version')


def load(libmagick):
    """Define MagickCore methods.
    We'll only define the bare-minimum methods to support the MagickWand
    library.

    .. seealso::

        #include <magick/MagickCore.h>
        // Or
        #include <MagickCore/MagickCore.h>

    :param libmagick: the loaded ``MagickCore`` library.
    :type libmagick: :class:`ctypes.CDLL`

    .. versionadded:: 0.5.0

    """
    libmagick.AcquireExceptionInfo.argtypes = []
    libmagick.AcquireExceptionInfo.restype = c_void_p
    libmagick.CloneImages.argtypes = [c_void_p, c_char_p, c_void_p]
    libmagick.CloneImages.restype = c_void_p
    libmagick.DestroyExceptionInfo.argtypes = [c_void_p]
    libmagick.DestroyExceptionInfo.restype = c_void_p
    libmagick.DestroyImage.argtypes = [c_void_p]
    libmagick.DestroyImage.restype = c_void_p
    libmagick.DestroyString.argtypes = [c_void_p]
    libmagick.DestroyString.restype = c_void_p
    try:
        libmagick.GetGeometry.argtypes = [c_char_p,
                                          POINTER(c_ssize_t),
                                          POINTER(c_ssize_t),
                                          POINTER(c_size_t),
                                          POINTER(c_size_t)]
        libmagick.GetGeometry.restype = c_int
    except AttributeError:
        libmagick.GetGeometry = None
    libmagick.GetMagickCopyright.argtypes = []
    libmagick.GetMagickCopyright.restype = c_char_p
    try:
        libmagick.GetMagickDelegates.argtypes = []
        libmagick.GetMagickDelegates.restype = c_char_p
    except AttributeError:
        libmagick.GetMagickDelegates = None
    libmagick.GetMagickFeatures.argtypes = []
    libmagick.GetMagickFeatures.restype = c_char_p
    try:
        libmagick.GetMagickLicense.argtypes = []
        libmagick.GetMagickLicense.restype = c_char_p
    except AttributeError:
        pass
    libmagick.GetMagickPackageName.argtypes = []
    libmagick.GetMagickPackageName.restype = c_char_p
    libmagick.GetMagickQuantumDepth.argtypes = [POINTER(c_size_t)]
    libmagick.GetMagickQuantumDepth.restype = c_char_p
    libmagick.GetMagickQuantumRange.argtypes = [POINTER(c_size_t)]
    libmagick.GetMagickQuantumRange.restype = c_char_p
    libmagick.GetMagickReleaseDate.argtypes = []
    libmagick.GetMagickReleaseDate.restype = c_char_p
    libmagick.GetMagickResource.argtypes = [c_int]
    libmagick.GetMagickResource.restype = c_ulonglong
    libmagick.GetMagickResourceLimit.argtypes = [c_int]
    libmagick.GetMagickResourceLimit.restype = c_ulonglong
    libmagick.GetMagickVersion.argtypes = [POINTER(c_size_t)]
    libmagick.GetMagickVersion.restype = c_char_p
    try:
        libmagick.GetPageGeometry.argtypes = [c_char_p]
        libmagick.GetPageGeometry.restype = c_void_p
    except AttributeError:
        libmagick.GetPageGeometry = None
    libmagick.GetNextImageInList.argtypes = [c_void_p]
    libmagick.GetNextImageInList.restype = c_void_p
    libmagick.MagickToMime.argtypes = [c_char_p]
    libmagick.MagickToMime.restype = c_void_p
    try:
        libmagick.ParseAbsoluteGeometry.argtypes = [c_char_p, c_void_p]
        libmagick.ParseAbsoluteGeometry.restype = c_int
    except AttributeError:
        libmagick.ParseAbsoluteGeometry = None
    try:
        libmagick.ParseChannelOption.argtypes = [c_char_p]
        libmagick.ParseChannelOption.restypes = c_ssize_t
    except AttributeError:
        libmagick.ParseChannelOption = None
    try:
        libmagick.ParseGeometry.argtypes = [c_char_p, c_void_p]
        libmagick.ParseGeometry.restype = c_int
        libmagick.ParseMetaGeometry.argtypes = [c_char_p,
                                                POINTER(c_ssize_t),
                                                POINTER(c_ssize_t),
                                                POINTER(c_size_t),
                                                POINTER(c_size_t)]
        libmagick.ParseMetaGeometry.restype = c_int
    except AttributeError:
        libmagick.ParseGeometry = None
        libmagick.ParseMetaGeometry = None
    libmagick.SetMagickResourceLimit.argtypes = [c_int, c_ulonglong]
    libmagick.SetMagickResourceLimit.restype = c_int


def load_with_version(libmagick, IM_VERSION):
    if IM_VERSION < 0x700:
        libmagick.AcquireKernelBuiltIn.argtypes = [c_int, c_void_p]
        libmagick.AcquireKernelBuiltIn.restype = c_void_p
        libmagick.AcquireKernelInfo.argtypes = [c_char_p]
        libmagick.AcquireKernelInfo.restype = c_void_p
    else:
        libmagick.AcquireKernelBuiltIn.argtypes = [c_int, c_void_p,
                                                   c_void_p]
        libmagick.AcquireKernelBuiltIn.restype = c_void_p
        libmagick.AcquireKernelInfo.argtypes = [c_char_p, c_void_p]
        libmagick.AcquireKernelInfo.restype = c_void_p
    libmagick.DestroyKernelInfo.argtypes = [c_void_p]
    libmagick.DestroyKernelInfo.restype = c_void_p
