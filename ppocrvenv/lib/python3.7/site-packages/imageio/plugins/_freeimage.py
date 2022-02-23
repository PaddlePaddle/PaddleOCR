# -*- coding: utf-8 -*-
# imageio is distributed under the terms of the (new) BSD License.

# styletest: ignore E261

""" Module imageio/freeimage.py

This module contains the wrapper code for the freeimage library.
The functions defined in this module are relatively thin; just thin
enough so that arguments and results are native Python/numpy data
types.

"""

import os
import sys
import ctypes
import threading
import logging
import numpy

from ..core import (
    get_remote_file,
    load_lib,
    Dict,
    resource_dirs,
    IS_PYPY,
    get_platform,
    InternetNotAllowedError,
    NeedDownloadError,
)

logger = logging.getLogger(__name__)

TEST_NUMPY_NO_STRIDES = False  # To test pypy fallback

FNAME_PER_PLATFORM = {
    "osx32": "libfreeimage-3.16.0-osx10.6.dylib",  # universal library
    "osx64": "libfreeimage-3.16.0-osx10.6.dylib",
    "win32": "FreeImage-3.15.4-win32.dll",
    "win64": "FreeImage-3.15.1-win64.dll",
    "linux32": "libfreeimage-3.16.0-linux32.so",
    "linux64": "libfreeimage-3.16.0-linux64.so",
}


def download(directory=None, force_download=False):
    """Download the FreeImage library to your computer.

    Parameters
    ----------
    directory : str | None
        The directory where the file will be cached if a download was
        required to obtain the file. By default, the appdata directory
        is used. This is also the first directory that is checked for
        a local version of the file.
    force_download : bool | str
        If True, the file will be downloaded even if a local copy exists
        (and this copy will be overwritten). Can also be a YYYY-MM-DD date
        to ensure a file is up-to-date (modified date of a file on disk,
        if present, is checked).
    """
    plat = get_platform()
    if plat and plat in FNAME_PER_PLATFORM:
        fname = "freeimage/" + FNAME_PER_PLATFORM[plat]
        get_remote_file(fname=fname, directory=directory, force_download=force_download)
        fi._lib = None  # allow trying again (needed to make tests work)


def get_freeimage_lib():
    """Ensure we have our version of the binary freeimage lib."""

    lib = os.getenv("IMAGEIO_FREEIMAGE_LIB", None)
    if lib:  # pragma: no cover
        return lib

    # Get filename to load
    # If we do not provide a binary, the system may still do ...
    plat = get_platform()
    if plat and plat in FNAME_PER_PLATFORM:
        try:
            return get_remote_file("freeimage/" + FNAME_PER_PLATFORM[plat], auto=False)
        except InternetNotAllowedError:
            pass
        except NeedDownloadError:
            raise NeedDownloadError(
                "Need FreeImage library. "
                "You can obtain it with either:\n"
                "  - download using the command: "
                "imageio_download_bin freeimage\n"
                "  - download by calling (in Python): "
                "imageio.plugins.freeimage.download()\n"
            )
        except RuntimeError as e:  # pragma: no cover
            logger.warning(str(e))


# Define function to encode a filename to bytes (for the current system)
def efn(x):
    return x.encode(sys.getfilesystemencoding())


# 4-byte quads of 0,v,v,v from 0,0,0,0 to 0,255,255,255
GREY_PALETTE = numpy.arange(0, 0x01000000, 0x00010101, dtype=numpy.uint32)


class FI_TYPES(object):
    FIT_UNKNOWN = 0
    FIT_BITMAP = 1
    FIT_UINT16 = 2
    FIT_INT16 = 3
    FIT_UINT32 = 4
    FIT_INT32 = 5
    FIT_FLOAT = 6
    FIT_DOUBLE = 7
    FIT_COMPLEX = 8
    FIT_RGB16 = 9
    FIT_RGBA16 = 10
    FIT_RGBF = 11
    FIT_RGBAF = 12

    dtypes = {
        FIT_BITMAP: numpy.uint8,
        FIT_UINT16: numpy.uint16,
        FIT_INT16: numpy.int16,
        FIT_UINT32: numpy.uint32,
        FIT_INT32: numpy.int32,
        FIT_FLOAT: numpy.float32,
        FIT_DOUBLE: numpy.float64,
        FIT_COMPLEX: numpy.complex128,
        FIT_RGB16: numpy.uint16,
        FIT_RGBA16: numpy.uint16,
        FIT_RGBF: numpy.float32,
        FIT_RGBAF: numpy.float32,
    }

    fi_types = {
        (numpy.uint8, 1): FIT_BITMAP,
        (numpy.uint8, 3): FIT_BITMAP,
        (numpy.uint8, 4): FIT_BITMAP,
        (numpy.uint16, 1): FIT_UINT16,
        (numpy.int16, 1): FIT_INT16,
        (numpy.uint32, 1): FIT_UINT32,
        (numpy.int32, 1): FIT_INT32,
        (numpy.float32, 1): FIT_FLOAT,
        (numpy.float64, 1): FIT_DOUBLE,
        (numpy.complex128, 1): FIT_COMPLEX,
        (numpy.uint16, 3): FIT_RGB16,
        (numpy.uint16, 4): FIT_RGBA16,
        (numpy.float32, 3): FIT_RGBF,
        (numpy.float32, 4): FIT_RGBAF,
    }

    extra_dims = {
        FIT_UINT16: [],
        FIT_INT16: [],
        FIT_UINT32: [],
        FIT_INT32: [],
        FIT_FLOAT: [],
        FIT_DOUBLE: [],
        FIT_COMPLEX: [],
        FIT_RGB16: [3],
        FIT_RGBA16: [4],
        FIT_RGBF: [3],
        FIT_RGBAF: [4],
    }


class IO_FLAGS(object):
    FIF_LOAD_NOPIXELS = 0x8000  # loading: load the image header only
    #                          # (not supported by all plugins)
    BMP_DEFAULT = 0
    BMP_SAVE_RLE = 1
    CUT_DEFAULT = 0
    DDS_DEFAULT = 0
    EXR_DEFAULT = 0  # save data as half with piz-based wavelet compression
    EXR_FLOAT = 0x0001  # save data as float instead of half (not recommended)
    EXR_NONE = 0x0002  # save with no compression
    EXR_ZIP = 0x0004  # save with zlib compression, in blocks of 16 scan lines
    EXR_PIZ = 0x0008  # save with piz-based wavelet compression
    EXR_PXR24 = 0x0010  # save with lossy 24-bit float compression
    EXR_B44 = 0x0020  # save with lossy 44% float compression
    #                # - goes to 22% when combined with EXR_LC
    EXR_LC = 0x0040  # save images with one luminance and two chroma channels,
    #               # rather than as RGB (lossy compression)
    FAXG3_DEFAULT = 0
    GIF_DEFAULT = 0
    GIF_LOAD256 = 1  # Load the image as a 256 color image with ununsed
    #               # palette entries, if it's 16 or 2 color
    GIF_PLAYBACK = 2  # 'Play' the GIF to generate each frame (as 32bpp)
    #                # instead of returning raw frame data when loading
    HDR_DEFAULT = 0
    ICO_DEFAULT = 0
    ICO_MAKEALPHA = 1  # convert to 32bpp and create an alpha channel from the
    #                 # AND-mask when loading
    IFF_DEFAULT = 0
    J2K_DEFAULT = 0  # save with a 16:1 rate
    JP2_DEFAULT = 0  # save with a 16:1 rate
    JPEG_DEFAULT = 0  # loading (see JPEG_FAST);
    #                # saving (see JPEG_QUALITYGOOD|JPEG_SUBSAMPLING_420)
    JPEG_FAST = 0x0001  # load the file as fast as possible,
    #                  # sacrificing some quality
    JPEG_ACCURATE = 0x0002  # load the file with the best quality,
    #                      # sacrificing some speed
    JPEG_CMYK = 0x0004  # load separated CMYK "as is"
    #                  # (use | to combine with other load flags)
    JPEG_EXIFROTATE = 0x0008  # load and rotate according to
    #                        # Exif 'Orientation' tag if available
    JPEG_QUALITYSUPERB = 0x80  # save with superb quality (100:1)
    JPEG_QUALITYGOOD = 0x0100  # save with good quality (75:1)
    JPEG_QUALITYNORMAL = 0x0200  # save with normal quality (50:1)
    JPEG_QUALITYAVERAGE = 0x0400  # save with average quality (25:1)
    JPEG_QUALITYBAD = 0x0800  # save with bad quality (10:1)
    JPEG_PROGRESSIVE = 0x2000  # save as a progressive-JPEG
    #                         # (use | to combine with other save flags)
    JPEG_SUBSAMPLING_411 = 0x1000  # save with high 4x1 chroma
    #                             # subsampling (4:1:1)
    JPEG_SUBSAMPLING_420 = 0x4000  # save with medium 2x2 medium chroma
    #                             # subsampling (4:2:0) - default value
    JPEG_SUBSAMPLING_422 = 0x8000  # save /w low 2x1 chroma subsampling (4:2:2)
    JPEG_SUBSAMPLING_444 = 0x10000  # save with no chroma subsampling (4:4:4)
    JPEG_OPTIMIZE = 0x20000  # on saving, compute optimal Huffman coding tables
    #                       # (can reduce a few percent of file size)
    JPEG_BASELINE = 0x40000  # save basic JPEG, without metadata or any markers
    KOALA_DEFAULT = 0
    LBM_DEFAULT = 0
    MNG_DEFAULT = 0
    PCD_DEFAULT = 0
    PCD_BASE = 1  # load the bitmap sized 768 x 512
    PCD_BASEDIV4 = 2  # load the bitmap sized 384 x 256
    PCD_BASEDIV16 = 3  # load the bitmap sized 192 x 128
    PCX_DEFAULT = 0
    PFM_DEFAULT = 0
    PICT_DEFAULT = 0
    PNG_DEFAULT = 0
    PNG_IGNOREGAMMA = 1  # loading: avoid gamma correction
    PNG_Z_BEST_SPEED = 0x0001  # save using ZLib level 1 compression flag
    #                         # (default value is 6)
    PNG_Z_DEFAULT_COMPRESSION = 0x0006  # save using ZLib level 6 compression
    #                                  # flag (default recommended value)
    PNG_Z_BEST_COMPRESSION = 0x0009  # save using ZLib level 9 compression flag
    #                               # (default value is 6)
    PNG_Z_NO_COMPRESSION = 0x0100  # save without ZLib compression
    PNG_INTERLACED = 0x0200  # save using Adam7 interlacing (use | to combine
    #                       # with other save flags)
    PNM_DEFAULT = 0
    PNM_SAVE_RAW = 0  # Writer saves in RAW format (i.e. P4, P5 or P6)
    PNM_SAVE_ASCII = 1  # Writer saves in ASCII format (i.e. P1, P2 or P3)
    PSD_DEFAULT = 0
    PSD_CMYK = 1  # reads tags for separated CMYK (default is conversion to RGB)
    PSD_LAB = 2  # reads tags for CIELab (default is conversion to RGB)
    RAS_DEFAULT = 0
    RAW_DEFAULT = 0  # load the file as linear RGB 48-bit
    RAW_PREVIEW = 1  # try to load the embedded JPEG preview with included
    #               # Exif Data or default to RGB 24-bit
    RAW_DISPLAY = 2  # load the file as RGB 24-bit
    SGI_DEFAULT = 0
    TARGA_DEFAULT = 0
    TARGA_LOAD_RGB888 = 1  # Convert RGB555 and ARGB8888 -> RGB888.
    TARGA_SAVE_RLE = 2  # Save with RLE compression
    TIFF_DEFAULT = 0
    TIFF_CMYK = 0x0001  # reads/stores tags for separated CMYK
    #                  # (use | to combine with compression flags)
    TIFF_PACKBITS = 0x0100  # save using PACKBITS compression
    TIFF_DEFLATE = 0x0200  # save using DEFLATE (a.k.a. ZLIB) compression
    TIFF_ADOBE_DEFLATE = 0x0400  # save using ADOBE DEFLATE compression
    TIFF_NONE = 0x0800  # save without any compression
    TIFF_CCITTFAX3 = 0x1000  # save using CCITT Group 3 fax encoding
    TIFF_CCITTFAX4 = 0x2000  # save using CCITT Group 4 fax encoding
    TIFF_LZW = 0x4000  # save using LZW compression
    TIFF_JPEG = 0x8000  # save using JPEG compression
    TIFF_LOGLUV = 0x10000  # save using LogLuv compression
    WBMP_DEFAULT = 0
    XBM_DEFAULT = 0
    XPM_DEFAULT = 0


class METADATA_MODELS(object):
    FIMD_COMMENTS = 0
    FIMD_EXIF_MAIN = 1
    FIMD_EXIF_EXIF = 2
    FIMD_EXIF_GPS = 3
    FIMD_EXIF_MAKERNOTE = 4
    FIMD_EXIF_INTEROP = 5
    FIMD_IPTC = 6
    FIMD_XMP = 7
    FIMD_GEOTIFF = 8
    FIMD_ANIMATION = 9


class METADATA_DATATYPE(object):
    FIDT_BYTE = 1  # 8-bit unsigned integer
    FIDT_ASCII = 2  # 8-bit bytes w/ last byte null
    FIDT_SHORT = 3  # 16-bit unsigned integer
    FIDT_LONG = 4  # 32-bit unsigned integer
    FIDT_RATIONAL = 5  # 64-bit unsigned fraction
    FIDT_SBYTE = 6  # 8-bit signed integer
    FIDT_UNDEFINED = 7  # 8-bit untyped data
    FIDT_SSHORT = 8  # 16-bit signed integer
    FIDT_SLONG = 9  # 32-bit signed integer
    FIDT_SRATIONAL = 10  # 64-bit signed fraction
    FIDT_FLOAT = 11  # 32-bit IEEE floating point
    FIDT_DOUBLE = 12  # 64-bit IEEE floating point
    FIDT_IFD = 13  # 32-bit unsigned integer (offset)
    FIDT_PALETTE = 14  # 32-bit RGBQUAD
    FIDT_LONG8 = 16  # 64-bit unsigned integer
    FIDT_SLONG8 = 17  # 64-bit signed integer
    FIDT_IFD8 = 18  # 64-bit unsigned integer (offset)

    dtypes = {
        FIDT_BYTE: numpy.uint8,
        FIDT_SHORT: numpy.uint16,
        FIDT_LONG: numpy.uint32,
        FIDT_RATIONAL: [("numerator", numpy.uint32), ("denominator", numpy.uint32)],
        FIDT_LONG8: numpy.uint64,
        FIDT_SLONG8: numpy.int64,
        FIDT_IFD8: numpy.uint64,
        FIDT_SBYTE: numpy.int8,
        FIDT_UNDEFINED: numpy.uint8,
        FIDT_SSHORT: numpy.int16,
        FIDT_SLONG: numpy.int32,
        FIDT_SRATIONAL: [("numerator", numpy.int32), ("denominator", numpy.int32)],
        FIDT_FLOAT: numpy.float32,
        FIDT_DOUBLE: numpy.float64,
        FIDT_IFD: numpy.uint32,
        FIDT_PALETTE: [
            ("R", numpy.uint8),
            ("G", numpy.uint8),
            ("B", numpy.uint8),
            ("A", numpy.uint8),
        ],
    }


class Freeimage(object):
    """Class to represent an interface to the FreeImage library.
    This class is relatively thin. It provides a Pythonic API that converts
    Freeimage objects to Python objects, but that's about it.
    The actual implementation should be provided by the plugins.

    The recommended way to call into the Freeimage library (so that
    errors and warnings show up in the right moment) is to use this
    object as a context manager:
    with imageio.fi as lib:
        lib.FreeImage_GetPalette()

    """

    _API = {
        # All we're doing here is telling ctypes that some of the
        # FreeImage functions return pointers instead of integers. (On
        # 64-bit systems, without this information the pointers get
        # truncated and crashes result). There's no need to list
        # functions that return ints, or the types of the parameters
        # to these or other functions -- that's fine to do implicitly.
        # Note that the ctypes immediately converts the returned void_p
        # back to a python int again! This is really not helpful,
        # because then passing it back to another library call will
        # cause truncation-to-32-bits on 64-bit systems. Thanks, ctypes!
        # So after these calls one must immediately re-wrap the int as
        # a c_void_p if it is to be passed back into FreeImage.
        "FreeImage_AllocateT": (ctypes.c_void_p, None),
        "FreeImage_FindFirstMetadata": (ctypes.c_void_p, None),
        "FreeImage_GetBits": (ctypes.c_void_p, None),
        "FreeImage_GetPalette": (ctypes.c_void_p, None),
        "FreeImage_GetTagKey": (ctypes.c_char_p, None),
        "FreeImage_GetTagValue": (ctypes.c_void_p, None),
        "FreeImage_CreateTag": (ctypes.c_void_p, None),
        "FreeImage_Save": (ctypes.c_void_p, None),
        "FreeImage_Load": (ctypes.c_void_p, None),
        "FreeImage_LoadFromMemory": (ctypes.c_void_p, None),
        "FreeImage_OpenMultiBitmap": (ctypes.c_void_p, None),
        "FreeImage_LoadMultiBitmapFromMemory": (ctypes.c_void_p, None),
        "FreeImage_LockPage": (ctypes.c_void_p, None),
        "FreeImage_OpenMemory": (ctypes.c_void_p, None),
        # 'FreeImage_ReadMemory': (ctypes.c_void_p, None),
        # 'FreeImage_CloseMemory': (ctypes.c_void_p, None),
        "FreeImage_GetVersion": (ctypes.c_char_p, None),
        "FreeImage_GetFIFExtensionList": (ctypes.c_char_p, None),
        "FreeImage_GetFormatFromFIF": (ctypes.c_char_p, None),
        "FreeImage_GetFIFDescription": (ctypes.c_char_p, None),
        "FreeImage_ColorQuantizeEx": (ctypes.c_void_p, None),
        # Pypy wants some extra definitions, so here we go ...
        "FreeImage_IsLittleEndian": (ctypes.c_int, None),
        "FreeImage_SetOutputMessage": (ctypes.c_void_p, None),
        "FreeImage_GetFIFCount": (ctypes.c_int, None),
        "FreeImage_IsPluginEnabled": (ctypes.c_int, None),
        "FreeImage_GetFileType": (ctypes.c_int, None),
        #
        "FreeImage_GetTagType": (ctypes.c_int, None),
        "FreeImage_GetTagLength": (ctypes.c_int, None),
        "FreeImage_FindNextMetadata": (ctypes.c_int, None),
        "FreeImage_FindCloseMetadata": (ctypes.c_void_p, None),
        #
        "FreeImage_GetFIFFromFilename": (ctypes.c_int, None),
        "FreeImage_FIFSupportsReading": (ctypes.c_int, None),
        "FreeImage_FIFSupportsWriting": (ctypes.c_int, None),
        "FreeImage_FIFSupportsExportType": (ctypes.c_int, None),
        "FreeImage_FIFSupportsExportBPP": (ctypes.c_int, None),
        "FreeImage_GetHeight": (ctypes.c_int, None),
        "FreeImage_GetWidth": (ctypes.c_int, None),
        "FreeImage_GetImageType": (ctypes.c_int, None),
        "FreeImage_GetBPP": (ctypes.c_int, None),
        "FreeImage_GetColorsUsed": (ctypes.c_int, None),
        "FreeImage_ConvertTo32Bits": (ctypes.c_void_p, None),
        "FreeImage_GetPitch": (ctypes.c_int, None),
        "FreeImage_Unload": (ctypes.c_void_p, None),
    }

    def __init__(self):

        # Initialize freeimage lib as None
        self._lib = None

        # A lock to create thread-safety
        self._lock = threading.RLock()

        # Init log messages lists
        self._messages = []

        # Select functype for error handler
        if sys.platform.startswith("win"):
            functype = ctypes.WINFUNCTYPE
        else:
            functype = ctypes.CFUNCTYPE

        # Create output message handler
        @functype(None, ctypes.c_int, ctypes.c_char_p)
        def error_handler(fif, message):
            message = message.decode("utf-8")
            self._messages.append(message)
            while (len(self._messages)) > 256:
                self._messages.pop(0)

        # Make sure to keep a ref to function
        self._error_handler = error_handler

    @property
    def lib(self):
        if self._lib is None:
            try:
                self.load_freeimage()
            except OSError as err:
                self._lib = "The freeimage library could not be loaded: "
                self._lib += str(err)
        if isinstance(self._lib, str):
            raise RuntimeError(self._lib)
        return self._lib

    def has_lib(self):
        try:
            self.lib
        except Exception:
            return False
        return True

    def load_freeimage(self):
        """Try to load the freeimage lib from the system. If not successful,
        try to download the imageio version and try again.
        """
        # Load library and register API
        success = False
        try:
            # Try without forcing a download, but giving preference
            # to the imageio-provided lib (if previously downloaded)
            self._load_freeimage()
            self._register_api()
            if self.lib.FreeImage_GetVersion().decode("utf-8") >= "3.15":
                success = True
        except OSError:
            pass

        if not success:
            # Ensure we have our own lib, try again
            get_freeimage_lib()
            self._load_freeimage()
            self._register_api()

        # Wrap up
        self.lib.FreeImage_SetOutputMessage(self._error_handler)
        self.lib_version = self.lib.FreeImage_GetVersion().decode("utf-8")

    def _load_freeimage(self):

        # Define names
        lib_names = ["freeimage", "libfreeimage"]
        exact_lib_names = [
            "FreeImage",
            "libfreeimage.dylib",
            "libfreeimage.so",
            "libfreeimage.so.3",
        ]
        # Add names of libraries that we provide (that file may not exist)
        res_dirs = resource_dirs()
        plat = get_platform()
        if plat:  # Can be None on e.g. FreeBSD
            fname = FNAME_PER_PLATFORM[plat]
            for dir in res_dirs:
                exact_lib_names.insert(0, os.path.join(dir, "freeimage", fname))

        # Add the path specified with IMAGEIO_FREEIMAGE_LIB:
        lib = os.getenv("IMAGEIO_FREEIMAGE_LIB", None)
        if lib is not None:
            exact_lib_names.insert(0, lib)

        # Load
        try:
            lib, fname = load_lib(exact_lib_names, lib_names, res_dirs)
        except OSError as err:  # pragma: no cover
            err_msg = str(err) + "\nPlease install the FreeImage library."
            raise OSError(err_msg)

        # Store
        self._lib = lib
        self.lib_fname = fname

    def _register_api(self):
        # Albert's ctypes pattern
        for f, (restype, argtypes) in self._API.items():
            func = getattr(self.lib, f)
            func.restype = restype
            func.argtypes = argtypes

    # Handling of output messages

    def __enter__(self):
        self._lock.acquire()
        return self.lib

    def __exit__(self, *args):
        self._show_any_warnings()
        self._lock.release()

    def _reset_log(self):
        """Reset the list of output messages. Call this before
        loading or saving an image with the FreeImage API.
        """
        self._messages = []

    def _get_error_message(self):
        """Get the output messages produced since the last reset as
        one string. Returns 'No known reason.' if there are no messages.
        Also resets the log.
        """
        if self._messages:
            res = " ".join(self._messages)
            self._reset_log()
            return res
        else:
            return "No known reason."

    def _show_any_warnings(self):
        """If there were any messages since the last reset, show them
        as a warning. Otherwise do nothing. Also resets the messages.
        """
        if self._messages:
            logger.warning("imageio.freeimage warning: " + self._get_error_message())
            self._reset_log()

    def get_output_log(self):
        """Return a list of the last 256 output messages
        (warnings and errors) produced by the FreeImage library.
        """
        # This message log is not cleared/reset, but kept to 256 elements.
        return [m for m in self._messages]

    def getFIF(self, filename, mode, bb=None):
        """Get the freeimage Format (FIF) from a given filename.
        If mode is 'r', will try to determine the format by reading
        the file, otherwise only the filename is used.

        This function also tests whether the format supports reading/writing.
        """
        with self as lib:

            # Init
            ftype = -1
            if mode not in "rw":
                raise ValueError('Invalid mode (must be "r" or "w").')

            # Try getting format from the content. Note that some files
            # do not have a header that allows reading the format from
            # the file.
            if mode == "r":
                if bb is not None:
                    fimemory = lib.FreeImage_OpenMemory(ctypes.c_char_p(bb), len(bb))
                    ftype = lib.FreeImage_GetFileTypeFromMemory(
                        ctypes.c_void_p(fimemory), len(bb)
                    )
                    lib.FreeImage_CloseMemory(ctypes.c_void_p(fimemory))
                if (ftype == -1) and os.path.isfile(filename):
                    ftype = lib.FreeImage_GetFileType(efn(filename), 0)
            # Try getting the format from the extension
            if ftype == -1:
                ftype = lib.FreeImage_GetFIFFromFilename(efn(filename))

            # Test if ok
            if ftype == -1:
                raise ValueError('Cannot determine format of file "%s"' % filename)
            elif mode == "w" and not lib.FreeImage_FIFSupportsWriting(ftype):
                raise ValueError('Cannot write the format of file "%s"' % filename)
            elif mode == "r" and not lib.FreeImage_FIFSupportsReading(ftype):
                raise ValueError('Cannot read the format of file "%s"' % filename)
            return ftype

    def create_bitmap(self, filename, ftype, flags=0):
        """create_bitmap(filename, ftype, flags=0)
        Create a wrapped bitmap object.
        """
        return FIBitmap(self, filename, ftype, flags)

    def create_multipage_bitmap(self, filename, ftype, flags=0):
        """create_multipage_bitmap(filename, ftype, flags=0)
        Create a wrapped multipage bitmap object.
        """
        return FIMultipageBitmap(self, filename, ftype, flags)


class FIBaseBitmap(object):
    def __init__(self, fi, filename, ftype, flags):
        self._fi = fi
        self._filename = filename
        self._ftype = ftype
        self._flags = flags
        self._bitmap = None
        self._close_funcs = []

    def __del__(self):
        self.close()

    def close(self):
        if (self._bitmap is not None) and self._close_funcs:
            for close_func in self._close_funcs:
                try:
                    with self._fi:
                        fun = close_func[0]
                        fun(*close_func[1:])
                except Exception:  # pragma: no cover
                    pass
            self._close_funcs = []
            self._bitmap = None

    def _set_bitmap(self, bitmap, close_func=None):
        """Function to set the bitmap and specify the function to unload it."""
        if self._bitmap is not None:
            pass  # bitmap is converted
        if close_func is None:
            close_func = self._fi.lib.FreeImage_Unload, bitmap

        self._bitmap = bitmap
        if close_func:
            self._close_funcs.append(close_func)

    def get_meta_data(self):

        # todo: there is also FreeImage_TagToString, is that useful?
        # and would that work well when reading and then saving?

        # Create a list of (model_name, number) tuples
        models = [
            (name[5:], number)
            for name, number in METADATA_MODELS.__dict__.items()
            if name.startswith("FIMD_")
        ]

        # Prepare
        metadata = Dict()
        tag = ctypes.c_void_p()

        with self._fi as lib:

            # Iterate over all FreeImage meta models
            for model_name, number in models:

                # Find beginning, get search handle
                mdhandle = lib.FreeImage_FindFirstMetadata(
                    number, self._bitmap, ctypes.byref(tag)
                )
                mdhandle = ctypes.c_void_p(mdhandle)
                if mdhandle:

                    # Iterate over all tags in this model
                    more = True
                    while more:
                        # Get info about tag
                        tag_name = lib.FreeImage_GetTagKey(tag).decode("utf-8")
                        tag_type = lib.FreeImage_GetTagType(tag)
                        byte_size = lib.FreeImage_GetTagLength(tag)
                        char_ptr = ctypes.c_char * byte_size
                        data = char_ptr.from_address(lib.FreeImage_GetTagValue(tag))
                        # Convert in a way compatible with Pypy
                        tag_bytes = bytes(bytearray(data))
                        # The default value is the raw bytes
                        tag_val = tag_bytes
                        # Convert to a Python value in the metadata dict
                        if tag_type == METADATA_DATATYPE.FIDT_ASCII:
                            tag_val = tag_bytes.decode("utf-8", "replace")
                        elif tag_type in METADATA_DATATYPE.dtypes:
                            dtype = METADATA_DATATYPE.dtypes[tag_type]
                            if IS_PYPY and isinstance(dtype, (list, tuple)):
                                pass  # pragma: no cover - or we get a segfault
                            else:
                                try:
                                    tag_val = numpy.frombuffer(
                                        tag_bytes, dtype=dtype
                                    ).copy()
                                    if len(tag_val) == 1:
                                        tag_val = tag_val[0]
                                except Exception:  # pragma: no cover
                                    pass
                        # Store data in dict
                        subdict = metadata.setdefault(model_name, Dict())
                        subdict[tag_name] = tag_val
                        # Next
                        more = lib.FreeImage_FindNextMetadata(
                            mdhandle, ctypes.byref(tag)
                        )

                    # Close search handle for current meta model
                    lib.FreeImage_FindCloseMetadata(mdhandle)

            # Done
            return metadata

    def set_meta_data(self, metadata):

        # Create a dict mapping model_name to number
        models = {}
        for name, number in METADATA_MODELS.__dict__.items():
            if name.startswith("FIMD_"):
                models[name[5:]] = number

        # Create a mapping from numpy.dtype to METADATA_DATATYPE
        def get_tag_type_number(dtype):
            for number, numpy_dtype in METADATA_DATATYPE.dtypes.items():
                if dtype == numpy_dtype:
                    return number
            else:
                return None

        with self._fi as lib:

            for model_name, subdict in metadata.items():

                # Get model number
                number = models.get(model_name, None)
                if number is None:
                    continue  # Unknown model, silent ignore

                for tag_name, tag_val in subdict.items():

                    # Create new tag
                    tag = lib.FreeImage_CreateTag()
                    tag = ctypes.c_void_p(tag)

                    try:
                        # Convert Python value to FI type, val
                        is_ascii = False
                        if isinstance(tag_val, str):
                            try:
                                tag_bytes = tag_val.encode("ascii")
                                is_ascii = True
                            except UnicodeError:
                                pass
                        if is_ascii:
                            tag_type = METADATA_DATATYPE.FIDT_ASCII
                            tag_count = len(tag_bytes)
                        else:
                            if not hasattr(tag_val, "dtype"):
                                tag_val = numpy.array([tag_val])
                            tag_type = get_tag_type_number(tag_val.dtype)
                            if tag_type is None:
                                logger.warning(
                                    "imageio.freeimage warning: Could not "
                                    "determine tag type of %r." % tag_name
                                )
                                continue
                            tag_bytes = tag_val.tobytes()
                            tag_count = tag_val.size
                        # Set properties
                        lib.FreeImage_SetTagKey(tag, tag_name.encode("utf-8"))
                        lib.FreeImage_SetTagType(tag, tag_type)
                        lib.FreeImage_SetTagCount(tag, tag_count)
                        lib.FreeImage_SetTagLength(tag, len(tag_bytes))
                        lib.FreeImage_SetTagValue(tag, tag_bytes)
                        # Store tag
                        tag_key = lib.FreeImage_GetTagKey(tag)
                        lib.FreeImage_SetMetadata(number, self._bitmap, tag_key, tag)

                    except Exception as err:  # pragma: no cover
                        logger.warning(
                            "imagio.freeimage warning: Could not set tag "
                            "%r: %s, %s"
                            % (tag_name, self._fi._get_error_message(), str(err))
                        )
                    finally:
                        lib.FreeImage_DeleteTag(tag)


class FIBitmap(FIBaseBitmap):
    """Wrapper for the FI bitmap object."""

    def allocate(self, array):

        # Prepare array
        assert isinstance(array, numpy.ndarray)
        shape = array.shape
        dtype = array.dtype

        # Get shape and channel info
        r, c = shape[:2]
        if len(shape) == 2:
            n_channels = 1
        elif len(shape) == 3:
            n_channels = shape[2]
        else:
            n_channels = shape[0]

        # Get fi_type
        try:
            fi_type = FI_TYPES.fi_types[(dtype.type, n_channels)]
            self._fi_type = fi_type
        except KeyError:
            raise ValueError("Cannot write arrays of given type and shape.")

        # Allocate bitmap
        with self._fi as lib:
            bpp = 8 * dtype.itemsize * n_channels
            bitmap = lib.FreeImage_AllocateT(fi_type, c, r, bpp, 0, 0, 0)
            bitmap = ctypes.c_void_p(bitmap)

            # Check and store
            if not bitmap:  # pragma: no cover
                raise RuntimeError(
                    "Could not allocate bitmap for storage: %s"
                    % self._fi._get_error_message()
                )
            self._set_bitmap(bitmap, (lib.FreeImage_Unload, bitmap))

    def load_from_filename(self, filename=None):
        if filename is None:
            filename = self._filename

        with self._fi as lib:
            # Create bitmap
            bitmap = lib.FreeImage_Load(self._ftype, efn(filename), self._flags)
            bitmap = ctypes.c_void_p(bitmap)

            # Check and store
            if not bitmap:  # pragma: no cover
                raise ValueError(
                    'Could not load bitmap "%s": %s'
                    % (self._filename, self._fi._get_error_message())
                )
            self._set_bitmap(bitmap, (lib.FreeImage_Unload, bitmap))

    # def load_from_bytes(self, bb):
    #     with self._fi as lib:
    #         # Create bitmap
    #         fimemory = lib.FreeImage_OpenMemory(
    #                                         ctypes.c_char_p(bb), len(bb))
    #         bitmap = lib.FreeImage_LoadFromMemory(
    #                         self._ftype, ctypes.c_void_p(fimemory), self._flags)
    #         bitmap = ctypes.c_void_p(bitmap)
    #         lib.FreeImage_CloseMemory(ctypes.c_void_p(fimemory))
    #
    #         # Check
    #         if not bitmap:
    #             raise ValueError('Could not load bitmap "%s": %s'
    #                         % (self._filename, self._fi._get_error_message()))
    #         else:
    #             self._set_bitmap(bitmap, (lib.FreeImage_Unload, bitmap))

    def save_to_filename(self, filename=None):
        if filename is None:
            filename = self._filename

        ftype = self._ftype
        bitmap = self._bitmap
        fi_type = self._fi_type  # element type

        with self._fi as lib:
            # Check if can write
            if fi_type == FI_TYPES.FIT_BITMAP:
                can_write = lib.FreeImage_FIFSupportsExportBPP(
                    ftype, lib.FreeImage_GetBPP(bitmap)
                )
            else:
                can_write = lib.FreeImage_FIFSupportsExportType(ftype, fi_type)
            if not can_write:
                raise TypeError("Cannot save image of this format to this file type")

            # Save to file
            res = lib.FreeImage_Save(ftype, bitmap, efn(filename), self._flags)
            # Check
            if res is None:  # pragma: no cover, we do so many checks, this is rare
                raise RuntimeError(
                    f"Could not save file `{self._filename}`: {self._fi._get_error_message()}"
                )

    # def save_to_bytes(self):
    #     ftype = self._ftype
    #     bitmap = self._bitmap
    #     fi_type = self._fi_type # element type
    #
    #     with self._fi as lib:
    #         # Check if can write
    #         if fi_type == FI_TYPES.FIT_BITMAP:
    #             can_write = lib.FreeImage_FIFSupportsExportBPP(ftype,
    #                                     lib.FreeImage_GetBPP(bitmap))
    #         else:
    #             can_write = lib.FreeImage_FIFSupportsExportType(ftype, fi_type)
    #         if not can_write:
    #             raise TypeError('Cannot save image of this format '
    #                             'to this file type')
    #
    #         # Extract the bytes
    #         fimemory = lib.FreeImage_OpenMemory(0, 0)
    #         res = lib.FreeImage_SaveToMemory(ftype, bitmap,
    #                                          ctypes.c_void_p(fimemory),
    #                                          self._flags)
    #         if res:
    #             N = lib.FreeImage_TellMemory(ctypes.c_void_p(fimemory))
    #             result = ctypes.create_string_buffer(N)
    #             lib.FreeImage_SeekMemory(ctypes.c_void_p(fimemory), 0)
    #             lib.FreeImage_ReadMemory(result, 1, N, ctypes.c_void_p(fimemory))
    #             result = result.raw
    #         lib.FreeImage_CloseMemory(ctypes.c_void_p(fimemory))
    #
    #         # Check
    #         if not res:
    #             raise RuntimeError('Could not save file "%s": %s'
    #                     % (self._filename, self._fi._get_error_message()))
    #
    #     # Done
    #     return result

    def get_image_data(self):
        dtype, shape, bpp = self._get_type_and_shape()
        array = self._wrap_bitmap_bits_in_array(shape, dtype, False)
        with self._fi as lib:
            isle = lib.FreeImage_IsLittleEndian()

        # swizzle the color components and flip the scanlines to go from
        # FreeImage's BGR[A] and upside-down internal memory format to
        # something more normal
        def n(arr):
            # return arr[..., ::-1].T  # Does not work on numpypy yet
            if arr.ndim == 1:  # pragma: no cover
                return arr[::-1].T
            elif arr.ndim == 2:  # Always the case here ...
                return arr[:, ::-1].T
            elif arr.ndim == 3:  # pragma: no cover
                return arr[:, :, ::-1].T
            elif arr.ndim == 4:  # pragma: no cover
                return arr[:, :, :, ::-1].T

        if len(shape) == 3 and isle and dtype.type == numpy.uint8:
            b = n(array[0])
            g = n(array[1])
            r = n(array[2])
            if shape[0] == 3:
                return numpy.dstack((r, g, b))
            elif shape[0] == 4:
                a = n(array[3])
                return numpy.dstack((r, g, b, a))
            else:  # pragma: no cover - we check this earlier
                raise ValueError("Cannot handle images of shape %s" % shape)

        # We need to copy because array does *not* own its memory
        # after bitmap is freed.
        a = n(array).copy()
        return a

    def set_image_data(self, array):

        # Prepare array
        assert isinstance(array, numpy.ndarray)
        shape = array.shape
        dtype = array.dtype
        with self._fi as lib:
            isle = lib.FreeImage_IsLittleEndian()

        # Calculate shape and channels
        r, c = shape[:2]
        if len(shape) == 2:
            n_channels = 1
            w_shape = (c, r)
        elif len(shape) == 3:
            n_channels = shape[2]
            w_shape = (n_channels, c, r)
        else:
            n_channels = shape[0]

        def n(arr):  # normalise to freeimage's in-memory format
            return arr[::-1].T

        wrapped_array = self._wrap_bitmap_bits_in_array(w_shape, dtype, True)
        # swizzle the color components and flip the scanlines to go to
        # FreeImage's BGR[A] and upside-down internal memory format
        # The BGR[A] order is only used for 8bits per channel images
        # on little endian machines. For everything else RGB[A] is
        # used.
        if len(shape) == 3 and isle and dtype.type == numpy.uint8:
            R = array[:, :, 0]
            G = array[:, :, 1]
            B = array[:, :, 2]
            wrapped_array[0] = n(B)
            wrapped_array[1] = n(G)
            wrapped_array[2] = n(R)
            if shape[2] == 4:
                A = array[:, :, 3]
                wrapped_array[3] = n(A)
        else:
            wrapped_array[:] = n(array)
        if self._need_finish:
            self._finish_wrapped_array(wrapped_array)

        if len(shape) == 2 and dtype.type == numpy.uint8:
            with self._fi as lib:
                palette = lib.FreeImage_GetPalette(self._bitmap)
            palette = ctypes.c_void_p(palette)
            if not palette:
                raise RuntimeError("Could not get image palette")
            try:
                palette_data = GREY_PALETTE.ctypes.data
            except Exception:  # pragma: no cover - IS_PYPY
                palette_data = GREY_PALETTE.__array_interface__["data"][0]
            ctypes.memmove(palette, palette_data, 1024)

    def _wrap_bitmap_bits_in_array(self, shape, dtype, save):
        """Return an ndarray view on the data in a FreeImage bitmap. Only
        valid for as long as the bitmap is loaded (if single page) / locked
        in memory (if multipage). This is used in loading data, but
        also during saving, to prepare a strided numpy array buffer.

        """
        # Get bitmap info
        with self._fi as lib:
            pitch = lib.FreeImage_GetPitch(self._bitmap)
            bits = lib.FreeImage_GetBits(self._bitmap)

        # Get more info
        height = shape[-1]
        byte_size = height * pitch
        itemsize = dtype.itemsize

        # Get strides
        if len(shape) == 3:
            strides = (itemsize, shape[0] * itemsize, pitch)
        else:
            strides = (itemsize, pitch)

        # Create numpy array and return
        data = (ctypes.c_char * byte_size).from_address(bits)
        try:
            self._need_finish = False
            if TEST_NUMPY_NO_STRIDES:
                raise NotImplementedError()
            return numpy.ndarray(shape, dtype=dtype, buffer=data, strides=strides)
        except NotImplementedError:
            # IS_PYPY - not very efficient. We create a C-contiguous
            # numpy array (because pypy does not support Fortran-order)
            # and shape it such that the rest of the code can remain.
            if save:
                self._need_finish = True  # Flag to use _finish_wrapped_array
                return numpy.zeros(shape, dtype=dtype)
            else:
                bb = bytes(bytearray(data))
                array = numpy.frombuffer(bb, dtype=dtype).copy()
                # Deal with strides
                if len(shape) == 3:
                    array.shape = shape[2], strides[-1] // shape[0], shape[0]
                    array2 = array[: shape[2], : shape[1], : shape[0]]
                    array = numpy.zeros(shape, dtype=array.dtype)
                    for i in range(shape[0]):
                        array[i] = array2[:, :, i].T
                else:
                    array.shape = shape[1], strides[-1]
                    array = array[: shape[1], : shape[0]].T
                return array

    def _finish_wrapped_array(self, array):  # IS_PYPY
        """Hardcore way to inject numpy array in bitmap."""
        # Get bitmap info
        with self._fi as lib:
            pitch = lib.FreeImage_GetPitch(self._bitmap)
            bits = lib.FreeImage_GetBits(self._bitmap)
            bpp = lib.FreeImage_GetBPP(self._bitmap)
        # Get channels and realwidth
        nchannels = bpp // 8 // array.itemsize
        realwidth = pitch // nchannels
        # Apply padding for pitch if necessary
        extra = realwidth - array.shape[-2]
        assert 0 <= extra < 10
        # Make sort of Fortran, also take padding (i.e. pitch) into account
        newshape = array.shape[-1], realwidth, nchannels
        array2 = numpy.zeros(newshape, array.dtype)
        if nchannels == 1:
            array2[:, : array.shape[-2], 0] = array.T
        else:
            for i in range(nchannels):
                array2[:, : array.shape[-2], i] = array[i, :, :].T
        # copy data
        data_ptr = array2.__array_interface__["data"][0]
        ctypes.memmove(bits, data_ptr, array2.nbytes)
        del array2

    def _get_type_and_shape(self):
        bitmap = self._bitmap

        # Get info on bitmap
        with self._fi as lib:
            w = lib.FreeImage_GetWidth(bitmap)
            h = lib.FreeImage_GetHeight(bitmap)
            self._fi_type = fi_type = lib.FreeImage_GetImageType(bitmap)
            if not fi_type:
                raise ValueError("Unknown image pixel type")

        # Determine required props for numpy array
        bpp = None
        dtype = FI_TYPES.dtypes[fi_type]

        if fi_type == FI_TYPES.FIT_BITMAP:
            with self._fi as lib:
                bpp = lib.FreeImage_GetBPP(bitmap)
                has_pallette = lib.FreeImage_GetColorsUsed(bitmap)
            if has_pallette:
                # Examine the palette. If it is grayscale, we return as such
                if has_pallette == 256:
                    palette = lib.FreeImage_GetPalette(bitmap)
                    palette = ctypes.c_void_p(palette)
                    p = (ctypes.c_uint8 * (256 * 4)).from_address(palette.value)
                    p = numpy.frombuffer(p, numpy.uint32).copy()
                    if (GREY_PALETTE == p).all():
                        extra_dims = []
                        return numpy.dtype(dtype), extra_dims + [w, h], bpp
                # Convert bitmap and call this method again
                newbitmap = lib.FreeImage_ConvertTo32Bits(bitmap)
                newbitmap = ctypes.c_void_p(newbitmap)
                self._set_bitmap(newbitmap)
                return self._get_type_and_shape()
            elif bpp == 8:
                extra_dims = []
            elif bpp == 24:
                extra_dims = [3]
            elif bpp == 32:
                extra_dims = [4]
            else:  # pragma: no cover
                # raise ValueError('Cannot convert %d BPP bitmap' % bpp)
                # Convert bitmap and call this method again
                newbitmap = lib.FreeImage_ConvertTo32Bits(bitmap)
                newbitmap = ctypes.c_void_p(newbitmap)
                self._set_bitmap(newbitmap)
                return self._get_type_and_shape()
        else:
            extra_dims = FI_TYPES.extra_dims[fi_type]

        # Return dtype and shape
        return numpy.dtype(dtype), extra_dims + [w, h], bpp

    def quantize(self, quantizer=0, palettesize=256):
        """Quantize the bitmap to make it 8-bit (paletted). Returns a new
        FIBitmap object.
        Only for 24 bit images.
        """
        with self._fi as lib:
            # New bitmap
            bitmap = lib.FreeImage_ColorQuantizeEx(
                self._bitmap, quantizer, palettesize, 0, None
            )
            bitmap = ctypes.c_void_p(bitmap)

            # Check and return
            if not bitmap:
                raise ValueError(
                    'Could not quantize bitmap "%s": %s'
                    % (self._filename, self._fi._get_error_message())
                )

            new = FIBitmap(self._fi, self._filename, self._ftype, self._flags)
            new._set_bitmap(bitmap, (lib.FreeImage_Unload, bitmap))
            new._fi_type = self._fi_type
            return new


# def convert_to_32bit(self):
#     """ Convert to 32bit image.
#     """
#     with self._fi as lib:
#         # New bitmap
#         bitmap = lib.FreeImage_ConvertTo32Bits(self._bitmap)
#         bitmap = ctypes.c_void_p(bitmap)
#
#         # Check and return
#         if not bitmap:
#             raise ValueError('Could not convert bitmap to 32bit "%s": %s' %
#                                 (self._filename,
#                                 self._fi._get_error_message()))
#         else:
#             new = FIBitmap(self._fi, self._filename, self._ftype,
#                             self._flags)
#             new._set_bitmap(bitmap, (lib.FreeImage_Unload, bitmap))
#             new._fi_type = self._fi_type
#             return new


class FIMultipageBitmap(FIBaseBitmap):
    """Wrapper for the multipage FI bitmap object."""

    def load_from_filename(self, filename=None):
        if filename is None:  # pragma: no cover
            filename = self._filename

        # Prepare
        create_new = False
        read_only = True
        keep_cache_in_memory = False

        # Try opening
        with self._fi as lib:

            # Create bitmap
            multibitmap = lib.FreeImage_OpenMultiBitmap(
                self._ftype,
                efn(filename),
                create_new,
                read_only,
                keep_cache_in_memory,
                self._flags,
            )
            multibitmap = ctypes.c_void_p(multibitmap)

            # Check
            if not multibitmap:  # pragma: no cover
                err = self._fi._get_error_message()
                raise ValueError(
                    'Could not open file "%s" as multi-image: %s'
                    % (self._filename, err)
                )
            self._set_bitmap(multibitmap, (lib.FreeImage_CloseMultiBitmap, multibitmap))

    # def load_from_bytes(self, bb):
    #     with self._fi as lib:
    #         # Create bitmap
    #         fimemory = lib.FreeImage_OpenMemory(
    #                                         ctypes.c_char_p(bb), len(bb))
    #         multibitmap = lib.FreeImage_LoadMultiBitmapFromMemory(
    #             self._ftype, ctypes.c_void_p(fimemory), self._flags)
    #         multibitmap = ctypes.c_void_p(multibitmap)
    #         #lib.FreeImage_CloseMemory(ctypes.c_void_p(fimemory))
    #         self._mem = fimemory
    #         self._bytes = bb
    #         # Check
    #         if not multibitmap:
    #             raise ValueError('Could not load multibitmap "%s": %s'
    #                         % (self._filename, self._fi._get_error_message()))
    #         else:
    #             self._set_bitmap(multibitmap,
    #                              (lib.FreeImage_CloseMultiBitmap, multibitmap))

    def save_to_filename(self, filename=None):
        if filename is None:  # pragma: no cover
            filename = self._filename

        # Prepare
        create_new = True
        read_only = False
        keep_cache_in_memory = False

        # Open the file
        # todo: Set flags at close func
        with self._fi as lib:
            multibitmap = lib.FreeImage_OpenMultiBitmap(
                self._ftype,
                efn(filename),
                create_new,
                read_only,
                keep_cache_in_memory,
                0,
            )
            multibitmap = ctypes.c_void_p(multibitmap)

            # Check
            if not multibitmap:  # pragma: no cover
                msg = 'Could not open file "%s" for writing multi-image: %s' % (
                    self._filename,
                    self._fi._get_error_message(),
                )
                raise ValueError(msg)
            self._set_bitmap(multibitmap, (lib.FreeImage_CloseMultiBitmap, multibitmap))

    def __len__(self):
        with self._fi as lib:
            return lib.FreeImage_GetPageCount(self._bitmap)

    def get_page(self, index):
        """Return the sub-bitmap for the given page index.
        Please close the returned bitmap when done.
        """
        with self._fi as lib:

            # Create low-level bitmap in freeimage
            bitmap = lib.FreeImage_LockPage(self._bitmap, index)
            bitmap = ctypes.c_void_p(bitmap)
            if not bitmap:  # pragma: no cover
                raise ValueError(
                    "Could not open sub-image %i in %r: %s"
                    % (index, self._filename, self._fi._get_error_message())
                )

            # Get bitmap object to wrap this bitmap
            bm = FIBitmap(self._fi, self._filename, self._ftype, self._flags)
            bm._set_bitmap(
                bitmap, (lib.FreeImage_UnlockPage, self._bitmap, bitmap, False)
            )
            return bm

    def append_bitmap(self, bitmap):
        """Add a sub-bitmap to the multi-page bitmap."""
        with self._fi as lib:
            # no return value
            lib.FreeImage_AppendPage(self._bitmap, bitmap._bitmap)


# Create instance
fi = Freeimage()
