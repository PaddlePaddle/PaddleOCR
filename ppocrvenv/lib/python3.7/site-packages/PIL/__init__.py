"""Pillow (Fork of the Python Imaging Library)

Pillow is the friendly PIL fork by Alex Clark and Contributors.
    https://github.com/python-pillow/Pillow/

Pillow is forked from PIL 1.1.7.

PIL is the Python Imaging Library by Fredrik Lundh and Contributors.
Copyright (c) 1999 by Secret Labs AB.

Use PIL.__version__ for this Pillow version.

;-)
"""

from . import _version

# VERSION was removed in Pillow 6.0.0.
# PILLOW_VERSION was removed in Pillow 9.0.0.
# Use __version__ instead.
__version__ = _version.__version__
del _version


_plugins = [
    "BlpImagePlugin",
    "BmpImagePlugin",
    "BufrStubImagePlugin",
    "CurImagePlugin",
    "DcxImagePlugin",
    "DdsImagePlugin",
    "EpsImagePlugin",
    "FitsStubImagePlugin",
    "FliImagePlugin",
    "FpxImagePlugin",
    "FtexImagePlugin",
    "GbrImagePlugin",
    "GifImagePlugin",
    "GribStubImagePlugin",
    "Hdf5StubImagePlugin",
    "IcnsImagePlugin",
    "IcoImagePlugin",
    "ImImagePlugin",
    "ImtImagePlugin",
    "IptcImagePlugin",
    "JpegImagePlugin",
    "Jpeg2KImagePlugin",
    "McIdasImagePlugin",
    "MicImagePlugin",
    "MpegImagePlugin",
    "MpoImagePlugin",
    "MspImagePlugin",
    "PalmImagePlugin",
    "PcdImagePlugin",
    "PcxImagePlugin",
    "PdfImagePlugin",
    "PixarImagePlugin",
    "PngImagePlugin",
    "PpmImagePlugin",
    "PsdImagePlugin",
    "SgiImagePlugin",
    "SpiderImagePlugin",
    "SunImagePlugin",
    "TgaImagePlugin",
    "TiffImagePlugin",
    "WebPImagePlugin",
    "WmfImagePlugin",
    "XbmImagePlugin",
    "XpmImagePlugin",
    "XVThumbImagePlugin",
]


class UnidentifiedImageError(OSError):
    """
    Raised in :py:meth:`PIL.Image.open` if an image cannot be opened and identified.
    """

    pass
