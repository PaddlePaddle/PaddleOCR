#
# The Python Imaging Library.
# $Id$
#
# GD file handling
#
# History:
# 1996-04-12 fl   Created
#
# Copyright (c) 1997 by Secret Labs AB.
# Copyright (c) 1996 by Fredrik Lundh.
#
# See the README file for information on usage and redistribution.
#


"""
.. note::
    This format cannot be automatically recognized, so the
    class is not registered for use with :py:func:`PIL.Image.open()`.  To open a
    gd file, use the :py:func:`PIL.GdImageFile.open()` function instead.

.. warning::
    THE GD FORMAT IS NOT DESIGNED FOR DATA INTERCHANGE.  This
    implementation is provided for convenience and demonstrational
    purposes only.
"""


from . import ImageFile, ImagePalette, UnidentifiedImageError
from ._binary import i16be as i16
from ._binary import i32be as i32


class GdImageFile(ImageFile.ImageFile):
    """
    Image plugin for the GD uncompressed format.  Note that this format
    is not supported by the standard :py:func:`PIL.Image.open()` function.  To use
    this plugin, you have to import the :py:mod:`PIL.GdImageFile` module and
    use the :py:func:`PIL.GdImageFile.open()` function.
    """

    format = "GD"
    format_description = "GD uncompressed images"

    def _open(self):

        # Header
        s = self.fp.read(1037)

        if not i16(s) in [65534, 65535]:
            raise SyntaxError("Not a valid GD 2.x .gd file")

        self.mode = "L"  # FIXME: "P"
        self._size = i16(s, 2), i16(s, 4)

        trueColor = s[6]
        trueColorOffset = 2 if trueColor else 0

        # transparency index
        tindex = i32(s, 7 + trueColorOffset)
        if tindex < 256:
            self.info["transparency"] = tindex

        self.palette = ImagePalette.raw(
            "XBGR", s[7 + trueColorOffset + 4 : 7 + trueColorOffset + 4 + 256 * 4]
        )

        self.tile = [
            ("raw", (0, 0) + self.size, 7 + trueColorOffset + 4 + 256 * 4, ("L", 0, 1))
        ]


def open(fp, mode="r"):
    """
    Load texture from a GD image file.

    :param filename: GD file name, or an opened file handle.
    :param mode: Optional mode.  In this version, if the mode argument
        is given, it must be "r".
    :returns: An image instance.
    :raises OSError: If the image could not be read.
    """
    if mode != "r":
        raise ValueError("bad mode")

    try:
        return GdImageFile(fp)
    except SyntaxError as e:
        raise UnidentifiedImageError("cannot identify this image file") from e
