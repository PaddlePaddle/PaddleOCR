#
# The Python Imaging Library
# $Id$
#
# FITS stub adapter
#
# Copyright (c) 1998-2003 by Fredrik Lundh
#
# See the README file for information on usage and redistribution.
#

from . import Image, ImageFile

_handler = None


def register_handler(handler):
    """
    Install application-specific FITS image handler.

    :param handler: Handler object.
    """
    global _handler
    _handler = handler


# --------------------------------------------------------------------
# Image adapter


def _accept(prefix):
    return prefix[:6] == b"SIMPLE"


class FITSStubImageFile(ImageFile.StubImageFile):

    format = "FITS"
    format_description = "FITS"

    def _open(self):
        offset = self.fp.tell()

        headers = {}
        while True:
            header = self.fp.read(80)
            if not header:
                raise OSError("Truncated FITS file")
            keyword = header[:8].strip()
            if keyword == b"END":
                break
            value = header[8:].strip()
            if value.startswith(b"="):
                value = value[1:].strip()
            if not headers and (not _accept(keyword) or value != b"T"):
                raise SyntaxError("Not a FITS file")
            headers[keyword] = value

        naxis = int(headers[b"NAXIS"])
        if naxis == 0:
            raise ValueError("No image data")
        elif naxis == 1:
            self._size = 1, int(headers[b"NAXIS1"])
        else:
            self._size = int(headers[b"NAXIS1"]), int(headers[b"NAXIS2"])

        number_of_bits = int(headers[b"BITPIX"])
        if number_of_bits == 8:
            self.mode = "L"
        elif number_of_bits == 16:
            self.mode = "I"
            # rawmode = "I;16S"
        elif number_of_bits == 32:
            self.mode = "I"
        elif number_of_bits in (-32, -64):
            self.mode = "F"
            # rawmode = "F" if number_of_bits == -32 else "F;64F"

        self.fp.seek(offset)

        loader = self._load()
        if loader:
            loader.open(self)

    def _load(self):
        return _handler


def _save(im, fp, filename):
    if _handler is None or not hasattr("_handler", "save"):
        raise OSError("FITS save handler not installed")
    _handler.save(im, fp, filename)


# --------------------------------------------------------------------
# Registry

Image.register_open(FITSStubImageFile.format, FITSStubImageFile, _accept)
Image.register_save(FITSStubImageFile.format, _save)

Image.register_extensions(FITSStubImageFile.format, [".fit", ".fits"])
