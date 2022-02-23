#
# The Python Imaging Library.
# $Id$
#
# MPO file handling
#
# See "Multi-Picture Format" (CIPA DC-007-Translation 2009, Standard of the
# Camera & Imaging Products Association)
#
# The multi-picture object combines multiple JPEG images (with a modified EXIF
# data format) into a single file. While it can theoretically be used much like
# a GIF animation, it is commonly used to represent 3D photographs and is (as
# of this writing) the most commonly used format by 3D cameras.
#
# History:
# 2014-03-13 Feneric   Created
#
# See the README file for information on usage and redistribution.
#

from . import Image, ImageFile, JpegImagePlugin
from ._binary import i16be as i16

# def _accept(prefix):
#     return JpegImagePlugin._accept(prefix)


def _save(im, fp, filename):
    # Note that we can only save the current frame at present
    return JpegImagePlugin._save(im, fp, filename)


##
# Image plugin for MPO images.


class MpoImageFile(JpegImagePlugin.JpegImageFile):

    format = "MPO"
    format_description = "MPO (CIPA DC-007)"
    _close_exclusive_fp_after_loading = False

    def _open(self):
        self.fp.seek(0)  # prep the fp in order to pass the JPEG test
        JpegImagePlugin.JpegImageFile._open(self)
        self._after_jpeg_open()

    def _after_jpeg_open(self, mpheader=None):
        self.mpinfo = mpheader if mpheader is not None else self._getmp()
        self.n_frames = self.mpinfo[0xB001]
        self.__mpoffsets = [
            mpent["DataOffset"] + self.info["mpoffset"] for mpent in self.mpinfo[0xB002]
        ]
        self.__mpoffsets[0] = 0
        # Note that the following assertion will only be invalid if something
        # gets broken within JpegImagePlugin.
        assert self.n_frames == len(self.__mpoffsets)
        del self.info["mpoffset"]  # no longer needed
        self.is_animated = self.n_frames > 1
        self.__fp = self.fp  # FIXME: hack
        self.__fp.seek(self.__mpoffsets[0])  # get ready to read first frame
        self.__frame = 0
        self.offset = 0
        # for now we can only handle reading and individual frame extraction
        self.readonly = 1

    def load_seek(self, pos):
        self.__fp.seek(pos)

    def seek(self, frame):
        if not self._seek_check(frame):
            return
        self.fp = self.__fp
        self.offset = self.__mpoffsets[frame]

        self.fp.seek(self.offset + 2)  # skip SOI marker
        segment = self.fp.read(2)
        if not segment:
            raise ValueError("No data found for frame")
        if i16(segment) == 0xFFE1:  # APP1
            n = i16(self.fp.read(2)) - 2
            self.info["exif"] = ImageFile._safe_read(self.fp, n)

            mptype = self.mpinfo[0xB002][frame]["Attribute"]["MPType"]
            if mptype.startswith("Large Thumbnail"):
                exif = self.getexif().get_ifd(0x8769)
                if 40962 in exif and 40963 in exif:
                    self._size = (exif[40962], exif[40963])
        elif "exif" in self.info:
            del self.info["exif"]

        self.tile = [("jpeg", (0, 0) + self.size, self.offset, (self.mode, ""))]
        self.__frame = frame

    def tell(self):
        return self.__frame

    def _close__fp(self):
        try:
            if self.__fp != self.fp:
                self.__fp.close()
        except AttributeError:
            pass
        finally:
            self.__fp = None

    @staticmethod
    def adopt(jpeg_instance, mpheader=None):
        """
        Transform the instance of JpegImageFile into
        an instance of MpoImageFile.
        After the call, the JpegImageFile is extended
        to be an MpoImageFile.

        This is essentially useful when opening a JPEG
        file that reveals itself as an MPO, to avoid
        double call to _open.
        """
        jpeg_instance.__class__ = MpoImageFile
        jpeg_instance._after_jpeg_open(mpheader)
        return jpeg_instance


# ---------------------------------------------------------------------
# Registry stuff

# Note that since MPO shares a factory with JPEG, we do not need to do a
# separate registration for it here.
# Image.register_open(MpoImageFile.format,
#                     JpegImagePlugin.jpeg_factory, _accept)
Image.register_save(MpoImageFile.format, _save)

Image.register_extension(MpoImageFile.format, ".mpo")

Image.register_mime(MpoImageFile.format, "image/mpo")
