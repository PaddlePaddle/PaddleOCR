#
# The Python Imaging Library.
# $Id$
#
# IM Tools support for PIL
#
# history:
# 1996-05-27 fl   Created (read 8-bit images only)
# 2001-02-17 fl   Use 're' instead of 'regex' (Python 2.1) (0.2)
#
# Copyright (c) Secret Labs AB 1997-2001.
# Copyright (c) Fredrik Lundh 1996-2001.
#
# See the README file for information on usage and redistribution.
#


import re

from . import Image, ImageFile

#
# --------------------------------------------------------------------

field = re.compile(br"([a-z]*) ([^ \r\n]*)")


##
# Image plugin for IM Tools images.


class ImtImageFile(ImageFile.ImageFile):

    format = "IMT"
    format_description = "IM Tools"

    def _open(self):

        # Quick rejection: if there's not a LF among the first
        # 100 bytes, this is (probably) not a text header.

        if b"\n" not in self.fp.read(100):
            raise SyntaxError("not an IM file")
        self.fp.seek(0)

        xsize = ysize = 0

        while True:

            s = self.fp.read(1)
            if not s:
                break

            if s == b"\x0C":

                # image data begins
                self.tile = [
                    ("raw", (0, 0) + self.size, self.fp.tell(), (self.mode, 0, 1))
                ]

                break

            else:

                # read key/value pair
                # FIXME: dangerous, may read whole file
                s = s + self.fp.readline()
                if len(s) == 1 or len(s) > 100:
                    break
                if s[0] == ord(b"*"):
                    continue  # comment

                m = field.match(s)
                if not m:
                    break
                k, v = m.group(1, 2)
                if k == "width":
                    xsize = int(v)
                    self._size = xsize, ysize
                elif k == "height":
                    ysize = int(v)
                    self._size = xsize, ysize
                elif k == "pixel" and v == "n8":
                    self.mode = "L"


#
# --------------------------------------------------------------------

Image.register_open(ImtImageFile.format, ImtImageFile)

#
# no extension registered (".im" is simply too common)
