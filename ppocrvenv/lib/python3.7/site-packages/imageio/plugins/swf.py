# -*- coding: utf-8 -*-
# imageio is distributed under the terms of the (new) BSD License.

""" Read/Write SWF files.

Backend: internal

Shockwave flash (SWF) is a media format designed for rich and
interactive animations. This plugin makes use of this format to
store a series of images in a lossless format with good compression
(zlib). The resulting images can be shown as an animation using
a flash player (such as the browser).

SWF stores images in RGBA format. RGB or grayscale images are
automatically converted. SWF does not support meta data.

Parameters for reading
----------------------
loop : bool
    If True, the video will rewind as soon as a frame is requested
    beyond the last frame. Otherwise, IndexError is raised. Default False.

Parameters for saving
---------------------
fps : int
    The speed to play the animation. Default 12.
loop : bool
    If True, add a tag to the end of the file to play again from
    the first frame. Most flash players will then play the movie
    in a loop. Note that the imageio SWF Reader does not check this
    tag. Default True.
html : bool
    If the output is a file on the file system, write an html file
    (in HTML5) that shows the animation. Default False.
compress : bool
    Whether to compress the swf file. Default False. You probably don't
    want to use this. This does not decrease the file size since
    the images are already compressed. It will result in slower
    read and write time. The only purpose of this feature is to
    create compressed SWF files, so that we can test the
    functionality to read them.

"""

import os
import zlib
import logging
from io import BytesIO

import numpy as np

from ..core import Format, read_n_bytes, image_as_uint


logger = logging.getLogger(__name__)

_swf = None  # lazily loaded in lib()


def load_lib():
    global _swf
    from . import _swf

    return _swf


class SWFFormat(Format):
    """See :mod:`imageio.plugins.swf`"""

    def _can_read(self, request):
        if request.mode[1] in (self.modes + "?"):
            tmp = request.firstbytes[0:3].decode("ascii", "ignore")
            if tmp in ("FWS", "CWS"):
                return True

    def _can_write(self, request):
        if request.mode[1] in (self.modes + "?"):
            if request.extension in self.extensions:
                return True

    # -- reader

    class Reader(Format.Reader):
        def _open(self, loop=False):
            if not _swf:
                load_lib()

            self._arg_loop = bool(loop)

            self._fp = self.request.get_file()

            # Check file ...
            tmp = self.request.firstbytes[0:3].decode("ascii", "ignore")
            if tmp == "FWS":
                pass  # OK
            elif tmp == "CWS":
                # Compressed, we need to decompress
                bb = self._fp.read()
                bb = bb[:8] + zlib.decompress(bb[8:])
                # Wrap up in a file object
                self._fp = BytesIO(bb)
            else:
                raise IOError("This does not look like a valid SWF file")

            # Skip first bytes. This also tests support got seeking ...
            try:
                self._fp.seek(8)
                self._streaming_mode = False
            except Exception:
                self._streaming_mode = True
                self._fp_read(8)

            # Skip header
            # Note that the number of frames is there, which we could
            # potentially use, but the number of frames does not necessarily
            # correspond to the number of images.
            nbits = _swf.bits2int(self._fp_read(1), 5)
            nbits = 5 + nbits * 4
            Lrect = nbits / 8.0
            if Lrect % 1:
                Lrect += 1
            Lrect = int(Lrect)
            self._fp_read(Lrect + 3)

            # Now the rest is basically tags ...
            self._imlocs = []  # tuple (loc, sze, T, L1)
            if not self._streaming_mode:
                # Collect locations of frame, while skipping through the data
                # This does not read any of the tag *data*.
                try:
                    while True:
                        isimage, sze, T, L1 = self._read_one_tag()
                        loc = self._fp.tell()
                        if isimage:
                            # Still need to check if the format is right
                            format = ord(self._fp_read(3)[2:])
                            if format == 5:  # RGB or RGBA lossless
                                self._imlocs.append((loc, sze, T, L1))
                        self._fp.seek(loc + sze)  # Skip over tag
                except IndexError:
                    pass  # done reading

        def _fp_read(self, n):
            return read_n_bytes(self._fp, n)

        def _close(self):
            pass

        def _get_length(self):
            if self._streaming_mode:
                return np.inf
            else:
                return len(self._imlocs)

        def _get_data(self, index):
            # Check index
            if index < 0:
                raise IndexError("Index in swf file must be > 0")
            if not self._streaming_mode:
                if self._arg_loop and self._imlocs:
                    index = index % len(self._imlocs)
                if index >= len(self._imlocs):
                    raise IndexError("Index out of bounds")

            if self._streaming_mode:
                # Walk over tags until we find an image
                while True:
                    isimage, sze, T, L1 = self._read_one_tag()
                    bb = self._fp_read(sze)  # always read data
                    if isimage:
                        im = _swf.read_pixels(bb, 0, T, L1)  # can be None
                        if im is not None:
                            return im, {}

            else:
                # Go to corresponding location, read data, and convert to image
                loc, sze, T, L1 = self._imlocs[index]
                self._fp.seek(loc)
                bb = self._fp_read(sze)
                # Read_pixels should return ndarry, since we checked format
                im = _swf.read_pixels(bb, 0, T, L1)
                return im, {}

        def _read_one_tag(self):
            """
            Return (True, loc, size, T, L1) if an image that we can read.
            Return (False, loc, size, T, L1) if any other tag.
            """

            # Get head
            head = self._fp_read(6)
            if not head:  # pragma: no cover
                raise IndexError("Reached end of swf movie")

            # Determine type and length
            T, L1, L2 = _swf.get_type_and_len(head)
            if not L2:  # pragma: no cover
                raise RuntimeError("Invalid tag length, could not proceed")

            # Read data
            isimage = False
            sze = L2 - 6
            # bb = self._fp_read(L2 - 6)

            # Parse tag
            if T == 0:
                raise IndexError("Reached end of swf movie")
            elif T in [20, 36]:
                isimage = True
                # im = _swf.read_pixels(bb, 0, T, L1)  # can be None
            elif T in [6, 21, 35, 90]:  # pragma: no cover
                logger.warning("Ignoring JPEG image: cannot read JPEG.")
            else:
                pass  # Not an image tag

            # Done.  Return image. Can be None
            # return im
            return isimage, sze, T, L1

        def _get_meta_data(self, index):
            return {}  # This format does not support meta data

    # -- writer

    class Writer(Format.Writer):
        def _open(self, fps=12, loop=True, html=False, compress=False):
            if not _swf:
                load_lib()

            self._arg_fps = int(fps)
            self._arg_loop = bool(loop)
            self._arg_html = bool(html)
            self._arg_compress = bool(compress)

            self._fp = self.request.get_file()
            self._framecounter = 0
            self._framesize = (100, 100)

            # For compress, we use an in-memory file object
            if self._arg_compress:
                self._fp_real = self._fp
                self._fp = BytesIO()

        def _close(self):
            self._complete()
            # Get size of (uncompressed) file
            sze = self._fp.tell()
            # set nframes, this is in the potentially compressed region
            self._fp.seek(self._location_to_save_nframes)
            self._fp.write(_swf.int2uint16(self._framecounter))
            # Compress body?
            if self._arg_compress:
                bb = self._fp.getvalue()
                self._fp = self._fp_real
                self._fp.write(bb[:8])
                self._fp.write(zlib.compress(bb[8:]))
                sze = self._fp.tell()  # renew sze value
            # set size
            self._fp.seek(4)
            self._fp.write(_swf.int2uint32(sze))
            self._fp = None  # Disable

            # Write html?
            if self._arg_html and os.path.isfile(self.request.filename):
                dirname, fname = os.path.split(self.request.filename)
                filename = os.path.join(dirname, fname[:-4] + ".html")
                w, h = self._framesize
                html = HTML % (fname, w, h, fname)
                with open(filename, "wb") as f:
                    f.write(html.encode("utf-8"))

        def _write_header(self, framesize, fps):
            self._framesize = framesize
            # Called as soon as we know framesize; when we get first frame
            bb = b""
            bb += "FC"[self._arg_compress].encode("ascii")
            bb += "WS".encode("ascii")  # signature bytes
            bb += _swf.int2uint8(8)  # version
            bb += "0000".encode("ascii")  # FileLength (leave open for now)
            bb += (
                _swf.Tag().make_rect_record(0, framesize[0], 0, framesize[1]).tobytes()
            )
            bb += _swf.int2uint8(0) + _swf.int2uint8(fps)  # FrameRate
            self._location_to_save_nframes = len(bb)
            bb += "00".encode("ascii")  # nframes (leave open for now)
            self._fp.write(bb)

            # Write some initial tags
            taglist = _swf.FileAttributesTag(), _swf.SetBackgroundTag(0, 0, 0)
            for tag in taglist:
                self._fp.write(tag.get_tag())

        def _complete(self):
            # What if no images were saved?
            if not self._framecounter:
                self._write_header((10, 10), self._arg_fps)
            # Write stop tag if we do not loop
            if not self._arg_loop:
                self._fp.write(_swf.DoActionTag("stop").get_tag())
            # finish with end tag
            self._fp.write("\x00\x00".encode("ascii"))

        def _append_data(self, im, meta):
            # Correct shape and type
            if im.ndim == 3 and im.shape[-1] == 1:
                im = im[:, :, 0]
            im = image_as_uint(im, bitdepth=8)
            # Get frame size
            wh = im.shape[1], im.shape[0]
            # Write header on first frame
            isfirstframe = False
            if self._framecounter == 0:
                isfirstframe = True
                self._write_header(wh, self._arg_fps)
            # Create tags
            bm = _swf.BitmapTag(im)
            sh = _swf.ShapeTag(bm.id, (0, 0), wh)
            po = _swf.PlaceObjectTag(1, sh.id, move=(not isfirstframe))
            sf = _swf.ShowFrameTag()
            # Write tags
            for tag in [bm, sh, po, sf]:
                self._fp.write(tag.get_tag())
            self._framecounter += 1

        def set_meta_data(self, meta):
            pass


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Show Flash animation %s</title>
</head>
<body>
    <embed width="%i" height="%i" src="%s">
</html>
"""
