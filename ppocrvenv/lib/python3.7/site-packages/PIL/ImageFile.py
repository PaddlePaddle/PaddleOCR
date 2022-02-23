#
# The Python Imaging Library.
# $Id$
#
# base class for image file handlers
#
# history:
# 1995-09-09 fl   Created
# 1996-03-11 fl   Fixed load mechanism.
# 1996-04-15 fl   Added pcx/xbm decoders.
# 1996-04-30 fl   Added encoders.
# 1996-12-14 fl   Added load helpers
# 1997-01-11 fl   Use encode_to_file where possible
# 1997-08-27 fl   Flush output in _save
# 1998-03-05 fl   Use memory mapping for some modes
# 1999-02-04 fl   Use memory mapping also for "I;16" and "I;16B"
# 1999-05-31 fl   Added image parser
# 2000-10-12 fl   Set readonly flag on memory-mapped images
# 2002-03-20 fl   Use better messages for common decoder errors
# 2003-04-21 fl   Fall back on mmap/map_buffer if map is not available
# 2003-10-30 fl   Added StubImageFile class
# 2004-02-25 fl   Made incremental parser more robust
#
# Copyright (c) 1997-2004 by Secret Labs AB
# Copyright (c) 1995-2004 by Fredrik Lundh
#
# See the README file for information on usage and redistribution.
#

import io
import itertools
import struct
import sys

from . import Image
from ._util import isPath

MAXBLOCK = 65536

SAFEBLOCK = 1024 * 1024

LOAD_TRUNCATED_IMAGES = False
"""Whether or not to load truncated image files. User code may change this."""

ERRORS = {
    -1: "image buffer overrun error",
    -2: "decoding error",
    -3: "unknown error",
    -8: "bad configuration",
    -9: "out of memory error",
}
"""Dict of known error codes returned from :meth:`.PyDecoder.decode`."""


#
# --------------------------------------------------------------------
# Helpers


def raise_oserror(error):
    try:
        message = Image.core.getcodecstatus(error)
    except AttributeError:
        message = ERRORS.get(error)
    if not message:
        message = f"decoder error {error}"
    raise OSError(message + " when reading image file")


def _tilesort(t):
    # sort on offset
    return t[2]


#
# --------------------------------------------------------------------
# ImageFile base class


class ImageFile(Image.Image):
    """Base class for image file format handlers."""

    def __init__(self, fp=None, filename=None):
        super().__init__()

        self._min_frame = 0

        self.custom_mimetype = None

        self.tile = None
        """ A list of tile descriptors, or ``None`` """

        self.readonly = 1  # until we know better

        self.decoderconfig = ()
        self.decodermaxblock = MAXBLOCK

        if isPath(fp):
            # filename
            self.fp = open(fp, "rb")
            self.filename = fp
            self._exclusive_fp = True
        else:
            # stream
            self.fp = fp
            self.filename = filename
            # can be overridden
            self._exclusive_fp = None

        try:
            try:
                self._open()
            except (
                IndexError,  # end of data
                TypeError,  # end of data (ord)
                KeyError,  # unsupported mode
                EOFError,  # got header but not the first frame
                struct.error,
            ) as v:
                raise SyntaxError(v) from v

            if not self.mode or self.size[0] <= 0:
                raise SyntaxError("not identified by this driver")
        except BaseException:
            # close the file only if we have opened it this constructor
            if self._exclusive_fp:
                self.fp.close()
            raise

    def get_format_mimetype(self):
        if self.custom_mimetype:
            return self.custom_mimetype
        if self.format is not None:
            return Image.MIME.get(self.format.upper())

    def verify(self):
        """Check file integrity"""

        # raise exception if something's wrong.  must be called
        # directly after open, and closes file when finished.
        if self._exclusive_fp:
            self.fp.close()
        self.fp = None

    def load(self):
        """Load image data based on tile list"""

        if self.tile is None:
            raise OSError("cannot load this image")

        pixel = Image.Image.load(self)
        if not self.tile:
            return pixel

        self.map = None
        use_mmap = self.filename and len(self.tile) == 1
        # As of pypy 2.1.0, memory mapping was failing here.
        use_mmap = use_mmap and not hasattr(sys, "pypy_version_info")

        readonly = 0

        # look for read/seek overrides
        try:
            read = self.load_read
            # don't use mmap if there are custom read/seek functions
            use_mmap = False
        except AttributeError:
            read = self.fp.read

        try:
            seek = self.load_seek
            use_mmap = False
        except AttributeError:
            seek = self.fp.seek

        if use_mmap:
            # try memory mapping
            decoder_name, extents, offset, args = self.tile[0]
            if (
                decoder_name == "raw"
                and len(args) >= 3
                and args[0] == self.mode
                and args[0] in Image._MAPMODES
            ):
                try:
                    # use mmap, if possible
                    import mmap

                    with open(self.filename) as fp:
                        self.map = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
                    self.im = Image.core.map_buffer(
                        self.map, self.size, decoder_name, offset, args
                    )
                    readonly = 1
                    # After trashing self.im,
                    # we might need to reload the palette data.
                    if self.palette:
                        self.palette.dirty = 1
                except (AttributeError, OSError, ImportError):
                    self.map = None

        self.load_prepare()
        err_code = -3  # initialize to unknown error
        if not self.map:
            # sort tiles in file order
            self.tile.sort(key=_tilesort)

            try:
                # FIXME: This is a hack to handle TIFF's JpegTables tag.
                prefix = self.tile_prefix
            except AttributeError:
                prefix = b""

            # Remove consecutive duplicates that only differ by their offset
            self.tile = [
                list(tiles)[-1]
                for _, tiles in itertools.groupby(
                    self.tile, lambda tile: (tile[0], tile[1], tile[3])
                )
            ]
            for decoder_name, extents, offset, args in self.tile:
                decoder = Image._getdecoder(
                    self.mode, decoder_name, args, self.decoderconfig
                )
                try:
                    seek(offset)
                    decoder.setimage(self.im, extents)
                    if decoder.pulls_fd:
                        decoder.setfd(self.fp)
                        status, err_code = decoder.decode(b"")
                    else:
                        b = prefix
                        while True:
                            try:
                                s = read(self.decodermaxblock)
                            except (IndexError, struct.error) as e:
                                # truncated png/gif
                                if LOAD_TRUNCATED_IMAGES:
                                    break
                                else:
                                    raise OSError("image file is truncated") from e

                            if not s:  # truncated jpeg
                                if LOAD_TRUNCATED_IMAGES:
                                    break
                                else:
                                    raise OSError(
                                        "image file is truncated "
                                        f"({len(b)} bytes not processed)"
                                    )

                            b = b + s
                            n, err_code = decoder.decode(b)
                            if n < 0:
                                break
                            b = b[n:]
                finally:
                    # Need to cleanup here to prevent leaks
                    decoder.cleanup()

        self.tile = []
        self.readonly = readonly

        self.load_end()

        if self._exclusive_fp and self._close_exclusive_fp_after_loading:
            self.fp.close()
        self.fp = None

        if not self.map and not LOAD_TRUNCATED_IMAGES and err_code < 0:
            # still raised if decoder fails to return anything
            raise_oserror(err_code)

        return Image.Image.load(self)

    def load_prepare(self):
        # create image memory if necessary
        if not self.im or self.im.mode != self.mode or self.im.size != self.size:
            self.im = Image.core.new(self.mode, self.size)
        # create palette (optional)
        if self.mode == "P":
            Image.Image.load(self)

    def load_end(self):
        # may be overridden
        pass

    # may be defined for contained formats
    # def load_seek(self, pos):
    #     pass

    # may be defined for blocked formats (e.g. PNG)
    # def load_read(self, bytes):
    #     pass

    def _seek_check(self, frame):
        if (
            frame < self._min_frame
            # Only check upper limit on frames if additional seek operations
            # are not required to do so
            or (
                not (hasattr(self, "_n_frames") and self._n_frames is None)
                and frame >= self.n_frames + self._min_frame
            )
        ):
            raise EOFError("attempt to seek outside sequence")

        return self.tell() != frame


class StubImageFile(ImageFile):
    """
    Base class for stub image loaders.

    A stub loader is an image loader that can identify files of a
    certain format, but relies on external code to load the file.
    """

    def _open(self):
        raise NotImplementedError("StubImageFile subclass must implement _open")

    def load(self):
        loader = self._load()
        if loader is None:
            raise OSError(f"cannot find loader for this {self.format} file")
        image = loader.load(self)
        assert image is not None
        # become the other object (!)
        self.__class__ = image.__class__
        self.__dict__ = image.__dict__

    def _load(self):
        """(Hook) Find actual image loader."""
        raise NotImplementedError("StubImageFile subclass must implement _load")


class Parser:
    """
    Incremental image parser.  This class implements the standard
    feed/close consumer interface.
    """

    incremental = None
    image = None
    data = None
    decoder = None
    offset = 0
    finished = 0

    def reset(self):
        """
        (Consumer) Reset the parser.  Note that you can only call this
        method immediately after you've created a parser; parser
        instances cannot be reused.
        """
        assert self.data is None, "cannot reuse parsers"

    def feed(self, data):
        """
        (Consumer) Feed data to the parser.

        :param data: A string buffer.
        :exception OSError: If the parser failed to parse the image file.
        """
        # collect data

        if self.finished:
            return

        if self.data is None:
            self.data = data
        else:
            self.data = self.data + data

        # parse what we have
        if self.decoder:

            if self.offset > 0:
                # skip header
                skip = min(len(self.data), self.offset)
                self.data = self.data[skip:]
                self.offset = self.offset - skip
                if self.offset > 0 or not self.data:
                    return

            n, e = self.decoder.decode(self.data)

            if n < 0:
                # end of stream
                self.data = None
                self.finished = 1
                if e < 0:
                    # decoding error
                    self.image = None
                    raise_oserror(e)
                else:
                    # end of image
                    return
            self.data = self.data[n:]

        elif self.image:

            # if we end up here with no decoder, this file cannot
            # be incrementally parsed.  wait until we've gotten all
            # available data
            pass

        else:

            # attempt to open this file
            try:
                with io.BytesIO(self.data) as fp:
                    im = Image.open(fp)
            except OSError:
                # traceback.print_exc()
                pass  # not enough data
            else:
                flag = hasattr(im, "load_seek") or hasattr(im, "load_read")
                if flag or len(im.tile) != 1:
                    # custom load code, or multiple tiles
                    self.decode = None
                else:
                    # initialize decoder
                    im.load_prepare()
                    d, e, o, a = im.tile[0]
                    im.tile = []
                    self.decoder = Image._getdecoder(im.mode, d, a, im.decoderconfig)
                    self.decoder.setimage(im.im, e)

                    # calculate decoder offset
                    self.offset = o
                    if self.offset <= len(self.data):
                        self.data = self.data[self.offset :]
                        self.offset = 0

                self.image = im

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        """
        (Consumer) Close the stream.

        :returns: An image object.
        :exception OSError: If the parser failed to parse the image file either
                            because it cannot be identified or cannot be
                            decoded.
        """
        # finish decoding
        if self.decoder:
            # get rid of what's left in the buffers
            self.feed(b"")
            self.data = self.decoder = None
            if not self.finished:
                raise OSError("image was incomplete")
        if not self.image:
            raise OSError("cannot parse this image")
        if self.data:
            # incremental parsing not possible; reopen the file
            # not that we have all data
            with io.BytesIO(self.data) as fp:
                try:
                    self.image = Image.open(fp)
                finally:
                    self.image.load()
        return self.image


# --------------------------------------------------------------------


def _save(im, fp, tile, bufsize=0):
    """Helper to save image based on tile list

    :param im: Image object.
    :param fp: File object.
    :param tile: Tile list.
    :param bufsize: Optional buffer size
    """

    im.load()
    if not hasattr(im, "encoderconfig"):
        im.encoderconfig = ()
    tile.sort(key=_tilesort)
    # FIXME: make MAXBLOCK a configuration parameter
    # It would be great if we could have the encoder specify what it needs
    # But, it would need at least the image size in most cases. RawEncode is
    # a tricky case.
    bufsize = max(MAXBLOCK, bufsize, im.size[0] * 4)  # see RawEncode.c
    try:
        fh = fp.fileno()
        fp.flush()
    except (AttributeError, io.UnsupportedOperation) as exc:
        # compress to Python file-compatible object
        for e, b, o, a in tile:
            e = Image._getencoder(im.mode, e, a, im.encoderconfig)
            if o > 0:
                fp.seek(o)
            e.setimage(im.im, b)
            if e.pushes_fd:
                e.setfd(fp)
                l, s = e.encode_to_pyfd()
            else:
                while True:
                    l, s, d = e.encode(bufsize)
                    fp.write(d)
                    if s:
                        break
            if s < 0:
                raise OSError(f"encoder error {s} when writing image file") from exc
            e.cleanup()
    else:
        # slight speedup: compress to real file object
        for e, b, o, a in tile:
            e = Image._getencoder(im.mode, e, a, im.encoderconfig)
            if o > 0:
                fp.seek(o)
            e.setimage(im.im, b)
            if e.pushes_fd:
                e.setfd(fp)
                l, s = e.encode_to_pyfd()
            else:
                s = e.encode_to_file(fh, bufsize)
            if s < 0:
                raise OSError(f"encoder error {s} when writing image file")
            e.cleanup()
    if hasattr(fp, "flush"):
        fp.flush()


def _safe_read(fp, size):
    """
    Reads large blocks in a safe way.  Unlike fp.read(n), this function
    doesn't trust the user.  If the requested size is larger than
    SAFEBLOCK, the file is read block by block.

    :param fp: File handle.  Must implement a <b>read</b> method.
    :param size: Number of bytes to read.
    :returns: A string containing <i>size</i> bytes of data.

    Raises an OSError if the file is truncated and the read cannot be completed

    """
    if size <= 0:
        return b""
    if size <= SAFEBLOCK:
        data = fp.read(size)
        if len(data) < size:
            raise OSError("Truncated File Read")
        return data
    data = []
    remaining_size = size
    while remaining_size > 0:
        block = fp.read(min(remaining_size, SAFEBLOCK))
        if not block:
            break
        data.append(block)
        remaining_size -= len(block)
    if sum(len(d) for d in data) < size:
        raise OSError("Truncated File Read")
    return b"".join(data)


class PyCodecState:
    def __init__(self):
        self.xsize = 0
        self.ysize = 0
        self.xoff = 0
        self.yoff = 0

    def extents(self):
        return (self.xoff, self.yoff, self.xoff + self.xsize, self.yoff + self.ysize)


class PyDecoder:
    """
    Python implementation of a format decoder. Override this class and
    add the decoding logic in the :meth:`decode` method.

    See :ref:`Writing Your Own File Decoder in Python<file-decoders-py>`
    """

    _pulls_fd = False

    def __init__(self, mode, *args):
        self.im = None
        self.state = PyCodecState()
        self.fd = None
        self.mode = mode
        self.init(args)

    def init(self, args):
        """
        Override to perform decoder specific initialization

        :param args: Array of args items from the tile entry
        :returns: None
        """
        self.args = args

    @property
    def pulls_fd(self):
        return self._pulls_fd

    def decode(self, buffer):
        """
        Override to perform the decoding process.

        :param buffer: A bytes object with the data to be decoded.
        :returns: A tuple of ``(bytes consumed, errcode)``.
            If finished with decoding return <0 for the bytes consumed.
            Err codes are from :data:`.ImageFile.ERRORS`.
        """
        raise NotImplementedError()

    def cleanup(self):
        """
        Override to perform decoder specific cleanup

        :returns: None
        """
        pass

    def setfd(self, fd):
        """
        Called from ImageFile to set the python file-like object

        :param fd: A python file-like object
        :returns: None
        """
        self.fd = fd

    def setimage(self, im, extents=None):
        """
        Called from ImageFile to set the core output image for the decoder

        :param im: A core image object
        :param extents: a 4 tuple of (x0, y0, x1, y1) defining the rectangle
            for this tile
        :returns: None
        """

        # following c code
        self.im = im

        if extents:
            (x0, y0, x1, y1) = extents
        else:
            (x0, y0, x1, y1) = (0, 0, 0, 0)

        if x0 == 0 and x1 == 0:
            self.state.xsize, self.state.ysize = self.im.size
        else:
            self.state.xoff = x0
            self.state.yoff = y0
            self.state.xsize = x1 - x0
            self.state.ysize = y1 - y0

        if self.state.xsize <= 0 or self.state.ysize <= 0:
            raise ValueError("Size cannot be negative")

        if (
            self.state.xsize + self.state.xoff > self.im.size[0]
            or self.state.ysize + self.state.yoff > self.im.size[1]
        ):
            raise ValueError("Tile cannot extend outside image")

    def set_as_raw(self, data, rawmode=None):
        """
        Convenience method to set the internal image from a stream of raw data

        :param data: Bytes to be set
        :param rawmode: The rawmode to be used for the decoder.
            If not specified, it will default to the mode of the image
        :returns: None
        """

        if not rawmode:
            rawmode = self.mode
        d = Image._getdecoder(self.mode, "raw", (rawmode))
        d.setimage(self.im, self.state.extents())
        s = d.decode(data)

        if s[0] >= 0:
            raise ValueError("not enough image data")
        if s[1] != 0:
            raise ValueError("cannot decode image data")
