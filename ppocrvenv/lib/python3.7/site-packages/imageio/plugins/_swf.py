# -*- coding: utf-8 -*-
# imageio is distributed under the terms of the (new) BSD License.
# This code was taken from https://github.com/almarklein/visvis/blob/master/vvmovie/images2swf.py

# styletest: ignore E261

"""
Provides a function (write_swf) to store a series of numpy arrays in an
SWF movie, that can be played on a wide range of OS's.

In desperation of wanting to share animated images, and then lacking a good
writer for animated gif or .avi, I decided to look into SWF. This format
is very well documented.

This is a pure python module to create an SWF file that shows a series
of images. The images are stored using the DEFLATE algorithm (same as
PNG and ZIP and which is included in the standard Python distribution).
As this compression algorithm is much more effective than that used in
GIF images, we obtain better quality (24 bit colors + alpha channel)
while still producesing smaller files (a test showed ~75%). Although
SWF also allows for JPEG compression, doing so would probably require
a third party library for the JPEG encoding/decoding, we could
perhaps do this via Pillow or freeimage.

sources and tools:

- SWF on wikipedia
- Adobes "SWF File Format Specification" version 10
  (http://www.adobe.com/devnet/swf/pdf/swf_file_format_spec_v10.pdf)
- swftools (swfdump in specific) for debugging
- iwisoft swf2avi can be used to convert swf to avi/mpg/flv with really
  good quality, while file size is reduced with factors 20-100.
  A good program in my opinion. The free version has the limitation
  of a watermark in the upper left corner.

"""

import os
import zlib
import time  # noqa
import logging

import numpy as np


logger = logging.getLogger(__name__)

# todo: use Pillow to support reading JPEG images from SWF?


# Base functions and classes


class BitArray:
    """Dynamic array of bits that automatically resizes
    with factors of two.
    Append bits using .append() or +=
    You can reverse bits using .reverse()
    """

    def __init__(self, initvalue=None):
        self.data = np.zeros((16,), dtype=np.uint8)
        self._len = 0
        if initvalue is not None:
            self.append(initvalue)

    def __len__(self):
        return self._len  # self.data.shape[0]

    def __repr__(self):
        return self.data[: self._len].tobytes().decode("ascii")

    def _checkSize(self):
        # check length... grow if necessary
        arraylen = self.data.shape[0]
        if self._len >= arraylen:
            tmp = np.zeros((arraylen * 2,), dtype=np.uint8)
            tmp[: self._len] = self.data[: self._len]
            self.data = tmp

    def __add__(self, value):
        self.append(value)
        return self

    def append(self, bits):

        # check input
        if isinstance(bits, BitArray):
            bits = str(bits)
        if isinstance(bits, int):  # pragma: no cover - we dont use it
            bits = str(bits)
        if not isinstance(bits, str):  # pragma: no cover
            raise ValueError("Append bits as strings or integers!")

        # add bits
        for bit in bits:
            self.data[self._len] = ord(bit)
            self._len += 1
            self._checkSize()

    def reverse(self):
        """In-place reverse."""
        tmp = self.data[: self._len].copy()
        self.data[: self._len] = tmp[::-1]

    def tobytes(self):
        """Convert to bytes. If necessary,
        zeros are padded to the end (right side).
        """
        bits = str(self)

        # determine number of bytes
        nbytes = 0
        while nbytes * 8 < len(bits):
            nbytes += 1
        # pad
        bits = bits.ljust(nbytes * 8, "0")

        # go from bits to bytes
        bb = bytes()
        for i in range(nbytes):
            tmp = int(bits[i * 8 : (i + 1) * 8], 2)
            bb += int2uint8(tmp)

        # done
        return bb


def int2uint32(i):
    return int(i).to_bytes(4, "little")


def int2uint16(i):
    return int(i).to_bytes(2, "little")


def int2uint8(i):
    return int(i).to_bytes(1, "little")


def int2bits(i, n=None):
    """convert int to a string of bits (0's and 1's in a string),
    pad to n elements. Convert back using int(ss,2)."""
    ii = i

    # make bits
    bb = BitArray()
    while ii > 0:
        bb += str(ii % 2)
        ii = ii >> 1
    bb.reverse()

    # justify
    if n is not None:
        if len(bb) > n:  # pragma: no cover
            raise ValueError("int2bits fail: len larger than padlength.")
        bb = str(bb).rjust(n, "0")

    # done
    return BitArray(bb)


def bits2int(bb, n=8):
    # Init
    value = ""

    # Get value in bits
    for i in range(len(bb)):
        b = bb[i : i + 1]
        tmp = bin(ord(b))[2:]
        # value += tmp.rjust(8,'0')
        value = tmp.rjust(8, "0") + value

    # Make decimal
    return int(value[:n], 2)


def get_type_and_len(bb):
    """bb should be 6 bytes at least
    Return (type, length, length_of_full_tag)
    """
    # Init
    value = ""

    # Get first 16 bits
    for i in range(2):
        b = bb[i : i + 1]
        tmp = bin(ord(b))[2:]
        # value += tmp.rjust(8,'0')
        value = tmp.rjust(8, "0") + value

    # Get type and length
    type = int(value[:10], 2)
    L = int(value[10:], 2)
    L2 = L + 2

    # Long tag header?
    if L == 63:  # '111111'
        value = ""
        for i in range(2, 6):
            b = bb[i : i + 1]  # becomes a single-byte bytes()
            tmp = bin(ord(b))[2:]
            # value += tmp.rjust(8,'0')
            value = tmp.rjust(8, "0") + value
        L = int(value, 2)
        L2 = L + 6

    # Done
    return type, L, L2


def signedint2bits(i, n=None):
    """convert signed int to a string of bits (0's and 1's in a string),
    pad to n elements. Negative numbers are stored in 2's complement bit
    patterns, thus positive numbers always start with a 0.
    """

    # negative number?
    ii = i
    if i < 0:
        # A negative number, -n, is represented as the bitwise opposite of
        ii = abs(ii) - 1  # the positive-zero number n-1.

    # make bits
    bb = BitArray()
    while ii > 0:
        bb += str(ii % 2)
        ii = ii >> 1
    bb.reverse()

    # justify
    bb = "0" + str(bb)  # always need the sign bit in front
    if n is not None:
        if len(bb) > n:  # pragma: no cover
            raise ValueError("signedint2bits fail: len larger than padlength.")
        bb = bb.rjust(n, "0")

    # was it negative? (then opposite bits)
    if i < 0:
        bb = bb.replace("0", "x").replace("1", "0").replace("x", "1")

    # done
    return BitArray(bb)


def twits2bits(arr):
    """Given a few (signed) numbers, store them
    as compactly as possible in the wat specifief by the swf format.
    The numbers are multiplied by 20, assuming they
    are twits.
    Can be used to make the RECT record.
    """

    # first determine length using non justified bit strings
    maxlen = 1
    for i in arr:
        tmp = len(signedint2bits(i * 20))
        if tmp > maxlen:
            maxlen = tmp

    # build array
    bits = int2bits(maxlen, 5)
    for i in arr:
        bits += signedint2bits(i * 20, maxlen)

    return bits


def floats2bits(arr):
    """Given a few (signed) numbers, convert them to bits,
    stored as FB (float bit values). We always use 16.16.
    Negative numbers are not (yet) possible, because I don't
    know how the're implemented (ambiguity).
    """
    bits = int2bits(31, 5)  # 32 does not fit in 5 bits!
    for i in arr:
        if i < 0:  # pragma: no cover
            raise ValueError("Dit not implement negative floats!")
        i1 = int(i)
        i2 = i - i1
        bits += int2bits(i1, 15)
        bits += int2bits(i2 * 2 ** 16, 16)
    return bits


# Base Tag


class Tag:
    def __init__(self):
        self.bytes = bytes()
        self.tagtype = -1

    def process_tag(self):
        """Implement this to create the tag."""
        raise NotImplementedError()

    def get_tag(self):
        """Calls processTag and attaches the header."""
        self.process_tag()

        # tag to binary
        bits = int2bits(self.tagtype, 10)

        # complete header uint16 thing
        bits += "1" * 6  # = 63 = 0x3f
        # make uint16
        bb = int2uint16(int(str(bits), 2))

        # now add 32bit length descriptor
        bb += int2uint32(len(self.bytes))

        # done, attach and return
        bb += self.bytes
        return bb

    def make_rect_record(self, xmin, xmax, ymin, ymax):
        """Simply uses makeCompactArray to produce
        a RECT Record."""
        return twits2bits([xmin, xmax, ymin, ymax])

    def make_matrix_record(self, scale_xy=None, rot_xy=None, trans_xy=None):

        # empty matrix?
        if scale_xy is None and rot_xy is None and trans_xy is None:
            return "0" * 8

        # init
        bits = BitArray()

        # scale
        if scale_xy:
            bits += "1"
            bits += floats2bits([scale_xy[0], scale_xy[1]])
        else:
            bits += "0"

        # rotation
        if rot_xy:
            bits += "1"
            bits += floats2bits([rot_xy[0], rot_xy[1]])
        else:
            bits += "0"

        # translation (no flag here)
        if trans_xy:
            bits += twits2bits([trans_xy[0], trans_xy[1]])
        else:
            bits += twits2bits([0, 0])

        # done
        return bits


# Control tags


class ControlTag(Tag):
    def __init__(self):
        Tag.__init__(self)


class FileAttributesTag(ControlTag):
    def __init__(self):
        ControlTag.__init__(self)
        self.tagtype = 69

    def process_tag(self):
        self.bytes = "\x00".encode("ascii") * (1 + 3)


class ShowFrameTag(ControlTag):
    def __init__(self):
        ControlTag.__init__(self)
        self.tagtype = 1

    def process_tag(self):
        self.bytes = bytes()


class SetBackgroundTag(ControlTag):
    """Set the color in 0-255, or 0-1 (if floats given)."""

    def __init__(self, *rgb):
        self.tagtype = 9
        if len(rgb) == 1:
            rgb = rgb[0]
        self.rgb = rgb

    def process_tag(self):
        bb = bytes()
        for i in range(3):
            clr = self.rgb[i]
            if isinstance(clr, float):  # pragma: no cover - not used
                clr = clr * 255
            bb += int2uint8(clr)
        self.bytes = bb


class DoActionTag(Tag):
    def __init__(self, action="stop"):
        Tag.__init__(self)
        self.tagtype = 12
        self.actions = [action]

    def append(self, action):  # pragma: no cover - not used
        self.actions.append(action)

    def process_tag(self):
        bb = bytes()

        for action in self.actions:
            action = action.lower()
            if action == "stop":
                bb += "\x07".encode("ascii")
            elif action == "play":  # pragma: no cover - not used
                bb += "\x06".encode("ascii")
            else:  # pragma: no cover
                logger.warning("unkown action: %s" % action)

        bb += int2uint8(0)
        self.bytes = bb


# Definition tags
class DefinitionTag(Tag):
    counter = 0  # to give automatically id's

    def __init__(self):
        Tag.__init__(self)
        DefinitionTag.counter += 1
        self.id = DefinitionTag.counter  # id in dictionary


class BitmapTag(DefinitionTag):
    def __init__(self, im):
        DefinitionTag.__init__(self)
        self.tagtype = 36  # DefineBitsLossless2

        # convert image (note that format is ARGB)
        # even a grayscale image is stored in ARGB, nevertheless,
        # the fabilous deflate compression will make it that not much
        # more data is required for storing (25% or so, and less than 10%
        # when storing RGB as ARGB).

        if len(im.shape) == 3:
            if im.shape[2] in [3, 4]:
                tmp = np.ones((im.shape[0], im.shape[1], 4), dtype=np.uint8) * 255
                for i in range(3):
                    tmp[:, :, i + 1] = im[:, :, i]
                if im.shape[2] == 4:
                    tmp[:, :, 0] = im[:, :, 3]  # swap channel where alpha is
            else:  # pragma: no cover
                raise ValueError("Invalid shape to be an image.")

        elif len(im.shape) == 2:
            tmp = np.ones((im.shape[0], im.shape[1], 4), dtype=np.uint8) * 255
            for i in range(3):
                tmp[:, :, i + 1] = im[:, :]
        else:  # pragma: no cover
            raise ValueError("Invalid shape to be an image.")

        # we changed the image to uint8 4 channels.
        # now compress!
        self._data = zlib.compress(tmp.tobytes(), zlib.DEFLATED)
        self.imshape = im.shape

    def process_tag(self):

        # build tag
        bb = bytes()
        bb += int2uint16(self.id)  # CharacterID
        bb += int2uint8(5)  # BitmapFormat
        bb += int2uint16(self.imshape[1])  # BitmapWidth
        bb += int2uint16(self.imshape[0])  # BitmapHeight
        bb += self._data  # ZlibBitmapData

        self.bytes = bb


class PlaceObjectTag(ControlTag):
    def __init__(self, depth, idToPlace=None, xy=(0, 0), move=False):
        ControlTag.__init__(self)
        self.tagtype = 26
        self.depth = depth
        self.idToPlace = idToPlace
        self.xy = xy
        self.move = move

    def process_tag(self):
        # retrieve stuff
        depth = self.depth
        xy = self.xy
        id = self.idToPlace

        # build PlaceObject2
        bb = bytes()
        if self.move:
            bb += "\x07".encode("ascii")
        else:
            # (8 bit flags): 4:matrix, 2:character, 1:move
            bb += "\x06".encode("ascii")
        bb += int2uint16(depth)  # Depth
        bb += int2uint16(id)  # character id
        bb += self.make_matrix_record(trans_xy=xy).tobytes()  # MATRIX record
        self.bytes = bb


class ShapeTag(DefinitionTag):
    def __init__(self, bitmapId, xy, wh):
        DefinitionTag.__init__(self)
        self.tagtype = 2
        self.bitmapId = bitmapId
        self.xy = xy
        self.wh = wh

    def process_tag(self):
        """Returns a defineshape tag. with a bitmap fill"""

        bb = bytes()
        bb += int2uint16(self.id)
        xy, wh = self.xy, self.wh
        tmp = self.make_rect_record(xy[0], wh[0], xy[1], wh[1])  # ShapeBounds
        bb += tmp.tobytes()

        # make SHAPEWITHSTYLE structure

        # first entry: FILLSTYLEARRAY with in it a single fill style
        bb += int2uint8(1)  # FillStyleCount
        bb += "\x41".encode("ascii")  # FillStyleType (0x41 or 0x43 unsmoothed)
        bb += int2uint16(self.bitmapId)  # BitmapId
        # bb += '\x00' # BitmapMatrix (empty matrix with leftover bits filled)
        bb += self.make_matrix_record(scale_xy=(20, 20)).tobytes()

        #         # first entry: FILLSTYLEARRAY with in it a single fill style
        #         bb += int2uint8(1)  # FillStyleCount
        #         bb += '\x00' # solid fill
        #         bb += '\x00\x00\xff' # color

        # second entry: LINESTYLEARRAY with a single line style
        bb += int2uint8(0)  # LineStyleCount
        # bb += int2uint16(0*20) # Width
        # bb += '\x00\xff\x00'  # Color

        # third and fourth entry: NumFillBits and NumLineBits (4 bits each)
        # I each give them four bits, so 16 styles possible.
        bb += "\x44".encode("ascii")

        self.bytes = bb

        # last entries: SHAPERECORDs ... (individual shape records not aligned)
        # STYLECHANGERECORD
        bits = BitArray()
        bits += self.make_style_change_record(0, 1, moveTo=(self.wh[0], self.wh[1]))
        # STRAIGHTEDGERECORD 4x
        bits += self.make_straight_edge_record(-self.wh[0], 0)
        bits += self.make_straight_edge_record(0, -self.wh[1])
        bits += self.make_straight_edge_record(self.wh[0], 0)
        bits += self.make_straight_edge_record(0, self.wh[1])

        # ENDSHAPRECORD
        bits += self.make_end_shape_record()

        self.bytes += bits.tobytes()

        # done
        # self.bytes = bb

    def make_style_change_record(self, lineStyle=None, fillStyle=None, moveTo=None):

        # first 6 flags
        # Note that we use FillStyle1. If we don't flash (at least 8) does not
        # recognize the frames properly when importing to library.

        bits = BitArray()
        bits += "0"  # TypeFlag (not an edge record)
        bits += "0"  # StateNewStyles (only for DefineShape2 and Defineshape3)
        if lineStyle:
            bits += "1"  # StateLineStyle
        else:
            bits += "0"
        if fillStyle:
            bits += "1"  # StateFillStyle1
        else:
            bits += "0"
        bits += "0"  # StateFillStyle0
        if moveTo:
            bits += "1"  # StateMoveTo
        else:
            bits += "0"

        # give information
        # todo: nbits for fillStyle and lineStyle is hard coded.

        if moveTo:
            bits += twits2bits([moveTo[0], moveTo[1]])
        if fillStyle:
            bits += int2bits(fillStyle, 4)
        if lineStyle:
            bits += int2bits(lineStyle, 4)

        return bits

    def make_straight_edge_record(self, *dxdy):
        if len(dxdy) == 1:
            dxdy = dxdy[0]

        # determine required number of bits
        xbits = signedint2bits(dxdy[0] * 20)
        ybits = signedint2bits(dxdy[1] * 20)
        nbits = max([len(xbits), len(ybits)])

        bits = BitArray()
        bits += "11"  # TypeFlag and StraightFlag
        bits += int2bits(nbits - 2, 4)
        bits += "1"  # GeneralLineFlag
        bits += signedint2bits(dxdy[0] * 20, nbits)
        bits += signedint2bits(dxdy[1] * 20, nbits)

        # note: I do not make use of vertical/horizontal only lines...

        return bits

    def make_end_shape_record(self):
        bits = BitArray()
        bits += "0"  # TypeFlag: no edge
        bits += "0" * 5  # EndOfShape
        return bits


def read_pixels(bb, i, tagType, L1):
    """With pf's seed after the recordheader, reads the pixeldata."""

    # Get info
    charId = bb[i : i + 2]  # noqa
    i += 2
    format = ord(bb[i : i + 1])
    i += 1
    width = bits2int(bb[i : i + 2], 16)
    i += 2
    height = bits2int(bb[i : i + 2], 16)
    i += 2

    # If we can, get pixeldata and make numpy array
    if format != 5:
        logger.warning("Can only read 24bit or 32bit RGB(A) lossless images.")
    else:
        # Read byte data
        offset = 2 + 1 + 2 + 2  # all the info bits
        bb2 = bb[i : i + (L1 - offset)]

        # Decompress and make numpy array
        data = zlib.decompress(bb2)
        a = np.frombuffer(data, dtype=np.uint8)

        # Set shape
        if tagType == 20:
            # DefineBitsLossless - RGB data
            try:
                a.shape = height, width, 3
            except Exception:
                # Byte align stuff might cause troubles
                logger.warning("Cannot read image due to byte alignment")
        if tagType == 36:
            # DefineBitsLossless2 - ARGB data
            a.shape = height, width, 4
            # Swap alpha channel to make RGBA
            b = a
            a = np.zeros_like(a)
            a[:, :, 0] = b[:, :, 1]
            a[:, :, 1] = b[:, :, 2]
            a[:, :, 2] = b[:, :, 3]
            a[:, :, 3] = b[:, :, 0]

        return a


# Last few functions


# These are the original public functions, we don't use them, but we
# keep it so that in principle this module can be used stand-alone.


def checkImages(images):  # pragma: no cover
    """checkImages(images)
    Check numpy images and correct intensity range etc.
    The same for all movie formats.
    """
    # Init results
    images2 = []

    for im in images:
        if isinstance(im, np.ndarray):
            # Check and convert dtype
            if im.dtype == np.uint8:
                images2.append(im)  # Ok
            elif im.dtype in [np.float32, np.float64]:
                theMax = im.max()
                if 128 < theMax < 300:
                    pass  # assume 0:255
                else:
                    im = im.copy()
                    im[im < 0] = 0
                    im[im > 1] = 1
                    im *= 255
                images2.append(im.astype(np.uint8))
            else:
                im = im.astype(np.uint8)
                images2.append(im)
            # Check size
            if im.ndim == 2:
                pass  # ok
            elif im.ndim == 3:
                if im.shape[2] not in [3, 4]:
                    raise ValueError("This array can not represent an image.")
            else:
                raise ValueError("This array can not represent an image.")
        else:
            raise ValueError("Invalid image type: " + str(type(im)))

    # Done
    return images2


def build_file(
    fp, taglist, nframes=1, framesize=(500, 500), fps=10, version=8
):  # pragma: no cover
    """Give the given file (as bytes) a header."""

    # compose header
    bb = bytes()
    bb += "F".encode("ascii")  # uncompressed
    bb += "WS".encode("ascii")  # signature bytes
    bb += int2uint8(version)  # version
    bb += "0000".encode("ascii")  # FileLength (leave open for now)
    bb += Tag().make_rect_record(0, framesize[0], 0, framesize[1]).tobytes()
    bb += int2uint8(0) + int2uint8(fps)  # FrameRate
    bb += int2uint16(nframes)
    fp.write(bb)

    # produce all tags
    for tag in taglist:
        fp.write(tag.get_tag())

    # finish with end tag
    fp.write("\x00\x00".encode("ascii"))

    # set size
    sze = fp.tell()
    fp.seek(4)
    fp.write(int2uint32(sze))


def write_swf(filename, images, duration=0.1, repeat=True):  # pragma: no cover
    """Write an swf-file from the specified images. If repeat is False,
    the movie is finished with a stop action. Duration may also
    be a list with durations for each frame (note that the duration
    for each frame is always an integer amount of the minimum duration.)

    Images should be a list consisting numpy arrays with values between
    0 and 255 for integer types, and between 0 and 1 for float types.

    """

    # Check images
    images2 = checkImages(images)

    # Init
    taglist = [FileAttributesTag(), SetBackgroundTag(0, 0, 0)]

    # Check duration
    if hasattr(duration, "__len__"):
        if len(duration) == len(images2):
            duration = [d for d in duration]
        else:
            raise ValueError("len(duration) doesn't match amount of images.")
    else:
        duration = [duration for im in images2]

    # Build delays list
    minDuration = float(min(duration))
    delays = [round(d / minDuration) for d in duration]
    delays = [max(1, int(d)) for d in delays]

    # Get FPS
    fps = 1.0 / minDuration

    # Produce series of tags for each image
    # t0 = time.time()
    nframes = 0
    for im in images2:
        bm = BitmapTag(im)
        wh = (im.shape[1], im.shape[0])
        sh = ShapeTag(bm.id, (0, 0), wh)
        po = PlaceObjectTag(1, sh.id, move=nframes > 0)
        taglist.extend([bm, sh, po])
        for i in range(delays[nframes]):
            taglist.append(ShowFrameTag())
        nframes += 1

    if not repeat:
        taglist.append(DoActionTag("stop"))

    # Build file
    # t1 = time.time()
    fp = open(filename, "wb")
    try:
        build_file(fp, taglist, nframes=nframes, framesize=wh, fps=fps)
    except Exception:
        raise
    finally:
        fp.close()
    # t2 = time.time()

    # logger.warning("Writing SWF took %1.2f and %1.2f seconds" % (t1-t0, t2-t1) )


def read_swf(filename):  # pragma: no cover
    """Read all images from an SWF (shockwave flash) file. Returns a list
    of numpy arrays.

    Limitation: only read the PNG encoded images (not the JPG encoded ones).
    """

    # Check whether it exists
    if not os.path.isfile(filename):
        raise IOError("File not found: " + str(filename))

    # Init images
    images = []

    # Open file and read all
    fp = open(filename, "rb")
    bb = fp.read()

    try:
        # Check opening tag
        tmp = bb[0:3].decode("ascii", "ignore")
        if tmp.upper() == "FWS":
            pass  # ok
        elif tmp.upper() == "CWS":
            # Decompress movie
            bb = bb[:8] + zlib.decompress(bb[8:])
        else:
            raise IOError("Not a valid SWF file: " + str(filename))

        # Set filepointer at first tag (skipping framesize RECT and two uin16's
        i = 8
        nbits = bits2int(bb[i : i + 1], 5)  # skip FrameSize
        nbits = 5 + nbits * 4
        Lrect = nbits / 8.0
        if Lrect % 1:
            Lrect += 1
        Lrect = int(Lrect)
        i += Lrect + 4

        # Iterate over the tags
        counter = 0
        while True:
            counter += 1

            # Get tag header
            head = bb[i : i + 6]
            if not head:
                break  # Done (we missed end tag)

            # Determine type and length
            T, L1, L2 = get_type_and_len(head)
            if not L2:
                logger.warning("Invalid tag length, could not proceed")
                break
            # logger.warning(T, L2)

            # Read image if we can
            if T in [20, 36]:
                im = read_pixels(bb, i + 6, T, L1)
                if im is not None:
                    images.append(im)
            elif T in [6, 21, 35, 90]:
                logger.warning("Ignoring JPEG image: cannot read JPEG.")
            else:
                pass  # Not an image tag

            # Detect end tag
            if T == 0:
                break

            # Next tag!
            i += L2

    finally:
        fp.close()

    # Done
    return images


# Backward compatibility; same public names as when this was images2swf.
writeSwf = write_swf
readSwf = read_swf
