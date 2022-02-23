"""
Miscellaneous Routines.
"""
import io
import pathlib
import struct
from typing import (Any, BinaryIO, Callable, Dict, Generic, Iterable, Iterator,
                    List, Optional, Set, TextIO, Tuple, TypeVar, Union,
                    TYPE_CHECKING, cast)
from html import escape

if TYPE_CHECKING:
    from .layout import LTComponent

import chardet  # For str encoding detection

# from sys import maxint as INF doesn't work anymore under Python3, but PDF
# still uses 32 bits ints
INF = (1 << 31) - 1


FileOrName = Union[pathlib.PurePath, str, io.IOBase]
AnyIO = Union[TextIO, BinaryIO]


class open_filename(object):
    """
    Context manager that allows opening a filename
    (str or pathlib.PurePath type is supported) and closes it on exit,
    (just like `open`), but does nothing for file-like objects.
    """
    def __init__(
        self,
        filename: FileOrName,
        *args: Any,
        **kwargs: Any
    ) -> None:
        if isinstance(filename, pathlib.PurePath):
            filename = str(filename)
        if isinstance(filename, str):
            self.file_handler: AnyIO = open(filename, *args, **kwargs)
            self.closing = True
        elif isinstance(filename, io.IOBase):
            self.file_handler = cast(AnyIO, filename)
            self.closing = False
        else:
            raise TypeError('Unsupported input type: %s' % type(filename))

    def __enter__(self) -> AnyIO:
        return self.file_handler

    def __exit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object
    ) -> None:
        if self.closing:
            self.file_handler.close()
        return


def make_compat_bytes(in_str: str) -> bytes:
    "Converts to bytes, encoding to unicode."
    assert isinstance(in_str, str), str(type(in_str))
    return in_str.encode()


def make_compat_str(o: object) -> str:
    """Converts everything to string, if bytes guessing the encoding."""
    if isinstance(o, bytes):
        enc = chardet.detect(o)
        return o.decode(enc['encoding'])
    else:
        return str(o)


def shorten_str(s: str, size: int) -> str:
    if size < 7:
        return s[:size]
    if len(s) > size:
        length = (size - 5) // 2
        return '{} ... {}'.format(s[:length], s[-length:])
    else:
        return s


def compatible_encode_method(
    bytesorstring: Union[bytes, str],
    encoding: str = 'utf-8',
    erraction: str = 'ignore'
) -> str:
    """When Py2 str.encode is called, it often means bytes.encode in Py3.

     This does either.
     """
    if isinstance(bytesorstring, str):
        return bytesorstring
    assert isinstance(bytesorstring, bytes), str(type(bytesorstring))
    return bytesorstring.decode(encoding, erraction)


def paeth_predictor(left: int, above: int, upper_left: int) -> int:
    # From http://www.libpng.org/pub/png/spec/1.2/PNG-Filters.html
    # Initial estimate
    p = left + above - upper_left
    # Distances to a,b,c
    pa = abs(p - left)
    pb = abs(p - above)
    pc = abs(p - upper_left)

    # Return nearest of a,b,c breaking ties in order a,b,c
    if pa <= pb and pa <= pc:
        return left
    elif pb <= pc:
        return above
    else:
        return upper_left


def apply_png_predictor(
    pred: int,
    colors: int,
    columns: int,
    bitspercomponent: int,
    data: bytes
) -> bytes:
    """Reverse the effect of the PNG predictor

    Documentation: http://www.libpng.org/pub/png/spec/1.2/PNG-Filters.html
    """
    if bitspercomponent != 8:
        msg = "Unsupported `bitspercomponent': %d" % bitspercomponent
        raise ValueError(msg)

    nbytes = colors * columns * bitspercomponent // 8
    bpp = colors * bitspercomponent // 8  # number of bytes per complete pixel
    buf = b''
    line_above = b'\x00' * columns
    for scanline_i in range(0, len(data), nbytes + 1):
        filter_type = data[scanline_i]
        line_encoded = data[scanline_i + 1:scanline_i + 1 + nbytes]
        raw = b''

        if filter_type == 0:
            # Filter type 0: None
            raw += line_encoded

        elif filter_type == 1:
            # Filter type 1: Sub
            # To reverse the effect of the Sub() filter after decompression,
            # output the following value:
            #   Raw(x) = Sub(x) + Raw(x - bpp)
            # (computed mod 256), where Raw() refers to the bytes already
            #  decoded.
            for j, sub_x in enumerate(line_encoded):
                if j - bpp < 0:
                    raw_x_bpp = 0
                else:
                    raw_x_bpp = int(raw[j - bpp])
                raw_x = (sub_x + raw_x_bpp) & 255
                raw += bytes((raw_x,))

        elif filter_type == 2:
            # Filter type 2: Up
            # To reverse the effect of the Up() filter after decompression,
            # output the following value:
            #   Raw(x) = Up(x) + Prior(x)
            # (computed mod 256), where Prior() refers to the decoded bytes of
            # the prior scanline.
            for (up_x, prior_x) in zip(line_encoded, line_above):
                raw_x = (up_x + prior_x) & 255
                raw += bytes((raw_x,))

        elif filter_type == 3:
            # Filter type 3: Average
            # To reverse the effect of the Average() filter after
            # decompression, output the following value:
            #    Raw(x) = Average(x) + floor((Raw(x-bpp)+Prior(x))/2)
            # where the result is computed mod 256, but the prediction is
            # calculated in the same way as for encoding. Raw() refers to the
            # bytes already decoded, and Prior() refers to the decoded bytes of
            # the prior scanline.
            for j, average_x in enumerate(line_encoded):
                if j - bpp < 0:
                    raw_x_bpp = 0
                else:
                    raw_x_bpp = int(raw[j - bpp])
                prior_x = int(line_above[j])
                raw_x = (average_x + (raw_x_bpp + prior_x) // 2) & 255
                raw += bytes((raw_x,))

        elif filter_type == 4:
            # Filter type 4: Paeth
            # To reverse the effect of the Paeth() filter after decompression,
            # output the following value:
            #    Raw(x) = Paeth(x)
            #             + PaethPredictor(Raw(x-bpp), Prior(x), Prior(x-bpp))
            # (computed mod 256), where Raw() and Prior() refer to bytes
            # already decoded. Exactly the same PaethPredictor() function is
            # used by both encoder and decoder.
            for j, paeth_x in enumerate(line_encoded):
                if j - bpp < 0:
                    raw_x_bpp = 0
                    prior_x_bpp = 0
                else:
                    raw_x_bpp = int(raw[j - bpp])
                    prior_x_bpp = int(line_above[j - bpp])
                prior_x = int(line_above[j])
                paeth = paeth_predictor(raw_x_bpp, prior_x, prior_x_bpp)
                raw_x = (paeth_x + paeth) & 255
                raw += bytes((raw_x,))

        else:
            raise ValueError("Unsupported predictor value: %d" % filter_type)

        buf += raw
        line_above = raw
    return buf


Point = Tuple[float, float]
Rect = Tuple[float, float, float, float]
Matrix = Tuple[float, float, float, float, float, float]
PathSegment = Union[
    Tuple[str],                                             # Literal['h']
    Tuple[str, float, float],                               # Literal['m', 'l']
    Tuple[str, float, float, float, float],                 # Literal['v', 'y']
    Tuple[str, float, float, float, float, float, float]]   # Literal['c']

#  Matrix operations
MATRIX_IDENTITY: Matrix = (1, 0, 0, 1, 0, 0)


def mult_matrix(m1: Matrix, m0: Matrix) -> Matrix:
    (a1, b1, c1, d1, e1, f1) = m1
    (a0, b0, c0, d0, e0, f0) = m0
    """Returns the multiplication of two matrices."""
    return (a0 * a1 + c0 * b1, b0 * a1 + d0 * b1,
            a0 * c1 + c0 * d1, b0 * c1 + d0 * d1,
            a0 * e1 + c0 * f1 + e0, b0 * e1 + d0 * f1 + f0)


def translate_matrix(m: Matrix, v: Point) -> Matrix:
    """Translates a matrix by (x, y)."""
    (a, b, c, d, e, f) = m
    (x, y) = v
    return a, b, c, d, x * a + y * c + e, x * b + y * d + f


def apply_matrix_pt(m: Matrix, v: Point) -> Point:
    (a, b, c, d, e, f) = m
    (x, y) = v
    """Applies a matrix to a point."""
    return a * x + c * y + e, b * x + d * y + f


def apply_matrix_norm(m: Matrix, v: Point) -> Point:
    """Equivalent to apply_matrix_pt(M, (p,q)) - apply_matrix_pt(M, (0,0))"""
    (a, b, c, d, e, f) = m
    (p, q) = v
    return a * p + c * q, b * p + d * q


#  Utility functions

def isnumber(x: object) -> bool:
    return isinstance(x, (int, float))


_T = TypeVar('_T')


def uniq(objs: Iterable[_T]) -> Iterator[_T]:
    """Eliminates duplicated elements."""
    done = set()
    for obj in objs:
        if obj in done:
            continue
        done.add(obj)
        yield obj
    return


def fsplit(
    pred: Callable[[_T], bool],
    objs: Iterable[_T]
) -> Tuple[List[_T], List[_T]]:
    """Split a list into two classes according to the predicate."""
    t = []
    f = []
    for obj in objs:
        if pred(obj):
            t.append(obj)
        else:
            f.append(obj)
    return t, f


def drange(v0: float, v1: float, d: int) -> range:
    """Returns a discrete range."""
    return range(int(v0) // d, int(v1 + d) // d)


def get_bound(pts: Iterable[Point]) -> Rect:
    """Compute a minimal rectangle that covers all the points."""
    limit: Rect = (INF, INF, -INF, -INF)
    (x0, y0, x1, y1) = limit
    for (x, y) in pts:
        x0 = min(x0, x)
        y0 = min(y0, y)
        x1 = max(x1, x)
        y1 = max(y1, y)
    return x0, y0, x1, y1


def pick(
    seq: Iterable[_T],
    func: Callable[[_T], float],
    maxobj: Optional[_T] = None
) -> Optional[_T]:
    """Picks the object obj where func(obj) has the highest value."""
    maxscore = None
    for obj in seq:
        score = func(obj)
        if maxscore is None or maxscore < score:
            (maxscore, maxobj) = (score, obj)
    return maxobj


def choplist(n: int, seq: Iterable[_T]) -> Iterator[Tuple[_T, ...]]:
    """Groups every n elements of the list."""
    r = []
    for x in seq:
        r.append(x)
        if len(r) == n:
            yield tuple(r)
            r = []
    return


def nunpack(s: bytes, default: int = 0) -> int:
    """Unpacks 1 to 4 or 8 byte integers (big endian)."""
    length = len(s)
    if not length:
        return default
    elif length == 1:
        return ord(s)
    elif length == 2:
        return cast(int, struct.unpack('>H', s)[0])
    elif length == 3:
        return cast(int, struct.unpack('>L', b'\x00' + s)[0])
    elif length == 4:
        return cast(int, struct.unpack('>L', s)[0])
    elif length == 8:
        return cast(int, struct.unpack('>Q', s)[0])
    else:
        raise TypeError('invalid length: %d' % length)


PDFDocEncoding = ''.join(chr(x) for x in (
    0x0000, 0x0001, 0x0002, 0x0003, 0x0004, 0x0005, 0x0006, 0x0007,
    0x0008, 0x0009, 0x000a, 0x000b, 0x000c, 0x000d, 0x000e, 0x000f,
    0x0010, 0x0011, 0x0012, 0x0013, 0x0014, 0x0015, 0x0017, 0x0017,
    0x02d8, 0x02c7, 0x02c6, 0x02d9, 0x02dd, 0x02db, 0x02da, 0x02dc,
    0x0020, 0x0021, 0x0022, 0x0023, 0x0024, 0x0025, 0x0026, 0x0027,
    0x0028, 0x0029, 0x002a, 0x002b, 0x002c, 0x002d, 0x002e, 0x002f,
    0x0030, 0x0031, 0x0032, 0x0033, 0x0034, 0x0035, 0x0036, 0x0037,
    0x0038, 0x0039, 0x003a, 0x003b, 0x003c, 0x003d, 0x003e, 0x003f,
    0x0040, 0x0041, 0x0042, 0x0043, 0x0044, 0x0045, 0x0046, 0x0047,
    0x0048, 0x0049, 0x004a, 0x004b, 0x004c, 0x004d, 0x004e, 0x004f,
    0x0050, 0x0051, 0x0052, 0x0053, 0x0054, 0x0055, 0x0056, 0x0057,
    0x0058, 0x0059, 0x005a, 0x005b, 0x005c, 0x005d, 0x005e, 0x005f,
    0x0060, 0x0061, 0x0062, 0x0063, 0x0064, 0x0065, 0x0066, 0x0067,
    0x0068, 0x0069, 0x006a, 0x006b, 0x006c, 0x006d, 0x006e, 0x006f,
    0x0070, 0x0071, 0x0072, 0x0073, 0x0074, 0x0075, 0x0076, 0x0077,
    0x0078, 0x0079, 0x007a, 0x007b, 0x007c, 0x007d, 0x007e, 0x0000,
    0x2022, 0x2020, 0x2021, 0x2026, 0x2014, 0x2013, 0x0192, 0x2044,
    0x2039, 0x203a, 0x2212, 0x2030, 0x201e, 0x201c, 0x201d, 0x2018,
    0x2019, 0x201a, 0x2122, 0xfb01, 0xfb02, 0x0141, 0x0152, 0x0160,
    0x0178, 0x017d, 0x0131, 0x0142, 0x0153, 0x0161, 0x017e, 0x0000,
    0x20ac, 0x00a1, 0x00a2, 0x00a3, 0x00a4, 0x00a5, 0x00a6, 0x00a7,
    0x00a8, 0x00a9, 0x00aa, 0x00ab, 0x00ac, 0x0000, 0x00ae, 0x00af,
    0x00b0, 0x00b1, 0x00b2, 0x00b3, 0x00b4, 0x00b5, 0x00b6, 0x00b7,
    0x00b8, 0x00b9, 0x00ba, 0x00bb, 0x00bc, 0x00bd, 0x00be, 0x00bf,
    0x00c0, 0x00c1, 0x00c2, 0x00c3, 0x00c4, 0x00c5, 0x00c6, 0x00c7,
    0x00c8, 0x00c9, 0x00ca, 0x00cb, 0x00cc, 0x00cd, 0x00ce, 0x00cf,
    0x00d0, 0x00d1, 0x00d2, 0x00d3, 0x00d4, 0x00d5, 0x00d6, 0x00d7,
    0x00d8, 0x00d9, 0x00da, 0x00db, 0x00dc, 0x00dd, 0x00de, 0x00df,
    0x00e0, 0x00e1, 0x00e2, 0x00e3, 0x00e4, 0x00e5, 0x00e6, 0x00e7,
    0x00e8, 0x00e9, 0x00ea, 0x00eb, 0x00ec, 0x00ed, 0x00ee, 0x00ef,
    0x00f0, 0x00f1, 0x00f2, 0x00f3, 0x00f4, 0x00f5, 0x00f6, 0x00f7,
    0x00f8, 0x00f9, 0x00fa, 0x00fb, 0x00fc, 0x00fd, 0x00fe, 0x00ff,
))


def decode_text(s: bytes) -> str:
    """Decodes a PDFDocEncoding string to Unicode."""
    if s.startswith(b'\xfe\xff'):
        return str(s[2:], 'utf-16be', 'ignore')
    else:
        return ''.join(PDFDocEncoding[c] for c in s)


def enc(x: str) -> str:
    """Encodes a string for SGML/XML/HTML"""
    if isinstance(x, bytes):
        return ''
    return escape(x)


def bbox2str(bbox: Rect) -> str:
    (x0, y0, x1, y1) = bbox
    return '{:.3f},{:.3f},{:.3f},{:.3f}'.format(x0, y0, x1, y1)


def matrix2str(m: Matrix) -> str:
    (a, b, c, d, e, f) = m
    return '[{:.2f},{:.2f},{:.2f},{:.2f}, ({:.2f},{:.2f})]'\
        .format(a, b, c, d, e, f)


def vecBetweenBoxes(obj1: "LTComponent", obj2: "LTComponent") -> Point:
    """A distance function between two TextBoxes.

    Consider the bounding rectangle for obj1 and obj2.
    Return vector between 2 boxes boundaries if they don't overlap, otherwise
    returns vector betweeen boxes centers

             +------+..........+ (x1, y1)
             | obj1 |          :
             +------+www+------+
             :          | obj2 |
    (x0, y0) +..........+------+
    """
    (x0, y0) = (min(obj1.x0, obj2.x0), min(obj1.y0, obj2.y0))
    (x1, y1) = (max(obj1.x1, obj2.x1), max(obj1.y1, obj2.y1))
    (ow, oh) = (x1 - x0, y1 - y0)
    (iw, ih) = (ow - obj1.width - obj2.width, oh - obj1.height - obj2.height)
    if iw < 0 and ih < 0:
        # if one is inside another we compute euclidean distance
        (xc1, yc1) = ((obj1.x0 + obj1.x1) / 2, (obj1.y0 + obj1.y1) / 2)
        (xc2, yc2) = ((obj2.x0 + obj2.x1) / 2, (obj2.y0 + obj2.y1) / 2)
        return xc1 - xc2, yc1 - yc2
    else:
        return max(0, iw), max(0, ih)


LTComponentT = TypeVar('LTComponentT', bound='LTComponent')


class Plane(Generic[LTComponentT]):
    """A set-like data structure for objects placed on a plane.

    Can efficiently find objects in a certain rectangular area.
    It maintains two parallel lists of objects, each of
    which is sorted by its x or y coordinate.
    """

    def __init__(self, bbox: Rect, gridsize: int = 50) -> None:
        self._seq: List[LTComponentT] = []  # preserve the object order.
        self._objs: Set[LTComponentT] = set()
        self._grid: Dict[Point, List[LTComponentT]] = {}
        self.gridsize = gridsize
        (self.x0, self.y0, self.x1, self.y1) = bbox

    def __repr__(self) -> str:
        return '<Plane objs=%r>' % list(self)

    def __iter__(self) -> Iterator[LTComponentT]:
        return (obj for obj in self._seq if obj in self._objs)

    def __len__(self) -> int:
        return len(self._objs)

    def __contains__(self, obj: object) -> bool:
        return obj in self._objs

    def _getrange(self, bbox: Rect) -> Iterator[Point]:
        (x0, y0, x1, y1) = bbox
        if x1 <= self.x0 or self.x1 <= x0 or y1 <= self.y0 or self.y1 <= y0:
            return
        x0 = max(self.x0, x0)
        y0 = max(self.y0, y0)
        x1 = min(self.x1, x1)
        y1 = min(self.y1, y1)
        for grid_y in drange(y0, y1, self.gridsize):
            for grid_x in drange(x0, x1, self.gridsize):
                yield (grid_x, grid_y)

    def extend(self, objs: Iterable[LTComponentT]) -> None:
        for obj in objs:
            self.add(obj)

    def add(self, obj: LTComponentT) -> None:
        """place an object."""
        for k in self._getrange((obj.x0, obj.y0, obj.x1, obj.y1)):
            if k not in self._grid:
                r: List[LTComponentT] = []
                self._grid[k] = r
            else:
                r = self._grid[k]
            r.append(obj)
        self._seq.append(obj)
        self._objs.add(obj)

    def remove(self, obj: LTComponentT) -> None:
        """displace an object."""
        for k in self._getrange((obj.x0, obj.y0, obj.x1, obj.y1)):
            try:
                self._grid[k].remove(obj)
            except (KeyError, ValueError):
                pass
        self._objs.remove(obj)

    def find(self, bbox: Rect) -> Iterator[LTComponentT]:
        """finds objects that are in a certain area."""
        (x0, y0, x1, y1) = bbox
        done = set()
        for k in self._getrange(bbox):
            if k not in self._grid:
                continue
            for obj in self._grid[k]:
                if obj in done:
                    continue
                done.add(obj)
                if obj.x1 <= x0 or x1 <= obj.x0 or obj.y1 <= y0 \
                        or y1 <= obj.y0:
                    continue
                yield obj
