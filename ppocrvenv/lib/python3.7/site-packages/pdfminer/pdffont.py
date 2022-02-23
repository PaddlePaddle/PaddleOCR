import logging
import struct
import sys
from io import BytesIO
from typing import (Any, BinaryIO, Dict, Iterable, Iterator, List, Mapping,
                    Optional, Tuple, Union, cast, TYPE_CHECKING)

from . import settings
from .cmapdb import CMap
from .cmapdb import CMapBase
from .cmapdb import CMapDB
from .cmapdb import CMapParser
from .cmapdb import UnicodeMap
from .cmapdb import FileUnicodeMap
from .encodingdb import EncodingDB
from .encodingdb import name2unicode
from .fontmetrics import FONT_METRICS
from .pdftypes import PDFException
from .pdftypes import PDFStream
from .pdftypes import dict_value
from .pdftypes import int_value
from .pdftypes import list_value
from .pdftypes import num_value
from .pdftypes import resolve1, resolve_all
from .pdftypes import stream_value
from .psparser import KWD
from .psparser import LIT
from .psparser import PSEOF
from .psparser import PSKeyword
from .psparser import PSLiteral
from .psparser import PSStackParser
from .psparser import literal_name
from .utils import Matrix, Point
from .utils import Rect
from .utils import apply_matrix_norm
from .utils import choplist
from .utils import nunpack

if TYPE_CHECKING:
    from .pdfinterp import PDFResourceManager

log = logging.getLogger(__name__)


def get_widths(seq: Iterable[object]) -> Dict[int, float]:
    """Build a mapping of character widths for horizontal writing."""
    widths: Dict[int, float] = {}
    r: List[float] = []
    for v in seq:
        if isinstance(v, list):
            if r:
                char1 = r[-1]
                for (i, w) in enumerate(v):
                    widths[cast(int, char1) + i] = w
                r = []
        elif isinstance(v, (int, float)):  # == utils.isnumber(v)
            r.append(v)
            if len(r) == 3:
                (char1, char2, w) = r
                for i in range(cast(int, char1), cast(int, char2) + 1):
                    widths[i] = w
                r = []
    return widths


def get_widths2(seq: Iterable[object]) -> Dict[int, Tuple[float, Point]]:
    """Build a mapping of character widths for vertical writing."""
    widths: Dict[int, Tuple[float, Point]] = {}
    r: List[float] = []
    for v in seq:
        if isinstance(v, list):
            if r:
                char1 = r[-1]
                for (i, (w, vx, vy)) in enumerate(choplist(3, v)):
                    widths[cast(int, char1) + i] = (w, (vx, vy))
                r = []
        elif isinstance(v, (int, float)):  # == utils.isnumber(v)
            r.append(v)
            if len(r) == 5:
                (char1, char2, w, vx, vy) = r
                for i in range(cast(int, char1), cast(int, char2) + 1):
                    widths[i] = (w, (vx, vy))
                r = []
    return widths


class FontMetricsDB:

    @classmethod
    def get_metrics(cls, fontname: str
                    ) -> Tuple[Dict[str, object], Dict[str, int]]:
        return FONT_METRICS[fontname]


# int here means that we're not extending PSStackParser with additional types.
class Type1FontHeaderParser(PSStackParser[int]):

    KEYWORD_BEGIN = KWD(b'begin')
    KEYWORD_END = KWD(b'end')
    KEYWORD_DEF = KWD(b'def')
    KEYWORD_PUT = KWD(b'put')
    KEYWORD_DICT = KWD(b'dict')
    KEYWORD_ARRAY = KWD(b'array')
    KEYWORD_READONLY = KWD(b'readonly')
    KEYWORD_FOR = KWD(b'for')

    def __init__(self, data: BinaryIO) -> None:
        PSStackParser.__init__(self, data)
        self._cid2unicode: Dict[int, str] = {}
        return

    def get_encoding(self) -> Dict[int, str]:
        """Parse the font encoding.

        The Type1 font encoding maps character codes to character names. These
        character names could either be standard Adobe glyph names, or
        character names associated with custom CharStrings for this font. A
        CharString is a sequence of operations that describe how the character
        should be drawn. Currently, this function returns '' (empty string)
        for character names that are associated with a CharStrings.

        Reference: Adobe Systems Incorporated, Adobe Type 1 Font Format

        :returns mapping of character identifiers (cid's) to unicode characters
        """
        while 1:
            try:
                (cid, name) = self.nextobject()
            except PSEOF:
                break
            try:
                self._cid2unicode[cid] = name2unicode(cast(str, name))
            except KeyError as e:
                log.debug(str(e))
        return self._cid2unicode

    def do_keyword(self, pos: int, token: PSKeyword) -> None:
        if token is self.KEYWORD_PUT:
            ((_, key), (_, value)) = self.pop(2)
            if (isinstance(key, int) and isinstance(value, PSLiteral)):
                self.add_results((key, literal_name(value)))
        return


NIBBLES = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', 'e', 'e-',
           None, '-')

# Mapping of cmap names. Original cmap name is kept if not in the mapping.
# (missing reference for why DLIdent is mapped to Identity)
IDENTITY_ENCODER = {
    'DLIdent-H': 'Identity-H',
    'DLIdent-V': 'Identity-V',
}


def getdict(data: bytes) -> Dict[int, List[Union[float, int]]]:
    d: Dict[int, List[Union[float, int]]] = {}
    fp = BytesIO(data)
    stack: List[Union[float, int]] = []
    while 1:
        c = fp.read(1)
        if not c:
            break
        b0 = ord(c)
        if b0 <= 21:
            d[b0] = stack
            stack = []
            continue
        if b0 == 30:
            s = ''
            loop = True
            while loop:
                b = ord(fp.read(1))
                for n in (b >> 4, b & 15):
                    if n == 15:
                        loop = False
                    else:
                        nibble = NIBBLES[n]
                        assert nibble is not None
                        s += nibble
            value = float(s)
        elif 32 <= b0 and b0 <= 246:
            value = b0-139
        else:
            b1 = ord(fp.read(1))
            if 247 <= b0 and b0 <= 250:
                value = ((b0-247) << 8)+b1+108
            elif 251 <= b0 and b0 <= 254:
                value = -((b0-251) << 8)-b1-108
            else:
                b2 = ord(fp.read(1))
                if 128 <= b1:
                    b1 -= 256
                if b0 == 28:
                    value = b1 << 8 | b2
                else:
                    value = b1 << 24 | b2 << 16 | \
                            struct.unpack('>H', fp.read(2))[0]
        stack.append(value)
    return d


class CFFFont:

    STANDARD_STRINGS = (
      '.notdef', 'space', 'exclam', 'quotedbl', 'numbersign',
      'dollar', 'percent', 'ampersand', 'quoteright', 'parenleft',
      'parenright', 'asterisk', 'plus', 'comma', 'hyphen', 'period',
      'slash', 'zero', 'one', 'two', 'three', 'four', 'five', 'six',
      'seven', 'eight', 'nine', 'colon', 'semicolon', 'less', 'equal',
      'greater', 'question', 'at', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
      'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
      'U', 'V', 'W', 'X', 'Y', 'Z', 'bracketleft', 'backslash',
      'bracketright', 'asciicircum', 'underscore', 'quoteleft', 'a',
      'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
      'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
      'braceleft', 'bar', 'braceright', 'asciitilde', 'exclamdown',
      'cent', 'sterling', 'fraction', 'yen', 'florin', 'section',
      'currency', 'quotesingle', 'quotedblleft', 'guillemotleft',
      'guilsinglleft', 'guilsinglright', 'fi', 'fl', 'endash',
      'dagger', 'daggerdbl', 'periodcentered', 'paragraph', 'bullet',
      'quotesinglbase', 'quotedblbase', 'quotedblright',
      'guillemotright', 'ellipsis', 'perthousand', 'questiondown',
      'grave', 'acute', 'circumflex', 'tilde', 'macron', 'breve',
      'dotaccent', 'dieresis', 'ring', 'cedilla', 'hungarumlaut',
      'ogonek', 'caron', 'emdash', 'AE', 'ordfeminine', 'Lslash',
      'Oslash', 'OE', 'ordmasculine', 'ae', 'dotlessi', 'lslash',
      'oslash', 'oe', 'germandbls', 'onesuperior', 'logicalnot', 'mu',
      'trademark', 'Eth', 'onehalf', 'plusminus', 'Thorn',
      'onequarter', 'divide', 'brokenbar', 'degree', 'thorn',
      'threequarters', 'twosuperior', 'registered', 'minus', 'eth',
      'multiply', 'threesuperior', 'copyright', 'Aacute',
      'Acircumflex', 'Adieresis', 'Agrave', 'Aring', 'Atilde',
      'Ccedilla', 'Eacute', 'Ecircumflex', 'Edieresis', 'Egrave',
      'Iacute', 'Icircumflex', 'Idieresis', 'Igrave', 'Ntilde',
      'Oacute', 'Ocircumflex', 'Odieresis', 'Ograve', 'Otilde',
      'Scaron', 'Uacute', 'Ucircumflex', 'Udieresis', 'Ugrave',
      'Yacute', 'Ydieresis', 'Zcaron', 'aacute', 'acircumflex',
      'adieresis', 'agrave', 'aring', 'atilde', 'ccedilla', 'eacute',
      'ecircumflex', 'edieresis', 'egrave', 'iacute', 'icircumflex',
      'idieresis', 'igrave', 'ntilde', 'oacute', 'ocircumflex',
      'odieresis', 'ograve', 'otilde', 'scaron', 'uacute',
      'ucircumflex', 'udieresis', 'ugrave', 'yacute', 'ydieresis',
      'zcaron', 'exclamsmall', 'Hungarumlautsmall', 'dollaroldstyle',
      'dollarsuperior', 'ampersandsmall', 'Acutesmall',
      'parenleftsuperior', 'parenrightsuperior', 'twodotenleader',
      'onedotenleader', 'zerooldstyle', 'oneoldstyle', 'twooldstyle',
      'threeoldstyle', 'fouroldstyle', 'fiveoldstyle', 'sixoldstyle',
      'sevenoldstyle', 'eightoldstyle', 'nineoldstyle',
      'commasuperior', 'threequartersemdash', 'periodsuperior',
      'questionsmall', 'asuperior', 'bsuperior', 'centsuperior',
      'dsuperior', 'esuperior', 'isuperior', 'lsuperior', 'msuperior',
      'nsuperior', 'osuperior', 'rsuperior', 'ssuperior', 'tsuperior',
      'ff', 'ffi', 'ffl', 'parenleftinferior', 'parenrightinferior',
      'Circumflexsmall', 'hyphensuperior', 'Gravesmall', 'Asmall',
      'Bsmall', 'Csmall', 'Dsmall', 'Esmall', 'Fsmall', 'Gsmall',
      'Hsmall', 'Ismall', 'Jsmall', 'Ksmall', 'Lsmall', 'Msmall',
      'Nsmall', 'Osmall', 'Psmall', 'Qsmall', 'Rsmall', 'Ssmall',
      'Tsmall', 'Usmall', 'Vsmall', 'Wsmall', 'Xsmall', 'Ysmall',
      'Zsmall', 'colonmonetary', 'onefitted', 'rupiah', 'Tildesmall',
      'exclamdownsmall', 'centoldstyle', 'Lslashsmall', 'Scaronsmall',
      'Zcaronsmall', 'Dieresissmall', 'Brevesmall', 'Caronsmall',
      'Dotaccentsmall', 'Macronsmall', 'figuredash', 'hypheninferior',
      'Ogoneksmall', 'Ringsmall', 'Cedillasmall', 'questiondownsmall',
      'oneeighth', 'threeeighths', 'fiveeighths', 'seveneighths',
      'onethird', 'twothirds', 'zerosuperior', 'foursuperior',
      'fivesuperior', 'sixsuperior', 'sevensuperior', 'eightsuperior',
      'ninesuperior', 'zeroinferior', 'oneinferior', 'twoinferior',
      'threeinferior', 'fourinferior', 'fiveinferior', 'sixinferior',
      'seveninferior', 'eightinferior', 'nineinferior',
      'centinferior', 'dollarinferior', 'periodinferior',
      'commainferior', 'Agravesmall', 'Aacutesmall',
      'Acircumflexsmall', 'Atildesmall', 'Adieresissmall',
      'Aringsmall', 'AEsmall', 'Ccedillasmall', 'Egravesmall',
      'Eacutesmall', 'Ecircumflexsmall', 'Edieresissmall',
      'Igravesmall', 'Iacutesmall', 'Icircumflexsmall',
      'Idieresissmall', 'Ethsmall', 'Ntildesmall', 'Ogravesmall',
      'Oacutesmall', 'Ocircumflexsmall', 'Otildesmall',
      'Odieresissmall', 'OEsmall', 'Oslashsmall', 'Ugravesmall',
      'Uacutesmall', 'Ucircumflexsmall', 'Udieresissmall',
      'Yacutesmall', 'Thornsmall', 'Ydieresissmall', '001.000',
      '001.001', '001.002', '001.003', 'Black', 'Bold', 'Book',
      'Light', 'Medium', 'Regular', 'Roman', 'Semibold',
    )

    class INDEX:

        def __init__(self, fp: BinaryIO) -> None:
            self.fp = fp
            self.offsets: List[int] = []
            (count, offsize) = struct.unpack('>HB', self.fp.read(3))
            for i in range(count+1):
                self.offsets.append(nunpack(self.fp.read(offsize)))
            self.base = self.fp.tell()-1
            self.fp.seek(self.base+self.offsets[-1])
            return

        def __repr__(self) -> str:
            return '<INDEX: size=%d>' % len(self)

        def __len__(self) -> int:
            return len(self.offsets)-1

        def __getitem__(self, i: int) -> bytes:
            self.fp.seek(self.base+self.offsets[i])
            return self.fp.read(self.offsets[i+1]-self.offsets[i])

        def __iter__(self) -> Iterator[bytes]:
            return iter(self[i] for i in range(len(self)))

    def __init__(self, name: str, fp: BinaryIO) -> None:
        self.name = name
        self.fp = fp
        # Header
        (_major, _minor, hdrsize, offsize) = struct.unpack('BBBB',
                                                           self.fp.read(4))
        self.fp.read(hdrsize-4)
        # Name INDEX
        self.name_index = self.INDEX(self.fp)
        # Top DICT INDEX
        self.dict_index = self.INDEX(self.fp)
        # String INDEX
        self.string_index = self.INDEX(self.fp)
        # Global Subr INDEX
        self.subr_index = self.INDEX(self.fp)
        # Top DICT DATA
        self.top_dict = getdict(self.dict_index[0])
        (charset_pos,) = self.top_dict.get(15, [0])
        (encoding_pos,) = self.top_dict.get(16, [0])
        (charstring_pos,) = self.top_dict.get(17, [0])
        # CharStrings
        self.fp.seek(cast(int, charstring_pos))
        self.charstring = self.INDEX(self.fp)
        self.nglyphs = len(self.charstring)
        # Encodings
        self.code2gid = {}
        self.gid2code = {}
        self.fp.seek(cast(int, encoding_pos))
        format = self.fp.read(1)
        if format == b'\x00':
            # Format 0
            (n,) = struct.unpack('B', self.fp.read(1))
            for (code, gid) in enumerate(struct.unpack('B'*n,
                                                       self.fp.read(n))):
                self.code2gid[code] = gid
                self.gid2code[gid] = code
        elif format == b'\x01':
            # Format 1
            (n,) = struct.unpack('B', self.fp.read(1))
            code = 0
            for i in range(n):
                (first, nleft) = struct.unpack('BB', self.fp.read(2))
                for gid in range(first, first+nleft+1):
                    self.code2gid[code] = gid
                    self.gid2code[gid] = code
                    code += 1
        else:
            raise ValueError('unsupported encoding format: %r' % format)
        # Charsets
        self.name2gid = {}
        self.gid2name = {}
        self.fp.seek(cast(int, charset_pos))
        format = self.fp.read(1)
        if format == b'\x00':
            # Format 0
            n = self.nglyphs-1
            for (gid, sid) in enumerate(
                    cast(Tuple[int, ...],
                         struct.unpack('>' + 'H' * n, self.fp.read(2 * n)))):
                gid += 1
                sidname = self.getstr(sid)
                self.name2gid[sidname] = gid
                self.gid2name[gid] = sidname
        elif format == b'\x01':
            # Format 1
            (n,) = struct.unpack('B', self.fp.read(1))
            sid = 0
            for i in range(n):
                (first, nleft) = struct.unpack('BB', self.fp.read(2))
                for gid in range(first, first+nleft+1):
                    sidname = self.getstr(sid)
                    self.name2gid[sidname] = gid
                    self.gid2name[gid] = sidname
                    sid += 1
        elif format == b'\x02':
            # Format 2
            assert False, str(('Unhandled', format))
        else:
            raise ValueError('unsupported charset format: %r' % format)
        return

    def getstr(self, sid: int) -> Union[str, bytes]:
        # This returns str for one of the STANDARD_STRINGS but bytes otherwise,
        # and appears to be a needless source of type complexity.
        if sid < len(self.STANDARD_STRINGS):
            return self.STANDARD_STRINGS[sid]
        return self.string_index[sid-len(self.STANDARD_STRINGS)]


class TrueTypeFont:

    class CMapNotFound(Exception):
        pass

    def __init__(self, name: str, fp: BinaryIO) -> None:
        self.name = name
        self.fp = fp
        self.tables: Dict[bytes, Tuple[int, int]] = {}
        self.fonttype = fp.read(4)
        try:
            (ntables, _1, _2, _3) = cast(Tuple[int, int, int, int],
                                         struct.unpack('>HHHH', fp.read(8)))
            for _ in range(ntables):
                (name_bytes, tsum, offset, length) = \
                    cast(Tuple[bytes, int, int, int],
                         struct.unpack('>4sLLL', fp.read(16)))
                self.tables[name_bytes] = (offset, length)
        except struct.error:
            # Do not fail if there are not enough bytes to read. Even for
            # corrupted PDFs we would like to get as much information as
            # possible, so continue.
            pass
        return

    def create_unicode_map(self) -> FileUnicodeMap:
        if b'cmap' not in self.tables:
            raise TrueTypeFont.CMapNotFound
        (base_offset, length) = self.tables[b'cmap']
        fp = self.fp
        fp.seek(base_offset)
        (version, nsubtables) = \
            cast(Tuple[int, int], struct.unpack('>HH', fp.read(4)))
        subtables: List[Tuple[int, int, int]] = []
        for i in range(nsubtables):
            subtables.append(
                cast(Tuple[int, int, int], struct.unpack('>HHL', fp.read(8))))
        char2gid: Dict[int, int] = {}
        # Only supports subtable type 0, 2 and 4.
        for (_1, _2, st_offset) in subtables:
            fp.seek(base_offset+st_offset)
            (fmttype, fmtlen, fmtlang) = \
                cast(Tuple[int, int, int], struct.unpack('>HHH', fp.read(6)))
            if fmttype == 0:
                char2gid.update(enumerate(
                    cast(Tuple[int, ...],
                         struct.unpack('>256B', fp.read(256)))))
            elif fmttype == 2:
                subheaderkeys = cast(Tuple[int, ...],
                                     struct.unpack('>256H', fp.read(512)))
                firstbytes = [0]*8192
                for (i, k) in enumerate(subheaderkeys):
                    firstbytes[k//8] = i
                nhdrs = max(subheaderkeys)//8 + 1
                hdrs: List[Tuple[int, int, int, int, int]] = []
                for i in range(nhdrs):
                    (firstcode, entcount, delta, offset) = \
                        cast(Tuple[int, int, int, int],
                             struct.unpack('>HHhH', fp.read(8)))
                    hdrs.append((i, firstcode, entcount, delta,
                                 fp.tell()-2+offset))
                for (i, firstcode, entcount, delta, pos) in hdrs:
                    if not entcount:
                        continue
                    first = firstcode + (firstbytes[i] << 8)
                    fp.seek(pos)
                    for c in range(entcount):
                        gid = cast(Tuple[int],
                                   struct.unpack('>H', fp.read(2)))[0]
                        if gid:
                            gid += delta
                        char2gid[first+c] = gid
            elif fmttype == 4:
                (segcount, _1, _2, _3) = \
                    cast(Tuple[int, int, int, int],
                         struct.unpack('>HHHH', fp.read(8)))
                segcount //= 2
                ecs = cast(Tuple[int, ...],
                           struct.unpack('>%dH' % segcount,
                                         fp.read(2*segcount)))
                fp.read(2)
                scs = cast(Tuple[int, ...],
                           struct.unpack('>%dH' % segcount,
                                         fp.read(2*segcount)))
                idds = cast(Tuple[int, ...],
                            struct.unpack('>%dh' % segcount,
                                          fp.read(2*segcount)))
                pos = fp.tell()
                idrs = cast(Tuple[int, ...],
                            struct.unpack('>%dH' % segcount,
                                          fp.read(2*segcount)))
                for (ec, sc, idd, idr) in zip(ecs, scs, idds, idrs):
                    if idr:
                        fp.seek(pos+idr)
                        for c in range(sc, ec+1):
                            b = cast(Tuple[int],
                                     struct.unpack('>H', fp.read(2)))[0]
                            char2gid[c] = (b + idd) & 0xffff
                    else:
                        for c in range(sc, ec+1):
                            char2gid[c] = (c + idd) & 0xffff
            else:
                assert False, str(('Unhandled', fmttype))
        # create unicode map
        unicode_map = FileUnicodeMap()
        for (char, gid) in char2gid.items():
            unicode_map.add_cid2unichr(gid, char)
        return unicode_map


class PDFFontError(PDFException):
    pass


class PDFUnicodeNotDefined(PDFFontError):
    pass


LITERAL_STANDARD_ENCODING = LIT('StandardEncoding')
LITERAL_TYPE1C = LIT('Type1C')

# Font widths are maintained in a dict type that maps from *either* unicode
# chars or integer character IDs.
FontWidthDict = Union[Dict[int, float], Dict[str, float]]


class PDFFont:

    def __init__(
        self,
        descriptor: Mapping[str, Any],
        widths: FontWidthDict,
        default_width: Optional[float] = None
    ) -> None:
        self.descriptor = descriptor
        self.widths: FontWidthDict = resolve_all(widths)
        self.fontname = resolve1(descriptor.get('FontName', 'unknown'))
        if isinstance(self.fontname, PSLiteral):
            self.fontname = literal_name(self.fontname)
        self.flags = int_value(descriptor.get('Flags', 0))
        self.ascent = num_value(descriptor.get('Ascent', 0))
        self.descent = num_value(descriptor.get('Descent', 0))
        self.italic_angle = num_value(descriptor.get('ItalicAngle', 0))
        if default_width is None:
            self.default_width = num_value(descriptor.get('MissingWidth', 0))
        else:
            self.default_width = default_width
        self.leading = num_value(descriptor.get('Leading', 0))
        self.bbox = cast(Rect, list_value(
            resolve_all(descriptor.get('FontBBox', (0, 0, 0, 0)))))
        self.hscale = self.vscale = .001

        # PDF RM 9.8.1 specifies /Descent should always be a negative number.
        # PScript5.dll seems to produce Descent with a positive number, but
        # text analysis will be wrong if this is taken as correct. So force
        # descent to negative.
        if self.descent > 0:
            self.descent = -self.descent
        return

    def __repr__(self) -> str:
        return '<PDFFont>'

    def is_vertical(self) -> bool:
        return False

    def is_multibyte(self) -> bool:
        return False

    def decode(self, bytes: bytes) -> Iterable[int]:
        return bytearray(bytes)  # map(ord, bytes)

    def get_ascent(self) -> float:
        """Ascent above the baseline, in text space units"""
        return self.ascent * self.vscale

    def get_descent(self) -> float:
        """Descent below the baseline, in text space units; always negative"""
        return self.descent * self.vscale

    def get_width(self) -> float:
        w = self.bbox[2]-self.bbox[0]
        if w == 0:
            w = -self.default_width
        return w * self.hscale

    def get_height(self) -> float:
        h = self.bbox[3]-self.bbox[1]
        if h == 0:
            h = self.ascent - self.descent
        return h * self.vscale

    def char_width(self, cid: int) -> float:
        # Because character widths may be mapping either IDs or strings,
        # we try to lookup the character ID first, then its str equivalent.
        try:
            return cast(Dict[int, float], self.widths)[cid] * self.hscale
        except KeyError:
            str_widths = cast(Dict[str, float], self.widths)
            try:
                return str_widths[self.to_unichr(cid)] * self.hscale
            except (KeyError, PDFUnicodeNotDefined):
                return self.default_width * self.hscale

    def char_disp(
        self,
        cid: int
    ) -> Union[float, Tuple[Optional[float], float]]:
        "Returns an integer for horizontal fonts, a tuple for vertical fonts."
        return 0

    def string_width(self, s: bytes) -> float:
        return sum(self.char_width(cid) for cid in self.decode(s))

    def to_unichr(self, cid: int) -> str:
        raise NotImplementedError


class PDFSimpleFont(PDFFont):

    def __init__(
        self,
        descriptor: Mapping[str, Any],
        widths: FontWidthDict,
        spec: Mapping[str, Any]
    ) -> None:
        # Font encoding is specified either by a name of
        # built-in encoding or a dictionary that describes
        # the differences.
        if 'Encoding' in spec:
            encoding = resolve1(spec['Encoding'])
        else:
            encoding = LITERAL_STANDARD_ENCODING
        if isinstance(encoding, dict):
            name = literal_name(encoding.get('BaseEncoding',
                                             LITERAL_STANDARD_ENCODING))
            diff = list_value(encoding.get('Differences', []))
            self.cid2unicode = EncodingDB.get_encoding(name, diff)
        else:
            self.cid2unicode = EncodingDB.get_encoding(literal_name(encoding))
        self.unicode_map: Optional[UnicodeMap] = None
        if 'ToUnicode' in spec:
            strm = stream_value(spec['ToUnicode'])
            self.unicode_map = FileUnicodeMap()
            CMapParser(self.unicode_map, BytesIO(strm.get_data())).run()
        PDFFont.__init__(self, descriptor, widths)
        return

    def to_unichr(self, cid: int) -> str:
        if self.unicode_map:
            try:
                return self.unicode_map.get_unichr(cid)
            except KeyError:
                pass
        try:
            return self.cid2unicode[cid]
        except KeyError:
            raise PDFUnicodeNotDefined(None, cid)


class PDFType1Font(PDFSimpleFont):

    def __init__(
        self,
        rsrcmgr: "PDFResourceManager",
        spec: Mapping[str, Any]
    ) -> None:
        try:
            self.basefont = literal_name(spec['BaseFont'])
        except KeyError:
            if settings.STRICT:
                raise PDFFontError('BaseFont is missing')
            self.basefont = 'unknown'

        widths: FontWidthDict
        try:
            (descriptor, int_widths) = FontMetricsDB.get_metrics(self.basefont)
            widths = cast(Dict[str, float], int_widths)  # implicit int->float
        except KeyError:
            descriptor = dict_value(spec.get('FontDescriptor', {}))
            firstchar = int_value(spec.get('FirstChar', 0))
            # lastchar = int_value(spec.get('LastChar', 255))
            width_list = list_value(spec.get('Widths', [0]*256))
            widths = {i+firstchar: w for (i, w) in enumerate(width_list)}
        PDFSimpleFont.__init__(self, descriptor, widths, spec)
        if 'Encoding' not in spec and 'FontFile' in descriptor:
            # try to recover the missing encoding info from the font file.
            self.fontfile = stream_value(descriptor.get('FontFile'))
            length1 = int_value(self.fontfile['Length1'])
            data = self.fontfile.get_data()[:length1]
            parser = Type1FontHeaderParser(BytesIO(data))
            self.cid2unicode = parser.get_encoding()
        return

    def __repr__(self) -> str:
        return '<PDFType1Font: basefont=%r>' % self.basefont


class PDFTrueTypeFont(PDFType1Font):

    def __repr__(self) -> str:
        return '<PDFTrueTypeFont: basefont=%r>' % self.basefont


class PDFType3Font(PDFSimpleFont):

    def __init__(
        self,
        rsrcmgr: "PDFResourceManager",
        spec: Mapping[str, Any]
    ) -> None:
        firstchar = int_value(spec.get('FirstChar', 0))
        # lastchar = int_value(spec.get('LastChar', 0))
        width_list = list_value(spec.get('Widths', [0]*256))
        widths = {i+firstchar: w for (i, w) in enumerate(width_list)}
        if 'FontDescriptor' in spec:
            descriptor = dict_value(spec['FontDescriptor'])
        else:
            descriptor = {'Ascent': 0, 'Descent': 0,
                          'FontBBox': spec['FontBBox']}
        PDFSimpleFont.__init__(self, descriptor, widths, spec)
        self.matrix = cast(Matrix, tuple(list_value(spec.get('FontMatrix'))))
        (_, self.descent, _, self.ascent) = self.bbox
        (self.hscale, self.vscale) = apply_matrix_norm(self.matrix, (1, 1))
        return

    def __repr__(self) -> str:
        return '<PDFType3Font>'


class PDFCIDFont(PDFFont):
    default_disp: Union[float, Tuple[Optional[float], float]]

    def __init__(
        self,
        rsrcmgr: "PDFResourceManager",
        spec: Mapping[str, Any],
        strict: bool = settings.STRICT
    ) -> None:
        try:
            self.basefont = literal_name(spec['BaseFont'])
        except KeyError:
            if strict:
                raise PDFFontError('BaseFont is missing')
            self.basefont = 'unknown'
        self.cidsysteminfo = dict_value(spec.get('CIDSystemInfo', {}))
        cid_registry = resolve1(
            self.cidsysteminfo.get('Registry', b'unknown')).decode("latin1")
        cid_ordering = resolve1(
            self.cidsysteminfo.get('Ordering', b'unknown')).decode("latin1")
        self.cidcoding = '{}-{}'.format(cid_registry, cid_ordering)
        self.cmap: CMapBase = self.get_cmap_from_spec(spec, strict)

        try:
            descriptor = dict_value(spec['FontDescriptor'])
        except KeyError:
            if strict:
                raise PDFFontError('FontDescriptor is missing')
            descriptor = {}
        ttf = None
        if 'FontFile2' in descriptor:
            self.fontfile = stream_value(descriptor.get('FontFile2'))
            ttf = TrueTypeFont(self.basefont,
                               BytesIO(self.fontfile.get_data()))
        self.unicode_map: Optional[UnicodeMap] = None
        if 'ToUnicode' in spec:
            strm = stream_value(spec['ToUnicode'])
            self.unicode_map = FileUnicodeMap()
            CMapParser(self.unicode_map, BytesIO(strm.get_data())).run()
        elif self.cidcoding in ('Adobe-Identity', 'Adobe-UCS'):
            if ttf:
                try:
                    self.unicode_map = ttf.create_unicode_map()
                except TrueTypeFont.CMapNotFound:
                    pass
        else:
            try:
                self.unicode_map = CMapDB.get_unicode_map(
                    self.cidcoding, self.cmap.is_vertical())
            except CMapDB.CMapNotFound:
                pass

        self.vertical = self.cmap.is_vertical()
        if self.vertical:
            # writing mode: vertical
            widths2 = get_widths2(list_value(spec.get('W2', [])))
            self.disps = {cid: (vx, vy)
                          for (cid, (_, (vx, vy))) in widths2.items()}
            (vy, w) = resolve1(spec.get('DW2', [880, -1000]))
            self.default_disp = (None, vy)
            widths = {cid: w for (cid, (w, _)) in widths2.items()}
            default_width = w
        else:
            # writing mode: horizontal
            self.disps = {}
            self.default_disp = 0
            widths = get_widths(list_value(spec.get('W', [])))
            default_width = spec.get('DW', 1000)
        PDFFont.__init__(self, descriptor, widths, default_width=default_width)
        return

    def get_cmap_from_spec(
        self,
        spec: Mapping[str, Any],
        strict: bool
    ) -> CMapBase:
        """Get cmap from font specification

        For certain PDFs, Encoding Type isn't mentioned as an attribute of
        Encoding but as an attribute of CMapName, where CMapName is an
        attribute of spec['Encoding'].
        The horizontal/vertical modes are mentioned with different name
        such as 'DLIdent-H/V','OneByteIdentityH/V','Identity-H/V'.
        """
        cmap_name = self._get_cmap_name(spec, strict)

        try:
            return CMapDB.get_cmap(cmap_name)
        except CMapDB.CMapNotFound as e:
            if strict:
                raise PDFFontError(e)
            return CMap()

    @staticmethod
    def _get_cmap_name(spec: Mapping[str, Any], strict: bool) -> str:
        """Get cmap name from font specification"""
        cmap_name = 'unknown'  # default value

        try:
            spec_encoding = spec['Encoding']
            if hasattr(spec_encoding, 'name'):
                cmap_name = literal_name(spec['Encoding'])
            else:
                cmap_name = literal_name(spec_encoding['CMapName'])
        except KeyError:
            if strict:
                raise PDFFontError('Encoding is unspecified')

        if type(cmap_name) is PDFStream:  # type: ignore[comparison-overlap]
            cmap_name_stream: PDFStream = cast(PDFStream, cmap_name)
            if 'CMapName' in cmap_name_stream:
                cmap_name = cmap_name_stream.get('CMapName').name
            else:
                if strict:
                    raise PDFFontError('CMapName unspecified for encoding')

        return IDENTITY_ENCODER.get(cmap_name, cmap_name)

    def __repr__(self) -> str:
        return '<PDFCIDFont: basefont={!r}, cidcoding={!r}>'\
            .format(self.basefont, self.cidcoding)

    def is_vertical(self) -> bool:
        return self.vertical

    def is_multibyte(self) -> bool:
        return True

    def decode(self, bytes: bytes) -> Iterable[int]:
        return self.cmap.decode(bytes)

    def char_disp(
        self,
        cid: int
    ) -> Union[float, Tuple[Optional[float], float]]:
        "Returns an integer for horizontal fonts, a tuple for vertical fonts."
        return self.disps.get(cid, self.default_disp)

    def to_unichr(self, cid: int) -> str:
        try:
            if not self.unicode_map:
                raise KeyError(cid)
            return self.unicode_map.get_unichr(cid)
        except KeyError:
            raise PDFUnicodeNotDefined(self.cidcoding, cid)


def main(argv: List[str]) -> None:
    for fname in argv[1:]:
        fp = open(fname, 'rb')
        font = CFFFont(fname, fp)
        print(font)
        fp.close()
    return


if __name__ == '__main__':
    main(sys.argv)
