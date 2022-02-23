import zlib
import logging
import sys
from typing import (TYPE_CHECKING, Any, Dict, Iterable, Optional, Union, List,
                    Tuple, cast)
from .lzw import lzwdecode
from .ascii85 import ascii85decode
from .ascii85 import asciihexdecode
from .runlength import rldecode
from .ccitt import ccittfaxdecode
from .psparser import PSException
from .psparser import PSObject
from .psparser import LIT
from . import settings
from .utils import apply_png_predictor

if TYPE_CHECKING:
    from .pdfdocument import PDFDocument


log = logging.getLogger(__name__)

LITERAL_CRYPT = LIT('Crypt')

# Abbreviation of Filter names in PDF 4.8.6. "Inline Images"
LITERALS_FLATE_DECODE = (LIT('FlateDecode'), LIT('Fl'))
LITERALS_LZW_DECODE = (LIT('LZWDecode'), LIT('LZW'))
LITERALS_ASCII85_DECODE = (LIT('ASCII85Decode'), LIT('A85'))
LITERALS_ASCIIHEX_DECODE = (LIT('ASCIIHexDecode'), LIT('AHx'))
LITERALS_RUNLENGTH_DECODE = (LIT('RunLengthDecode'), LIT('RL'))
LITERALS_CCITTFAX_DECODE = (LIT('CCITTFaxDecode'), LIT('CCF'))
LITERALS_DCT_DECODE = (LIT('DCTDecode'), LIT('DCT'))
LITERALS_JBIG2_DECODE = (LIT('JBIG2Decode'),)


if sys.version_info >= (3, 8):
    from typing import Protocol

    class DecipherCallable(Protocol):
        """Fully typed a decipher callback, with optional parameter."""
        def __call__(self, objid: int, genno: int, data: bytes,
                     attrs: Optional[Dict[str, Any]] = None) -> bytes:
            raise NotImplementedError

else:  # Fallback for older Python
    from typing import Callable

    DecipherCallable = Callable[..., bytes]


class PDFObject(PSObject):
    pass


class PDFException(PSException):
    pass


class PDFTypeError(PDFException):
    pass


class PDFValueError(PDFException):
    pass


class PDFObjectNotFound(PDFException):
    pass


class PDFNotImplementedError(PDFException):
    pass


class PDFObjRef(PDFObject):

    def __init__(
        self,
        doc: Optional["PDFDocument"],
        objid: int,
        _: object
    ) -> None:
        if objid == 0:
            if settings.STRICT:
                raise PDFValueError('PDF object id cannot be 0.')
        self.doc = doc
        self.objid = objid
        return

    def __repr__(self) -> str:
        return '<PDFObjRef:%d>' % (self.objid)

    def resolve(self, default: object = None) -> Any:
        assert self.doc is not None
        try:
            return self.doc.getobj(self.objid)
        except PDFObjectNotFound:
            return default


def resolve1(x: object, default: object = None) -> Any:
    """Resolves an object.

    If this is an array or dictionary, it may still contains
    some indirect objects inside.
    """
    while isinstance(x, PDFObjRef):
        x = x.resolve(default=default)
    return x


def resolve_all(x: object, default: object = None) -> Any:
    """Recursively resolves the given object and all the internals.

    Make sure there is no indirect reference within the nested object.
    This procedure might be slow.
    """
    while isinstance(x, PDFObjRef):
        x = x.resolve(default=default)
    if isinstance(x, list):
        x = [resolve_all(v, default=default) for v in x]
    elif isinstance(x, dict):
        for (k, v) in x.items():
            x[k] = resolve_all(v, default=default)
    return x


def decipher_all(
    decipher: DecipherCallable,
    objid: int,
    genno: int,
    x: object
) -> Any:
    """Recursively deciphers the given object.
    """
    if isinstance(x, bytes):
        return decipher(objid, genno, x)
    if isinstance(x, list):
        x = [decipher_all(decipher, objid, genno, v) for v in x]
    elif isinstance(x, dict):
        for (k, v) in x.items():
            x[k] = decipher_all(decipher, objid, genno, v)
    return x


def int_value(x: object) -> int:
    x = resolve1(x)
    if not isinstance(x, int):
        if settings.STRICT:
            raise PDFTypeError('Integer required: %r' % x)
        return 0
    return x


def float_value(x: object) -> float:
    x = resolve1(x)
    if not isinstance(x, float):
        if settings.STRICT:
            raise PDFTypeError('Float required: %r' % x)
        return 0.0
    return x


def num_value(x: object) -> float:
    x = resolve1(x)
    if not isinstance(x, (int, float)):  # == utils.isnumber(x)
        if settings.STRICT:
            raise PDFTypeError('Int or Float required: %r' % x)
        return 0
    return x


def uint_value(x: object, n_bits: int) -> int:
    """Resolve number and interpret it as a two's-complement unsigned number"""
    xi = int_value(x)
    if xi > 0:
        return xi
    else:
        return xi + cast(int, 2**n_bits)


def str_value(x: object) -> bytes:
    x = resolve1(x)
    if not isinstance(x, bytes):
        if settings.STRICT:
            raise PDFTypeError('String required: %r' % x)
        return b''
    return x


def list_value(x: object) -> Union[List[Any], Tuple[Any, ...]]:
    x = resolve1(x)
    if not isinstance(x, (list, tuple)):
        if settings.STRICT:
            raise PDFTypeError('List required: %r' % x)
        return []
    return x


def dict_value(x: object) -> Dict[Any, Any]:
    x = resolve1(x)
    if not isinstance(x, dict):
        if settings.STRICT:
            log.error('PDFTypeError : Dict required: %r', x)
            raise PDFTypeError('Dict required: %r' % x)
        return {}
    return x


def stream_value(x: object) -> "PDFStream":
    x = resolve1(x)
    if not isinstance(x, PDFStream):
        if settings.STRICT:
            raise PDFTypeError('PDFStream required: %r' % x)
        return PDFStream({}, b'')
    return x


class PDFStream(PDFObject):

    def __init__(
        self,
        attrs: Dict[str, Any],
        rawdata: bytes,
        decipher: Optional[DecipherCallable] = None
    ) -> None:
        assert isinstance(attrs, dict), str(type(attrs))
        self.attrs = attrs
        self.rawdata: Optional[bytes] = rawdata
        self.decipher = decipher
        self.data: Optional[bytes] = None
        self.objid: Optional[int] = None
        self.genno: Optional[int] = None
        return

    def set_objid(self, objid: int, genno: int) -> None:
        self.objid = objid
        self.genno = genno
        return

    def __repr__(self) -> str:
        if self.data is None:
            assert self.rawdata is not None
            return '<PDFStream(%r): raw=%d, %r>' % \
                   (self.objid, len(self.rawdata), self.attrs)
        else:
            assert self.data is not None
            return '<PDFStream(%r): len=%d, %r>' % \
                   (self.objid, len(self.data), self.attrs)

    def __contains__(self, name: object) -> bool:
        return name in self.attrs

    def __getitem__(self, name: str) -> Any:
        return self.attrs[name]

    def get(self, name: str, default: object = None) -> Any:
        return self.attrs.get(name, default)

    def get_any(self, names: Iterable[str], default: object = None) -> Any:
        for name in names:
            if name in self.attrs:
                return self.attrs[name]
        return default

    def get_filters(self) -> List[Tuple[Any, Any]]:
        filters = self.get_any(('F', 'Filter'))
        params = self.get_any(('DP', 'DecodeParms', 'FDecodeParms'), {})
        if not filters:
            return []
        if not isinstance(filters, list):
            filters = [filters]
        if not isinstance(params, list):
            # Make sure the parameters list is the same as filters.
            params = [params] * len(filters)
        if settings.STRICT and len(params) != len(filters):
            raise PDFException("Parameters len filter mismatch")
        # resolve filter if possible
        _filters = []
        for fltr in filters:
            if hasattr(fltr, 'resolve'):
                fltr = fltr.resolve()[0]
            _filters.append(fltr)
        # return list solves https://github.com/pdfminer/pdfminer.six/issues/15
        return list(zip(_filters, params))

    def decode(self) -> None:
        assert self.data is None \
               and self.rawdata is not None, str((self.data, self.rawdata))
        data = self.rawdata
        if self.decipher:
            # Handle encryption
            assert self.objid is not None
            assert self.genno is not None
            data = self.decipher(self.objid, self.genno, data, self.attrs)
        filters = self.get_filters()
        if not filters:
            self.data = data
            self.rawdata = None
            return
        for (f, params) in filters:
            if f in LITERALS_FLATE_DECODE:
                # will get errors if the document is encrypted.
                try:
                    data = zlib.decompress(data)
                except zlib.error as e:
                    if settings.STRICT:
                        error_msg = 'Invalid zlib bytes: {!r}, {!r}'\
                            .format(e, data)
                        raise PDFException(error_msg)
                    data = b''
            elif f in LITERALS_LZW_DECODE:
                data = lzwdecode(data)
            elif f in LITERALS_ASCII85_DECODE:
                data = ascii85decode(data)
            elif f in LITERALS_ASCIIHEX_DECODE:
                data = asciihexdecode(data)
            elif f in LITERALS_RUNLENGTH_DECODE:
                data = rldecode(data)
            elif f in LITERALS_CCITTFAX_DECODE:
                data = ccittfaxdecode(data, params)
            elif f in LITERALS_DCT_DECODE:
                # This is probably a JPG stream
                # it does not need to be decoded twice.
                # Just return the stream to the user.
                pass
            elif f in LITERALS_JBIG2_DECODE:
                pass
            elif f == LITERAL_CRYPT:
                # not yet..
                raise PDFNotImplementedError('/Crypt filter is unsupported')
            else:
                raise PDFNotImplementedError('Unsupported filter: %r' % f)
            # apply predictors
            if params and 'Predictor' in params:
                pred = int_value(params['Predictor'])
                if pred == 1:
                    # no predictor
                    pass
                elif 10 <= pred:
                    # PNG predictor
                    colors = int_value(params.get('Colors', 1))
                    columns = int_value(params.get('Columns', 1))
                    raw_bits_per_component = params.get('BitsPerComponent', 8)
                    bitspercomponent = int_value(raw_bits_per_component)
                    data = apply_png_predictor(pred, colors, columns,
                                               bitspercomponent, data)
                else:
                    error_msg = 'Unsupported predictor: %r' % pred
                    raise PDFNotImplementedError(error_msg)
        self.data = data
        self.rawdata = None
        return

    def get_data(self) -> bytes:
        if self.data is None:
            self.decode()
            assert self.data is not None
        return self.data

    def get_rawdata(self) -> Optional[bytes]:
        return self.rawdata
