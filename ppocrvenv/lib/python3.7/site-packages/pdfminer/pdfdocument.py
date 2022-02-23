import logging
import re
import struct
from hashlib import sha256, md5, sha384, sha512
from typing import (Any, Callable, Dict, Iterable, Iterator, KeysView, List,
                    Optional, Sequence, Tuple, Type, Union, cast)

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from . import settings
from .arcfour import Arcfour
from .pdfparser import PDFSyntaxError, PDFParser, PDFStreamParser
from .pdftypes import DecipherCallable, PDFException, PDFTypeError, PDFStream,\
    PDFObjectNotFound, decipher_all, int_value, str_value, list_value, \
    uint_value, dict_value, stream_value
from .psparser import PSEOF, literal_name, LIT, KWD
from .utils import choplist, nunpack, decode_text

log = logging.getLogger(__name__)


class PDFNoValidXRef(PDFSyntaxError):
    pass


class PDFNoValidXRefWarning(SyntaxWarning):
    pass


class PDFNoOutlines(PDFException):
    pass


class PDFDestinationNotFound(PDFException):
    pass


class PDFEncryptionError(PDFException):
    pass


class PDFPasswordIncorrect(PDFEncryptionError):
    pass


class PDFTextExtractionNotAllowedWarning(UserWarning):
    pass


class PDFTextExtractionNotAllowed(PDFEncryptionError):
    pass


class PDFTextExtractionNotAllowedError(PDFTextExtractionNotAllowed):
    def __init__(self, *args: object) -> None:
        from warnings import warn
        warn('PDFTextExtractionNotAllowedError will be removed in the future. '
             'Use PDFTextExtractionNotAllowed instead.', DeprecationWarning)
        super().__init__(*args)


# some predefined literals and keywords.
LITERAL_OBJSTM = LIT('ObjStm')
LITERAL_XREF = LIT('XRef')
LITERAL_CATALOG = LIT('Catalog')


class PDFBaseXRef:
    def get_trailer(self) -> Dict[str, Any]:
        raise NotImplementedError

    def get_objids(self) -> Iterable[int]:
        return []

    # Must return
    #     (strmid, index, genno)
    #  or (None, pos, genno)
    def get_pos(self, objid: int) -> Tuple[Optional[int], int, int]:
        raise KeyError(objid)

    def load(self, parser: PDFParser) -> None:
        raise NotImplementedError


class PDFXRef(PDFBaseXRef):

    def __init__(self) -> None:
        self.offsets: Dict[int, Tuple[Optional[int], int, int]] = {}
        self.trailer: Dict[str, Any] = {}
        return

    def __repr__(self) -> str:
        return '<PDFXRef: offsets=%r>' % (self.offsets.keys())

    def load(self, parser: PDFParser) -> None:
        while True:
            try:
                (pos, line) = parser.nextline()
                line = line.strip()
                if not line:
                    continue
            except PSEOF:
                raise PDFNoValidXRef('Unexpected EOF - file corrupted?')
            if line.startswith(b'trailer'):
                parser.seek(pos)
                break
            f = line.split(b' ')
            if len(f) != 2:
                error_msg = 'Trailer not found: {!r}: line={!r}'\
                    .format(parser, line)
                raise PDFNoValidXRef(error_msg)
            try:
                (start, nobjs) = map(int, f)
            except ValueError:
                error_msg = 'Invalid line: {!r}: line={!r}'\
                    .format(parser, line)
                raise PDFNoValidXRef(error_msg)
            for objid in range(start, start+nobjs):
                try:
                    (_, line) = parser.nextline()
                    line = line.strip()
                except PSEOF:
                    raise PDFNoValidXRef('Unexpected EOF - file corrupted?')
                f = line.split(b' ')
                if len(f) != 3:
                    error_msg = 'Invalid XRef format: {!r}, line={!r}'\
                        .format(parser, line)
                    raise PDFNoValidXRef(error_msg)
                (pos_b, genno_b, use_b) = f
                if use_b != b'n':
                    continue
                self.offsets[objid] = (None, int(pos_b), int(genno_b))
        log.info('xref objects: %r', self.offsets)
        self.load_trailer(parser)
        return

    def load_trailer(self, parser: PDFParser) -> None:
        try:
            (_, kwd) = parser.nexttoken()
            assert kwd is KWD(b'trailer'), str(kwd)
            (_, dic) = parser.nextobject()
        except PSEOF:
            x = parser.pop(1)
            if not x:
                raise PDFNoValidXRef('Unexpected EOF - file corrupted')
            (_, dic) = x[0]
        self.trailer.update(dict_value(dic))
        log.debug('trailer=%r', self.trailer)
        return

    def get_trailer(self) -> Dict[str, Any]:
        return self.trailer

    def get_objids(self) -> KeysView[int]:
        return self.offsets.keys()

    def get_pos(self, objid: int) -> Tuple[Optional[int], int, int]:
        try:
            return self.offsets[objid]
        except KeyError:
            raise


class PDFXRefFallback(PDFXRef):

    def __repr__(self) -> str:
        return '<PDFXRefFallback: offsets=%r>' % (self.offsets.keys())

    PDFOBJ_CUE = re.compile(r'^(\d+)\s+(\d+)\s+obj\b')

    def load(self, parser: PDFParser) -> None:
        parser.seek(0)
        while 1:
            try:
                (pos, line_bytes) = parser.nextline()
            except PSEOF:
                break
            if line_bytes.startswith(b'trailer'):
                parser.seek(pos)
                self.load_trailer(parser)
                log.info('trailer: %r', self.trailer)
                break
            line = line_bytes.decode('latin-1')  # default pdf encoding
            m = self.PDFOBJ_CUE.match(line)
            if not m:
                continue
            (objid_s, genno_s) = m.groups()
            objid = int(objid_s)
            genno = int(genno_s)
            self.offsets[objid] = (None, pos, genno)
            # expand ObjStm.
            parser.seek(pos)
            (_, obj) = parser.nextobject()
            if isinstance(obj, PDFStream) \
                    and obj.get('Type') is LITERAL_OBJSTM:
                stream = stream_value(obj)
                try:
                    n = stream['N']
                except KeyError:
                    if settings.STRICT:
                        raise PDFSyntaxError('N is not defined: %r' % stream)
                    n = 0
                parser1 = PDFStreamParser(stream.get_data())
                objs: List[int] = []
                try:
                    while 1:
                        (_, obj) = parser1.nextobject()
                        objs.append(cast(int, obj))
                except PSEOF:
                    pass
                n = min(n, len(objs)//2)
                for index in range(n):
                    objid1 = objs[index*2]
                    self.offsets[objid1] = (objid, index, 0)
        return


class PDFXRefStream(PDFBaseXRef):

    def __init__(self) -> None:
        self.data: Optional[bytes] = None
        self.entlen: Optional[int] = None
        self.fl1: Optional[int] = None
        self.fl2: Optional[int] = None
        self.fl3: Optional[int] = None
        self.ranges: List[Tuple[int, int]] = []
        return

    def __repr__(self) -> str:
        return '<PDFXRefStream: ranges=%r>' % (self.ranges)

    def load(self, parser: PDFParser) -> None:
        (_, objid) = parser.nexttoken()  # ignored
        (_, genno) = parser.nexttoken()  # ignored
        (_, kwd) = parser.nexttoken()
        (_, stream) = parser.nextobject()
        if not isinstance(stream, PDFStream) \
                or stream.get('Type') is not LITERAL_XREF:
            raise PDFNoValidXRef('Invalid PDF stream spec.')
        size = stream['Size']
        index_array = stream.get('Index', (0, size))
        if len(index_array) % 2 != 0:
            raise PDFSyntaxError('Invalid index number')
        self.ranges.extend(cast(Iterator[Tuple[int, int]],
                                choplist(2, index_array)))
        (self.fl1, self.fl2, self.fl3) = stream['W']
        assert (self.fl1 is not None and self.fl2 is not None
                and self.fl3 is not None)
        self.data = stream.get_data()
        self.entlen = self.fl1+self.fl2+self.fl3
        self.trailer = stream.attrs
        log.info('xref stream: objid=%s, fields=%d,%d,%d',
                 ', '.join(map(repr, self.ranges)),
                 self.fl1, self.fl2, self.fl3)
        return

    def get_trailer(self) -> Dict[str, Any]:
        return self.trailer

    def get_objids(self) -> Iterator[int]:
        for (start, nobjs) in self.ranges:
            for i in range(nobjs):
                assert self.entlen is not None
                assert self.data is not None
                offset = self.entlen * i
                ent = self.data[offset:offset+self.entlen]
                f1 = nunpack(ent[:self.fl1], 1)
                if f1 == 1 or f1 == 2:
                    yield start+i
        return

    def get_pos(self, objid: int) -> Tuple[Optional[int], int, int]:
        index = 0
        for (start, nobjs) in self.ranges:
            if start <= objid and objid < start+nobjs:
                index += objid - start
                break
            else:
                index += nobjs
        else:
            raise KeyError(objid)
        assert self.entlen is not None
        assert self.data is not None
        assert (self.fl1 is not None and self.fl2 is not None
                and self.fl3 is not None)
        offset = self.entlen * index
        ent = self.data[offset:offset+self.entlen]
        f1 = nunpack(ent[:self.fl1], 1)
        f2 = nunpack(ent[self.fl1:self.fl1+self.fl2])
        f3 = nunpack(ent[self.fl1+self.fl2:])
        if f1 == 1:
            return (None, f2, f3)
        elif f1 == 2:
            return (f2, f3, 0)
        else:
            # this is a free object
            raise KeyError(objid)


class PDFStandardSecurityHandler:

    PASSWORD_PADDING = (b'(\xbfN^Nu\x8aAd\x00NV\xff\xfa\x01\x08'
                        b'..\x00\xb6\xd0h>\x80/\x0c\xa9\xfedSiz')
    supported_revisions: Tuple[int, ...] = (2, 3)

    def __init__(
        self,
        docid: Sequence[bytes],
        param: Dict[str, Any],
        password: str = ''
    ) -> None:
        self.docid = docid
        self.param = param
        self.password = password
        self.init()
        return

    def init(self) -> None:
        self.init_params()
        if self.r not in self.supported_revisions:
            error_msg = 'Unsupported revision: param=%r' % self.param
            raise PDFEncryptionError(error_msg)
        self.init_key()
        return

    def init_params(self) -> None:
        self.v = int_value(self.param.get('V', 0))
        self.r = int_value(self.param['R'])
        self.p = uint_value(self.param['P'], 32)
        self.o = str_value(self.param['O'])
        self.u = str_value(self.param['U'])
        self.length = int_value(self.param.get('Length', 40))
        return

    def init_key(self) -> None:
        self.key = self.authenticate(self.password)
        if self.key is None:
            raise PDFPasswordIncorrect
        return

    def is_printable(self) -> bool:
        return bool(self.p & 4)

    def is_modifiable(self) -> bool:
        return bool(self.p & 8)

    def is_extractable(self) -> bool:
        return bool(self.p & 16)

    def compute_u(self, key: bytes) -> bytes:
        if self.r == 2:
            # Algorithm 3.4
            return Arcfour(key).encrypt(self.PASSWORD_PADDING)  # 2
        else:
            # Algorithm 3.5
            hash = md5(self.PASSWORD_PADDING)  # 2
            hash.update(self.docid[0])  # 3
            result = Arcfour(key).encrypt(hash.digest())  # 4
            for i in range(1, 20):  # 5
                k = b''.join(bytes((c ^ i,)) for c in iter(key))
                result = Arcfour(k).encrypt(result)
            result += result  # 6
            return result

    def compute_encryption_key(self, password: bytes) -> bytes:
        # Algorithm 3.2
        password = (password + self.PASSWORD_PADDING)[:32]  # 1
        hash = md5(password)  # 2
        hash.update(self.o)  # 3
        # See https://github.com/pdfminer/pdfminer.six/issues/186
        hash.update(struct.pack('<L', self.p))  # 4
        hash.update(self.docid[0])  # 5
        if self.r >= 4:
            if not cast(PDFStandardSecurityHandlerV4, self).encrypt_metadata:
                hash.update(b'\xff\xff\xff\xff')
        result = hash.digest()
        n = 5
        if self.r >= 3:
            n = self.length // 8
            for _ in range(50):
                result = md5(result[:n]).digest()
        return result[:n]

    def authenticate(self, password: str) -> Optional[bytes]:
        password_bytes = password.encode("latin1")
        key = self.authenticate_user_password(password_bytes)
        if key is None:
            key = self.authenticate_owner_password(password_bytes)
        return key

    def authenticate_user_password(self, password: bytes) -> Optional[bytes]:
        key = self.compute_encryption_key(password)
        if self.verify_encryption_key(key):
            return key
        else:
            return None

    def verify_encryption_key(self, key: bytes) -> bool:
        # Algorithm 3.6
        u = self.compute_u(key)
        if self.r == 2:
            return u == self.u
        return u[:16] == self.u[:16]

    def authenticate_owner_password(self, password: bytes) -> Optional[bytes]:
        # Algorithm 3.7
        password = (password + self.PASSWORD_PADDING)[:32]
        hash = md5(password)
        if self.r >= 3:
            for _ in range(50):
                hash = md5(hash.digest())
        n = 5
        if self.r >= 3:
            n = self.length // 8
        key = hash.digest()[:n]
        if self.r == 2:
            user_password = Arcfour(key).decrypt(self.o)
        else:
            user_password = self.o
            for i in range(19, -1, -1):
                k = b''.join(bytes((c ^ i,)) for c in iter(key))
                user_password = Arcfour(k).decrypt(user_password)
        return self.authenticate_user_password(user_password)

    def decrypt(
        self,
        objid: int,
        genno: int,
        data: bytes,
        attrs: Optional[Dict[str, Any]] = None
    ) -> bytes:
        return self.decrypt_rc4(objid, genno, data)

    def decrypt_rc4(self, objid: int, genno: int, data: bytes) -> bytes:
        assert self.key is not None
        key = self.key + struct.pack('<L', objid)[:3] \
            + struct.pack('<L', genno)[:2]
        hash = md5(key)
        key = hash.digest()[:min(len(key), 16)]
        return Arcfour(key).decrypt(data)


class PDFStandardSecurityHandlerV4(PDFStandardSecurityHandler):

    supported_revisions: Tuple[int, ...] = (4,)

    def init_params(self) -> None:
        super().init_params()
        self.length = 128
        self.cf = dict_value(self.param.get('CF'))
        self.stmf = literal_name(self.param['StmF'])
        self.strf = literal_name(self.param['StrF'])
        self.encrypt_metadata = bool(self.param.get('EncryptMetadata', True))
        if self.stmf != self.strf:
            error_msg = 'Unsupported crypt filter: param=%r' % self.param
            raise PDFEncryptionError(error_msg)
        self.cfm = {}
        for k, v in self.cf.items():
            f = self.get_cfm(literal_name(v['CFM']))
            if f is None:
                error_msg = 'Unknown crypt filter method: param=%r' \
                            % self.param
                raise PDFEncryptionError(error_msg)
            self.cfm[k] = f
        self.cfm['Identity'] = self.decrypt_identity
        if self.strf not in self.cfm:
            error_msg = 'Undefined crypt filter: param=%r' % self.param
            raise PDFEncryptionError(error_msg)
        return

    def get_cfm(
        self,
        name: str
    ) -> Optional[Callable[[int, int, bytes], bytes]]:
        if name == 'V2':
            return self.decrypt_rc4
        elif name == 'AESV2':
            return self.decrypt_aes128
        else:
            return None

    def decrypt(
        self,
        objid: int,
        genno: int,
        data: bytes,
        attrs: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None
    ) -> bytes:
        if not self.encrypt_metadata and attrs is not None:
            t = attrs.get('Type')
            if t is not None and literal_name(t) == 'Metadata':
                return data
        if name is None:
            name = self.strf
        return self.cfm[name](objid, genno, data)

    def decrypt_identity(self, objid: int, genno: int, data: bytes) -> bytes:
        return data

    def decrypt_aes128(self, objid: int, genno: int, data: bytes) -> bytes:
        assert self.key is not None
        key = self.key + struct.pack('<L', objid)[:3] \
            + struct.pack('<L', genno)[:2] + b'sAlT'
        hash = md5(key)
        key = hash.digest()[:min(len(key), 16)]
        initialization_vector = data[:16]
        ciphertext = data[16:]
        cipher = Cipher(algorithms.AES(key),
                        modes.CBC(initialization_vector),
                        backend=default_backend())  # type: ignore
        return cipher.decryptor().update(ciphertext)  # type: ignore


class PDFStandardSecurityHandlerV5(PDFStandardSecurityHandlerV4):

    supported_revisions = (5, 6)

    def init_params(self) -> None:
        super().init_params()
        self.length = 256
        self.oe = str_value(self.param['OE'])
        self.ue = str_value(self.param['UE'])
        self.o_hash = self.o[:32]
        self.o_validation_salt = self.o[32:40]
        self.o_key_salt = self.o[40:]
        self.u_hash = self.u[:32]
        self.u_validation_salt = self.u[32:40]
        self.u_key_salt = self.u[40:]
        return

    def get_cfm(
        self,
        name: str
    ) -> Optional[Callable[[int, int, bytes], bytes]]:
        if name == 'AESV3':
            return self.decrypt_aes256
        else:
            return None

    def authenticate(self, password: str) -> Optional[bytes]:
        password_b = self._normalize_password(password)
        hash = self._password_hash(password_b, self.o_validation_salt, self.u)
        if hash == self.o_hash:
            hash = self._password_hash(password_b, self.o_key_salt, self.u)
            cipher = Cipher(algorithms.AES(hash),
                            modes.CBC(b'\0' * 16),
                            backend=default_backend())  # type: ignore
            return cipher.decryptor().update(self.oe)  # type: ignore
        hash = self._password_hash(password_b, self.u_validation_salt)
        if hash == self.u_hash:
            hash = self._password_hash(password_b, self.u_key_salt)
            cipher = Cipher(algorithms.AES(hash),
                            modes.CBC(b'\0' * 16),
                            backend=default_backend())  # type: ignore
            return cipher.decryptor().update(self.ue)  # type: ignore
        return None

    def _normalize_password(self, password: str) -> bytes:
        if self.r == 6:
            # saslprep expects non-empty strings, apparently
            if not password:
                return b''
            from ._saslprep import saslprep
            password = saslprep(password)
        return password.encode('utf-8')[:127]

    def _password_hash(
        self,
        password: bytes,
        salt: bytes,
        vector: Optional[bytes] = None
    ) -> bytes:
        """
        Compute password hash depending on revision number
        """
        if self.r == 5:
            return self._r5_password(password, salt, vector)
        return self._r6_password(password, salt[0:8], vector)

    def _r5_password(
        self,
        password: bytes,
        salt: bytes,
        vector: Optional[bytes] = None
    ) -> bytes:
        """
        Compute the password for revision 5
        """
        hash = sha256(password)
        hash.update(salt)
        if vector is not None:
            hash.update(vector)
        return hash.digest()

    def _r6_password(
        self,
        password: bytes,
        salt: bytes,
        vector: Optional[bytes] = None
    ) -> bytes:
        """
        Compute the password for revision 6
        """
        initial_hash = sha256(password)
        initial_hash.update(salt)
        if vector is not None:
            initial_hash.update(vector)
        k = initial_hash.digest()
        hashes = (sha256, sha384, sha512)
        round_no = last_byte_val = 0
        while round_no < 64 or last_byte_val > round_no - 32:
            k1 = (password + k + (vector or b'')) * 64
            e = self._aes_cbc_encrypt(
                key=k[:16], iv=k[16:32], data=k1
            )
            # compute the first 16 bytes of e,
            # interpreted as an unsigned integer mod 3
            next_hash = hashes[self._bytes_mod_3(e[:16])]
            k = next_hash(e).digest()
            last_byte_val = e[len(e) - 1]
            round_no += 1
        return k[:32]

    @staticmethod
    def _bytes_mod_3(input_bytes: bytes) -> int:
        # 256 is 1 mod 3, so we can just sum 'em
        return sum(b % 3 for b in input_bytes) % 3

    def _aes_cbc_encrypt(
        self,
        key: bytes,
        iv: bytes,
        data: bytes
    ) -> bytes:
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        encryptor = cipher.encryptor()  # type: ignore
        return encryptor.update(data) + encryptor.finalize()  # type: ignore

    def decrypt_aes256(self, objid: int, genno: int, data: bytes) -> bytes:
        initialization_vector = data[:16]
        ciphertext = data[16:]
        assert self.key is not None
        cipher = Cipher(algorithms.AES(self.key),
                        modes.CBC(initialization_vector),
                        backend=default_backend())  # type: ignore
        return cipher.decryptor().update(ciphertext)  # type: ignore


class PDFDocument:
    """PDFDocument object represents a PDF document.

    Since a PDF file can be very big, normally it is not loaded at
    once. So PDF document has to cooperate with a PDF parser in order to
    dynamically import the data as processing goes.

    Typical usage:
      doc = PDFDocument(parser, password)
      obj = doc.getobj(objid)

    """

    security_handler_registry: Dict[int, Type[PDFStandardSecurityHandler]] = {
        1: PDFStandardSecurityHandler,
        2: PDFStandardSecurityHandler,
        4: PDFStandardSecurityHandlerV4,
        5: PDFStandardSecurityHandlerV5,
    }

    def __init__(
        self,
        parser: PDFParser,
        password: str = '',
        caching: bool = True,
        fallback: bool = True
    ) -> None:
        "Set the document to use a given PDFParser object."
        self.caching = caching
        self.xrefs: List[PDFBaseXRef] = []
        self.info = []
        self.catalog: Dict[str, Any] = {}
        self.encryption: Optional[Tuple[Any, Any]] = None
        self.decipher: Optional[DecipherCallable] = None
        self._parser = None
        self._cached_objs: Dict[int, Tuple[object, int]] = {}
        self._parsed_objs: Dict[int, Tuple[List[object], int]] = {}
        self._parser = parser
        self._parser.set_document(self)
        self.is_printable = self.is_modifiable = self.is_extractable = True
        # Retrieve the information of each header that was appended
        # (maybe multiple times) at the end of the document.
        try:
            pos = self.find_xref(parser)
            self.read_xref_from(parser, pos, self.xrefs)
        except PDFNoValidXRef:
            pass  # fallback = True
        if fallback:
            parser.fallback = True
            newxref = PDFXRefFallback()
            newxref.load(parser)
            self.xrefs.append(newxref)
        for xref in self.xrefs:
            trailer = xref.get_trailer()
            if not trailer:
                continue
            # If there's an encryption info, remember it.
            if 'Encrypt' in trailer:
                if 'ID' in trailer:
                    id_value = list_value(trailer['ID'])
                else:
                    # Some documents may not have a /ID, use two empty
                    # byte strings instead. Solves
                    # https://github.com/pdfminer/pdfminer.six/issues/594
                    id_value = (b'', b'')
                self.encryption = (id_value,
                                   dict_value(trailer['Encrypt']))
                self._initialize_password(password)
            if 'Info' in trailer:
                self.info.append(dict_value(trailer['Info']))
            if 'Root' in trailer:
                # Every PDF file must have exactly one /Root dictionary.
                self.catalog = dict_value(trailer['Root'])
                break
        else:
            raise PDFSyntaxError('No /Root object! - Is this really a PDF?')
        if self.catalog.get('Type') is not LITERAL_CATALOG:
            if settings.STRICT:
                raise PDFSyntaxError('Catalog not found!')
        return

    KEYWORD_OBJ = KWD(b'obj')

    # _initialize_password(password=b'')
    #   Perform the initialization with a given password.
    def _initialize_password(self, password: str = '') -> None:
        assert self.encryption is not None
        (docid, param) = self.encryption
        if literal_name(param.get('Filter')) != 'Standard':
            raise PDFEncryptionError('Unknown filter: param=%r' % param)
        v = int_value(param.get('V', 0))
        factory = self.security_handler_registry.get(v)
        if factory is None:
            raise PDFEncryptionError('Unknown algorithm: param=%r' % param)
        handler = factory(docid, param, password)
        self.decipher = handler.decrypt
        self.is_printable = handler.is_printable()
        self.is_modifiable = handler.is_modifiable()
        self.is_extractable = handler.is_extractable()
        assert self._parser is not None
        self._parser.fallback = False  # need to read streams with exact length
        return

    def _getobj_objstm(
        self,
        stream: PDFStream,
        index: int,
        objid: int
    ) -> object:
        if stream.objid in self._parsed_objs:
            (objs, n) = self._parsed_objs[stream.objid]
        else:
            (objs, n) = self._get_objects(stream)
            if self.caching:
                assert stream.objid is not None
                self._parsed_objs[stream.objid] = (objs, n)
        i = n*2+index
        try:
            obj = objs[i]
        except IndexError:
            raise PDFSyntaxError('index too big: %r' % index)
        return obj

    def _get_objects(self, stream: PDFStream) -> Tuple[List[object], int]:
        if stream.get('Type') is not LITERAL_OBJSTM:
            if settings.STRICT:
                raise PDFSyntaxError('Not a stream object: %r' % stream)
        try:
            n = cast(int, stream['N'])
        except KeyError:
            if settings.STRICT:
                raise PDFSyntaxError('N is not defined: %r' % stream)
            n = 0
        parser = PDFStreamParser(stream.get_data())
        parser.set_document(self)
        objs: List[object] = []
        try:
            while 1:
                (_, obj) = parser.nextobject()
                objs.append(obj)
        except PSEOF:
            pass
        return (objs, n)

    def _getobj_parse(self, pos: int, objid: int) -> object:
        assert self._parser is not None
        self._parser.seek(pos)
        (_, objid1) = self._parser.nexttoken()  # objid
        (_, genno) = self._parser.nexttoken()  # genno
        (_, kwd) = self._parser.nexttoken()
        # hack around malformed pdf files
        # copied from https://github.com/jaepil/pdfminer3k/blob/master/
        # pdfminer/pdfparser.py#L399
        # to solve https://github.com/pdfminer/pdfminer.six/issues/56
        # assert objid1 == objid, str((objid1, objid))
        if objid1 != objid:
            x = []
            while kwd is not self.KEYWORD_OBJ:
                (_, kwd) = self._parser.nexttoken()
                x.append(kwd)
            if len(x) >= 2:
                objid1 = x[-2]
        # #### end hack around malformed pdf files
        if objid1 != objid:
            raise PDFSyntaxError('objid mismatch: {!r}={!r}'
                                 .format(objid1, objid))

        if kwd != KWD(b'obj'):
            raise PDFSyntaxError('Invalid object spec: offset=%r' % pos)
        (_, obj) = self._parser.nextobject()
        return obj

    # can raise PDFObjectNotFound
    def getobj(self, objid: int) -> object:
        """Get object from PDF

        :raises PDFException if PDFDocument is not initialized
        :raises PDFObjectNotFound if objid does not exist in PDF
        """
        if not self.xrefs:
            raise PDFException('PDFDocument is not initialized')
        log.debug('getobj: objid=%r', objid)
        if objid in self._cached_objs:
            (obj, genno) = self._cached_objs[objid]
        else:
            for xref in self.xrefs:
                try:
                    (strmid, index, genno) = xref.get_pos(objid)
                except KeyError:
                    continue
                try:
                    if strmid is not None:
                        stream = stream_value(self.getobj(strmid))
                        obj = self._getobj_objstm(stream, index, objid)
                    else:
                        obj = self._getobj_parse(index, objid)
                        if self.decipher:
                            obj = decipher_all(self.decipher, objid, genno,
                                               obj)

                    if isinstance(obj, PDFStream):
                        obj.set_objid(objid, genno)
                    break
                except (PSEOF, PDFSyntaxError):
                    continue
            else:
                raise PDFObjectNotFound(objid)
            log.debug('register: objid=%r: %r', objid, obj)
            if self.caching:
                self._cached_objs[objid] = (obj, genno)
        return obj

    OutlineType = Tuple[Any, Any, Any, Any, Any]

    def get_outlines(self) -> Iterator[OutlineType]:
        if 'Outlines' not in self.catalog:
            raise PDFNoOutlines

        def search(entry: object, level: int
                   ) -> Iterator[PDFDocument.OutlineType]:
            entry = dict_value(entry)
            if 'Title' in entry:
                if 'A' in entry or 'Dest' in entry:
                    title = decode_text(str_value(entry['Title']))
                    dest = entry.get('Dest')
                    action = entry.get('A')
                    se = entry.get('SE')
                    yield (level, title, dest, action, se)
            if 'First' in entry and 'Last' in entry:
                yield from search(entry['First'], level+1)
            if 'Next' in entry:
                yield from search(entry['Next'], level)
            return
        return search(self.catalog['Outlines'], 0)

    def lookup_name(
        self,
        cat: str,
        key: Union[str, bytes]
    ) -> Any:
        try:
            names = dict_value(self.catalog['Names'])
        except (PDFTypeError, KeyError):
            raise KeyError((cat, key))
        # may raise KeyError
        d0 = dict_value(names[cat])

        def lookup(d: Dict[str, Any]) -> Any:
            if 'Limits' in d:
                (k1, k2) = list_value(d['Limits'])
                if key < k1 or k2 < key:
                    return None
            if 'Names' in d:
                objs = list_value(d['Names'])
                names = dict(cast(Iterator[Tuple[Union[str, bytes], Any]],
                                  choplist(2, objs)))
                return names[key]
            if 'Kids' in d:
                for c in list_value(d['Kids']):
                    v = lookup(dict_value(c))
                    if v:
                        return v
            raise KeyError((cat, key))
        return lookup(d0)

    def get_dest(self, name: Union[str, bytes]) -> Any:
        try:
            # PDF-1.2 or later
            obj = self.lookup_name('Dests', name)
        except KeyError:
            # PDF-1.1 or prior
            if 'Dests' not in self.catalog:
                raise PDFDestinationNotFound(name)
            d0 = dict_value(self.catalog['Dests'])
            if name not in d0:
                raise PDFDestinationNotFound(name)
            obj = d0[name]
        return obj

    # find_xref
    def find_xref(self, parser: PDFParser) -> int:
        """Internal function used to locate the first XRef."""
        # search the last xref table by scanning the file backwards.
        prev = None
        for line in parser.revreadlines():
            line = line.strip()
            log.debug('find_xref: %r', line)
            if line == b'startxref':
                break
            if line:
                prev = line
        else:
            raise PDFNoValidXRef('Unexpected EOF')
        log.info('xref found: pos=%r', prev)
        assert prev is not None
        return int(prev)

    # read xref table
    def read_xref_from(
        self,
        parser: PDFParser,
        start: int,
        xrefs: List[PDFBaseXRef]
    ) -> None:
        """Reads XRefs from the given location."""
        parser.seek(start)
        parser.reset()
        try:
            (pos, token) = parser.nexttoken()
        except PSEOF:
            raise PDFNoValidXRef('Unexpected EOF')
        log.info('read_xref_from: start=%d, token=%r', start, token)
        if isinstance(token, int):
            # XRefStream: PDF-1.5
            parser.seek(pos)
            parser.reset()
            xref: PDFBaseXRef = PDFXRefStream()
            xref.load(parser)
        else:
            if token is parser.KEYWORD_XREF:
                parser.nextline()
            xref = PDFXRef()
            xref.load(parser)
        xrefs.append(xref)
        trailer = xref.get_trailer()
        log.info('trailer: %r', trailer)
        if 'XRefStm' in trailer:
            pos = int_value(trailer['XRefStm'])
            self.read_xref_from(parser, pos, xrefs)
        if 'Prev' in trailer:
            # find previous xref
            pos = int_value(trailer['Prev'])
            self.read_xref_from(parser, pos, xrefs)
        return
