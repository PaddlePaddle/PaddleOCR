from io import BytesIO
import logging
from typing import BinaryIO, Iterator, List, Optional, cast


logger = logging.getLogger(__name__)


class CorruptDataError(Exception):
    pass


class LZWDecoder:

    def __init__(self, fp: BinaryIO) -> None:
        self.fp = fp
        self.buff = 0
        self.bpos = 8
        self.nbits = 9
        # NB: self.table stores None only in indices 256 and 257
        self.table: Optional[List[Optional[bytes]]] = None
        self.prevbuf: Optional[bytes] = None
        return

    def readbits(self, bits: int) -> int:
        v = 0
        while 1:
            # the number of remaining bits we can get from the current buffer.
            r = 8-self.bpos
            if bits <= r:
                # |-----8-bits-----|
                # |-bpos-|-bits-|  |
                # |      |----r----|
                v = (v << bits) | ((self.buff >> (r-bits)) & ((1 << bits)-1))
                self.bpos += bits
                break
            else:
                # |-----8-bits-----|
                # |-bpos-|---bits----...
                # |      |----r----|
                v = (v << r) | (self.buff & ((1 << r)-1))
                bits -= r
                x = self.fp.read(1)
                if not x:
                    raise EOFError
                self.buff = ord(x)
                self.bpos = 0
        return v

    def feed(self, code: int) -> bytes:
        x = b''
        if code == 256:
            self.table = [bytes((c,)) for c in range(256)]  # 0-255
            self.table.append(None)  # 256
            self.table.append(None)  # 257
            self.prevbuf = b''
            self.nbits = 9
        elif code == 257:
            pass
        elif not self.prevbuf:
            assert self.table is not None
            x = self.prevbuf = cast(bytes, self.table[code])  # assume not None
        else:
            assert self.table is not None
            if code < len(self.table):
                x = cast(bytes, self.table[code])  # assume not None
                self.table.append(self.prevbuf+x[:1])
            elif code == len(self.table):
                self.table.append(self.prevbuf+self.prevbuf[:1])
                x = cast(bytes, self.table[code])
            else:
                raise CorruptDataError
            table_length = len(self.table)
            if table_length == 511:
                self.nbits = 10
            elif table_length == 1023:
                self.nbits = 11
            elif table_length == 2047:
                self.nbits = 12
            self.prevbuf = x
        return x

    def run(self) -> Iterator[bytes]:
        while 1:
            try:
                code = self.readbits(self.nbits)
            except EOFError:
                break
            try:
                x = self.feed(code)
            except CorruptDataError:
                # just ignore corrupt data and stop yielding there
                break
            yield x
            assert self.table is not None
            logger.debug('nbits=%d, code=%d, output=%r, table=%r'
                         % (self.nbits, code, x, self.table[258:]))
        return


def lzwdecode(data: bytes) -> bytes:
    fp = BytesIO(data)
    s = LZWDecoder(fp).run()
    return b''.join(s)
