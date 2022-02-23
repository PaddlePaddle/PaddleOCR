from typing import Union, Iterable

from Crypto.Cipher._mode_ecb import EcbMode

MODE_ECB: int

Buffer = Union[bytes, bytearray, memoryview]

def new(key: Buffer,
        mode: int,
	salt: Buffer,
	cost: int) -> EcbMode: ...

block_size: int
key_size: Iterable[int]
