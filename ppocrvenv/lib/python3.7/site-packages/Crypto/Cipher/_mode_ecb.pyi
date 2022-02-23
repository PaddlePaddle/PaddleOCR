from typing import Union, overload

from Crypto.Util._raw_api import SmartPointer

Buffer = Union[bytes, bytearray, memoryview]

__all__ = [ 'EcbMode' ]

class EcbMode(object):
    def __init__(self, block_cipher: SmartPointer) -> None: ...
    @overload
    def encrypt(self, plaintext: Buffer) -> bytes: ...
    @overload
    def encrypt(self, plaintext: Buffer, output: Union[bytearray, memoryview]) -> None: ...
    @overload
    def decrypt(self, plaintext: Buffer) -> bytes: ...
    @overload
    def decrypt(self, plaintext: Buffer, output: Union[bytearray, memoryview]) -> None: ...

