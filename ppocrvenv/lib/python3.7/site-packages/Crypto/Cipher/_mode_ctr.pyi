from typing import Union, overload

from Crypto.Util._raw_api import SmartPointer

Buffer = Union[bytes, bytearray, memoryview]

__all__ = ['CtrMode']

class CtrMode(object):
    block_size: int
    nonce: bytes

    def __init__(self,
                 block_cipher: SmartPointer,
                 initial_counter_block: Buffer,
                 prefix_len: int,
                 counter_len: int,
                 little_endian: bool) -> None: ...
    @overload
    def encrypt(self, plaintext: Buffer) -> bytes: ...
    @overload
    def encrypt(self, plaintext: Buffer, output: Union[bytearray, memoryview]) -> None: ...
    @overload
    def decrypt(self, plaintext: Buffer) -> bytes: ...
    @overload
    def decrypt(self, plaintext: Buffer, output: Union[bytearray, memoryview]) -> None: ...

