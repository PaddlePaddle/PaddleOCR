from typing import Union, Tuple, Optional, overload


Buffer = Union[bytes, bytearray, memoryview]

class Salsa20Cipher:
    nonce: bytes
    block_size: int
    key_size: int

    def __init__(self,
                 key: Buffer,
                 nonce: Buffer) -> None: ...
    @overload
    def encrypt(self, plaintext: Buffer) -> bytes: ...
    @overload
    def encrypt(self, plaintext: Buffer, output: Union[bytearray, memoryview]) -> None: ...
    @overload
    def decrypt(self, plaintext: Buffer) -> bytes: ...
    @overload
    def decrypt(self, plaintext: Buffer, output: Union[bytearray, memoryview]) -> None: ...

def new(key: Buffer, nonce: Optional[Buffer] = ...) -> Salsa20Cipher: ...

block_size: int
key_size: Tuple[int, int]

