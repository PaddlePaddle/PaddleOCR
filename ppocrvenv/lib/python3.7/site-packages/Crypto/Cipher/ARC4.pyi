from typing import Any, Union, Iterable

Buffer = Union[bytes, bytearray, memoryview]

class ARC4Cipher:
    block_size: int
    key_size: int

    def __init__(self, key: Buffer, *args: Any, **kwargs: Any) -> None: ...
    def encrypt(self, plaintext: Buffer) -> bytes: ...
    def decrypt(self, ciphertext: Buffer) -> bytes: ...

def new(key: Buffer, drop : int = ...) -> ARC4Cipher: ...

block_size: int
key_size: Iterable[int]
