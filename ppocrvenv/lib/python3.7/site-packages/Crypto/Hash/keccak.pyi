from typing import Union, Any

Buffer = Union[bytes, bytearray, memoryview]

class Keccak_Hash(object):
    digest_size: int
    def __init__(self,
                 data: Buffer,
                 digest_bytes: int,
                 update_after_digest: bool) -> None: ...
    def update(self, data: Buffer) -> Keccak_Hash: ...
    def digest(self) -> bytes: ...
    def hexdigest(self) -> str: ...
    def new(self,
            data: Buffer = ...,
            digest_bytes: int = ...,
            digest_bits: int = ...,
            update_after_digest: bool = ...) -> Keccak_Hash: ...

def new(data: Buffer = ...,
        digest_bytes: int = ...,
        digest_bits: int = ...,
        update_after_digest: bool = ...) -> Keccak_Hash: ...
