from types import ModuleType
from typing import Union

Buffer = Union[bytes, bytearray, memoryview]

class Poly1305_MAC(object):
    block_size: int
    digest_size: int
    oid: str

    def __init__(self,
                 r : int,
                 s : int,
                 data : Buffer) -> None: ...
    def update(self, data: Buffer) -> Poly1305_MAC: ...
    def digest(self) -> bytes: ...
    def hexdigest(self) -> str: ...
    def verify(self, mac_tag: Buffer) -> None: ...
    def hexverify(self, hex_mac_tag: str) -> None: ...

def new(key: Buffer,
        cipher: ModuleType,
        nonce: Buffer = ...,
        data: Buffer = ...) -> Poly1305_MAC: ...
