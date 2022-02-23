from typing import Any, Union
from types import ModuleType

Buffer = Union[bytes, bytearray, memoryview]

class BLAKE2b_Hash(object):
    block_size: int
    digest_size: int
    oid: str

    def __init__(self,
                 data: Buffer,
		 key: Buffer,
		 digest_bytes: bytes,
		 update_after_digest: bool) -> None: ...
    def update(self, data: Buffer) -> BLAKE2b_Hash: ...
    def digest(self) -> bytes: ...
    def hexdigest(self) -> str: ...
    def verify(self, mac_tag: Buffer) -> None: ...
    def hexverify(self, hex_mac_tag: str) -> None: ...
    def new(self,
            data: Buffer = ...,
	    digest_bytes: int = ...,
	    digest_bits: int = ...,
	    key: Buffer = ...,
	    update_after_digest: bool = ...) -> BLAKE2b_Hash: ...

def new(data: Buffer = ...,
	digest_bytes: int = ...,
	digest_bits: int = ...,
	key: Buffer = ...,
	update_after_digest: bool = ...) -> BLAKE2b_Hash: ...
