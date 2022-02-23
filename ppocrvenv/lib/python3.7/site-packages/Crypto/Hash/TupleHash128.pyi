from typing import Any, Union
from types import ModuleType

Buffer = Union[bytes, bytearray, memoryview]

class TupleHash(object):
    digest_size: int
    def __init__(self,
		         custom: bytes,
                 cshake: ModuleType,
                 digest_size: int) -> None: ...
    def update(self, data: Buffer) -> TupleHash: ...
    def digest(self) -> bytes: ...
    def hexdigest(self) -> str: ...
    def new(self,
	        digest_bytes: int = ...,
	        digest_bits: int = ...,
            custom: int = ...) -> TupleHash: ...

def new(digest_bytes: int = ...,
	    digest_bits: int = ...,
        custom: int = ...) -> TupleHash: ...
