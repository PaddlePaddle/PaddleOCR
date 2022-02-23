from typing import Union

from .KMAC128 import KMAC_Hash

Buffer = Union[bytes, bytearray, memoryview]

def new(key: Buffer,
        data: Buffer = ...,
	    mac_len: int = ...,
        custom: Buffer = ...) -> KMAC_Hash: ...
