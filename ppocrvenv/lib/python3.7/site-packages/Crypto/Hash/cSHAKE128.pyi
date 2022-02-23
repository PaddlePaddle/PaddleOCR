from typing import Union, Optional

Buffer = Union[bytes, bytearray, memoryview]

class cSHAKE_XOF(object):
    def __init__(self,
                 data:     Optional[Buffer] = ...,
                 function: Optional[bytes] = ...,
                 custom:   Optional[bytes] = ...) -> None: ...
    def update(self, data: Buffer) -> cSHAKE_XOF: ...
    def read(self, length: int) -> bytes: ...

def new(data:     Optional[Buffer] = ...,
        custom:   Optional[Buffer] = ...) -> cSHAKE_XOF: ...
