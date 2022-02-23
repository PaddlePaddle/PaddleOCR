from types import ModuleType
from typing import Union, Dict

Buffer = Union[bytes, bytearray, memoryview]

__all__ = ['OpenPgpMode']

class OpenPgpMode(object):
    block_size: int
    iv: Union[bytes, bytearray, memoryview]
    IV: Union[bytes, bytearray, memoryview]
    
    def __init__(self,
                 factory: ModuleType,
                 key: Buffer,
                 iv: Buffer,
                 cipher_params: Dict) -> None: ...
    def encrypt(self, plaintext: Buffer) -> bytes: ...
    def decrypt(self, plaintext: Buffer) -> bytes: ...

