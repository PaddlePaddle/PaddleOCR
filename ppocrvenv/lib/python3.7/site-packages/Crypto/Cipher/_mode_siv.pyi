from types import ModuleType
from typing import Union, Tuple, Dict, Optional, overload

Buffer = Union[bytes, bytearray, memoryview]

__all__ = ['SivMode']

class SivMode(object):
    block_size: int
    nonce: bytes
    
    def __init__(self,
                 factory: ModuleType,
                 key: Buffer,
                 nonce: Buffer,
                 kwargs: Dict) -> None: ...
    
    def update(self, component: Buffer) -> SivMode: ...

    def encrypt(self, plaintext: Buffer) -> bytes: ...
    def decrypt(self, plaintext: Buffer) -> bytes: ...

    def digest(self) -> bytes: ...
    def hexdigest(self) -> str: ...
    def verify(self, received_mac_tag: Buffer) -> None: ...
    def hexverify(self, hex_mac_tag: str) -> None: ...

    @overload
    def encrypt_and_digest(self,
                           plaintext: Buffer) -> Tuple[bytes, bytes]: ...
    @overload
    def encrypt_and_digest(self,
                           plaintext: Buffer,
                           output: Buffer) -> Tuple[None, bytes]: ...
    def decrypt_and_verify(self,
                           ciphertext: Buffer,
                           received_mac_tag: Buffer,
                           output: Optional[Union[bytearray, memoryview]] = ...) -> bytes: ...
