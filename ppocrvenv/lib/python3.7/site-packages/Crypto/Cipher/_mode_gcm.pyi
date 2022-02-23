from types import ModuleType
from typing import Union, Tuple, Dict, overload, Optional

__all__ = ['GcmMode']

Buffer = Union[bytes, bytearray, memoryview]

class GcmMode(object):
    block_size: int
    nonce: Buffer
    
    def __init__(self,
                 factory: ModuleType,
                 key: Buffer,
                 nonce: Buffer,
                 mac_len: int,
                 cipher_params: Dict) -> None: ...
    
    def update(self, assoc_data: Buffer) -> GcmMode: ...

    @overload
    def encrypt(self, plaintext: Buffer) -> bytes: ...
    @overload
    def encrypt(self, plaintext: Buffer, output: Union[bytearray, memoryview]) -> None: ...
    @overload
    def decrypt(self, plaintext: Buffer) -> bytes: ...
    @overload
    def decrypt(self, plaintext: Buffer, output: Union[bytearray, memoryview]) -> None: ...

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
