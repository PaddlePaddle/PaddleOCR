from typing import Callable, Union, Any, Optional, TypeVar

from Crypto.PublicKey.RSA import RsaKey

Buffer = Union[bytes, bytearray, memoryview]
T = TypeVar('T')

class PKCS115_Cipher:
    def __init__(self,
                 key: RsaKey,
                 randfunc: Callable[[int], bytes]) -> None: ...
    def can_encrypt(self) -> bool: ...
    def can_decrypt(self) -> bool: ...
    def encrypt(self, message: Buffer) -> bytes: ...
    def decrypt(self, ciphertext: Buffer,
                sentinel: T,
                expected_pt_len: Optional[int] = ...) -> Union[bytes, T]: ...

def new(key: RsaKey,
        randfunc: Optional[Callable[[int], bytes]] = ...) -> PKCS115_Cipher: ...
