from typing import Optional, Union, Callable, Any, overload
from typing_extensions import Protocol

from Crypto.PublicKey.RSA import RsaKey

class HashLikeClass(Protocol):
    digest_size : int
    def new(self, data: Optional[bytes] = ...) -> Any: ...

class HashLikeModule(Protocol):
    digest_size : int
    @staticmethod
    def new(data: Optional[bytes] = ...) -> Any: ...

HashLike = Union[HashLikeClass, HashLikeModule]

Buffer = Union[bytes, bytearray, memoryview]

class PKCS1OAEP_Cipher:
    def __init__(self,
                 key: RsaKey,
                 hashAlgo: HashLike,
                 mgfunc: Callable[[bytes, int], bytes],
                 label: Buffer,
                 randfunc: Callable[[int], bytes]) -> None: ...
    def can_encrypt(self) -> bool: ...
    def can_decrypt(self) -> bool: ...
    def encrypt(self, message: Buffer) -> bytes: ...
    def decrypt(self, ciphertext: Buffer) -> bytes: ...

def new(key: RsaKey,
        hashAlgo: Optional[HashLike] = ...,
        mgfunc: Optional[Callable[[bytes, int], bytes]] = ...,
        label: Optional[Buffer] = ...,
        randfunc: Optional[Callable[[int], bytes]] = ...) -> PKCS1OAEP_Cipher: ...
