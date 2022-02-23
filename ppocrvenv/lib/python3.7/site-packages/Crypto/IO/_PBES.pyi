from typing import Dict, Optional, Callable

class PbesError(ValueError):
    ...

class PBES1(object):
    @staticmethod
    def decrypt(data: bytes, passphrase: bytes) -> bytes: ...

class PBES2(object):
    @staticmethod
    def encrypt(data: bytes,
                passphrase: bytes,
		protection: str,
		prot_params: Optional[Dict] = ...,
		randfunc: Optional[Callable[[int],bytes]] = ...) -> bytes: ...

    @staticmethod
    def decrypt(data:bytes, passphrase: bytes) -> bytes: ...
