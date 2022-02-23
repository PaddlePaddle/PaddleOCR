from typing import Optional, Callable

from Crypto.PublicKey.RSA import RsaKey
from Crypto.Signature.pss import PSS_SigScheme


def new(rsa_key: RsaKey, mgfunc: Optional[Callable]=None, saltLen: Optional[int]=None, randfunc: Optional[Callable]=None) -> PSS_SigScheme: ...
