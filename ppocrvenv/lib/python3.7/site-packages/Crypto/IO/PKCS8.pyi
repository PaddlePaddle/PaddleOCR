from typing import Dict, Tuple, Optional, Union, Callable

from Crypto.Util.asn1 import DerObject

def wrap(private_key: bytes,
         key_oid: str,
	 passphrase: Union[bytes, str] = ...,
	 protection: str = ...,
         prot_params: Dict = ...,
	 key_params: DerObject = ...,
	 randfunc: Optional[Callable[[int],str]]  = ...) -> bytes: ...


def unwrap(p8_private_key: bytes, passphrase: Optional[Union[bytes, str]] = ...) -> Tuple[str, bytes, Optional[bytes]]: ...
