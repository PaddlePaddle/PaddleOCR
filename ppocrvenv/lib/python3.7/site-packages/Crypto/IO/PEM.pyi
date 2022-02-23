from typing import Tuple, Optional, Callable

def encode(data: bytes,
           marke: str,
	   passphrase: Optional[bytes] = ...,
	   randfunc: Optional[Callable[[int],bytes]] = ...) -> str: ...


def decode(pem_data: str,
           passphrase: Optional[bytes] = ...) -> Tuple[bytes, str, bool]: ...
