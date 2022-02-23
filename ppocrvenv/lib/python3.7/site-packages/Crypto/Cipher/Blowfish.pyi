from typing import Union, Dict, Iterable

from Crypto.Cipher._mode_ecb import EcbMode
from Crypto.Cipher._mode_cbc import CbcMode
from Crypto.Cipher._mode_cfb import CfbMode
from Crypto.Cipher._mode_ofb import OfbMode
from Crypto.Cipher._mode_ctr import CtrMode
from Crypto.Cipher._mode_openpgp import OpenPgpMode
from Crypto.Cipher._mode_eax import EaxMode

BlowfishMode = int

MODE_ECB: BlowfishMode
MODE_CBC: BlowfishMode
MODE_CFB: BlowfishMode
MODE_OFB: BlowfishMode
MODE_CTR: BlowfishMode
MODE_OPENPGP: BlowfishMode
MODE_EAX: BlowfishMode

Buffer = Union[bytes, bytearray, memoryview]

def new(key: Buffer,
        mode: BlowfishMode,
        iv : Buffer = ...,
        IV : Buffer = ...,
        nonce : Buffer = ...,
        segment_size : int = ...,
        mac_len : int = ...,
        initial_value : Union[int, Buffer] = ...,
        counter : Dict = ...) -> \
        Union[EcbMode, CbcMode, CfbMode, OfbMode, CtrMode, OpenPgpMode]: ...

block_size: int
key_size: Iterable[int]
