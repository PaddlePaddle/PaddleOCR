from typing import Union, Dict, Tuple

from Crypto.Cipher._mode_ecb import EcbMode
from Crypto.Cipher._mode_cbc import CbcMode
from Crypto.Cipher._mode_cfb import CfbMode
from Crypto.Cipher._mode_ofb import OfbMode
from Crypto.Cipher._mode_ctr import CtrMode
from Crypto.Cipher._mode_openpgp import OpenPgpMode
from Crypto.Cipher._mode_eax import EaxMode

def adjust_key_parity(key_in: bytes) -> bytes: ...

DES3Mode = int

MODE_ECB: DES3Mode
MODE_CBC: DES3Mode
MODE_CFB: DES3Mode
MODE_OFB: DES3Mode
MODE_CTR: DES3Mode
MODE_OPENPGP: DES3Mode
MODE_EAX: DES3Mode

Buffer = Union[bytes, bytearray, memoryview]

def new(key: Buffer,
        mode: DES3Mode,
        iv : Buffer = ...,
        IV : Buffer = ...,
        nonce : Buffer = ...,
        segment_size : int = ...,
        mac_len : int = ...,
        initial_value : Union[int, Buffer] = ...,
        counter : Dict = ...) -> \
        Union[EcbMode, CbcMode, CfbMode, OfbMode, CtrMode, OpenPgpMode]: ...

block_size: int
key_size: Tuple[int, int]
