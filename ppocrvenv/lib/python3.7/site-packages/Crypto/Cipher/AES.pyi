from typing import Union, Tuple, Optional, Dict

from Crypto.Cipher._mode_ecb import EcbMode
from Crypto.Cipher._mode_cbc import CbcMode
from Crypto.Cipher._mode_cfb import CfbMode
from Crypto.Cipher._mode_ofb import OfbMode
from Crypto.Cipher._mode_ctr import CtrMode
from Crypto.Cipher._mode_openpgp import OpenPgpMode
from Crypto.Cipher._mode_ccm import CcmMode
from Crypto.Cipher._mode_eax import EaxMode
from Crypto.Cipher._mode_gcm import GcmMode
from Crypto.Cipher._mode_siv import SivMode
from Crypto.Cipher._mode_ocb import OcbMode

AESMode = int

MODE_ECB: AESMode
MODE_CBC: AESMode
MODE_CFB: AESMode
MODE_OFB: AESMode
MODE_CTR: AESMode
MODE_OPENPGP: AESMode
MODE_CCM: AESMode
MODE_EAX: AESMode
MODE_GCM: AESMode
MODE_SIV: AESMode
MODE_OCB: AESMode

Buffer = Union[bytes, bytearray, memoryview]

def new(key: Buffer,
        mode: AESMode,
        iv : Buffer = ...,
        IV : Buffer = ...,
        nonce : Buffer = ...,
        segment_size : int = ...,
        mac_len : int = ...,
        assoc_len : int = ...,
        initial_value : Union[int, Buffer] = ...,
        counter : Dict = ...,
        use_aesni : bool = ...) -> \
        Union[EcbMode, CbcMode, CfbMode, OfbMode, CtrMode,
              OpenPgpMode, CcmMode, EaxMode, GcmMode,
              SivMode, OcbMode]: ...

block_size: int
key_size: Tuple[int, int, int]
