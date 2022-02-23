# -*- coding: utf-8 -*-
#
#  Cipher/AES.py : AES
#
# ===================================================================
# The contents of this file are dedicated to the public domain.  To
# the extent that dedication to the public domain is not available,
# everyone is granted a worldwide, perpetual, royalty-free,
# non-exclusive license to exercise all rights associated with the
# contents of this file for any purpose whatsoever.
# No rights are reserved.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ===================================================================
"""
Module's constants for the modes of operation supported with AES:

:var MODE_ECB: :ref:`Electronic Code Book (ECB) <ecb_mode>`
:var MODE_CBC: :ref:`Cipher-Block Chaining (CBC) <cbc_mode>`
:var MODE_CFB: :ref:`Cipher FeedBack (CFB) <cfb_mode>`
:var MODE_OFB: :ref:`Output FeedBack (OFB) <ofb_mode>`
:var MODE_CTR: :ref:`CounTer Mode (CTR) <ctr_mode>`
:var MODE_OPENPGP:  :ref:`OpenPGP Mode <openpgp_mode>`
:var MODE_CCM: :ref:`Counter with CBC-MAC (CCM) Mode <ccm_mode>`
:var MODE_EAX: :ref:`EAX Mode <eax_mode>`
:var MODE_GCM: :ref:`Galois Counter Mode (GCM) <gcm_mode>`
:var MODE_SIV: :ref:`Syntethic Initialization Vector (SIV) <siv_mode>`
:var MODE_OCB: :ref:`Offset Code Book (OCB) <ocb_mode>`
"""

import sys

from Crypto.Cipher import _create_cipher
from Crypto.Util._raw_api import (load_pycryptodome_raw_lib,
                                  VoidPointer, SmartPointer,
                                  c_size_t, c_uint8_ptr)

from Crypto.Util import _cpu_features
from Crypto.Random import get_random_bytes


_cproto = """
        int AES_start_operation(const uint8_t key[],
                                size_t key_len,
                                void **pResult);
        int AES_encrypt(const void *state,
                        const uint8_t *in,
                        uint8_t *out,
                        size_t data_len);
        int AES_decrypt(const void *state,
                        const uint8_t *in,
                        uint8_t *out,
                        size_t data_len);
        int AES_stop_operation(void *state);
        """


# Load portable AES
_raw_aes_lib = load_pycryptodome_raw_lib("Crypto.Cipher._raw_aes",
                                         _cproto)

# Try to load AES with AES NI instructions
try:
    _raw_aesni_lib = None
    if _cpu_features.have_aes_ni():
        _raw_aesni_lib = load_pycryptodome_raw_lib("Crypto.Cipher._raw_aesni",
                                                   _cproto.replace("AES",
                                                                   "AESNI"))
# _raw_aesni may not have been compiled in
except OSError:
    pass


def _create_base_cipher(dict_parameters):
    """This method instantiates and returns a handle to a low-level
    base cipher. It will absorb named parameters in the process."""

    use_aesni = dict_parameters.pop("use_aesni", True)

    try:
        key = dict_parameters.pop("key")
    except KeyError:
        raise TypeError("Missing 'key' parameter")

    if len(key) not in key_size:
        raise ValueError("Incorrect AES key length (%d bytes)" % len(key))

    if use_aesni and _raw_aesni_lib:
        start_operation = _raw_aesni_lib.AESNI_start_operation
        stop_operation = _raw_aesni_lib.AESNI_stop_operation
    else:
        start_operation = _raw_aes_lib.AES_start_operation
        stop_operation = _raw_aes_lib.AES_stop_operation

    cipher = VoidPointer()
    result = start_operation(c_uint8_ptr(key),
                             c_size_t(len(key)),
                             cipher.address_of())
    if result:
        raise ValueError("Error %X while instantiating the AES cipher"
                         % result)
    return SmartPointer(cipher.get(), stop_operation)


def _derive_Poly1305_key_pair(key, nonce):
    """Derive a tuple (r, s, nonce) for a Poly1305 MAC.

    If nonce is ``None``, a new 16-byte nonce is generated.
    """

    if len(key) != 32:
        raise ValueError("Poly1305 with AES requires a 32-byte key")

    if nonce is None:
        nonce = get_random_bytes(16)
    elif len(nonce) != 16:
        raise ValueError("Poly1305 with AES requires a 16-byte nonce")

    s = new(key[:16], MODE_ECB).encrypt(nonce)
    return key[16:], s, nonce


def new(key, mode, *args, **kwargs):
    """Create a new AES cipher.

    :param key:
        The secret key to use in the symmetric cipher.

        It must be 16, 24 or 32 bytes long (respectively for *AES-128*,
        *AES-192* or *AES-256*).

        For ``MODE_SIV`` only, it doubles to 32, 48, or 64 bytes.
    :type key: bytes/bytearray/memoryview

    :param mode:
        The chaining mode to use for encryption or decryption.
        If in doubt, use ``MODE_EAX``.
    :type mode: One of the supported ``MODE_*`` constants

    :Keyword Arguments:
        *   **iv** (*bytes*, *bytearray*, *memoryview*) --
            (Only applicable for ``MODE_CBC``, ``MODE_CFB``, ``MODE_OFB``,
            and ``MODE_OPENPGP`` modes).

            The initialization vector to use for encryption or decryption.

            For ``MODE_CBC``, ``MODE_CFB``, and ``MODE_OFB`` it must be 16 bytes long.

            For ``MODE_OPENPGP`` mode only,
            it must be 16 bytes long for encryption
            and 18 bytes for decryption (in the latter case, it is
            actually the *encrypted* IV which was prefixed to the ciphertext).

            If not provided, a random byte string is generated (you must then
            read its value with the :attr:`iv` attribute).

        *   **nonce** (*bytes*, *bytearray*, *memoryview*) --
            (Only applicable for ``MODE_CCM``, ``MODE_EAX``, ``MODE_GCM``,
            ``MODE_SIV``, ``MODE_OCB``, and ``MODE_CTR``).

            A value that must never be reused for any other encryption done
            with this key (except possibly for ``MODE_SIV``, see below).

            For ``MODE_EAX``, ``MODE_GCM`` and ``MODE_SIV`` there are no
            restrictions on its length (recommended: **16** bytes).

            For ``MODE_CCM``, its length must be in the range **[7..13]**.
            Bear in mind that with CCM there is a trade-off between nonce
            length and maximum message size. Recommendation: **11** bytes.

            For ``MODE_OCB``, its length must be in the range **[1..15]**
            (recommended: **15**).

            For ``MODE_CTR``, its length must be in the range **[0..15]**
            (recommended: **8**).

            For ``MODE_SIV``, the nonce is optional, if it is not specified,
            then no nonce is being used, which renders the encryption
            deterministic.

            If not provided, for modes other than ``MODE_SIV```, a random
            byte string of the recommended length is used (you must then
            read its value with the :attr:`nonce` attribute).

        *   **segment_size** (*integer*) --
            (Only ``MODE_CFB``).The number of **bits** the plaintext and ciphertext
            are segmented in. It must be a multiple of 8.
            If not specified, it will be assumed to be 8.

        *   **mac_len** : (*integer*) --
            (Only ``MODE_EAX``, ``MODE_GCM``, ``MODE_OCB``, ``MODE_CCM``)
            Length of the authentication tag, in bytes.

            It must be even and in the range **[4..16]**.
            The recommended value (and the default, if not specified) is **16**.

        *   **msg_len** : (*integer*) --
            (Only ``MODE_CCM``). Length of the message to (de)cipher.
            If not specified, ``encrypt`` must be called with the entire message.
            Similarly, ``decrypt`` can only be called once.

        *   **assoc_len** : (*integer*) --
            (Only ``MODE_CCM``). Length of the associated data.
            If not specified, all associated data is buffered internally,
            which may represent a problem for very large messages.

        *   **initial_value** : (*integer* or *bytes/bytearray/memoryview*) --
            (Only ``MODE_CTR``).
            The initial value for the counter. If not present, the cipher will
            start counting from 0. The value is incremented by one for each block.
            The counter number is encoded in big endian mode.

        *   **counter** : (*object*) --
            Instance of ``Crypto.Util.Counter``, which allows full customization
            of the counter block. This parameter is incompatible to both ``nonce``
            and ``initial_value``.

        *   **use_aesni** : (*boolean*) --
            Use Intel AES-NI hardware extensions (default: use if available).

    :Return: an AES object, of the applicable mode.
    """

    kwargs["add_aes_modes"] = True
    return _create_cipher(sys.modules[__name__], key, mode, *args, **kwargs)


MODE_ECB = 1
MODE_CBC = 2
MODE_CFB = 3
MODE_OFB = 5
MODE_CTR = 6
MODE_OPENPGP = 7
MODE_CCM = 8
MODE_EAX = 9
MODE_SIV = 10
MODE_GCM = 11
MODE_OCB = 12

# Size of a data block (in bytes)
block_size = 16
# Size of a key (in bytes)
key_size = (16, 24, 32)
