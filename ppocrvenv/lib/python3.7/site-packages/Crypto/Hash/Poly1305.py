# -*- coding: utf-8 -*-
#
# Hash/Poly1305.py - Implements the Poly1305 MAC
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

from binascii import unhexlify

from Crypto.Util.py3compat import bord, tobytes, _copy_bytes

from Crypto.Hash import BLAKE2s
from Crypto.Random import get_random_bytes
from Crypto.Util._raw_api import (load_pycryptodome_raw_lib,
                                  VoidPointer, SmartPointer,
                                  create_string_buffer,
                                  get_raw_buffer, c_size_t,
                                  c_uint8_ptr)


_raw_poly1305 = load_pycryptodome_raw_lib("Crypto.Hash._poly1305",
                        """
                        int poly1305_init(void **state,
                                          const uint8_t *r,
                                          size_t r_len,
                                          const uint8_t *s,
                                          size_t s_len);
                        int poly1305_destroy(void *state);
                        int poly1305_update(void *state,
                                            const uint8_t *in,
                                            size_t len);
                        int poly1305_digest(const void *state,
                                            uint8_t *digest,
                                            size_t len);
                        """)


class Poly1305_MAC(object):
    """An Poly1305 MAC object.
    Do not instantiate directly. Use the :func:`new` function.

    :ivar digest_size: the size in bytes of the resulting MAC tag
    :vartype digest_size: integer
    """

    digest_size = 16

    def __init__(self, r, s, data):

        if len(r) != 16:
            raise ValueError("Parameter r is not 16 bytes long")
        if len(s) != 16:
            raise ValueError("Parameter s is not 16 bytes long")

        self._mac_tag = None

        state = VoidPointer()
        result = _raw_poly1305.poly1305_init(state.address_of(),
                                             c_uint8_ptr(r),
                                             c_size_t(len(r)),
                                             c_uint8_ptr(s),
                                             c_size_t(len(s))
                                             )
        if result:
            raise ValueError("Error %d while instantiating Poly1305" % result)
        self._state = SmartPointer(state.get(),
                                   _raw_poly1305.poly1305_destroy)
        if data:
            self.update(data)

    def update(self, data):
        """Authenticate the next chunk of message.

        Args:
            data (byte string/byte array/memoryview): The next chunk of data
        """

        if self._mac_tag:
            raise TypeError("You can only call 'digest' or 'hexdigest' on this object")

        result = _raw_poly1305.poly1305_update(self._state.get(),
                                               c_uint8_ptr(data),
                                               c_size_t(len(data)))
        if result:
            raise ValueError("Error %d while hashing Poly1305 data" % result)
        return self

    def copy(self):
        raise NotImplementedError()

    def digest(self):
        """Return the **binary** (non-printable) MAC tag of the message
        authenticated so far.

        :return: The MAC tag digest, computed over the data processed so far.
                 Binary form.
        :rtype: byte string
        """

        if self._mac_tag:
            return self._mac_tag
        
        bfr = create_string_buffer(16)
        result = _raw_poly1305.poly1305_digest(self._state.get(),
                                               bfr,
                                               c_size_t(len(bfr)))
        if result:
            raise ValueError("Error %d while creating Poly1305 digest" % result)

        self._mac_tag = get_raw_buffer(bfr)
        return self._mac_tag

    def hexdigest(self):
        """Return the **printable** MAC tag of the message authenticated so far.

        :return: The MAC tag, computed over the data processed so far.
                 Hexadecimal encoded.
        :rtype: string
        """

        return "".join(["%02x" % bord(x)
                        for x in tuple(self.digest())])

    def verify(self, mac_tag):
        """Verify that a given **binary** MAC (computed by another party)
        is valid.

        Args:
          mac_tag (byte string/byte string/memoryview): the expected MAC of the message.

        Raises:
            ValueError: if the MAC does not match. It means that the message
                has been tampered with or that the MAC key is incorrect.
        """

        secret = get_random_bytes(16)

        mac1 = BLAKE2s.new(digest_bits=160, key=secret, data=mac_tag)
        mac2 = BLAKE2s.new(digest_bits=160, key=secret, data=self.digest())

        if mac1.digest() != mac2.digest():
            raise ValueError("MAC check failed")

    def hexverify(self, hex_mac_tag):
        """Verify that a given **printable** MAC (computed by another party)
        is valid.

        Args:
            hex_mac_tag (string): the expected MAC of the message,
                as a hexadecimal string.

        Raises:
            ValueError: if the MAC does not match. It means that the message
                has been tampered with or that the MAC key is incorrect.
        """

        self.verify(unhexlify(tobytes(hex_mac_tag)))



def new(**kwargs):
    """Create a new Poly1305 MAC object.

    Args:
        key (bytes/bytearray/memoryview):
            The 32-byte key for the Poly1305 object.
        cipher (module from ``Crypto.Cipher``):
            The cipher algorithm to use for deriving the Poly1305
            key pair *(r, s)*.
            It can only be ``Crypto.Cipher.AES`` or ``Crypto.Cipher.ChaCha20``.
        nonce (bytes/bytearray/memoryview):
            Optional. The non-repeatable value to use for the MAC of this message.
            It must be 16 bytes long for ``AES`` and 8 or 12 bytes for ``ChaCha20``.
            If not passed, a random nonce is created; you will find it in the
            ``nonce`` attribute of the new object.
        data (bytes/bytearray/memoryview):
            Optional. The very first chunk of the message to authenticate.
            It is equivalent to an early call to ``update()``.

    Returns:
        A :class:`Poly1305_MAC` object
    """

    cipher = kwargs.pop("cipher", None)
    if not hasattr(cipher, '_derive_Poly1305_key_pair'):
        raise ValueError("Parameter 'cipher' must be AES or ChaCha20")

    cipher_key = kwargs.pop("key", None)
    if cipher_key is None:
        raise TypeError("You must pass a parameter 'key'")

    nonce = kwargs.pop("nonce", None)
    data = kwargs.pop("data", None)
    
    if kwargs:
        raise TypeError("Unknown parameters: " + str(kwargs))

    r, s, nonce = cipher._derive_Poly1305_key_pair(cipher_key, nonce)
    
    new_mac = Poly1305_MAC(r, s, data)
    new_mac.nonce = _copy_bytes(None, None, nonce)  # nonce may still be just a memoryview
    return new_mac
