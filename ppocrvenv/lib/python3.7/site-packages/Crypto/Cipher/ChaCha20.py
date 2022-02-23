# ===================================================================
#
# Copyright (c) 2014, Legrandin <helderijs@gmail.com>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ===================================================================

from Crypto.Random import get_random_bytes

from Crypto.Util.py3compat import _copy_bytes
from Crypto.Util._raw_api import (load_pycryptodome_raw_lib,
                                  create_string_buffer,
                                  get_raw_buffer, VoidPointer,
                                  SmartPointer, c_size_t,
                                  c_uint8_ptr, c_ulong,
                                  is_writeable_buffer)

_raw_chacha20_lib = load_pycryptodome_raw_lib("Crypto.Cipher._chacha20",
                    """
                    int chacha20_init(void **pState,
                                      const uint8_t *key,
                                      size_t keySize,
                                      const uint8_t *nonce,
                                      size_t nonceSize);

                    int chacha20_destroy(void *state);

                    int chacha20_encrypt(void *state,
                                         const uint8_t in[],
                                         uint8_t out[],
                                         size_t len);

                    int chacha20_seek(void *state,
                                      unsigned long block_high,
                                      unsigned long block_low,
                                      unsigned offset);
                    int hchacha20(  const uint8_t key[32],
                                    const uint8_t nonce16[16],
                                    uint8_t subkey[32]);
                    """)


def _HChaCha20(key, nonce):

    assert(len(key) == 32)
    assert(len(nonce) == 16)

    subkey = bytearray(32)
    result = _raw_chacha20_lib.hchacha20(
                c_uint8_ptr(key),
                c_uint8_ptr(nonce),
                c_uint8_ptr(subkey))
    if result:
        raise ValueError("Error %d when deriving subkey with HChaCha20" % result)

    return subkey


class ChaCha20Cipher(object):
    """ChaCha20 (or XChaCha20) cipher object.
    Do not create it directly. Use :py:func:`new` instead.

    :var nonce: The nonce with length 8, 12 or 24 bytes
    :vartype nonce: bytes
    """

    block_size = 1

    def __init__(self, key, nonce):
        """Initialize a ChaCha20/XChaCha20 cipher object

        See also `new()` at the module level."""

        self.nonce = _copy_bytes(None, None, nonce)

        # XChaCha20 requires a key derivation with HChaCha20
        # See 2.3 in https://tools.ietf.org/html/draft-arciszewski-xchacha-03
        if len(nonce) == 24:
            key = _HChaCha20(key, nonce[:16])
            nonce = b'\x00' * 4 + nonce[16:]
            self._name = "XChaCha20"
        else:
            self._name = "ChaCha20"
            nonce = self.nonce

        self._next = ( self.encrypt, self.decrypt )

        self._state = VoidPointer()
        result = _raw_chacha20_lib.chacha20_init(
                        self._state.address_of(),
                        c_uint8_ptr(key),
                        c_size_t(len(key)),
                        nonce,
                        c_size_t(len(nonce)))
        if result:
            raise ValueError("Error %d instantiating a %s cipher" % (result,
                                                                     self._name))
        self._state = SmartPointer(self._state.get(),
                                   _raw_chacha20_lib.chacha20_destroy)

    def encrypt(self, plaintext, output=None):
        """Encrypt a piece of data.

        Args:
          plaintext(bytes/bytearray/memoryview): The data to encrypt, of any size.
        Keyword Args:
          output(bytes/bytearray/memoryview): The location where the ciphertext
            is written to. If ``None``, the ciphertext is returned.
        Returns:
          If ``output`` is ``None``, the ciphertext is returned as ``bytes``.
          Otherwise, ``None``.
        """

        if self.encrypt not in self._next:
            raise TypeError("Cipher object can only be used for decryption")
        self._next = ( self.encrypt, )
        return self._encrypt(plaintext, output)

    def _encrypt(self, plaintext, output):
        """Encrypt without FSM checks"""

        if output is None:
            ciphertext = create_string_buffer(len(plaintext))
        else:
            ciphertext = output

            if not is_writeable_buffer(output):
                raise TypeError("output must be a bytearray or a writeable memoryview")

            if len(plaintext) != len(output):
                raise ValueError("output must have the same length as the input"
                                 "  (%d bytes)" % len(plaintext))

        result = _raw_chacha20_lib.chacha20_encrypt(
                                         self._state.get(),
                                         c_uint8_ptr(plaintext),
                                         c_uint8_ptr(ciphertext),
                                         c_size_t(len(plaintext)))
        if result:
            raise ValueError("Error %d while encrypting with %s" % (result, self._name))

        if output is None:
            return get_raw_buffer(ciphertext)
        else:
            return None

    def decrypt(self, ciphertext, output=None):
        """Decrypt a piece of data.

        Args:
          ciphertext(bytes/bytearray/memoryview): The data to decrypt, of any size.
        Keyword Args:
          output(bytes/bytearray/memoryview): The location where the plaintext
            is written to. If ``None``, the plaintext is returned.
        Returns:
          If ``output`` is ``None``, the plaintext is returned as ``bytes``.
          Otherwise, ``None``.
        """

        if self.decrypt not in self._next:
            raise TypeError("Cipher object can only be used for encryption")
        self._next = ( self.decrypt, )

        try:
            return self._encrypt(ciphertext, output)
        except ValueError as e:
            raise ValueError(str(e).replace("enc", "dec"))

    def seek(self, position):
        """Seek to a certain position in the key stream.

        Args:
          position (integer):
            The absolute position within the key stream, in bytes.
        """

        position, offset = divmod(position, 64)
        block_low = position & 0xFFFFFFFF
        block_high = position >> 32

        result = _raw_chacha20_lib.chacha20_seek(
                                                 self._state.get(),
                                                 c_ulong(block_high),
                                                 c_ulong(block_low),
                                                 offset
                                                 )
        if result:
            raise ValueError("Error %d while seeking with %s" % (result, self._name))


def _derive_Poly1305_key_pair(key, nonce):
    """Derive a tuple (r, s, nonce) for a Poly1305 MAC.

    If nonce is ``None``, a new 12-byte nonce is generated.
    """

    if len(key) != 32:
        raise ValueError("Poly1305 with ChaCha20 requires a 32-byte key")

    if nonce is None:
        padded_nonce = nonce = get_random_bytes(12)
    elif len(nonce) == 8:
        # See RFC7538, 2.6: [...] ChaCha20 as specified here requires a 96-bit
        # nonce.  So if the provided nonce is only 64-bit, then the first 32
        # bits of the nonce will be set to a constant number.
        # This will usually be zero, but for protocols with multiple senders it may be
        # different for each sender, but should be the same for all
        # invocations of the function with the same key by a particular
        # sender.
        padded_nonce = b'\x00\x00\x00\x00' + nonce
    elif len(nonce) == 12:
        padded_nonce = nonce
    else:
        raise ValueError("Poly1305 with ChaCha20 requires an 8- or 12-byte nonce")

    rs = new(key=key, nonce=padded_nonce).encrypt(b'\x00' * 32)
    return rs[:16], rs[16:], nonce


def new(**kwargs):
    """Create a new ChaCha20 or XChaCha20 cipher

    Keyword Args:
        key (bytes/bytearray/memoryview): The secret key to use.
            It must be 32 bytes long.
        nonce (bytes/bytearray/memoryview): A mandatory value that
            must never be reused for any other encryption
            done with this key.

            For ChaCha20, it must be 8 or 12 bytes long.

            For XChaCha20, it must be 24 bytes long.

            If not provided, 8 bytes will be randomly generated
            (you can find them back in the ``nonce`` attribute).

    :Return: a :class:`Crypto.Cipher.ChaCha20.ChaCha20Cipher` object
    """

    try:
        key = kwargs.pop("key")
    except KeyError as e:
        raise TypeError("Missing parameter %s" % e)

    nonce = kwargs.pop("nonce", None)
    if nonce is None:
        nonce = get_random_bytes(8)

    if len(key) != 32:
        raise ValueError("ChaCha20/XChaCha20 key must be 32 bytes long")

    if len(nonce) not in (8, 12, 24):
        raise ValueError("Nonce must be 8/12 bytes(ChaCha20) or 24 bytes (XChaCha20)")

    if kwargs:
        raise TypeError("Unknown parameters: " + str(kwargs))

    return ChaCha20Cipher(key, nonce)

# Size of a data block (in bytes)
block_size = 1

# Size of a key (in bytes)
key_size = 32
