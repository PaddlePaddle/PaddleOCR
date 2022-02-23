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

"""
Counter with CBC-MAC (CCM) mode.
"""

__all__ = ['CcmMode']

import struct
from binascii import unhexlify

from Crypto.Util.py3compat import (byte_string, bord,
                                   _copy_bytes)
from Crypto.Util._raw_api import is_writeable_buffer

from Crypto.Util.strxor import strxor
from Crypto.Util.number import long_to_bytes

from Crypto.Hash import BLAKE2s
from Crypto.Random import get_random_bytes


def enum(**enums):
    return type('Enum', (), enums)

MacStatus = enum(NOT_STARTED=0, PROCESSING_AUTH_DATA=1, PROCESSING_PLAINTEXT=2)


class CcmMode(object):
    """Counter with CBC-MAC (CCM).

    This is an Authenticated Encryption with Associated Data (`AEAD`_) mode.
    It provides both confidentiality and authenticity.

    The header of the message may be left in the clear, if needed, and it will
    still be subject to authentication. The decryption step tells the receiver
    if the message comes from a source that really knowns the secret key.
    Additionally, decryption detects if any part of the message - including the
    header - has been modified or corrupted.

    This mode requires a nonce. The nonce shall never repeat for two
    different messages encrypted with the same key, but it does not need
    to be random.
    Note that there is a trade-off between the size of the nonce and the
    maximum size of a single message you can encrypt.

    It is important to use a large nonce if the key is reused across several
    messages and the nonce is chosen randomly.

    It is acceptable to us a short nonce if the key is only used a few times or
    if the nonce is taken from a counter.

    The following table shows the trade-off when the nonce is chosen at
    random. The column on the left shows how many messages it takes
    for the keystream to repeat **on average**. In practice, you will want to
    stop using the key way before that.

    +--------------------+---------------+-------------------+
    | Avg. # of messages |    nonce      |     Max. message  |
    | before keystream   |    size       |     size          |
    | repeats            |    (bytes)    |     (bytes)       |
    +====================+===============+===================+
    |       2^52         |      13       |        64K        |
    +--------------------+---------------+-------------------+
    |       2^48         |      12       |        16M        |
    +--------------------+---------------+-------------------+
    |       2^44         |      11       |         4G        |
    +--------------------+---------------+-------------------+
    |       2^40         |      10       |         1T        |
    +--------------------+---------------+-------------------+
    |       2^36         |       9       |        64P        |
    +--------------------+---------------+-------------------+
    |       2^32         |       8       |        16E        |
    +--------------------+---------------+-------------------+

    This mode is only available for ciphers that operate on 128 bits blocks
    (e.g. AES but not TDES).

    See `NIST SP800-38C`_ or RFC3610_.

    .. _`NIST SP800-38C`: http://csrc.nist.gov/publications/nistpubs/800-38C/SP800-38C.pdf
    .. _RFC3610: https://tools.ietf.org/html/rfc3610
    .. _AEAD: http://blog.cryptographyengineering.com/2012/05/how-to-choose-authenticated-encryption.html

    :undocumented: __init__
    """

    def __init__(self, factory, key, nonce, mac_len, msg_len, assoc_len,
                 cipher_params):

        self.block_size = factory.block_size
        """The block size of the underlying cipher, in bytes."""

        self.nonce = _copy_bytes(None, None, nonce)
        """The nonce used for this cipher instance"""

        self._factory = factory
        self._key = _copy_bytes(None, None, key)
        self._mac_len = mac_len
        self._msg_len = msg_len
        self._assoc_len = assoc_len
        self._cipher_params = cipher_params

        self._mac_tag = None  # Cache for MAC tag

        if self.block_size != 16:
            raise ValueError("CCM mode is only available for ciphers"
                             " that operate on 128 bits blocks")

        # MAC tag length (Tlen)
        if mac_len not in (4, 6, 8, 10, 12, 14, 16):
            raise ValueError("Parameter 'mac_len' must be even"
                             " and in the range 4..16 (not %d)" % mac_len)

        # Nonce value
        if not (nonce and 7 <= len(nonce) <= 13):
            raise ValueError("Length of parameter 'nonce' must be"
                             " in the range 7..13 bytes")

        # Create MAC object (the tag will be the last block
        # bytes worth of ciphertext)
        self._mac = self._factory.new(key,
                                      factory.MODE_CBC,
                                      iv=b'\x00' * 16,
                                      **cipher_params)
        self._mac_status = MacStatus.NOT_STARTED
        self._t = None

        # Allowed transitions after initialization
        self._next = [self.update, self.encrypt, self.decrypt,
                      self.digest, self.verify]

        # Cumulative lengths
        self._cumul_assoc_len = 0
        self._cumul_msg_len = 0

        # Cache for unaligned associated data/plaintext.
        # This is a list with byte strings, but when the MAC starts,
        # it will become a binary string no longer than the block size.
        self._cache = []

        # Start CTR cipher, by formatting the counter (A.3)
        q = 15 - len(nonce)  # length of Q, the encoded message length
        self._cipher = self._factory.new(key,
                                         self._factory.MODE_CTR,
                                         nonce=struct.pack("B", q - 1) + self.nonce,
                                         **cipher_params)

        # S_0, step 6 in 6.1 for j=0
        self._s_0 = self._cipher.encrypt(b'\x00' * 16)

        # Try to start the MAC
        if None not in (assoc_len, msg_len):
            self._start_mac()

    def _start_mac(self):

        assert(self._mac_status == MacStatus.NOT_STARTED)
        assert(None not in (self._assoc_len, self._msg_len))
        assert(isinstance(self._cache, list))

        # Formatting control information and nonce (A.2.1)
        q = 15 - len(self.nonce)  # length of Q, the encoded message length
        flags = (64 * (self._assoc_len > 0) + 8 * ((self._mac_len - 2) // 2) +
                 (q - 1))
        b_0 = struct.pack("B", flags) + self.nonce + long_to_bytes(self._msg_len, q)

        # Formatting associated data (A.2.2)
        # Encoded 'a' is concatenated with the associated data 'A'
        assoc_len_encoded = b''
        if self._assoc_len > 0:
            if self._assoc_len < (2 ** 16 - 2 ** 8):
                enc_size = 2
            elif self._assoc_len < (2 ** 32):
                assoc_len_encoded = b'\xFF\xFE'
                enc_size = 4
            else:
                assoc_len_encoded = b'\xFF\xFF'
                enc_size = 8
            assoc_len_encoded += long_to_bytes(self._assoc_len, enc_size)

        # b_0 and assoc_len_encoded must be processed first
        self._cache.insert(0, b_0)
        self._cache.insert(1, assoc_len_encoded)

        # Process all the data cached so far
        first_data_to_mac = b"".join(self._cache)
        self._cache = b""
        self._mac_status = MacStatus.PROCESSING_AUTH_DATA
        self._update(first_data_to_mac)

    def _pad_cache_and_update(self):

        assert(self._mac_status != MacStatus.NOT_STARTED)
        assert(len(self._cache) < self.block_size)

        # Associated data is concatenated with the least number
        # of zero bytes (possibly none) to reach alignment to
        # the 16 byte boundary (A.2.3)
        len_cache = len(self._cache)
        if len_cache > 0:
            self._update(b'\x00' * (self.block_size - len_cache))

    def update(self, assoc_data):
        """Protect associated data

        If there is any associated data, the caller has to invoke
        this function one or more times, before using
        ``decrypt`` or ``encrypt``.

        By *associated data* it is meant any data (e.g. packet headers) that
        will not be encrypted and will be transmitted in the clear.
        However, the receiver is still able to detect any modification to it.
        In CCM, the *associated data* is also called
        *additional authenticated data* (AAD).

        If there is no associated data, this method must not be called.

        The caller may split associated data in segments of any size, and
        invoke this method multiple times, each time with the next segment.

        :Parameters:
          assoc_data : bytes/bytearray/memoryview
            A piece of associated data. There are no restrictions on its size.
        """

        if self.update not in self._next:
            raise TypeError("update() can only be called"
                            " immediately after initialization")

        self._next = [self.update, self.encrypt, self.decrypt,
                      self.digest, self.verify]

        self._cumul_assoc_len += len(assoc_data)
        if self._assoc_len is not None and \
           self._cumul_assoc_len > self._assoc_len:
            raise ValueError("Associated data is too long")

        self._update(assoc_data)
        return self

    def _update(self, assoc_data_pt=b""):
        """Update the MAC with associated data or plaintext
           (without FSM checks)"""

        # If MAC has not started yet, we just park the data into a list.
        # If the data is mutable, we create a copy and store that instead.
        if self._mac_status == MacStatus.NOT_STARTED:
            if is_writeable_buffer(assoc_data_pt):
                assoc_data_pt = _copy_bytes(None, None, assoc_data_pt)
            self._cache.append(assoc_data_pt)
            return

        assert(len(self._cache) < self.block_size)

        if len(self._cache) > 0:
            filler = min(self.block_size - len(self._cache),
                         len(assoc_data_pt))
            self._cache += _copy_bytes(None, filler, assoc_data_pt)
            assoc_data_pt = _copy_bytes(filler, None, assoc_data_pt)

            if len(self._cache) < self.block_size:
                return

            # The cache is exactly one block
            self._t = self._mac.encrypt(self._cache)
            self._cache = b""

        update_len = len(assoc_data_pt) // self.block_size * self.block_size
        self._cache = _copy_bytes(update_len, None, assoc_data_pt)
        if update_len > 0:
            self._t = self._mac.encrypt(assoc_data_pt[:update_len])[-16:]

    def encrypt(self, plaintext, output=None):
        """Encrypt data with the key set at initialization.

        A cipher object is stateful: once you have encrypted a message
        you cannot encrypt (or decrypt) another message using the same
        object.

        This method can be called only **once** if ``msg_len`` was
        not passed at initialization.

        If ``msg_len`` was given, the data to encrypt can be broken
        up in two or more pieces and `encrypt` can be called
        multiple times.

        That is, the statement:

            >>> c.encrypt(a) + c.encrypt(b)

        is equivalent to:

             >>> c.encrypt(a+b)

        This function does not add any padding to the plaintext.

        :Parameters:
          plaintext : bytes/bytearray/memoryview
            The piece of data to encrypt.
            It can be of any length.
        :Keywords:
          output : bytearray/memoryview
            The location where the ciphertext must be written to.
            If ``None``, the ciphertext is returned.
        :Return:
          If ``output`` is ``None``, the ciphertext as ``bytes``.
          Otherwise, ``None``.
        """

        if self.encrypt not in self._next:
            raise TypeError("encrypt() can only be called after"
                            " initialization or an update()")
        self._next = [self.encrypt, self.digest]

        # No more associated data allowed from now
        if self._assoc_len is None:
            assert(isinstance(self._cache, list))
            self._assoc_len = sum([len(x) for x in self._cache])
            if self._msg_len is not None:
                self._start_mac()
        else:
            if self._cumul_assoc_len < self._assoc_len:
                raise ValueError("Associated data is too short")

        # Only once piece of plaintext accepted if message length was
        # not declared in advance
        if self._msg_len is None:
            self._msg_len = len(plaintext)
            self._start_mac()
            self._next = [self.digest]

        self._cumul_msg_len += len(plaintext)
        if self._cumul_msg_len > self._msg_len:
            raise ValueError("Message is too long")

        if self._mac_status == MacStatus.PROCESSING_AUTH_DATA:
            # Associated data is concatenated with the least number
            # of zero bytes (possibly none) to reach alignment to
            # the 16 byte boundary (A.2.3)
            self._pad_cache_and_update()
            self._mac_status = MacStatus.PROCESSING_PLAINTEXT

        self._update(plaintext)
        return self._cipher.encrypt(plaintext, output=output)

    def decrypt(self, ciphertext, output=None):
        """Decrypt data with the key set at initialization.

        A cipher object is stateful: once you have decrypted a message
        you cannot decrypt (or encrypt) another message with the same
        object.

        This method can be called only **once** if ``msg_len`` was
        not passed at initialization.

        If ``msg_len`` was given, the data to decrypt can be
        broken up in two or more pieces and `decrypt` can be
        called multiple times.

        That is, the statement:

            >>> c.decrypt(a) + c.decrypt(b)

        is equivalent to:

             >>> c.decrypt(a+b)

        This function does not remove any padding from the plaintext.

        :Parameters:
          ciphertext : bytes/bytearray/memoryview
            The piece of data to decrypt.
            It can be of any length.
        :Keywords:
          output : bytearray/memoryview
            The location where the plaintext must be written to.
            If ``None``, the plaintext is returned.
        :Return:
          If ``output`` is ``None``, the plaintext as ``bytes``.
          Otherwise, ``None``.
        """

        if self.decrypt not in self._next:
            raise TypeError("decrypt() can only be called"
                            " after initialization or an update()")
        self._next = [self.decrypt, self.verify]

        # No more associated data allowed from now
        if self._assoc_len is None:
            assert(isinstance(self._cache, list))
            self._assoc_len = sum([len(x) for x in self._cache])
            if self._msg_len is not None:
                self._start_mac()
        else:
            if self._cumul_assoc_len < self._assoc_len:
                raise ValueError("Associated data is too short")

        # Only once piece of ciphertext accepted if message length was
        # not declared in advance
        if self._msg_len is None:
            self._msg_len = len(ciphertext)
            self._start_mac()
            self._next = [self.verify]

        self._cumul_msg_len += len(ciphertext)
        if self._cumul_msg_len > self._msg_len:
            raise ValueError("Message is too long")

        if self._mac_status == MacStatus.PROCESSING_AUTH_DATA:
            # Associated data is concatenated with the least number
            # of zero bytes (possibly none) to reach alignment to
            # the 16 byte boundary (A.2.3)
            self._pad_cache_and_update()
            self._mac_status = MacStatus.PROCESSING_PLAINTEXT

        # Encrypt is equivalent to decrypt with the CTR mode
        plaintext = self._cipher.encrypt(ciphertext, output=output)
        if output is None:
            self._update(plaintext)
        else:
            self._update(output)
        return plaintext

    def digest(self):
        """Compute the *binary* MAC tag.

        The caller invokes this function at the very end.

        This method returns the MAC that shall be sent to the receiver,
        together with the ciphertext.

        :Return: the MAC, as a byte string.
        """

        if self.digest not in self._next:
            raise TypeError("digest() cannot be called when decrypting"
                            " or validating a message")
        self._next = [self.digest]
        return self._digest()

    def _digest(self):
        if self._mac_tag:
            return self._mac_tag

        if self._assoc_len is None:
            assert(isinstance(self._cache, list))
            self._assoc_len = sum([len(x) for x in self._cache])
            if self._msg_len is not None:
                self._start_mac()
        else:
            if self._cumul_assoc_len < self._assoc_len:
                raise ValueError("Associated data is too short")

        if self._msg_len is None:
            self._msg_len = 0
            self._start_mac()

        if self._cumul_msg_len != self._msg_len:
            raise ValueError("Message is too short")

        # Both associated data and payload are concatenated with the least
        # number of zero bytes (possibly none) that align it to the
        # 16 byte boundary (A.2.2 and A.2.3)
        self._pad_cache_and_update()

        # Step 8 in 6.1 (T xor MSB_Tlen(S_0))
        self._mac_tag = strxor(self._t, self._s_0)[:self._mac_len]

        return self._mac_tag

    def hexdigest(self):
        """Compute the *printable* MAC tag.

        This method is like `digest`.

        :Return: the MAC, as a hexadecimal string.
        """
        return "".join(["%02x" % bord(x) for x in self.digest()])

    def verify(self, received_mac_tag):
        """Validate the *binary* MAC tag.

        The caller invokes this function at the very end.

        This method checks if the decrypted message is indeed valid
        (that is, if the key is correct) and it has not been
        tampered with while in transit.

        :Parameters:
          received_mac_tag : bytes/bytearray/memoryview
            This is the *binary* MAC, as received from the sender.
        :Raises ValueError:
            if the MAC does not match. The message has been tampered with
            or the key is incorrect.
        """

        if self.verify not in self._next:
            raise TypeError("verify() cannot be called"
                            " when encrypting a message")
        self._next = [self.verify]

        self._digest()
        secret = get_random_bytes(16)

        mac1 = BLAKE2s.new(digest_bits=160, key=secret, data=self._mac_tag)
        mac2 = BLAKE2s.new(digest_bits=160, key=secret, data=received_mac_tag)

        if mac1.digest() != mac2.digest():
            raise ValueError("MAC check failed")

    def hexverify(self, hex_mac_tag):
        """Validate the *printable* MAC tag.

        This method is like `verify`.

        :Parameters:
          hex_mac_tag : string
            This is the *printable* MAC, as received from the sender.
        :Raises ValueError:
            if the MAC does not match. The message has been tampered with
            or the key is incorrect.
        """

        self.verify(unhexlify(hex_mac_tag))

    def encrypt_and_digest(self, plaintext, output=None):
        """Perform encrypt() and digest() in one step.

        :Parameters:
          plaintext : bytes/bytearray/memoryview
            The piece of data to encrypt.
        :Keywords:
          output : bytearray/memoryview
            The location where the ciphertext must be written to.
            If ``None``, the ciphertext is returned.
        :Return:
            a tuple with two items:

            - the ciphertext, as ``bytes``
            - the MAC tag, as ``bytes``

            The first item becomes ``None`` when the ``output`` parameter
            specified a location for the result.
        """

        return self.encrypt(plaintext, output=output), self.digest()

    def decrypt_and_verify(self, ciphertext, received_mac_tag, output=None):
        """Perform decrypt() and verify() in one step.

        :Parameters:
          ciphertext : bytes/bytearray/memoryview
            The piece of data to decrypt.
          received_mac_tag : bytes/bytearray/memoryview
            This is the *binary* MAC, as received from the sender.
        :Keywords:
          output : bytearray/memoryview
            The location where the plaintext must be written to.
            If ``None``, the plaintext is returned.
        :Return: the plaintext as ``bytes`` or ``None`` when the ``output``
            parameter specified a location for the result.
        :Raises ValueError:
            if the MAC does not match. The message has been tampered with
            or the key is incorrect.
        """

        plaintext = self.decrypt(ciphertext, output=output)
        self.verify(received_mac_tag)
        return plaintext


def _create_ccm_cipher(factory, **kwargs):
    """Create a new block cipher, configured in CCM mode.

    :Parameters:
      factory : module
        A symmetric cipher module from `Crypto.Cipher` (like
        `Crypto.Cipher.AES`).

    :Keywords:
      key : bytes/bytearray/memoryview
        The secret key to use in the symmetric cipher.

      nonce : bytes/bytearray/memoryview
        A value that must never be reused for any other encryption.

        Its length must be in the range ``[7..13]``.
        11 or 12 bytes are reasonable values in general. Bear in
        mind that with CCM there is a trade-off between nonce length and
        maximum message size.

        If not specified, a 11 byte long random string is used.

      mac_len : integer
        Length of the MAC, in bytes. It must be even and in
        the range ``[4..16]``. The default is 16.

      msg_len : integer
        Length of the message to (de)cipher.
        If not specified, ``encrypt`` or ``decrypt`` may only be called once.

      assoc_len : integer
        Length of the associated data.
        If not specified, all data is internally buffered.
    """

    try:
        key = key = kwargs.pop("key")
    except KeyError as e:
        raise TypeError("Missing parameter: " + str(e))

    nonce = kwargs.pop("nonce", None)  # N
    if nonce is None:
        nonce = get_random_bytes(11)
    mac_len = kwargs.pop("mac_len", factory.block_size)
    msg_len = kwargs.pop("msg_len", None)      # p
    assoc_len = kwargs.pop("assoc_len", None)  # a
    cipher_params = dict(kwargs)

    return CcmMode(factory, key, nonce, mac_len, msg_len,
                   assoc_len, cipher_params)
