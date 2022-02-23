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
EAX mode.
"""

__all__ = ['EaxMode']

import struct
from binascii import unhexlify

from Crypto.Util.py3compat import byte_string, bord, _copy_bytes

from Crypto.Util._raw_api import is_buffer

from Crypto.Util.strxor import strxor
from Crypto.Util.number import long_to_bytes, bytes_to_long

from Crypto.Hash import CMAC, BLAKE2s
from Crypto.Random import get_random_bytes


class EaxMode(object):
    """*EAX* mode.

    This is an Authenticated Encryption with Associated Data
    (`AEAD`_) mode. It provides both confidentiality and authenticity.

    The header of the message may be left in the clear, if needed,
    and it will still be subject to authentication.

    The decryption step tells the receiver if the message comes
    from a source that really knowns the secret key.
    Additionally, decryption detects if any part of the message -
    including the header - has been modified or corrupted.

    This mode requires a *nonce*.

    This mode is only available for ciphers that operate on 64 or
    128 bits blocks.

    There are no official standards defining EAX.
    The implementation is based on `a proposal`__ that
    was presented to NIST.

    .. _AEAD: http://blog.cryptographyengineering.com/2012/05/how-to-choose-authenticated-encryption.html
    .. __: http://csrc.nist.gov/groups/ST/toolkit/BCM/documents/proposedmodes/eax/eax-spec.pdf

    :undocumented: __init__
    """

    def __init__(self, factory, key, nonce, mac_len, cipher_params):
        """EAX cipher mode"""

        self.block_size = factory.block_size
        """The block size of the underlying cipher, in bytes."""

        self.nonce = _copy_bytes(None, None, nonce)
        """The nonce originally used to create the object."""

        self._mac_len = mac_len
        self._mac_tag = None  # Cache for MAC tag

        # Allowed transitions after initialization
        self._next = [self.update, self.encrypt, self.decrypt,
                      self.digest, self.verify]

        # MAC tag length
        if not (4 <= self._mac_len <= self.block_size):
            raise ValueError("Parameter 'mac_len' must not be larger than %d"
                             % self.block_size)

        # Nonce cannot be empty and must be a byte string
        if len(self.nonce) == 0:
            raise ValueError("Nonce cannot be empty in EAX mode")
        if not is_buffer(nonce):
            raise TypeError("nonce must be bytes, bytearray or memoryview")

        self._omac = [
                CMAC.new(key,
                         b'\x00' * (self.block_size - 1) + struct.pack('B', i),
                         ciphermod=factory,
                         cipher_params=cipher_params)
                for i in range(0, 3)
                ]

        # Compute MAC of nonce
        self._omac[0].update(self.nonce)
        self._signer = self._omac[1]

        # MAC of the nonce is also the initial counter for CTR encryption
        counter_int = bytes_to_long(self._omac[0].digest())
        self._cipher = factory.new(key,
                                   factory.MODE_CTR,
                                   initial_value=counter_int,
                                   nonce=b"",
                                   **cipher_params)

    def update(self, assoc_data):
        """Protect associated data

        If there is any associated data, the caller has to invoke
        this function one or more times, before using
        ``decrypt`` or ``encrypt``.

        By *associated data* it is meant any data (e.g. packet headers) that
        will not be encrypted and will be transmitted in the clear.
        However, the receiver is still able to detect any modification to it.

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

        self._signer.update(assoc_data)
        return self

    def encrypt(self, plaintext, output=None):
        """Encrypt data with the key and the parameters set at initialization.

        A cipher object is stateful: once you have encrypted a message
        you cannot encrypt (or decrypt) another message using the same
        object.

        The data to encrypt can be broken up in two or
        more pieces and `encrypt` can be called multiple times.

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
        ct = self._cipher.encrypt(plaintext, output=output)
        if output is None:
            self._omac[2].update(ct)
        else:
            self._omac[2].update(output)
        return ct

    def decrypt(self, ciphertext, output=None):
        """Decrypt data with the key and the parameters set at initialization.

        A cipher object is stateful: once you have decrypted a message
        you cannot decrypt (or encrypt) another message with the same
        object.

        The data to decrypt can be broken up in two or
        more pieces and `decrypt` can be called multiple times.

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
        self._omac[2].update(ciphertext)
        return self._cipher.decrypt(ciphertext, output=output)

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

        if not self._mac_tag:
            tag = b'\x00' * self.block_size
            for i in range(3):
                tag = strxor(tag, self._omac[i].digest())
            self._mac_tag = tag[:self._mac_len]

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
        :Raises MacMismatchError:
            if the MAC does not match. The message has been tampered with
            or the key is incorrect.
        """

        if self.verify not in self._next:
            raise TypeError("verify() cannot be called"
                                " when encrypting a message")
        self._next = [self.verify]

        if not self._mac_tag:
            tag = b'\x00' * self.block_size
            for i in range(3):
                tag = strxor(tag, self._omac[i].digest())
            self._mac_tag = tag[:self._mac_len]

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
        :Raises MacMismatchError:
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
        :Raises MacMismatchError:
            if the MAC does not match. The message has been tampered with
            or the key is incorrect.
        """

        pt = self.decrypt(ciphertext, output=output)
        self.verify(received_mac_tag)
        return pt


def _create_eax_cipher(factory, **kwargs):
    """Create a new block cipher, configured in EAX mode.

    :Parameters:
      factory : module
        A symmetric cipher module from `Crypto.Cipher` (like
        `Crypto.Cipher.AES`).

    :Keywords:
      key : bytes/bytearray/memoryview
        The secret key to use in the symmetric cipher.

      nonce : bytes/bytearray/memoryview
        A value that must never be reused for any other encryption.
        There are no restrictions on its length, but it is recommended to use
        at least 16 bytes.

        The nonce shall never repeat for two different messages encrypted with
        the same key, but it does not need to be random.

        If not specified, a 16 byte long random string is used.

      mac_len : integer
        Length of the MAC, in bytes. It must be no larger than the cipher
        block bytes (which is the default).
    """

    try:
        key = kwargs.pop("key")
        nonce = kwargs.pop("nonce", None)
        if nonce is None:
            nonce = get_random_bytes(16)
        mac_len = kwargs.pop("mac_len", factory.block_size)
    except KeyError as e:
        raise TypeError("Missing parameter: " + str(e))

    return EaxMode(factory, key, nonce, mac_len, kwargs)
