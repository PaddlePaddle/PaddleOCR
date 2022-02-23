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
OpenPGP mode.
"""

__all__ = ['OpenPgpMode']

from Crypto.Util.py3compat import _copy_bytes
from Crypto.Random import get_random_bytes

class OpenPgpMode(object):
    """OpenPGP mode.

    This mode is a variant of CFB, and it is only used in PGP and
    OpenPGP_ applications. If in doubt, use another mode.

    An Initialization Vector (*IV*) is required.

    Unlike CFB, the *encrypted* IV (not the IV itself) is
    transmitted to the receiver.

    The IV is a random data block. For legacy reasons, two of its bytes are
    duplicated to act as a checksum for the correctness of the key, which is now
    known to be insecure and is ignored. The encrypted IV is therefore 2 bytes
    longer than the clean IV.

    .. _OpenPGP: http://tools.ietf.org/html/rfc4880

    :undocumented: __init__
    """

    def __init__(self, factory, key, iv, cipher_params):

        #: The block size of the underlying cipher, in bytes.
        self.block_size = factory.block_size

        self._done_first_block = False  # True after the first encryption

        # Instantiate a temporary cipher to process the IV
        IV_cipher = factory.new(
                        key,
                        factory.MODE_CFB,
                        IV=b'\x00' * self.block_size,
                        segment_size=self.block_size * 8,
                        **cipher_params)

        iv = _copy_bytes(None, None, iv)

        # The cipher will be used for...
        if len(iv) == self.block_size:
            # ... encryption
            self._encrypted_IV = IV_cipher.encrypt(iv + iv[-2:])
        elif len(iv) == self.block_size + 2:
            # ... decryption
            self._encrypted_IV = iv
            # Last two bytes are for a deprecated "quick check" feature that
            # should not be used. (https://eprint.iacr.org/2005/033)
            iv = IV_cipher.decrypt(iv)[:-2]
        else:
            raise ValueError("Length of IV must be %d or %d bytes"
                             " for MODE_OPENPGP"
                             % (self.block_size, self.block_size + 2))

        self.iv = self.IV = iv

        # Instantiate the cipher for the real PGP data
        self._cipher = factory.new(
                            key,
                            factory.MODE_CFB,
                            IV=self._encrypted_IV[-self.block_size:],
                            segment_size=self.block_size * 8,
                            **cipher_params)

    def encrypt(self, plaintext):
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

        :Return:
            the encrypted data, as a byte string.
            It is as long as *plaintext* with one exception:
            when encrypting the first message chunk,
            the encypted IV is prepended to the returned ciphertext.
        """

        res = self._cipher.encrypt(plaintext)
        if not self._done_first_block:
            res = self._encrypted_IV + res
            self._done_first_block = True
        return res

    def decrypt(self, ciphertext):
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

        :Return: the decrypted data (byte string).
        """

        return self._cipher.decrypt(ciphertext)


def _create_openpgp_cipher(factory, **kwargs):
    """Create a new block cipher, configured in OpenPGP mode.

    :Parameters:
      factory : module
        The module.

    :Keywords:
      key : bytes/bytearray/memoryview
        The secret key to use in the symmetric cipher.

      IV : bytes/bytearray/memoryview
        The initialization vector to use for encryption or decryption.

        For encryption, the IV must be as long as the cipher block size.

        For decryption, it must be 2 bytes longer (it is actually the
        *encrypted* IV which was prefixed to the ciphertext).
    """

    iv = kwargs.pop("IV", None)
    IV = kwargs.pop("iv", None)

    if (None, None) == (iv, IV):
        iv = get_random_bytes(factory.block_size)
    if iv is not None:
        if IV is not None:
            raise TypeError("You must either use 'iv' or 'IV', not both")
    else:
        iv = IV

    try:
        key = kwargs.pop("key")
    except KeyError as e:
        raise TypeError("Missing component: " + str(e))

    return OpenPgpMode(factory, key, iv, kwargs)
