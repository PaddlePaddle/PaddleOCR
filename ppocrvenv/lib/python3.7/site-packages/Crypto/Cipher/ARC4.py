# -*- coding: utf-8 -*-
#
#  Cipher/ARC4.py : ARC4
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

from Crypto.Util.py3compat import b

from Crypto.Util._raw_api import (load_pycryptodome_raw_lib, VoidPointer,
                                  create_string_buffer, get_raw_buffer,
                                  SmartPointer, c_size_t, c_uint8_ptr)


_raw_arc4_lib = load_pycryptodome_raw_lib("Crypto.Cipher._ARC4", """
                    int ARC4_stream_encrypt(void *rc4State, const uint8_t in[],
                                            uint8_t out[], size_t len);
                    int ARC4_stream_init(uint8_t *key, size_t keylen,
                                         void **pRc4State);
                    int ARC4_stream_destroy(void *rc4State);
                    """)


class ARC4Cipher:
    """ARC4 cipher object. Do not create it directly. Use
    :func:`Crypto.Cipher.ARC4.new` instead.
    """

    def __init__(self, key, *args, **kwargs):
        """Initialize an ARC4 cipher object

        See also `new()` at the module level."""

        if len(args) > 0:
            ndrop = args[0]
            args = args[1:]
        else:
            ndrop = kwargs.pop('drop', 0)

        if len(key) not in key_size:
            raise ValueError("Incorrect ARC4 key length (%d bytes)" %
                             len(key))

        self._state = VoidPointer()
        result = _raw_arc4_lib.ARC4_stream_init(c_uint8_ptr(key),
                                                c_size_t(len(key)),
                                                self._state.address_of())
        if result != 0:
            raise ValueError("Error %d while creating the ARC4 cipher"
                             % result)
        self._state = SmartPointer(self._state.get(),
                                   _raw_arc4_lib.ARC4_stream_destroy)

        if ndrop > 0:
            # This is OK even if the cipher is used for decryption,
            # since encrypt and decrypt are actually the same thing
            # with ARC4.
            self.encrypt(b'\x00' * ndrop)

        self.block_size = 1
        self.key_size = len(key)

    def encrypt(self, plaintext):
        """Encrypt a piece of data.

        :param plaintext: The data to encrypt, of any size.
        :type plaintext: bytes, bytearray, memoryview
        :returns: the encrypted byte string, of equal length as the
          plaintext.
        """

        ciphertext = create_string_buffer(len(plaintext))
        result = _raw_arc4_lib.ARC4_stream_encrypt(self._state.get(),
                                                   c_uint8_ptr(plaintext),
                                                   ciphertext,
                                                   c_size_t(len(plaintext)))
        if result:
            raise ValueError("Error %d while encrypting with RC4" % result)
        return get_raw_buffer(ciphertext)

    def decrypt(self, ciphertext):
        """Decrypt a piece of data.

        :param ciphertext: The data to decrypt, of any size.
        :type ciphertext: bytes, bytearray, memoryview
        :returns: the decrypted byte string, of equal length as the
          ciphertext.
        """

        try:
            return self.encrypt(ciphertext)
        except ValueError as e:
            raise ValueError(str(e).replace("enc", "dec"))


def new(key, *args, **kwargs):
    """Create a new ARC4 cipher.

    :param key:
        The secret key to use in the symmetric cipher.
        Its length must be in the range ``[5..256]``.
        The recommended length is 16 bytes.
    :type key: bytes, bytearray, memoryview

    :Keyword Arguments:
        *   *drop* (``integer``) --
            The amount of bytes to discard from the initial part of the keystream.
            In fact, such part has been found to be distinguishable from random
            data (while it shouldn't) and also correlated to key.

            The recommended value is 3072_ bytes. The default value is 0.

    :Return: an `ARC4Cipher` object

    .. _3072: http://eprint.iacr.org/2002/067.pdf
    """
    return ARC4Cipher(key, *args, **kwargs)

# Size of a data block (in bytes)
block_size = 1
# Size of a key (in bytes)
key_size = range(5, 256+1)
