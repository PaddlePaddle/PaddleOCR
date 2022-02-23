# -*- coding: utf-8 -*-
#
#  Cipher/mode_cfb.py : CFB mode
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
Counter Feedback (CFB) mode.
"""

__all__ = ['CfbMode']

from Crypto.Util.py3compat import _copy_bytes
from Crypto.Util._raw_api import (load_pycryptodome_raw_lib, VoidPointer,
                                  create_string_buffer, get_raw_buffer,
                                  SmartPointer, c_size_t, c_uint8_ptr,
                                  is_writeable_buffer)

from Crypto.Random import get_random_bytes

raw_cfb_lib = load_pycryptodome_raw_lib("Crypto.Cipher._raw_cfb","""
                    int CFB_start_operation(void *cipher,
                                            const uint8_t iv[],
                                            size_t iv_len,
                                            size_t segment_len, /* In bytes */
                                            void **pResult);
                    int CFB_encrypt(void *cfbState,
                                    const uint8_t *in,
                                    uint8_t *out,
                                    size_t data_len);
                    int CFB_decrypt(void *cfbState,
                                    const uint8_t *in,
                                    uint8_t *out,
                                    size_t data_len);
                    int CFB_stop_operation(void *state);"""
                    )


class CfbMode(object):
    """*Cipher FeedBack (CFB)*.

    This mode is similar to CFB, but it transforms
    the underlying block cipher into a stream cipher.

    Plaintext and ciphertext are processed in *segments*
    of **s** bits. The mode is therefore sometimes
    labelled **s**-bit CFB.

    An Initialization Vector (*IV*) is required.

    See `NIST SP800-38A`_ , Section 6.3.

    .. _`NIST SP800-38A` : http://csrc.nist.gov/publications/nistpubs/800-38a/sp800-38a.pdf

    :undocumented: __init__
    """

    def __init__(self, block_cipher, iv, segment_size):
        """Create a new block cipher, configured in CFB mode.

        :Parameters:
          block_cipher : C pointer
            A smart pointer to the low-level block cipher instance.

          iv : bytes/bytearray/memoryview
            The initialization vector to use for encryption or decryption.
            It is as long as the cipher block.

            **The IV must be unpredictable**. Ideally it is picked randomly.

            Reusing the *IV* for encryptions performed with the same key
            compromises confidentiality.

          segment_size : integer
            The number of bytes the plaintext and ciphertext are segmented in.
        """

        self._state = VoidPointer()
        result = raw_cfb_lib.CFB_start_operation(block_cipher.get(),
                                                 c_uint8_ptr(iv),
                                                 c_size_t(len(iv)),
                                                 c_size_t(segment_size),
                                                 self._state.address_of())
        if result:
            raise ValueError("Error %d while instantiating the CFB mode" % result)

        # Ensure that object disposal of this Python object will (eventually)
        # free the memory allocated by the raw library for the cipher mode
        self._state = SmartPointer(self._state.get(),
                                   raw_cfb_lib.CFB_stop_operation)

        # Memory allocated for the underlying block cipher is now owed
        # by the cipher mode
        block_cipher.release()

        self.block_size = len(iv)
        """The block size of the underlying cipher, in bytes."""

        self.iv = _copy_bytes(None, None, iv)
        """The Initialization Vector originally used to create the object.
        The value does not change."""

        self.IV = self.iv
        """Alias for `iv`"""

        self._next = [ self.encrypt, self.decrypt ]

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
          If ``output`` is ``None``, the ciphertext is returned as ``bytes``.
          Otherwise, ``None``.
        """

        if self.encrypt not in self._next:
            raise TypeError("encrypt() cannot be called after decrypt()")
        self._next = [ self.encrypt ]
        
        if output is None:
            ciphertext = create_string_buffer(len(plaintext))
        else:
            ciphertext = output
            
            if not is_writeable_buffer(output):
                raise TypeError("output must be a bytearray or a writeable memoryview")
        
            if len(plaintext) != len(output):
                raise ValueError("output must have the same length as the input"
                                 "  (%d bytes)" % len(plaintext))

        result = raw_cfb_lib.CFB_encrypt(self._state.get(),
                                         c_uint8_ptr(plaintext),
                                         c_uint8_ptr(ciphertext),
                                         c_size_t(len(plaintext)))
        if result:
            raise ValueError("Error %d while encrypting in CFB mode" % result)

        if output is None:
            return get_raw_buffer(ciphertext)
        else:
            return None

    def decrypt(self, ciphertext,  output=None):
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
          If ``output`` is ``None``, the plaintext is returned as ``bytes``.
          Otherwise, ``None``.
        """

        if self.decrypt not in self._next:
            raise TypeError("decrypt() cannot be called after encrypt()")
        self._next = [ self.decrypt ]
        
        if output is None:
            plaintext = create_string_buffer(len(ciphertext))
        else:
            plaintext = output

            if not is_writeable_buffer(output):
                raise TypeError("output must be a bytearray or a writeable memoryview")
            
            if len(ciphertext) != len(output):
                raise ValueError("output must have the same length as the input"
                                 "  (%d bytes)" % len(plaintext))
        
        result = raw_cfb_lib.CFB_decrypt(self._state.get(),
                                         c_uint8_ptr(ciphertext),
                                         c_uint8_ptr(plaintext),
                                         c_size_t(len(ciphertext)))
        if result:
            raise ValueError("Error %d while decrypting in CFB mode" % result)

        if output is None:
            return get_raw_buffer(plaintext)
        else:
            return None


def _create_cfb_cipher(factory, **kwargs):
    """Instantiate a cipher object that performs CFB encryption/decryption.

    :Parameters:
      factory : module
        The underlying block cipher, a module from ``Crypto.Cipher``.

    :Keywords:
      iv : bytes/bytearray/memoryview
        The IV to use for CFB.

      IV : bytes/bytearray/memoryview
        Alias for ``iv``.

      segment_size : integer
        The number of bit the plaintext and ciphertext are segmented in.
        If not present, the default is 8.

    Any other keyword will be passed to the underlying block cipher.
    See the relevant documentation for details (at least ``key`` will need
    to be present).
    """

    cipher_state = factory._create_base_cipher(kwargs)

    iv = kwargs.pop("IV", None)
    IV = kwargs.pop("iv", None)

    if (None, None) == (iv, IV):
        iv = get_random_bytes(factory.block_size)
    if iv is not None:
        if IV is not None:
            raise TypeError("You must either use 'iv' or 'IV', not both")
    else:
        iv = IV

    if len(iv) != factory.block_size:
        raise ValueError("Incorrect IV length (it must be %d bytes long)" %
                factory.block_size)

    segment_size_bytes, rem = divmod(kwargs.pop("segment_size", 8), 8)
    if segment_size_bytes == 0 or rem != 0:
        raise ValueError("'segment_size' must be positive and multiple of 8 bits")

    if kwargs:
        raise TypeError("Unknown parameters for CFB: %s" % str(kwargs))
    return CfbMode(cipher_state, iv, segment_size_bytes)
