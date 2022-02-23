# -*- coding: utf-8 -*-
#
#  Cipher/mode_ecb.py : ECB mode
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
Electronic Code Book (ECB) mode.
"""

__all__ = [ 'EcbMode' ]

from Crypto.Util._raw_api import (load_pycryptodome_raw_lib,
                                  VoidPointer, create_string_buffer,
                                  get_raw_buffer, SmartPointer,
                                  c_size_t, c_uint8_ptr,
                                  is_writeable_buffer)

raw_ecb_lib = load_pycryptodome_raw_lib("Crypto.Cipher._raw_ecb", """
                    int ECB_start_operation(void *cipher,
                                            void **pResult);
                    int ECB_encrypt(void *ecbState,
                                    const uint8_t *in,
                                    uint8_t *out,
                                    size_t data_len);
                    int ECB_decrypt(void *ecbState,
                                    const uint8_t *in,
                                    uint8_t *out,
                                    size_t data_len);
                    int ECB_stop_operation(void *state);
                    """
                                        )


class EcbMode(object):
    """*Electronic Code Book (ECB)*.

    This is the simplest encryption mode. Each of the plaintext blocks
    is directly encrypted into a ciphertext block, independently of
    any other block.

    This mode is dangerous because it exposes frequency of symbols
    in your plaintext. Other modes (e.g. *CBC*) should be used instead.

    See `NIST SP800-38A`_ , Section 6.1.

    .. _`NIST SP800-38A` : http://csrc.nist.gov/publications/nistpubs/800-38a/sp800-38a.pdf

    :undocumented: __init__
    """

    def __init__(self, block_cipher):
        """Create a new block cipher, configured in ECB mode.

        :Parameters:
          block_cipher : C pointer
            A smart pointer to the low-level block cipher instance.
        """
        self.block_size = block_cipher.block_size

        self._state = VoidPointer()
        result = raw_ecb_lib.ECB_start_operation(block_cipher.get(),
                                                 self._state.address_of())
        if result:
            raise ValueError("Error %d while instantiating the ECB mode"
                             % result)

        # Ensure that object disposal of this Python object will (eventually)
        # free the memory allocated by the raw library for the cipher
        # mode
        self._state = SmartPointer(self._state.get(),
                                   raw_ecb_lib.ECB_stop_operation)

        # Memory allocated for the underlying block cipher is now owned
        # by the cipher mode
        block_cipher.release()

    def encrypt(self, plaintext, output=None):
        """Encrypt data with the key set at initialization.

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
            The length must be multiple of the cipher block length.
        :Keywords:
          output : bytearray/memoryview
            The location where the ciphertext must be written to.
            If ``None``, the ciphertext is returned.
        :Return:
          If ``output`` is ``None``, the ciphertext is returned as ``bytes``.
          Otherwise, ``None``.
        """

        if output is None:
            ciphertext = create_string_buffer(len(plaintext))
        else:
            ciphertext = output
            
            if not is_writeable_buffer(output):
                raise TypeError("output must be a bytearray or a writeable memoryview")
        
            if len(plaintext) != len(output):
                raise ValueError("output must have the same length as the input"
                                 "  (%d bytes)" % len(plaintext))

        result = raw_ecb_lib.ECB_encrypt(self._state.get(),
                                         c_uint8_ptr(plaintext),
                                         c_uint8_ptr(ciphertext),
                                         c_size_t(len(plaintext)))
        if result:
            if result == 3:
                raise ValueError("Data must be aligned to block boundary in ECB mode")
            raise ValueError("Error %d while encrypting in ECB mode" % result)
        
        if output is None:
            return get_raw_buffer(ciphertext)
        else:
            return None

    def decrypt(self, ciphertext, output=None):
        """Decrypt data with the key set at initialization.

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
            The length must be multiple of the cipher block length.
        :Keywords:
          output : bytearray/memoryview
            The location where the plaintext must be written to.
            If ``None``, the plaintext is returned.
        :Return:
          If ``output`` is ``None``, the plaintext is returned as ``bytes``.
          Otherwise, ``None``.
        """
        
        if output is None:
            plaintext = create_string_buffer(len(ciphertext))
        else:
            plaintext = output

            if not is_writeable_buffer(output):
                raise TypeError("output must be a bytearray or a writeable memoryview")
            
            if len(ciphertext) != len(output):
                raise ValueError("output must have the same length as the input"
                                 "  (%d bytes)" % len(plaintext))

        result = raw_ecb_lib.ECB_decrypt(self._state.get(),
                                         c_uint8_ptr(ciphertext),
                                         c_uint8_ptr(plaintext),
                                         c_size_t(len(ciphertext)))
        if result:
            if result == 3:
                raise ValueError("Data must be aligned to block boundary in ECB mode")
            raise ValueError("Error %d while decrypting in ECB mode" % result)

        if output is None:
            return get_raw_buffer(plaintext)
        else:
            return None


def _create_ecb_cipher(factory, **kwargs):
    """Instantiate a cipher object that performs ECB encryption/decryption.

    :Parameters:
      factory : module
        The underlying block cipher, a module from ``Crypto.Cipher``.

    All keywords are passed to the underlying block cipher.
    See the relevant documentation for details (at least ``key`` will need
    to be present"""

    cipher_state = factory._create_base_cipher(kwargs)
    cipher_state.block_size = factory.block_size
    if kwargs:
        raise TypeError("Unknown parameters for ECB: %s" % str(kwargs))
    return EcbMode(cipher_state)
