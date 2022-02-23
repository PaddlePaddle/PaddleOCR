# -*- coding: utf-8 -*-
#
#  Cipher/mode_ctr.py : CTR mode
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
Counter (CTR) mode.
"""

__all__ = ['CtrMode']

import struct

from Crypto.Util._raw_api import (load_pycryptodome_raw_lib, VoidPointer,
                                  create_string_buffer, get_raw_buffer,
                                  SmartPointer, c_size_t, c_uint8_ptr,
                                  is_writeable_buffer)

from Crypto.Random import get_random_bytes
from Crypto.Util.py3compat import _copy_bytes, is_native_int
from Crypto.Util.number import long_to_bytes

raw_ctr_lib = load_pycryptodome_raw_lib("Crypto.Cipher._raw_ctr", """
                    int CTR_start_operation(void *cipher,
                                            uint8_t   initialCounterBlock[],
                                            size_t    initialCounterBlock_len,
                                            size_t    prefix_len,
                                            unsigned  counter_len,
                                            unsigned  littleEndian,
                                            void **pResult);
                    int CTR_encrypt(void *ctrState,
                                    const uint8_t *in,
                                    uint8_t *out,
                                    size_t data_len);
                    int CTR_decrypt(void *ctrState,
                                    const uint8_t *in,
                                    uint8_t *out,
                                    size_t data_len);
                    int CTR_stop_operation(void *ctrState);"""
                                        )


class CtrMode(object):
    """*CounTeR (CTR)* mode.

    This mode is very similar to ECB, in that
    encryption of one block is done independently of all other blocks.

    Unlike ECB, the block *position* contributes to the encryption
    and no information leaks about symbol frequency.

    Each message block is associated to a *counter* which
    must be unique across all messages that get encrypted
    with the same key (not just within the same message).
    The counter is as big as the block size.

    Counters can be generated in several ways. The most
    straightword one is to choose an *initial counter block*
    (which can be made public, similarly to the *IV* for the
    other modes) and increment its lowest **m** bits by one
    (modulo *2^m*) for each block. In most cases, **m** is
    chosen to be half the block size.

    See `NIST SP800-38A`_, Section 6.5 (for the mode) and
    Appendix B (for how to manage the *initial counter block*).

    .. _`NIST SP800-38A` : http://csrc.nist.gov/publications/nistpubs/800-38a/sp800-38a.pdf

    :undocumented: __init__
    """

    def __init__(self, block_cipher, initial_counter_block,
                 prefix_len, counter_len, little_endian):
        """Create a new block cipher, configured in CTR mode.

        :Parameters:
          block_cipher : C pointer
            A smart pointer to the low-level block cipher instance.

          initial_counter_block : bytes/bytearray/memoryview
            The initial plaintext to use to generate the key stream.

            It is as large as the cipher block, and it embeds
            the initial value of the counter.

            This value must not be reused.
            It shall contain a nonce or a random component.
            Reusing the *initial counter block* for encryptions
            performed with the same key compromises confidentiality.

          prefix_len : integer
            The amount of bytes at the beginning of the counter block
            that never change.

          counter_len : integer
            The length in bytes of the counter embedded in the counter
            block.

          little_endian : boolean
            True if the counter in the counter block is an integer encoded
            in little endian mode. If False, it is big endian.
        """

        if len(initial_counter_block) == prefix_len + counter_len:
            self.nonce = _copy_bytes(None, prefix_len, initial_counter_block)
            """Nonce; not available if there is a fixed suffix"""

        self._state = VoidPointer()
        result = raw_ctr_lib.CTR_start_operation(block_cipher.get(),
                                                 c_uint8_ptr(initial_counter_block),
                                                 c_size_t(len(initial_counter_block)),
                                                 c_size_t(prefix_len),
                                                 counter_len,
                                                 little_endian,
                                                 self._state.address_of())
        if result:
            raise ValueError("Error %X while instantiating the CTR mode"
                             % result)

        # Ensure that object disposal of this Python object will (eventually)
        # free the memory allocated by the raw library for the cipher mode
        self._state = SmartPointer(self._state.get(),
                                   raw_ctr_lib.CTR_stop_operation)

        # Memory allocated for the underlying block cipher is now owed
        # by the cipher mode
        block_cipher.release()

        self.block_size = len(initial_counter_block)
        """The block size of the underlying cipher, in bytes."""

        self._next = [self.encrypt, self.decrypt]

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
        self._next = [self.encrypt]

        if output is None:
            ciphertext = create_string_buffer(len(plaintext))
        else:
            ciphertext = output

            if not is_writeable_buffer(output):
                raise TypeError("output must be a bytearray or a writeable memoryview")

            if len(plaintext) != len(output):
                raise ValueError("output must have the same length as the input"
                                 "  (%d bytes)" % len(plaintext))

        result = raw_ctr_lib.CTR_encrypt(self._state.get(),
                                         c_uint8_ptr(plaintext),
                                         c_uint8_ptr(ciphertext),
                                         c_size_t(len(plaintext)))
        if result:
            if result == 0x60002:
                raise OverflowError("The counter has wrapped around in"
                                    " CTR mode")
            raise ValueError("Error %X while encrypting in CTR mode" % result)

        if output is None:
            return get_raw_buffer(ciphertext)
        else:
            return None

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
          If ``output`` is ``None``, the plaintext is returned as ``bytes``.
          Otherwise, ``None``.
        """

        if self.decrypt not in self._next:
            raise TypeError("decrypt() cannot be called after encrypt()")
        self._next = [self.decrypt]

        if output is None:
            plaintext = create_string_buffer(len(ciphertext))
        else:
            plaintext = output

            if not is_writeable_buffer(output):
                raise TypeError("output must be a bytearray or a writeable memoryview")

            if len(ciphertext) != len(output):
                raise ValueError("output must have the same length as the input"
                                 "  (%d bytes)" % len(plaintext))

        result = raw_ctr_lib.CTR_decrypt(self._state.get(),
                                         c_uint8_ptr(ciphertext),
                                         c_uint8_ptr(plaintext),
                                         c_size_t(len(ciphertext)))
        if result:
            if result == 0x60002:
                raise OverflowError("The counter has wrapped around in"
                                    " CTR mode")
            raise ValueError("Error %X while decrypting in CTR mode" % result)

        if output is None:
            return get_raw_buffer(plaintext)
        else:
            return None


def _create_ctr_cipher(factory, **kwargs):
    """Instantiate a cipher object that performs CTR encryption/decryption.

    :Parameters:
      factory : module
        The underlying block cipher, a module from ``Crypto.Cipher``.

    :Keywords:
      nonce : bytes/bytearray/memoryview
        The fixed part at the beginning of the counter block - the rest is
        the counter number that gets increased when processing the next block.
        The nonce must be such that no two messages are encrypted under the
        same key and the same nonce.

        The nonce must be shorter than the block size (it can have
        zero length; the counter is then as long as the block).

        If this parameter is not present, a random nonce will be created with
        length equal to half the block size. No random nonce shorter than
        64 bits will be created though - you must really think through all
        security consequences of using such a short block size.

      initial_value : posive integer or bytes/bytearray/memoryview
        The initial value for the counter. If not present, the cipher will
        start counting from 0. The value is incremented by one for each block.
        The counter number is encoded in big endian mode.

      counter : object
        Instance of ``Crypto.Util.Counter``, which allows full customization
        of the counter block. This parameter is incompatible to both ``nonce``
        and ``initial_value``.

    Any other keyword will be passed to the underlying block cipher.
    See the relevant documentation for details (at least ``key`` will need
    to be present).
    """

    cipher_state = factory._create_base_cipher(kwargs)

    counter = kwargs.pop("counter", None)
    nonce = kwargs.pop("nonce", None)
    initial_value = kwargs.pop("initial_value", None)
    if kwargs:
        raise TypeError("Invalid parameters for CTR mode: %s" % str(kwargs))

    if counter is not None and (nonce, initial_value) != (None, None):
        raise TypeError("'counter' and 'nonce'/'initial_value'"
                        " are mutually exclusive")

    if counter is None:
        # Crypto.Util.Counter is not used
        if nonce is None:
            if factory.block_size < 16:
                raise TypeError("Impossible to create a safe nonce for short"
                                " block sizes")
            nonce = get_random_bytes(factory.block_size // 2)
        else:
            if len(nonce) >= factory.block_size:
                raise ValueError("Nonce is too long")

        # What is not nonce is counter
        counter_len = factory.block_size - len(nonce)

        if initial_value is None:
            initial_value = 0

        if is_native_int(initial_value):
            if (1 << (counter_len * 8)) - 1 < initial_value:
                raise ValueError("Initial counter value is too large")
            initial_counter_block = nonce + long_to_bytes(initial_value, counter_len)
        else:
            if len(initial_value) != counter_len:
                raise ValueError("Incorrect length for counter byte string (%d bytes, expected %d)" %
                                 (len(initial_value), counter_len))
            initial_counter_block = nonce + initial_value

        return CtrMode(cipher_state,
                       initial_counter_block,
                       len(nonce),                     # prefix
                       counter_len,
                       False)                          # little_endian

    # Crypto.Util.Counter is used

    # 'counter' used to be a callable object, but now it is
    # just a dictionary for backward compatibility.
    _counter = dict(counter)
    try:
        counter_len = _counter.pop("counter_len")
        prefix = _counter.pop("prefix")
        suffix = _counter.pop("suffix")
        initial_value = _counter.pop("initial_value")
        little_endian = _counter.pop("little_endian")
    except KeyError:
        raise TypeError("Incorrect counter object"
                        " (use Crypto.Util.Counter.new)")

    # Compute initial counter block
    words = []
    while initial_value > 0:
        words.append(struct.pack('B', initial_value & 255))
        initial_value >>= 8
    words += [b'\x00'] * max(0, counter_len - len(words))
    if not little_endian:
        words.reverse()
    initial_counter_block = prefix + b"".join(words) + suffix

    if len(initial_counter_block) != factory.block_size:
        raise ValueError("Size of the counter block (%d bytes) must match"
                         " block size (%d)" % (len(initial_counter_block),
                                               factory.block_size))

    return CtrMode(cipher_state, initial_counter_block,
                   len(prefix), counter_len, little_endian)
