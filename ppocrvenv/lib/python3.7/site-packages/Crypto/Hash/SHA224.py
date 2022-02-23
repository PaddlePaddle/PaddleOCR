# -*- coding: utf-8 -*-
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

from Crypto.Util.py3compat import bord

from Crypto.Util._raw_api import (load_pycryptodome_raw_lib,
                                  VoidPointer, SmartPointer,
                                  create_string_buffer,
                                  get_raw_buffer, c_size_t,
                                  c_uint8_ptr)

_raw_sha224_lib = load_pycryptodome_raw_lib("Crypto.Hash._SHA224",
                        """
                        int SHA224_init(void **shaState);
                        int SHA224_destroy(void *shaState);
                        int SHA224_update(void *hs,
                                          const uint8_t *buf,
                                          size_t len);
                        int SHA224_digest(const void *shaState,
                                          uint8_t *digest,
                                          size_t digest_size);
                        int SHA224_copy(const void *src, void *dst);

                        int SHA224_pbkdf2_hmac_assist(const void *inner,
                                            const void *outer,
                                            const uint8_t *first_digest,
                                            uint8_t *final_digest,
                                            size_t iterations,
                                            size_t digest_size);
                        """)

class SHA224Hash(object):
    """A SHA-224 hash object.
    Do not instantiate directly.
    Use the :func:`new` function.

    :ivar oid: ASN.1 Object ID
    :vartype oid: string

    :ivar block_size: the size in bytes of the internal message block,
                      input to the compression function
    :vartype block_size: integer

    :ivar digest_size: the size in bytes of the resulting hash
    :vartype digest_size: integer
    """

    # The size of the resulting hash in bytes.
    digest_size = 28
    # The internal block size of the hash algorithm in bytes.
    block_size = 64
    # ASN.1 Object ID
    oid = '2.16.840.1.101.3.4.2.4'

    def __init__(self, data=None):
        state = VoidPointer()
        result = _raw_sha224_lib.SHA224_init(state.address_of())
        if result:
            raise ValueError("Error %d while instantiating SHA224"
                             % result)
        self._state = SmartPointer(state.get(),
                                   _raw_sha224_lib.SHA224_destroy)
        if data:
            self.update(data)

    def update(self, data):
        """Continue hashing of a message by consuming the next chunk of data.

        Args:
            data (byte string/byte array/memoryview): The next chunk of the message being hashed.
        """

        result = _raw_sha224_lib.SHA224_update(self._state.get(),
                                               c_uint8_ptr(data),
                                               c_size_t(len(data)))
        if result:
            raise ValueError("Error %d while hashing data with SHA224"
                             % result)

    def digest(self):
        """Return the **binary** (non-printable) digest of the message that has been hashed so far.

        :return: The hash digest, computed over the data processed so far.
                 Binary form.
        :rtype: byte string
        """

        bfr = create_string_buffer(self.digest_size)
        result = _raw_sha224_lib.SHA224_digest(self._state.get(),
                                               bfr,
                                               c_size_t(self.digest_size))
        if result:
            raise ValueError("Error %d while making SHA224 digest"
                             % result)

        return get_raw_buffer(bfr)

    def hexdigest(self):
        """Return the **printable** digest of the message that has been hashed so far.

        :return: The hash digest, computed over the data processed so far.
                 Hexadecimal encoded.
        :rtype: string
        """

        return "".join(["%02x" % bord(x) for x in self.digest()])

    def copy(self):
        """Return a copy ("clone") of the hash object.

        The copy will have the same internal state as the original hash
        object.
        This can be used to efficiently compute the digests of strings that
        share a common initial substring.

        :return: A hash object of the same type
        """

        clone = SHA224Hash()
        result = _raw_sha224_lib.SHA224_copy(self._state.get(),
                                             clone._state.get())
        if result:
            raise ValueError("Error %d while copying SHA224" % result)
        return clone

    def new(self, data=None):
        """Create a fresh SHA-224 hash object."""

        return SHA224Hash(data)


def new(data=None):
    """Create a new hash object.

    :parameter data:
        Optional. The very first chunk of the message to hash.
        It is equivalent to an early call to :meth:`SHA224Hash.update`.
    :type data: byte string/byte array/memoryview

    :Return: A :class:`SHA224Hash` hash object
    """
    return SHA224Hash().new(data)


# The size of the resulting hash in bytes.
digest_size = SHA224Hash.digest_size

# The internal block size of the hash algorithm in bytes.
block_size = SHA224Hash.block_size


def _pbkdf2_hmac_assist(inner, outer, first_digest, iterations):
    """Compute the expensive inner loop in PBKDF-HMAC."""

    assert iterations > 0

    bfr = create_string_buffer(len(first_digest));
    result = _raw_sha224_lib.SHA224_pbkdf2_hmac_assist(
                    inner._state.get(),
                    outer._state.get(),
                    first_digest,
                    bfr,
                    c_size_t(iterations),
                    c_size_t(len(first_digest)))

    if result:
        raise ValueError("Error %d with PBKDF2-HMAC assist for SHA224" % result)

    return get_raw_buffer(bfr)
