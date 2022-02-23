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

from Crypto.Util.py3compat import *

from Crypto.Util._raw_api import (load_pycryptodome_raw_lib,
                                  VoidPointer, SmartPointer,
                                  create_string_buffer,
                                  get_raw_buffer, c_size_t,
                                  c_uint8_ptr)

_raw_md5_lib = load_pycryptodome_raw_lib("Crypto.Hash._MD5",
                        """
                        #define MD5_DIGEST_SIZE 16

                        int MD5_init(void **shaState);
                        int MD5_destroy(void *shaState);
                        int MD5_update(void *hs,
                                          const uint8_t *buf,
                                          size_t len);
                        int MD5_digest(const void *shaState,
                                          uint8_t digest[MD5_DIGEST_SIZE]);
                        int MD5_copy(const void *src, void *dst);

                        int MD5_pbkdf2_hmac_assist(const void *inner,
                                            const void *outer,
                                            const uint8_t first_digest[MD5_DIGEST_SIZE],
                                            uint8_t final_digest[MD5_DIGEST_SIZE],
                                            size_t iterations);
                        """)

class MD5Hash(object):
    """A MD5 hash object.
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
    digest_size = 16
    # The internal block size of the hash algorithm in bytes.
    block_size = 64
    # ASN.1 Object ID
    oid = "1.2.840.113549.2.5"

    def __init__(self, data=None):
        state = VoidPointer()
        result = _raw_md5_lib.MD5_init(state.address_of())
        if result:
            raise ValueError("Error %d while instantiating MD5"
                             % result)
        self._state = SmartPointer(state.get(),
                                   _raw_md5_lib.MD5_destroy)
        if data:
            self.update(data)

    def update(self, data):
        """Continue hashing of a message by consuming the next chunk of data.

        Args:
            data (byte string/byte array/memoryview): The next chunk of the message being hashed.
        """

        result = _raw_md5_lib.MD5_update(self._state.get(),
                                         c_uint8_ptr(data),
                                         c_size_t(len(data)))
        if result:
            raise ValueError("Error %d while instantiating MD5"
                             % result)

    def digest(self):
        """Return the **binary** (non-printable) digest of the message that has been hashed so far.

        :return: The hash digest, computed over the data processed so far.
                 Binary form.
        :rtype: byte string
        """

        bfr = create_string_buffer(self.digest_size)
        result = _raw_md5_lib.MD5_digest(self._state.get(),
                                           bfr)
        if result:
            raise ValueError("Error %d while instantiating MD5"
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

        clone = MD5Hash()
        result = _raw_md5_lib.MD5_copy(self._state.get(),
                                         clone._state.get())
        if result:
            raise ValueError("Error %d while copying MD5" % result)
        return clone

    def new(self, data=None):
        """Create a fresh SHA-1 hash object."""

        return MD5Hash(data)


def new(data=None):
    """Create a new hash object.

    :parameter data:
        Optional. The very first chunk of the message to hash.
        It is equivalent to an early call to :meth:`MD5Hash.update`.
    :type data: byte string/byte array/memoryview

    :Return: A :class:`MD5Hash` hash object
    """
    return MD5Hash().new(data)

# The size of the resulting hash in bytes.
digest_size = 16

# The internal block size of the hash algorithm in bytes.
block_size = 64


def _pbkdf2_hmac_assist(inner, outer, first_digest, iterations):
    """Compute the expensive inner loop in PBKDF-HMAC."""

    assert len(first_digest) == digest_size
    assert iterations > 0

    bfr = create_string_buffer(digest_size);
    result = _raw_md5_lib.MD5_pbkdf2_hmac_assist(
                    inner._state.get(),
                    outer._state.get(),
                    first_digest,
                    bfr,
                    c_size_t(iterations))

    if result:
        raise ValueError("Error %d with PBKDF2-HMAC assis for MD5" % result)

    return get_raw_buffer(bfr)
